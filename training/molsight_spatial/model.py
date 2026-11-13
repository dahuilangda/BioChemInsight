"""Spatial detection heads over MolSight image features.

The regular path consumes only image features and learned queries. Optional
denoising anchors run through an independent attention path with shared weights;
they never enter regular-query attention or determine its cardinality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpatialDetectorConfig:
    feature_dim: int = 512
    dim: int = 256
    layers: int = 4
    heads: int = 8
    points: int = 4
    queries: int = 512

    def __post_init__(self) -> None:
        for name in ("feature_dim", "dim", "layers", "heads", "points", "queries"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")


def inverse_sigmoid(points: Tensor) -> Tensor:
    points = points.clamp(1e-5, 1 - 1e-5)
    return points.log() - torch.log1p(-points)


class CoordFourierEncoder(nn.Module):
    """True sine/cosine features of both normalized coordinate components."""

    def __init__(self, dim: int, frequencies=(1., 2., 4., 8.)):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))
        self.project = nn.Sequential(
            nn.Linear(4 * len(frequencies), dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def encode(self, points: Tensor) -> Tensor:
        phase = (points * 2 - 1).unsqueeze(-1) * self.frequencies * math.pi
        return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)

    def forward(self, points: Tensor) -> Tensor:
        return self.project(self.encode(points))


class PairGeometry(nn.Module):
    def __init__(self, frequencies=(.5, 1., 2., 4., 8., 16.)):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))
        self.output_dim = 2 * len(frequencies)

    def forward(self, points: Tensor) -> Tensor:
        difference = points.unsqueeze(2) - points.unsqueeze(1)
        distance = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
        phase = distance * self.frequencies * math.pi
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class DeformableCrossAttention(nn.Module):
    """Per-head spatial sampling with distinct radial points from initialization."""

    def __init__(self, dim: int, heads: int = 8, points: int = 4):
        super().__init__()
        if dim % heads or min(heads, points) < 1:
            raise ValueError("Invalid deformable-attention dimensions")
        self.heads, self.points = heads, points
        self.offsets = nn.Linear(dim, heads * points * 2)
        self.weights = nn.Linear(dim, heads * points)
        self.value_proj = nn.Conv2d(dim, dim, 1)
        self.output_proj = nn.Linear(dim, dim)
        # Uniform weights are safe only when their sampled values differ.
        nn.init.zeros_(self.offsets.weight)
        angles = torch.arange(heads, dtype=torch.float32) * (2 * math.pi / heads)
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        radii = torch.arange(1, points + 1, dtype=torch.float32) / points
        radial = directions[:, None, :] * radii[None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(radial.flatten())
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)

    def sampling_locations(self, queries: Tensor, references: Tensor) -> Tensor:
        batch, count, _ = queries.shape
        offsets = self.offsets(queries).reshape(batch, count, self.heads, self.points, 2)
        return (references[:, :, None, None, :] + offsets.tanh() * .08).clamp(0, 1)

    def forward(self, queries: Tensor, references: Tensor, features: Tensor) -> Tensor:
        batch, count, dim = queries.shape
        _, _, height, width = features.shape
        per_head = dim // self.heads
        values = self.value_proj(features).reshape(batch, self.heads, per_head, height, width)
        values = values.reshape(batch * self.heads, per_head, height, width)
        locations = self.sampling_locations(queries, references)
        grid = (locations * 2 - 1).permute(0, 2, 1, 3, 4)
        grid = grid.reshape(batch * self.heads, count, self.points, 2)
        # grid_sample has no CPU BF16 implementation; FP32 is also a stable
        # sampling boundary under CUDA autocast. Both casts retain gradients.
        sampled = F.grid_sample(values.float(), grid.float(), mode="bilinear",
                                padding_mode="border", align_corners=False)
        weights = self.weights(queries).reshape(batch, count, self.heads, self.points)
        weights = weights.softmax(-1).permute(0, 2, 1, 3)
        weights = weights.reshape(batch * self.heads, 1, count, self.points)
        contexts = (sampled * weights).sum(-1)
        contexts = contexts.reshape(batch, self.heads, per_head, count)
        contexts = contexts.permute(0, 3, 1, 2).reshape(batch, count, dim)
        return self.output_proj(contexts.to(queries.dtype))


class SpatialLayer(nn.Module):
    def __init__(self, config: SpatialDetectorConfig, geometry_dim: int):
        super().__init__()
        dim = config.dim
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, config.heads, batch_first=True)
        self.geometry_bias = nn.Sequential(
            nn.Linear(geometry_dim, dim), nn.GELU(), nn.Linear(dim, config.heads)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = DeformableCrossAttention(dim, config.heads, config.points)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.coord_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        nn.init.zeros_(self.coord_head[-1].weight)
        nn.init.zeros_(self.coord_head[-1].bias)

    def forward(self, queries: Tensor, references: Tensor, features: Tensor,
                geometry: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        batch, count, _ = queries.shape
        bias = self.geometry_bias(geometry).permute(0, 3, 1, 2)
        bias = bias.reshape(batch * self.heads, count, count)
        normalized = self.norm1(queries)
        attended, _ = self.self_attn(normalized + position, normalized + position,
                                     normalized, attn_mask=bias, need_weights=False)
        queries = queries + attended
        queries = queries + self.cross_attn(self.norm2(queries) + position, references, features)
        queries = queries + self.ffn(self.norm3(queries))
        coordinates = (inverse_sigmoid(references) + self.coord_head(queries)).sigmoid()
        return queries, coordinates


class LowRankPairHead(nn.Module):
    """Class-specific biaffine scores without a B×Q×Q×(2D) feature tensor."""

    def __init__(self, dim: int, classes: int, geometry_dim: int, rank: int = 32):
        super().__init__()
        self.classes, self.rank = classes, min(rank, dim)
        self.head = nn.Linear(dim, classes * self.rank)
        self.tail = nn.Linear(dim, classes * self.rank)
        self.first = nn.Linear(dim, classes)
        self.second = nn.Linear(dim, classes)
        self.geometry = nn.Linear(geometry_dim, classes)

    def forward(self, hidden: Tensor, geometry: Tensor) -> Tensor:
        batch, count, _ = hidden.shape
        head = self.head(hidden).reshape(batch, count, self.classes, self.rank)
        tail = self.tail(hidden).reshape(batch, count, self.classes, self.rank)
        score = torch.einsum("bicr,bjcr->bijc", head, tail) / math.sqrt(self.rank)
        score = score + self.first(hidden).unsqueeze(2) + self.second(hidden).unsqueeze(1)
        return score + self.geometry(geometry)


class SpatialDetector(nn.Module):
    """Fixed-cardinality visual proposals and explicit chemical-state logits.

    Atomic-number classes are 0..118, charge classes map to -8..8, isotope
    classes to 0..512, and H/radical classes to 0..8. All returned fields are
    logits except normalized coordinates and their differentiable auxiliaries.
    Pair heads are directed during training; chemical symmetrization belongs
    to the decoder, whereas wedges preserve their explicit apex direction.
    """

    STATE_CLASSES = {"elements": 119, "charge": 17, "isotope": 513,
                     "explicit_h": 9, "no_implicit": 2, "radicals": 9, "aromatic": 2}

    def __init__(self, config: SpatialDetectorConfig | None = None):
        super().__init__()
        self.config = config or SpatialDetectorConfig()
        config = self.config
        self.feature_proj = nn.Sequential(nn.Conv2d(config.feature_dim, config.dim, 1),
                                          nn.GroupNorm(math.gcd(8, config.dim), config.dim))
        self.coord_encoder = CoordFourierEncoder(config.dim)
        self.geometry = PairGeometry()
        self.query_embed = nn.Embedding(config.queries, config.dim)
        self.dn_query_embed = nn.Parameter(torch.empty(1, config.dim))
        nn.init.normal_(self.query_embed.weight, std=.02)
        nn.init.normal_(self.dn_query_embed, std=.02)
        side = math.ceil(math.sqrt(config.queries))
        centers = (torch.arange(side, dtype=torch.float32) + .5) / side
        y, x = torch.meshgrid(centers, centers, indexing="ij")
        anchors = torch.stack((x.flatten(), y.flatten()), -1)[:config.queries]
        self.anchor_logits = nn.Parameter(inverse_sigmoid(anchors))
        self.layers = nn.ModuleList([SpatialLayer(config, self.geometry.output_dim)
                                     for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.dim)
        self.objectness = nn.Linear(config.dim, 1)
        self.state_heads = nn.ModuleDict({name: nn.Linear(config.dim, count)
                                          for name, count in self.STATE_CLASSES.items()})
        self.edges = LowRankPairHead(config.dim, 5, self.geometry.output_dim)
        self.wedges = LowRankPairHead(config.dim, 3, self.geometry.output_dim)
        self.double_stereo = LowRankPairHead(config.dim, 2, self.geometry.output_dim)

    def _run_queries(self, features: Tensor, references: Tensor, queries: Tensor) -> dict:
        auxiliary = []
        for layer in self.layers:
            # Both signals follow the current refined coordinates, not the
            # initial grid/noisy anchors from before previous refinement.
            geometry = self.geometry(references)
            position = self.coord_encoder(references)
            queries, references = layer(queries, references, features, geometry, position)
            auxiliary.append(references)
        queries = self.final_norm(queries)
        geometry = self.geometry(references)
        return {"objectness": self.objectness(queries).squeeze(-1),
                "coords": references, "aux_coords": auxiliary,
                **{name: head(queries) for name, head in self.state_heads.items()},
                "edges": self.edges(queries, geometry),
                "wedges": self.wedges(queries, geometry),
                "double_stereo": self.double_stereo(queries, geometry)}

    def forward(self, features: Tensor, dn_points: Tensor | None = None) -> dict:
        if features.ndim != 4 or features.shape[1] != self.config.feature_dim:
            raise ValueError("Expected B×feature_dim×H×W MolSight features")
        if min(features.shape[0], features.shape[2], features.shape[3]) < 1:
            raise ValueError("Image features must have nonempty batch/spatial dimensions")
        if dn_points is not None:
            if (dn_points.ndim != 3 or dn_points.shape[0] != features.shape[0]
                    or dn_points.shape[-1] != 2 or dn_points.shape[1] < 1):
                raise ValueError("Denoising anchors must have shape B×N×2")
            if dn_points.device != features.device or not torch.isfinite(dn_points).all():
                raise ValueError("Denoising anchors must be finite and on the feature device")
            if dn_points.lt(0).any() or dn_points.gt(1).any():
                raise ValueError("Denoising anchors must be normalized to [0,1]")
        mapped = self.feature_proj(features)
        batch = features.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        references = self.anchor_logits.sigmoid().unsqueeze(0).expand(batch, -1, -1)
        result = self._run_queries(mapped, references, queries)
        if dn_points is not None:
            dn_queries = self.dn_query_embed.unsqueeze(0).expand(batch, dn_points.shape[1], -1)
            result["dn"] = self._run_queries(mapped, dn_points, dn_queries)
        return result
