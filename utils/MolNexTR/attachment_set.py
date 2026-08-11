"""Set-prediction head for routed Markush/fragment attachment experts.

The autoregressive MolNexTR stream is a strong molecular backbone parser but
is a poor place to learn sparse variable endpoints from coordinate-free patent
labels.  This module follows DETR's set-prediction formulation: learned queries
predict attachment objectness, image-space points, bonded/standalone state,
bond type, and global cardinality directly from frozen encoder features.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttachmentSetHead(nn.Module):
    """DETR-style sparse attachment detector over MolNexTR encoder tokens."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 256,
        num_queries: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        max_count: int = 30,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.max_count = int(max_count)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_embedding = nn.Embedding(self.num_queries, self.hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.objectness = nn.Linear(self.hidden_dim, 1)
        self.point = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.bonded = nn.Linear(self.hidden_dim, 2)
        self.bond_type = nn.Linear(self.hidden_dim, 7)
        self.cardinality = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.max_count + 1),
        )

    @staticmethod
    def _positions(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height = max(1, int(math.floor(math.sqrt(max(1, length)))))
        width = max(1, int(math.ceil(length / height)))
        index = torch.arange(length, device=device)
        y = torch.div(index, width, rounding_mode="floor").to(dtype=dtype)
        x = (index % width).to(dtype=dtype)
        x = x / max(1, width - 1)
        y = y / max(1, height - 1)
        return torch.stack([x, y, x * x, y * y], dim=-1)

    @staticmethod
    def _positions_2d(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, max(1, height), device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, max(1, width), device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([x, y, x * x, y * y], dim=-1).reshape(-1, 4)

    def _outputs_from_memory(
        self,
        memory: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        queries = self.query_embedding.weight.to(dtype=memory.dtype).unsqueeze(0)
        queries = queries.expand(memory.size(0), -1, -1)
        decoded = self.output_norm(self.decoder(queries, memory))
        global_memory = memory.mean(dim=1)
        global_query = decoded.amax(dim=1)
        outputs = {
            "objectness_logits": self.objectness(decoded).squeeze(-1),
            "points": torch.sigmoid(self.point(decoded)),
            "bonded_logits": self.bonded(decoded),
            "bond_type_logits": self.bond_type(decoded),
            "cardinality_logits": self.cardinality(
                torch.cat([global_memory, global_query], dim=-1)
            ),
        }
        return outputs, decoded

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.size(-1) != self.feature_dim:
            raise ValueError(
                "attachment set head expects (B,L,C) encoder features with "
                f"C={self.feature_dim}, got {tuple(features.shape)}"
            )
        memory = self.input_projection(features)
        position = self._positions(
            memory.size(1), memory.device, memory.dtype
        )
        memory = memory + self.position_projection(position).unsqueeze(0)
        outputs, _decoded = self._outputs_from_memory(memory)
        return outputs


class MultiScaleAttachmentSetHead(AttachmentSetHead):
    """Multi-scale query decoder with a learned spatial anchor pointer.

    The final MolNexTR Swin map is only 12x12. Variable labels and wavy bond
    endpoints require earlier 48x48/24x24 evidence, so the production head
    projects and concatenates the last three frozen encoder stages. A separate
    anchor point predicts the bonded backbone atom location; inference resolves
    that point against atoms decoded by frozen expert0.
    """

    def __init__(
        self,
        feature_dims: tuple[int, ...] | list[int],
        *,
        hidden_dim: int = 256,
        num_queries: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        max_count: int = 30,
        dropout: float = 0.1,
        max_feature_size: int = 32,
    ) -> None:
        dims = tuple(int(value) for value in feature_dims)
        if len(dims) < 2 or any(value <= 0 for value in dims):
            raise ValueError(
                f"multi-scale attachment head needs >=2 positive feature dims, got {dims}"
            )
        super().__init__(
            dims[-1],
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            max_count=max_count,
            dropout=dropout,
        )
        self.feature_dims = dims
        self.max_feature_size = max(4, int(max_feature_size))
        self.earlier_input_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.hidden_dim))
            for dim in dims[:-1]
        )
        self.level_embedding = nn.Parameter(
            torch.empty(len(dims), self.hidden_dim)
        )
        nn.init.normal_(self.level_embedding, std=0.02)
        self.anchor_point = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def _project_level(
        self,
        features: torch.Tensor,
        level_index: int,
    ) -> torch.Tensor:
        expected_dim = self.feature_dims[level_index]
        if features.ndim != 3 or features.size(-1) != expected_dim:
            raise ValueError(
                f"attachment feature level {level_index} expects (B,L,{expected_dim}), "
                f"got {tuple(features.shape)}"
            )
        length = int(features.size(1))
        side = int(round(math.sqrt(length)))
        if side * side != length:
            raise ValueError(
                "multi-scale attachment features must form a square map, "
                f"got length={length}"
            )
        spatial = features.transpose(1, 2).reshape(
            features.size(0), expected_dim, side, side
        )
        if side > self.max_feature_size:
            spatial = F.adaptive_avg_pool2d(
                spatial,
                (self.max_feature_size, self.max_feature_size),
            )
        height, width = spatial.shape[-2:]
        flattened = spatial.flatten(2).transpose(1, 2)
        projection = (
            self.input_projection
            if level_index == len(self.feature_dims) - 1
            else self.earlier_input_projections[level_index]
        )
        memory = projection(flattened)
        position = self._positions_2d(
            height,
            width,
            memory.device,
            memory.dtype,
        )
        return (
            memory
            + self.position_projection(position).unsqueeze(0)
            + self.level_embedding[level_index].to(dtype=memory.dtype).view(1, 1, -1)
        )

    def forward(
        self,
        features: torch.Tensor,
        multi_scale_features: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        projected_levels = self._project_multiscale_levels(multi_scale_features)
        memory = torch.cat(projected_levels, dim=1)
        outputs, decoded = self._outputs_from_memory(memory)
        outputs["anchor_points"] = torch.sigmoid(self.anchor_point(decoded))
        outputs["_query_features"] = decoded
        return outputs

    def _project_multiscale_levels(
        self,
        multi_scale_features: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    ) -> list[torch.Tensor]:
        levels = list(multi_scale_features or [])
        if len(levels) < len(self.feature_dims):
            raise ValueError(
                f"multi-scale attachment head requires {len(self.feature_dims)} encoder "
                f"levels, got {len(levels)}"
            )
        levels = levels[-len(self.feature_dims):]
        return [
            self._project_level(level, index)
            for index, level in enumerate(levels)
        ]


class MultiScaleAttachmentPointerHead(MultiScaleAttachmentSetHead):
    """Multi-scale set detector with a pointer over expert0 decoder atoms.

    Continuous image coordinates remain useful localization supervision, but
    the production edit must ultimately choose a discrete atom in expert0's
    graph. The pointer scores every decoded atom from its autoregressive hidden
    state and, when available, its image coordinate. This closes the previous
    train/deploy gap where an L1-regressed point was resolved by an arbitrary
    distance radius after decoding.
    """

    def __init__(
        self,
        feature_dims: tuple[int, ...] | list[int],
        *,
        atom_feature_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        max_count: int = 30,
        dropout: float = 0.1,
        max_feature_size: int = 48,
    ) -> None:
        super().__init__(
            feature_dims,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            max_count=max_count,
            dropout=dropout,
            max_feature_size=max_feature_size,
        )
        self.atom_feature_dim = int(atom_feature_dim)
        self.pointer_query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pointer_atom = nn.Sequential(
            nn.LayerNorm(self.atom_feature_dim),
            nn.Linear(self.atom_feature_dim, self.hidden_dim),
        )
        geometry_hidden = max(32, self.hidden_dim // 2)
        self.pointer_geometry = nn.Sequential(
            nn.Linear(5, geometry_hidden),
            nn.SiLU(),
            nn.Linear(geometry_hidden, 1),
        )
        self.dummy_pointer_query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dummy_pointer_atom = nn.Sequential(
            nn.LayerNorm(self.atom_feature_dim),
            nn.Linear(self.atom_feature_dim, self.hidden_dim),
        )
        self.dummy_pointer_geometry = nn.Sequential(
            nn.Linear(5, geometry_hidden),
            nn.SiLU(),
            nn.Linear(geometry_hidden, 1),
        )
        self.pointer_scale = self.hidden_dim ** -0.5

    def _atom_pointer_logits(
        self,
        outputs: dict[str, torch.Tensor],
        atom_features: torch.Tensor,
        *,
        query_projection: nn.Module,
        atom_projection: nn.Module,
        geometry_projection: nn.Module,
        query_points: torch.Tensor,
        atom_coords: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_features = outputs.get("_query_features")
        if query_features is None:
            raise ValueError("pointer head outputs are missing query features")
        if atom_features.ndim != 3 or atom_features.size(-1) != self.atom_feature_dim:
            raise ValueError(
                "atom pointer expects (B,N,C) decoder features with "
                f"C={self.atom_feature_dim}, got {tuple(atom_features.shape)}"
            )
        query = query_projection(query_features)
        atom = atom_projection(atom_features)
        logits = torch.einsum("bqh,bnh->bqn", query, atom) * self.pointer_scale

        if atom_coords is not None and atom_coords.numel():
            coords = atom_coords.to(device=logits.device, dtype=logits.dtype)
            if coords.size(1) < atom_features.size(1):
                coords = F.pad(
                    coords,
                    (0, 0, 0, atom_features.size(1) - coords.size(1)),
                    value=-1.0,
                )
            coords = coords[:, : atom_features.size(1)]
            points = query_points.to(dtype=logits.dtype)
            delta = coords.unsqueeze(1) - points.unsqueeze(2)
            geometry = torch.cat(
                [delta, delta.abs(), delta.square().sum(dim=-1, keepdim=True)],
                dim=-1,
            )
            geometry_logits = geometry_projection(geometry).squeeze(-1)
            valid_coords = coords.ge(0.0).all(dim=-1) & coords.le(1.0).all(dim=-1)
            logits = logits + torch.where(
                valid_coords.unsqueeze(1),
                geometry_logits,
                torch.zeros_like(geometry_logits),
            )

        if atom_mask is not None:
            mask = atom_mask.to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~mask.unsqueeze(1), -1.0e4)
        return logits

    def anchor_pointer_logits(
        self,
        outputs: dict[str, torch.Tensor],
        atom_features: torch.Tensor,
        *,
        atom_coords: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._atom_pointer_logits(
            outputs,
            atom_features,
            query_projection=self.pointer_query,
            atom_projection=self.pointer_atom,
            geometry_projection=self.pointer_geometry,
            query_points=outputs["anchor_points"],
            atom_coords=atom_coords,
            atom_mask=atom_mask,
        )

    def dummy_pointer_logits(
        self,
        outputs: dict[str, torch.Tensor],
        atom_features: torch.Tensor,
        *,
        atom_coords: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match each visual attachment query to its decoder-owned dummy atom."""
        return self._atom_pointer_logits(
            outputs,
            atom_features,
            query_projection=self.dummy_pointer_query,
            atom_projection=self.dummy_pointer_atom,
            geometry_projection=self.dummy_pointer_geometry,
            query_points=outputs["points"],
            atom_coords=atom_coords,
            atom_mask=atom_mask,
        )


class MultiScaleAttachmentHeatmapPointerHead(MultiScaleAttachmentPointerHead):
    """Decoder-atom pointer plus query-conditioned dense point localization.

    The categorical atom pointer resolves attachment topology. A separate
    48x48 query heatmap models the visible dummy/R-label center as a spatial
    distribution instead of forcing one MLP to regress a potentially
    multi-modal coordinate. The inherited continuous point head supplies a
    warm-started spatial prior; a learned sub-cell residual removes grid
    quantization at deployment.
    """

    def __init__(
        self,
        feature_dims: tuple[int, ...] | list[int],
        *,
        atom_feature_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        max_count: int = 30,
        dropout: float = 0.1,
        max_feature_size: int = 48,
        heatmap_logit_scale: float = 4.0,
        heatmap_prior_precision: float = 8.0,
    ) -> None:
        super().__init__(
            feature_dims,
            atom_feature_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            max_count=max_count,
            dropout=dropout,
            max_feature_size=max_feature_size,
        )
        self.heatmap_logit_scale = float(heatmap_logit_scale)
        self.heatmap_prior_precision = float(heatmap_prior_precision)
        if self.heatmap_logit_scale <= 0.0:
            raise ValueError("heatmap_logit_scale must be positive")
        if self.heatmap_prior_precision < 0.0:
            raise ValueError("heatmap_prior_precision must be non-negative")
        self.point_query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.point_memory = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.point_offset = nn.Linear(self.hidden_dim, 2)
        nn.init.eye_(self.point_query[-1].weight)
        nn.init.zeros_(self.point_query[-1].bias)
        nn.init.eye_(self.point_memory[-1].weight)
        nn.init.zeros_(self.point_memory[-1].bias)
        nn.init.zeros_(self.point_offset.weight)
        nn.init.zeros_(self.point_offset.bias)

    def forward(
        self,
        features: torch.Tensor,
        multi_scale_features: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        projected_levels = self._project_multiscale_levels(multi_scale_features)
        memory = torch.cat(projected_levels, dim=1)
        outputs, decoded = self._outputs_from_memory(memory)
        coarse_points = outputs["points"]
        outputs["coarse_points"] = coarse_points
        outputs["anchor_points"] = torch.sigmoid(self.anchor_point(decoded))
        outputs["_query_features"] = decoded

        high_resolution = projected_levels[0]
        side = int(round(math.sqrt(int(high_resolution.size(1)))))
        if side * side != int(high_resolution.size(1)):
            raise ValueError(
                "attachment heatmap memory must form a square map, "
                f"got length={int(high_resolution.size(1))}"
            )
        grid = self._positions_2d(
            side,
            side,
            high_resolution.device,
            high_resolution.dtype,
        )[:, :2]
        query = self.point_query(decoded)
        spatial = self.point_memory(high_resolution)
        logits = (
            torch.einsum("bqh,bnh->bqn", query, spatial)
            * (self.hidden_dim ** -0.5)
            * self.heatmap_logit_scale
        )
        delta = grid.view(1, 1, -1, 2) - coarse_points.unsqueeze(2)
        logits = logits - self.heatmap_prior_precision * delta.square().sum(dim=-1)
        probabilities_float = logits.float().softmax(dim=-1)
        probabilities = probabilities_float.to(dtype=decoded.dtype)
        dense_points = torch.einsum(
            "bqn,nc->bqc", probabilities, grid
        )
        normalized_entropy = -(
            probabilities_float
            * probabilities_float.clamp_min(1.0e-8).log()
        ).sum(dim=-1) / max(1.0, math.log(max(2, int(logits.size(-1)))))
        heatmap_confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)
        cell_scale = decoded.new_tensor(
            [0.5 / max(1, side - 1), 0.5 / max(1, side - 1)]
        )
        offsets = torch.tanh(self.point_offset(decoded)) * cell_scale
        outputs["points"] = (dense_points + offsets).clamp(0.0, 1.0)
        outputs["point_heatmap_logits"] = logits.reshape(
            logits.size(0), logits.size(1), side, side
        )
        outputs["point_heatmap_confidence"] = heatmap_confidence
        outputs["point_heatmap_peak_probability"] = probabilities_float.amax(
            dim=-1
        )
        return outputs


def _bilinear_spatial_cross_entropy(
    logits: torch.Tensor,
    target_points: torch.Tensor,
) -> torch.Tensor:
    """Dimension-normalized spatial CE with bilinear sub-cell targets."""
    matrix = _bilinear_spatial_nll_matrix(logits, target_points)
    if matrix.size(0) != matrix.size(1):
        raise ValueError(
            "matched spatial CE requires one heatmap per target, got "
            f"{tuple(matrix.shape)}"
        )
    return matrix.diagonal().mean()


def _bilinear_spatial_nll_matrix(
    logits: torch.Tensor,
    target_points: torch.Tensor,
) -> torch.Tensor:
    """Return normalized bilinear NLL for every heatmap/target pair."""
    if logits.ndim != 3:
        raise ValueError(f"point heatmap logits must be (N,H,W), got {logits.shape}")
    height, width = int(logits.size(-2)), int(logits.size(-1))
    targets = target_points.float().clamp(0.0, 1.0)
    x = targets[:, 0] * max(1, width - 1)
    y = targets[:, 1] * max(1, height - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    fx = x - x0.float()
    fy = y - y0.float()
    indices = torch.stack(
        [
            y0 * width + x0,
            y0 * width + x1,
            y1 * width + x0,
            y1 * width + x1,
        ],
        dim=1,
    )
    weights = torch.stack(
        [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ],
        dim=1,
    )
    log_probabilities = F.log_softmax(logits.float().flatten(1), dim=1)
    gathered = log_probabilities[:, indices]
    normalizer = max(1.0, math.log(max(2, height * width)))
    return -(weights.unsqueeze(0) * gathered).sum(dim=-1) / normalizer


def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
        return (
            torch.as_tensor(row, dtype=torch.long, device=cost.device),
            torch.as_tensor(col, dtype=torch.long, device=cost.device),
        )
    except Exception:
        # Deterministic fallback for minimal runtime environments. Formal
        # training installs scipy and therefore uses exact Hungarian matching.
        work = cost.detach().clone()
        rows, cols = [], []
        for _ in range(min(work.size(0), work.size(1))):
            flat = int(work.argmin().item())
            row = flat // work.size(1)
            col = flat % work.size(1)
            rows.append(row)
            cols.append(col)
            work[row, :] = float("inf")
            work[:, col] = float("inf")
        return (
            torch.tensor(rows, dtype=torch.long, device=cost.device),
            torch.tensor(cols, dtype=torch.long, device=cost.device),
        )


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * ce


def attachment_set_loss(
    outputs_by_expert: list[dict[str, torch.Tensor]],
    *,
    structure_labels: torch.Tensor,
    points: torch.Tensor,
    point_mask: torch.Tensor,
    set_complete: torch.Tensor,
    counts: torch.Tensor,
    bonded: torch.Tensor | None = None,
    bond_types: torch.Tensor | None = None,
    anchor_points: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    anchor_indices: torch.Tensor | None = None,
    dummy_indices: torch.Tensor | None = None,
    atom_candidate_mask: torch.Tensor | None = None,
    anchor_candidate_mask: torch.Tensor | None = None,
    dummy_candidate_mask: torch.Tensor | None = None,
    point_cost_weight: float = 5.0,
    point_loss_weight: float = 5.0,
    objectness_loss_weight: float = 1.0,
    cardinality_loss_weight: float = 1.0,
    relation_loss_weight: float = 1.0,
    anchor_loss_weight: float = 5.0,
    pointer_loss_weight: float = 3.0,
    dummy_pointer_loss_weight: float = 3.0,
    heatmap_loss_weight: float = 2.0,
    heatmap_cost_weight: float = 2.0,
    heatmap_diversity_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Hungarian set loss with partial-label support for real patent OCR cells.

    Complete sets supervise unmatched queries as ``no object``. Partial sets
    supervise matched positives only, while the independently known dummy count
    trains cardinality on every row. This avoids treating an unmatched real OCR
    variable as a negative label.
    """

    if not outputs_by_expert:
        zero = points.sum() * 0.0
        return zero, {}
    device = points.device
    labels = structure_labels.to(device).long()
    point_mask = point_mask.to(device, dtype=torch.bool)
    set_complete = set_complete.to(device, dtype=torch.bool)
    counts = counts.to(device).long()
    if anchor_candidate_mask is None:
        anchor_candidate_mask = atom_candidate_mask
    if dummy_candidate_mask is None:
        dummy_candidate_mask = atom_candidate_mask
    components: dict[str, list[torch.Tensor]] = {
        "objectness": [],
        "point": [],
        "cardinality": [],
        "bonded": [],
        "bond_type": [],
        "anchor": [],
        "pointer": [],
        "dummy_pointer": [],
        "heatmap": [],
        "heatmap_diversity": [],
    }

    for sidecar_index, outputs in enumerate(outputs_by_expert, start=1):
        active_rows = labels.eq(sidecar_index) | labels.eq(0)
        for batch_index in torch.nonzero(active_rows, as_tuple=False).flatten().tolist():
            logits = outputs["objectness_logits"][batch_index]
            predicted_points = outputs["points"][batch_index]
            target_indices = torch.nonzero(
                point_mask[batch_index], as_tuple=False
            ).flatten()
            matched_queries = torch.empty(0, dtype=torch.long, device=device)
            matched_targets = torch.empty(0, dtype=torch.long, device=device)
            if target_indices.numel():
                target_points = points[batch_index, target_indices]
                cost = (
                    float(point_cost_weight)
                    * torch.cdist(predicted_points.float(), target_points.float(), p=1)
                    - torch.sigmoid(logits.float()).unsqueeze(1)
                )
                if outputs.get("point_heatmap_logits") is not None:
                    cost = cost + float(heatmap_cost_weight) * (
                        _bilinear_spatial_nll_matrix(
                            outputs["point_heatmap_logits"][batch_index],
                            target_points,
                        )
                    )
                matched_queries, local_targets = _linear_sum_assignment(cost)
                matched_targets = target_indices[local_targets]

            object_targets = torch.zeros_like(logits)
            if matched_queries.numel():
                object_targets[matched_queries] = 1.0
            if bool(set_complete[batch_index]):
                object_mask = torch.ones_like(logits, dtype=torch.bool)
            else:
                object_mask = torch.zeros_like(logits, dtype=torch.bool)
                object_mask[matched_queries] = True
            if bool(object_mask.any()):
                components["objectness"].append(
                    _focal_bce_with_logits(
                        logits[object_mask].float(),
                        object_targets[object_mask].float(),
                    ).mean()
                )
            if matched_queries.numel():
                components["point"].append(
                    F.smooth_l1_loss(
                        predicted_points[matched_queries].float(),
                        points[batch_index, matched_targets].float(),
                    )
                )
                if outputs.get("point_heatmap_logits") is not None:
                    components["heatmap"].append(
                        _bilinear_spatial_cross_entropy(
                            outputs["point_heatmap_logits"][
                                batch_index, matched_queries
                            ],
                            points[batch_index, matched_targets],
                        )
                    )
                if bonded is not None:
                    bonded_targets = bonded[batch_index, matched_targets].long()
                    valid = bonded_targets.ge(0)
                    if bool(valid.any()):
                        components["bonded"].append(
                            F.cross_entropy(
                                outputs["bonded_logits"][batch_index, matched_queries[valid]].float(),
                                bonded_targets[valid],
                            )
                        )
                if bond_types is not None:
                    bond_targets = bond_types[batch_index, matched_targets].long()
                    valid = bond_targets.ge(0)
                    if bool(valid.any()):
                        components["bond_type"].append(
                            F.cross_entropy(
                                outputs["bond_type_logits"][batch_index, matched_queries[valid]].float(),
                                bond_targets[valid].clamp(min=0, max=6),
                            )
                        )
                if (
                    anchor_points is not None
                    and anchor_mask is not None
                    and outputs.get("anchor_points") is not None
                ):
                    valid = anchor_mask[batch_index, matched_targets].bool()
                    if bool(valid.any()):
                        components["anchor"].append(
                            F.smooth_l1_loss(
                                outputs["anchor_points"][
                                    batch_index, matched_queries[valid]
                                ].float(),
                                anchor_points[
                                    batch_index, matched_targets[valid]
                                ].float(),
                            )
                        )
                if (
                    anchor_indices is not None
                    and outputs.get("anchor_pointer_logits") is not None
                ):
                    pointer_targets = anchor_indices[
                        batch_index, matched_targets
                    ].long()
                    pointer_width = outputs["anchor_pointer_logits"].size(-1)
                    valid = pointer_targets.ge(0) & pointer_targets.lt(pointer_width)
                    if anchor_candidate_mask is not None and bool(valid.any()):
                        candidate_mask = anchor_candidate_mask[batch_index].bool()
                        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
                        target_is_candidate = candidate_mask[
                            pointer_targets[valid_positions]
                        ]
                        valid[valid_positions] &= target_is_candidate
                    if bool(valid.any()):
                        components["pointer"].append(
                            F.cross_entropy(
                                outputs["anchor_pointer_logits"][
                                    batch_index, matched_queries[valid]
                                ].float(),
                                pointer_targets[valid],
                            )
                        )
                if (
                    dummy_indices is not None
                    and outputs.get("dummy_pointer_logits") is not None
                ):
                    dummy_targets = dummy_indices[
                        batch_index, matched_targets
                    ].long()
                    pointer_width = outputs["dummy_pointer_logits"].size(-1)
                    valid = dummy_targets.ge(0) & dummy_targets.lt(pointer_width)
                    if dummy_candidate_mask is not None and bool(valid.any()):
                        candidate_mask = dummy_candidate_mask[batch_index].bool()
                        valid_positions = torch.nonzero(
                            valid, as_tuple=False
                        ).flatten()
                        target_is_candidate = candidate_mask[
                            dummy_targets[valid_positions]
                        ]
                        valid[valid_positions] &= target_is_candidate
                    if bool(valid.any()):
                        components["dummy_pointer"].append(
                            F.cross_entropy(
                                outputs["dummy_pointer_logits"][
                                    batch_index, matched_queries[valid]
                                ].float(),
                                dummy_targets[valid],
                            )
                        )

        sidecar_rows = labels.eq(sidecar_index)
        complete_rows = labels.eq(0)
        count_rows = sidecar_rows | complete_rows
        if bool(count_rows.any()):
            count_targets = torch.where(
                sidecar_rows,
                counts.clamp(min=0, max=outputs["cardinality_logits"].size(1) - 1),
                torch.zeros_like(counts),
            )
            components["cardinality"].append(
                F.cross_entropy(
                    outputs["cardinality_logits"][count_rows].float(),
                    count_targets[count_rows],
                )
            )
        if outputs.get("point_heatmap_logits") is not None:
            for batch_index in torch.nonzero(
                sidecar_rows, as_tuple=False
            ).flatten().tolist():
                selected_count = min(
                    int(counts[batch_index].clamp(min=0).item()),
                    int(outputs["objectness_logits"].size(1)),
                )
                if selected_count <= 1:
                    continue
                top_queries = outputs["objectness_logits"][
                    batch_index
                ].detach().topk(selected_count).indices
                distributions = outputs["point_heatmap_logits"][
                    batch_index, top_queries
                ].float().flatten(1).softmax(dim=-1)
                normalized = F.normalize(distributions, p=2, dim=1)
                overlap = normalized @ normalized.transpose(0, 1)
                upper = torch.triu(
                    torch.ones_like(overlap, dtype=torch.bool), diagonal=1
                )
                components["heatmap_diversity"].append(overlap[upper].mean())

    def mean_or_zero(name: str) -> torch.Tensor:
        values = components[name]
        return torch.stack(values).mean() if values else points.sum() * 0.0

    object_loss = mean_or_zero("objectness")
    point_loss = mean_or_zero("point")
    cardinality_loss = mean_or_zero("cardinality")
    bonded_loss = mean_or_zero("bonded")
    bond_type_loss = mean_or_zero("bond_type")
    anchor_loss = mean_or_zero("anchor")
    pointer_loss = mean_or_zero("pointer")
    dummy_pointer_loss = mean_or_zero("dummy_pointer")
    heatmap_loss = mean_or_zero("heatmap")
    heatmap_diversity_loss = mean_or_zero("heatmap_diversity")
    total = (
        float(objectness_loss_weight) * object_loss
        + float(point_loss_weight) * point_loss
        + float(cardinality_loss_weight) * cardinality_loss
        + float(relation_loss_weight) * (bonded_loss + bond_type_loss)
        + float(anchor_loss_weight) * anchor_loss
        + float(pointer_loss_weight) * pointer_loss
        + float(dummy_pointer_loss_weight) * dummy_pointer_loss
        + float(heatmap_loss_weight) * heatmap_loss
        + float(heatmap_diversity_loss_weight) * heatmap_diversity_loss
    )
    return total, {
        "attachment_set_objectness_loss": object_loss,
        "attachment_set_point_loss": point_loss,
        "attachment_set_cardinality_loss": cardinality_loss,
        "attachment_set_bonded_loss": bonded_loss,
        "attachment_set_bond_type_loss": bond_type_loss,
        "attachment_set_anchor_loss": anchor_loss,
        "attachment_set_pointer_loss": pointer_loss,
        "attachment_set_dummy_pointer_loss": dummy_pointer_loss,
        "attachment_set_heatmap_loss": heatmap_loss,
        "attachment_set_heatmap_diversity_loss": heatmap_diversity_loss,
    }


@torch.no_grad()
def select_attachment_queries(
    outputs: dict[str, torch.Tensor],
    *,
    batch_index: int,
    expected_type: str,
    min_confidence: float,
    max_count: int,
) -> list[dict[str, Any]]:
    probabilities = torch.sigmoid(outputs["objectness_logits"][batch_index].float())
    count = int(outputs["cardinality_logits"][batch_index].argmax().item())
    if str(expected_type or "").strip().lower() == "fragment":
        count = min(1, max(0, count))
    count = min(int(max_count), max(0, count), probabilities.numel())
    if count <= 0:
        return []
    top_probabilities, top_indices = probabilities.topk(count)
    selected = []
    for confidence, query_index in zip(top_probabilities, top_indices):
        confidence_value = float(confidence.item())
        if confidence_value < float(min_confidence):
            continue
        query = int(query_index.item())
        point = outputs["points"][batch_index, query]
        anchor_point = (
            outputs["anchor_points"][batch_index, query]
            if outputs.get("anchor_points") is not None
            else point
        )
        bonded_prob = torch.softmax(
            outputs["bonded_logits"][batch_index, query].float(), dim=-1
        )
        bond_prob = torch.softmax(
            outputs["bond_type_logits"][batch_index, query].float(), dim=-1
        )
        selected.append(
            {
                "query_index": query,
                "confidence": confidence_value,
                "x": float(point[0].item()),
                "y": float(point[1].item()),
                "anchor_x": float(anchor_point[0].item()),
                "anchor_y": float(anchor_point[1].item()),
                "bonded": bool(int(bonded_prob.argmax().item())),
                "bonded_confidence": float(bonded_prob.max().item()),
                "bond_type": int(bond_prob.argmax().item()),
                "bond_type_confidence": float(bond_prob.max().item()),
                "point_heatmap_confidence": (
                    float(
                        outputs["point_heatmap_confidence"][
                            batch_index, query
                        ].item()
                    )
                    if outputs.get("point_heatmap_confidence") is not None
                    else None
                ),
                "point_heatmap_peak_probability": (
                    float(
                        outputs["point_heatmap_peak_probability"][
                            batch_index, query
                        ].item()
                    )
                    if outputs.get("point_heatmap_peak_probability") is not None
                    else None
                ),
            }
        )
    return selected
