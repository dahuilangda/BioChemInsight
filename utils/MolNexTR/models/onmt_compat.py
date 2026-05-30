"""
Self-contained implementations of OpenNMT-py components used by MolNexTR.

This module inlines the subset of onmt (OpenNMT-py) classes and functions
that MolNexTR depends on, removing the need for the opennmt-py package.

Components:
    - DecoderBase: base class for transformer decoders
    - MultiHeadedAttention: scaled dot-product multi-head attention with KV-cache
    - AverageAttention: stub (not used in default MolNexTR config)
    - PositionwiseFeedForward: two-layer feed-forward network
    - ActivationFunction: enum-like class for activation function selection
    - sequence_mask: create a boolean mask from sequence lengths
    - Elementwise: apply an element-wise merge operation across a list of modules
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ActivationFunction
# ---------------------------------------------------------------------------

class ActivationFunction:
    """Enum-like container for activation function names.

    Mirrors ``onmt.modules.position_ffn.ActivationFunction``.
    """

    relu = "relu"
    gelu = "gelu"


# ---------------------------------------------------------------------------
# sequence_mask
# ---------------------------------------------------------------------------

def sequence_mask(lengths, max_len=None):
    """Create a boolean mask from sequence lengths.

    Args:
        lengths (LongTensor): ``(batch,)`` tensor of sequence lengths.
        max_len (int, optional): maximum length. If *None*, uses
            ``lengths.max()``.

    Returns:
        ByteTensor: ``(batch, max_len)`` mask that is *True* for valid
        positions and *False* for padded positions.
    """
    if max_len is None:
        max_len = lengths.max()
    # (batch, max_len)
    arange = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    # lengths.unsqueeze(1): (batch, 1),  arange: (max_len,)
    return arange < lengths.unsqueeze(1)


# ---------------------------------------------------------------------------
# DecoderBase
# ---------------------------------------------------------------------------

class DecoderBase(nn.Module):
    """Base class for decoders, mirroring ``onmt.decoders.decoder.DecoderBase``.

    Provides the interface that ``TransformerDecoderBase`` expects: a
    ``state`` dictionary and the methods ``init_state``, ``map_state``,
    ``detach_state`` and ``forward``.
    """

    def __init__(self):
        super(DecoderBase, self).__init__()

    # Subclasses override the following methods as needed.

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        pass

    def map_state(self, fn):
        """Apply *fn* to every tensor inside ``self.state``."""
        pass

    def detach_state(self):
        """Detach tensors in ``self.state`` from the computation graph."""
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MultiHeadedAttention
# ---------------------------------------------------------------------------

class MultiHeadedAttention(nn.Module):
    """Multi-head scaled dot-product attention with KV-cache support.

    Mirrors ``onmt.modules.MultiHeadedAttention``.

    Args:
        head_count (int): number of attention heads.
        model_dim (int): total model dimension (must be divisible by
            *head_count*).
        dropout (float): dropout probability applied to attention weights.
        max_relative_positions (int): if > 0, add relative position
            embeddings (not commonly used in MolNexTR).
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        super(MultiHeadedAttention, self).__init__()

        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V and output. These names and bias
        # settings match the OpenNMT-py modules used by the MolNexTR checkpoint.
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head, bias=True)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head, bias=True)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.final_linear = nn.Linear(head_count * self.dim_per_head, model_dim, bias=True)

        # Relative position embeddings (optional, kept for interface compat)
        self.max_relative_positions = max_relative_positions
        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head
            )

    def _relative_position_bucket(self, relative_positions):
        """Map relative positions to embedding indices (clamped)."""
        return relative_positions

    def forward(self, key, value, query, mask=None, layer_cache=None,
                attn_type=None):
        """Compute multi-head attention.

        Args:
            key (FloatTensor): ``(batch, key_len, model_dim)``
            value (FloatTensor): ``(batch, key_len, model_dim)``
            query (FloatTensor): ``(batch, query_len, model_dim)``
            mask (BoolTensor or None): ``(batch, 1, key_len)`` or
                ``(batch, query_len, key_len)``. Positions that are *True*
                are **masked out** (set to -inf).
            layer_cache (dict or None): cache dict for autoregressive
                decoding. Expected keys depend on *attn_type*:
                - ``"self"``: uses ``"self_keys"`` / ``"self_values"``
                - ``"context"``: uses ``"memory_keys"`` / ``"memory_values"``
            attn_type (str or None): ``"self"`` or ``"context"``. Determines
                which cache keys to read/write.

        Returns:
            (FloatTensor, FloatTensor):
                - output ``(batch, query_len, model_dim)``
                - attention weights ``(batch, head_count, query_len, key_len)``
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """(batch, seq, model_dim) -> (batch, head, seq, dim_per_head)"""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """(batch, head, seq, dim_per_head) -> (batch, seq, model_dim)"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # --- Project Q / K / V ---
        query_heads = shape(self.linear_query(query))   # (B, H, Q, d)
        key_heads = shape(self.linear_keys(key))         # (B, H, K, d)
        value_heads = shape(self.linear_values(value))   # (B, H, K, d)

        # --- KV-cache handling ---
        if layer_cache is not None:
            if attn_type == "self":
                # Self-attention: concatenate previous keys/values with new ones
                if layer_cache.get("self_keys") is not None:
                    key_heads = torch.cat(
                        [layer_cache["self_keys"], key_heads], dim=2
                    )
                if layer_cache.get("self_values") is not None:
                    value_heads = torch.cat(
                        [layer_cache["self_values"], value_heads], dim=2
                    )
                # Store for next step
                layer_cache["self_keys"] = key_heads
                layer_cache["self_values"] = value_heads
            elif attn_type == "context":
                # Cross-attention: cache memory keys/values once, then reuse
                if layer_cache.get("memory_keys") is None:
                    layer_cache["memory_keys"] = key_heads
                    layer_cache["memory_values"] = value_heads
                else:
                    key_heads = layer_cache["memory_keys"]
                    value_heads = layer_cache["memory_values"]

        key_len = key_heads.size(2)
        query_len = query_heads.size(2)

        # --- Scaled dot-product attention ---
        # (B, H, Q, d) x (B, H, d, K) -> (B, H, Q, K)
        scale = math.sqrt(dim_per_head)
        scores = torch.matmul(query_heads, key_heads.transpose(2, 3)) / scale

        # Optional relative position bias
        if self.max_relative_positions > 0 and attn_type == "self":
            range_vec = torch.arange(
                key_len, device=scores.device, dtype=torch.long
            )
            relative_pos = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
            # Clamp to valid range
            rp = relative_pos.clamp(-self.max_relative_positions,
                                     self.max_relative_positions)
            rp = rp + self.max_relative_positions  # shift to >= 0
            rp_emb = self.relative_positions_embeddings(rp)  # (K, K, d)
            # (B, H, Q, d) x (K, K, d) broadcast -> (B, H, Q, K)
            # Use Einstein summation: for each head, query . rp_emb
            # Simplified: add per-position bias
            scores = scores + torch.einsum(
                "bhid,jkd->bhij", query_heads, rp_emb
            )

        # Apply mask (True positions are masked out)
        if mask is not None:
            # mask shape can be (B, 1, K) or (B, Q, K)
            # Expand to (B, H, Q, K) if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, Q, K) or (B, 1, 1, K)
            scores = scores.masked_fill(mask, -1e18)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        # (B, H, Q, K) x (B, H, K, d) -> (B, H, Q, d)
        context = torch.matmul(drop_attn, value_heads)

        # --- Output projection ---
        output = self.final_linear(unshape(context))

        # Return attention weights (detached so they don't affect gradients)
        return output, attn.detach()

    def update_dropout(self, dropout):
        self.dropout.p = dropout


# ---------------------------------------------------------------------------
# AverageAttention (stub)
# ---------------------------------------------------------------------------

class AverageAttention(nn.Module):
    """Stub for ``onmt.modules.AverageAttention``.

    Average attention is **not** used in the default MolNexTR configuration
    (which uses ``self_attn_type="scaled-dot"``).  This stub exists so that
    the ``isinstance`` checks in ``TransformerDecoderLayerBase`` compile
    correctly.  Instantiating it is fine, but calling ``forward`` will raise.
    """

    def __init__(self, model_dim, dropout=0.1, aan_useffn=False):
        super(AverageAttention, self).__init__()
        self.model_dim = model_dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "AverageAttention is not used in MolNexTR's default config. "
            "If you see this error, the model was configured with "
            "self_attn_type='average', which is not supported."
        )

    def update_dropout(self, dropout):
        pass


# ---------------------------------------------------------------------------
# PositionwiseFeedForward
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network with residual-style activation.

    Mirrors ``onmt.modules.position_ffn.PositionwiseFeedForward``.

    Args:
        d_model (int): input/output dimension.
        d_ff (int): inner-layer dimension.
        dropout (float): dropout probability.
        activation_fn (str or ActivationFunction): activation to use
            between the two linear layers.  Accepted values are
            ``"relu"`` / ``ActivationFunction.relu`` and
            ``"gelu"`` / ``ActivationFunction.gelu``.
    """

    def __init__(self, d_model, d_ff, dropout=0.1, activation_fn=ActivationFunction.relu):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Normalise activation_fn to a string
        if isinstance(activation_fn, str):
            act_name = activation_fn
        else:
            act_name = str(activation_fn)

        if act_name in ("gelu", "ActivationFunction.gelu"):
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        """FFN forward: ``x -> w_2(dropout(act(w_1(x))))``."""
        inter = self.w_1(self.layer_norm(x))
        output = self.w_2(self.dropout(self.activation(inter)))
        return output + x

    def update_dropout(self, dropout):
        self.dropout.p = dropout


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------

class Elementwise(nn.Module):
    """Apply an element-wise merge to the outputs of a list of modules.

    Mirrors ``onmt.modules.util_class.Elementwise``.

    Given a list of ``nn.Module`` objects (typically ``nn.Embedding``
    layers), this module:

    1. Applies each module to the corresponding feature column of the input.
    2. Merges the results according to *merge_mode*:
       - ``"concat"``: concatenate along the last dimension.
       - ``"sum"``: element-wise summation.
       - ``"mlp"``: element-wise summation (same as ``"sum"`` here;
         the caller is expected to add an MLP layer separately).

    Args:
        merge_mode (str): one of ``"concat"``, ``"sum"``, ``"mlp"``.
        modules (list[nn.Module]): list of modules to apply element-wise.
    """

    def __init__(self, merge_mode, modules):
        assert merge_mode in ("concat", "sum", "mlp")
        super(Elementwise, self).__init__()
        self.merge_mode = merge_mode
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, inputs):
        """Apply each module to its corresponding input feature and merge.

        Args:
            inputs (LongTensor): ``(seq_len, batch, n_features)``.

        Returns:
            Tensor: merged embeddings with shape depending on *merge_mode*:
                - ``"concat"``: ``(seq_len, batch, sum_of_dims)``
                - ``"sum"``/``"mlp"``: ``(seq_len, batch, dim)``
        """
        # inputs: (seq_len, batch, n_features)
        n_features = inputs.size(-1)
        outputs = []
        for i in range(n_features):
            # Each module processes its corresponding feature column
            outputs.append(self._modules[str(i)](inputs[:, :, i]))

        if self.merge_mode == "concat":
            return torch.cat(outputs, dim=-1)
        else:  # "sum" or "mlp"
            return torch.stack(outputs, dim=0).sum(dim=0)

    def __getitem__(self, idx):
        return self._modules[str(idx)]
