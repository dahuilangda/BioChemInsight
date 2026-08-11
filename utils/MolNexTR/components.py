""" model components"""
import math
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from .utils import FORMAT_INFO, to_device
from .abbrs import ABBREVIATIONS, ELEMENTS
from .attachment_extension import (
    AttachmentExtensionError,
    allowed_extension_token_ids,
    materialize_attachment_extension,
)
from .tokenization import SOS_ID, EOS_ID, PAD_ID, MASK_ID
from .decoding import GreedySearch
from .models import TransformerDecoder, Embeddings

MAX_DECODE_ATOMS = 128
MAX_DECODE_BONDS = 160
MAX_DECODE_BOND_RATIO = 2.5
MAX_DECODE_SYMBOL_LEN = 8
ELEMENT_SET = set(ELEMENTS)

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )



    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)

        return tuple(outputs)



class Encoder(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        model_name = args.encoder
        self.model_name = model_name
        if model_name.startswith('resnet'):
            self.model_type = 'resnet'
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.cnn.num_features  # encoder_dim
            self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        elif model_name.startswith('convnext'):
            self.model_type = 'convnext'
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = 1024  # encoder_dim
            # self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        elif model_name.startswith('swin'):
            self.model_type = 'vision_transformer'
            self.transformer = timm.create_model(model_name, pretrained=pretrained, pretrained_strict=False,
                                                 use_checkpoint=args.use_checkpoint)
            self.n_features = self.transformer.num_features
            self.transformer.head = nn.Identity()
        else:
            raise NotImplemented

    def forwards(self, transformer, x):
        x = transformer.patch_embed(x)
        if transformer.absolute_pos_embed is not None:
            x = x + transformer.absolute_pos_embed
        x = transformer.pos_drop(x)

        def layer_forward(layer, x, hiddens):
            for blk in layer.blocks:
                if not torch.jit.is_scripting() and layer.use_checkpoint:
                    # Non-reentrant checkpointing supports trainable parameters
                    # even when the incoming tensor comes from frozen earlier
                    # encoder stages and therefore has requires_grad=False.
                    x = torch.utils.checkpoint.checkpoint(
                        blk,
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = blk(x)
            H, W = layer.input_resolution
            B, L, C = x.shape
            hiddens.append(x.view(B, H, W, C))
            if layer.downsample is not None:
                x = layer.downsample(x)
            return x, hiddens

        hiddens = []
        for layer in transformer.layers:
            x, hiddens = layer_forward(layer, x, hiddens)
        x = transformer.norm(x)  # B L C
        hiddens[-1] = x.view_as(hiddens[-1])
        return x, hiddens

    def forward(self, x, refs=None):
        if self.model_type in ['resnet', 'efficientnet', 'convnext']:
            features = self.cnn(x)
            features = features.permute(0, 2, 3, 1)
            hiddens = []
        elif self.model_type == 'vision_transformer':
            if 'patch' in self.model_name:
                features, hiddens = self.forwards(self.transformer, x)
            else:
                features, hiddens = self.transformer(x)
        else:
            raise NotImplemented
        return features, hiddens


class TransformerDecoderBase(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.enc_trans_layer = nn.Sequential(
            nn.Linear(args.encoder_dim, args.dec_hidden_size)
            # nn.LayerNorm(args.dec_hidden_size, eps=1e-6)
        )
        self.enc_pos_emb = nn.Embedding(144, args.encoder_dim) if args.enc_pos_emb else None

        self.decoder = TransformerDecoder(
            num_layers=args.dec_num_layers,
            d_model=args.dec_hidden_size,
            heads=args.dec_attn_heads,
            d_ff=args.dec_hidden_size * 4,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=args.hidden_dropout,
            attention_dropout=args.attn_dropout,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=0,
            alignment_heads=0,
            pos_ffn_activation_fn='gelu'
        )

    def enc_transform(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        max_len = encoder_out.size(1)
        device = encoder_out.device
        if self.enc_pos_emb:
            pos_emb = self.enc_pos_emb(torch.arange(max_len, device=device)).unsqueeze(0)
            encoder_out = encoder_out + pos_emb
        encoder_out = self.enc_trans_layer(encoder_out)
        return encoder_out


class TransformerDecoderAR(TransformerDecoderBase):
    """Autoregressive Transformer Decoder"""

    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.output_layer = nn.Linear(args.dec_hidden_size, self.vocab_size, bias=True)
        self.embeddings = Embeddings(
            word_vec_size=args.dec_hidden_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=PAD_ID,
            position_encoding=True,
            dropout=args.hidden_dropout)

    def dec_embedding(self, tgt, step=None):
        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt.data.eq(pad_idx).transpose(1, 2)  # [B, 1, T_tgt]
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # batch x len x embedding_dim
        return emb, tgt_pad_mask

    def forward(self, encoder_out, labels, label_lengths, logit_bias=None):
        """Training mode"""
        batch_size, max_len, _ = encoder_out.size()
        memory_bank = self.enc_transform(encoder_out)

        tgt = labels.unsqueeze(-1)  # (b, t, 1)
        tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
        dec_out, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank, tgt_pad_mask=tgt_pad_mask)

        logits = self.output_layer(dec_out)  # (b, t, h) -> (b, t, v)
        if logit_bias is not None:
            logits = logits + logit_bias.to(logits.device, dtype=logits.dtype).unsqueeze(1)
        return logits[:, :-1], labels[:, 1:], dec_out

    def decode(
        self,
        encoder_out,
        beam_size: int,
        n_best: int,
        min_length: int = 1,
        max_length: int = 256,
        labels=None,
        logit_bias=None,
        decode_constraints=None,
        mask_y_coordinate_context: bool = False,
        terminal_action_head=None,
        terminal_star_token_id: int | None = None,
        allow_sep: bool = False,
    ):
        """Inference mode. Autoregressively decode the sequence — **GREEDY ONLY**.

        MolNexTR is trained greedily and the inherited OpenNMT ``BeamSearch``
        path is NOT used: under ``beam_size>1`` it failed to emit EOS within
        ``max_length`` (``molnextr_decode_missing_eos`` → empty SMILES), and
        beam search on a greedily-trained OCSR decoder does not help (both
        MolNexTR and MolScribe ship greedy-only). Any ``beam_size > 1`` is
        clamped to 1 here with a one-time warning; ``n_best`` is implicitly 1
        (greedy returns the single best sequence). ``labels`` is used for
        partial prediction (part of the sequence is given); standard decoding
        passes ``labels=None``.
        """
        if int(beam_size) != 1:
            import warnings
            warnings.warn(
                "MolNexTR decoding is greedy-only (beam search is unsupported and "
                "was removed); clamping beam_size=%d to 1." % int(beam_size),
                stacklevel=2,
            )
            beam_size = 1
        batch_size, max_len, _ = encoder_out.size()
        memory_bank = self.enc_transform(encoder_out)
        orig_labels = labels

        decode_strategy = GreedySearch(
            sampling_temp=0.0, keep_topk=1, batch_size=batch_size, min_length=min_length, max_length=max_length,
            pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
            return_attention=False, return_hidden=True)

        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        _, memory_bank = decode_strategy.initialize(memory_bank=memory_bank)
        star_token_id = terminal_star_token_id
        star_budgets = None
        star_logit_bias = None
        generated_star_counts = None
        # Phase 2: <sep> token id (chartok_coords). When allow_sep is False (the
        # default — frozen base / expert0 / complete rows), <sep> is masked to -inf
        # every step so it is never emitted ⇒ byte-identical to pre-<sep> decoding.
        sep_id = getattr(self.tokenizer, "sep_id", None)
        if decode_constraints:
            star_token_id = decode_constraints.get(
                "star_token_id", star_token_id
            )
            star_budgets = decode_constraints.get("star_budgets")
            star_logit_bias = decode_constraints.get("star_logit_bias")
            if star_token_id is not None and star_budgets is not None:
                star_budgets = star_budgets.to(memory_bank.device, dtype=torch.long).view(-1)
                if star_budgets.numel() != memory_bank.size(0):
                    star_budgets = star_budgets[: memory_bank.size(0)]
                generated_star_counts = torch.zeros_like(star_budgets)
            if star_token_id is not None and star_logit_bias is not None:
                star_logit_bias = star_logit_bias.to(memory_bank.device, dtype=torch.float32).view(-1)
                if star_logit_bias.numel() != memory_bank.size(0):
                    star_logit_bias = star_logit_bias[: memory_bank.size(0)]
        if (
            terminal_action_head is not None
            and star_token_id is not None
            and generated_star_counts is None
        ):
            generated_star_counts = torch.zeros(
                memory_bank.size(0),
                dtype=torch.long,
                device=memory_bank.device,
            )

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            emitted_tgt = decode_strategy.current_predictions.view(-1, 1, 1)
            tgt = emitted_tgt
            if labels is not None:
                label = labels[:, step].view(-1, 1, 1)
                mask = label.eq(MASK_ID).long()
                tgt = tgt * mask + label * (1 - mask)
            if mask_y_coordinate_context:
                # Preserve the emitted y coordinate in the output sequence, but
                # feed a stable sentinel before the next topology token. This
                # aligns direct-sidecar inference with real patent graph rows,
                # where coordinates are unavailable rather than fabricated.
                is_y = torch.tensor(
                    [
                        self.tokenizer.is_y(int(token))
                        for token in emitted_tgt.view(-1).tolist()
                    ],
                    dtype=torch.bool,
                    device=emitted_tgt.device,
                ).view_as(emitted_tgt)
                tgt = torch.where(
                    is_y,
                    torch.full_like(tgt, MASK_ID),
                    tgt,
                )
            tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
            dec_out, dec_attn, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank,
                                                 tgt_pad_mask=tgt_pad_mask, step=step)

            attn = dec_attn.get("std", None)

            dec_logits = self.output_layer(dec_out)  # [b, t, h] => [b, t, v]
            dec_logits = dec_logits.squeeze(1)
            # Phase 2: forbid <sep> unless the caller opts in (trainable markush/
            # fragment decoder). Default False ⇒ base/expert0 never emit <sep>.
            if (
                not allow_sep
                and sep_id is not None
                and 0 <= int(sep_id) < dec_logits.size(-1)
            ):
                dec_logits[:, int(sep_id)] = float("-inf")
            if logit_bias is not None:
                step_bias = logit_bias.to(dec_logits.device, dtype=dec_logits.dtype)
                if step_bias.size(0) != dec_logits.size(0):
                    step_bias = step_bias[: dec_logits.size(0)]
                dec_logits = dec_logits + step_bias
            if (
                star_token_id is not None
                and star_logit_bias is not None
                and 0 <= int(star_token_id) < dec_logits.size(-1)
            ):
                step_star_bias = star_logit_bias.to(dec_logits.device, dtype=dec_logits.dtype)
                if step_star_bias.size(0) != dec_logits.size(0):
                    step_star_bias = step_star_bias[: dec_logits.size(0)]
                dec_logits[:, int(star_token_id)] = dec_logits[:, int(star_token_id)] + step_star_bias
            output_mask = None
            if self.tokenizer.output_constraint:
                output_mask = [
                    self.tokenizer.get_output_mask(id)
                    for id in emitted_tgt.view(-1).tolist()
                ]
                output_mask = torch.tensor(
                    output_mask,
                    dtype=torch.bool,
                    device=dec_logits.device,
                )
            if (
                terminal_action_head is not None
                and star_token_id is not None
                and 0 <= int(star_token_id) < dec_logits.size(-1)
            ):
                constrained_for_top = dec_logits
                if output_mask is not None:
                    constrained_for_top = constrained_for_top.masked_fill(
                        output_mask,
                        float("-inf"),
                    )
                if step < int(min_length):
                    constrained_for_top = constrained_for_top.clone()
                    constrained_for_top[:, EOS_ID] = float("-inf")
                if (
                    star_budgets is not None
                    and generated_star_counts is not None
                ):
                    exhausted = generated_star_counts.ge(star_budgets)
                    if bool(exhausted.any()):
                        constrained_for_top = constrained_for_top.clone()
                        constrained_for_top[
                            exhausted, int(star_token_id)
                        ] = float("-inf")
                base_top = constrained_for_top.argmax(dim=-1)
                has_attachment = (
                    generated_star_counts.gt(0)
                    if generated_star_counts is not None
                    else torch.zeros_like(base_top, dtype=torch.bool)
                )
                eligible = (
                    base_top.eq(EOS_ID)
                    | base_top.eq(int(star_token_id))
                    | has_attachment
                )
                step_fraction = torch.full(
                    (dec_logits.size(0),),
                    float(step) / float(max(1, decode_strategy.max_length - 1)),
                    dtype=torch.float32,
                    device=dec_logits.device,
                )
                (
                    dec_logits,
                    _terminal_pair,
                    _terminal_stop_pair,
                ) = terminal_action_head.adjust_logits(
                    dec_out.squeeze(1),
                    dec_logits,
                    eos_token_id=EOS_ID,
                    star_token_id=int(star_token_id),
                    has_attachment=has_attachment,
                    step_fraction=step_fraction,
                    eligible=eligible,
                )
            log_probs = F.log_softmax(dec_logits, dim=-1)

            if output_mask is not None:
                # Grammar follows what was emitted. The MASK sentinel is only
                # decoder context and must not allow another coordinate token.
                log_probs.masked_fill_(output_mask, -10000)
            if (
                star_token_id is not None
                and star_budgets is not None
                and generated_star_counts is not None
                and 0 <= int(star_token_id) < log_probs.size(-1)
            ):
                exhausted = generated_star_counts.ge(star_budgets)
                if bool(exhausted.any()):
                    log_probs[exhausted, int(star_token_id)] = -10000

            if allow_sep and sep_id is not None:
                # Once the model emits <sep>, decode under the extension's
                # finite grammar. The model still owns <sep> emission and the
                # anchor digits; malformed/missing semantic output is rejected
                # after decoding rather than repaired.
                for row_index, emitted_sequence in enumerate(
                    decode_strategy.alive_seq.tolist()
                ):
                    allowed_ids = allowed_extension_token_ids(
                        self.tokenizer, emitted_sequence
                    )
                    if allowed_ids is None:
                        continue
                    grammar_mask = torch.ones_like(
                        log_probs[row_index], dtype=torch.bool
                    )
                    if allowed_ids:
                        grammar_mask[list(allowed_ids)] = False
                    else:
                        # This path can only be reached from a pre-existing
                        # invalid prefix. End it so strict materialization can
                        # report an auditable failure instead of running to the
                        # maximum decode length.
                        grammar_mask[EOS_ID] = False
                    log_probs[row_index].masked_fill_(grammar_mask, -10000)

            label = labels[:, step + 1] if labels is not None and step + 1 < labels.size(1) else None
            decode_strategy.advance(log_probs, attn, dec_out, label)
            if (
                star_token_id is not None
                and generated_star_counts is not None
                and decode_strategy.current_predictions.numel() == generated_star_counts.numel()
            ):
                generated_star_counts = generated_star_counts + decode_strategy.current_predictions.eq(
                    int(star_token_id)
                ).long()
            any_finished = decode_strategy.is_finished.any()
            select_indices = decode_strategy.select_indices
            if select_indices is not None and beam_size > 1:
                memory_bank = memory_bank.index_select(0, select_indices)
                if labels is not None:
                    labels = labels.index_select(0, select_indices)
                if logit_bias is not None:
                    logit_bias = logit_bias.index_select(0, select_indices.to(logit_bias.device))
                if star_budgets is not None:
                    star_budgets = star_budgets.index_select(0, select_indices.to(star_budgets.device))
                if star_logit_bias is not None:
                    star_logit_bias = star_logit_bias.index_select(0, select_indices.to(star_logit_bias.device))
                if generated_star_counts is not None:
                    generated_star_counts = generated_star_counts.index_select(
                        0,
                        select_indices.to(generated_star_counts.device),
                    )
                self.map_state(lambda state, dim: state.index_select(dim, select_indices))
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if any_finished and beam_size == 1:
                # Reorder states.
                memory_bank = memory_bank.index_select(0, select_indices)
                if labels is not None:
                    labels = labels.index_select(0, select_indices)
                if logit_bias is not None:
                    logit_bias = logit_bias.index_select(0, select_indices.to(logit_bias.device))
                if star_budgets is not None:
                    star_budgets = star_budgets.index_select(0, select_indices.to(star_budgets.device))
                if star_logit_bias is not None:
                    star_logit_bias = star_logit_bias.index_select(0, select_indices.to(star_logit_bias.device))
                if generated_star_counts is not None:
                    generated_star_counts = generated_star_counts.index_select(
                        0,
                        select_indices.to(generated_star_counts.device),
                    )
                self.map_state(lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores  # fixed to be average of token scores
        results["token_scores"] = decode_strategy.token_scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["hidden"] = decode_strategy.hidden
        if orig_labels is not None:
            for i in range(batch_size):
                pred = results["predictions"][i][0]
                label = orig_labels[i][1:len(pred) + 1]
                mask = label.eq(MASK_ID).long()
                pred = pred[:len(label)]
                results["predictions"][i][0] = pred * mask + label * (1 - mask)

        return results["predictions"], results['scores'], results["token_scores"], results["hidden"]

    # adapted from onmt.decoders.models
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])

    def _step_logits(self, tgt_token, memory_bank, step, logit_bias=None):
        """Run ONE autoregressive decode step for this expert and return its
        token logits + hidden state.

        This mirrors the step body of ``TransformerDecoderAR.decode``
        (components.py:321-338) exactly, so that given an identical
        ``memory_bank`` and ``tgt_token`` the returned logits are bit-identical
        to the single-expert greedy path. It exists so that ``MoEDecoder`` can
        run K experts in lockstep and mix their per-step logits into a soft
        mixture, while keeping the frozen dominant expert's output unchanged.

        Args:
            tgt_token: LongTensor ``[B]`` — the shared current prediction.
            memory_bank: ``[B, L, dec_hidden]`` — this expert's projected
                encoder memory (from ``enc_transform``).
            step: int decode step (drives the transformer KV-cache).
            logit_bias: optional ``[B]`` bias added to every token logit.

        Returns:
            (logits ``[B, vocab]``, dec_out ``[B, 1, dec_hidden]``).
        """
        tgt = tgt_token.view(-1, 1, 1)
        tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
        dec_out, _dec_attn, *_ = self.decoder(
            tgt_emb=tgt_emb, memory_bank=memory_bank,
            tgt_pad_mask=tgt_pad_mask, step=step)
        dec_logits = self.output_layer(dec_out).squeeze(1)  # [b, v]
        if logit_bias is not None:
            step_bias = logit_bias.to(dec_logits.device, dtype=dec_logits.dtype)
            if step_bias.size(0) != dec_logits.size(0):
                step_bias = step_bias[: dec_logits.size(0)]
            dec_logits = dec_logits + step_bias
        return dec_logits, dec_out


class GraphPredictor(nn.Module):

    def __init__(self, decoder_dim, coords=False):
        super(GraphPredictor, self).__init__()
        self.coords = coords
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, 7)
        )
        if coords:
            self.coords_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
                nn.Linear(decoder_dim, 2)
            )

    def forward(self, hidden, indices=None):
        b, l, dim = hidden.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            hidden = hidden[:, index]
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            hidden = hidden[batch_id, indices].view(b, -1, dim)
        b, l, dim = hidden.size()
        results = {}
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)
        results['edges'] = self.mlp(hh).permute(0, 3, 1, 2)
        if self.coords:
            results['coords'] = self.coords_mlp(hidden)
        return results


def get_edge_prediction(edge_prob):
    edge_prob = np.asarray(edge_prob)
    if edge_prob.size == 0:
        return [], {}
    n = edge_prob.shape[0]
    if n == 0:
        return [], {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(5):
                edge_prob[i][j][k] = (edge_prob[i][j][k] + edge_prob[j][i][k]) / 2
                edge_prob[j][i][k] = edge_prob[i][j][k]
            edge_prob[i][j][5] = (edge_prob[i][j][5] + edge_prob[j][i][6]) / 2
            edge_prob[i][j][6] = (edge_prob[i][j][6] + edge_prob[j][i][5]) / 2
            edge_prob[j][i][5] = edge_prob[i][j][6]
            edge_prob[j][i][6] = edge_prob[i][j][5]
    prediction = np.argmax(edge_prob, axis=2)
    score = np.max(edge_prob, axis=2)
    bond_scores = {
        (i, j): float(score[i][j])
        for i in range(n)
        for j in range(i + 1, n)
        if int(prediction[i][j]) != 0
    }
    return prediction.tolist(), bond_scores


def decode_terminal_dummy_single_bond_map(
    edge_prob,
    edge_prediction,
    edge_scores,
    symbols,
):
    """Conditional MAP edge decode for a terminal fragment dummy.

    The v3 fragment contract contains exactly one terminal dummy with exactly
    one single bond.  Pairwise edge logits remain the model evidence; this
    routine solves the constrained global argmax over all possible anchors and
    leaves every backbone-backbone decision untouched.
    """
    symbols = list(symbols or [])
    dummy_indices = [
        index for index, symbol in enumerate(symbols)
        if "*" in str(symbol)
    ]
    if len(dummy_indices) != 1:
        return edge_prediction, edge_scores, None
    dummy_index = int(dummy_indices[0])
    anchors = [index for index in range(len(symbols)) if index != dummy_index]
    probabilities = np.asarray(edge_prob)
    if (
        not anchors
        or probabilities.ndim != 3
        or probabilities.shape[:2] != (len(symbols), len(symbols))
        or probabilities.shape[2] < 2
    ):
        return edge_prediction, edge_scores, None
    action_scores = []
    single_probabilities = []
    for anchor_index in anchors:
        no_bond = max(
            1.0e-12,
            0.5 * (
                float(probabilities[dummy_index, anchor_index, 0])
                + float(probabilities[anchor_index, dummy_index, 0])
            ),
        )
        single = max(
            1.0e-12,
            0.5 * (
                float(probabilities[dummy_index, anchor_index, 1])
                + float(probabilities[anchor_index, dummy_index, 1])
            ),
        )
        action_scores.append(math.log(single) - math.log(no_bond))
        single_probabilities.append(single)
    constrained = [list(row) for row in edge_prediction]
    for index in range(len(symbols)):
        constrained[dummy_index][index] = 0
        constrained[index][dummy_index] = 0
    constrained_scores = {
        key: value
        for key, value in dict(edge_scores or {}).items()
        if dummy_index not in key
    }

    valid_anchor_locals = []
    try:
        from rdkit import Chem, rdBase
        from .chemical import _atom_from_predicted_symbol

        log_block = rdBase.BlockLogs()
        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
            5: Chem.BondType.SINGLE,
            6: Chem.BondType.SINGLE,
        }
        for local_index, anchor_index in enumerate(anchors):
            try:
                editable = Chem.RWMol()
                atom_indices = []
                for symbol in symbols:
                    atom = (
                        Chem.Atom(0)
                        if "*" in str(symbol)
                        else _atom_from_predicted_symbol(str(symbol))
                    )
                    atom_indices.append(editable.AddAtom(atom))
                for left in range(len(symbols)):
                    if left == dummy_index:
                        continue
                    for right in range(left + 1, len(symbols)):
                        if right == dummy_index:
                            continue
                        bond_type = bond_types.get(int(edge_prediction[left][right]))
                        if bond_type is not None:
                            editable.AddBond(
                                atom_indices[left],
                                atom_indices[right],
                                bond_type,
                            )
                editable.AddBond(
                    atom_indices[dummy_index],
                    atom_indices[anchor_index],
                    Chem.BondType.SINGLE,
                )
                candidate = editable.GetMol()
                Chem.SanitizeMol(candidate)
                valid_anchor_locals.append(local_index)
            except Exception:
                continue
        del log_block
    except Exception:
        # Missing chemistry support is an explicit no-action result; silently
        # decoding an unconstrained invalid edge would violate the contract.
        valid_anchor_locals = []

    if not valid_anchor_locals:
        return constrained, constrained_scores, {
            "dummy_index": dummy_index,
            "anchor_index": -1,
            "bond_type": 0,
            "action_score": None,
            "candidate_count": len(anchors),
            "valid_candidate_count": 0,
        }
    best_local = max(valid_anchor_locals, key=lambda index: action_scores[index])
    anchor_index = int(anchors[best_local])
    constrained[dummy_index][anchor_index] = 1
    constrained[anchor_index][dummy_index] = 1
    constrained_scores[tuple(sorted((dummy_index, anchor_index)))] = float(
        single_probabilities[best_local]
    )
    return constrained, constrained_scores, {
        "dummy_index": dummy_index,
        "anchor_index": anchor_index,
        "bond_type": 1,
        "action_score": float(action_scores[best_local]),
        "candidate_count": len(anchors),
        "valid_candidate_count": len(valid_anchor_locals),
    }


def decode_bond_limit(atom_count):
    return max(MAX_DECODE_BONDS, int(np.ceil(MAX_DECODE_BOND_RATIO * max(atom_count, 1))))


def invalid_decode_symbol(symbol):
    text = str(symbol or "").strip()
    if len(text) > MAX_DECODE_SYMBOL_LEN:
        return text
    if not (text.startswith("[") and text.endswith("]")):
        return ""
    label = text[1:-1].strip()
    if not label or len(label) > MAX_DECODE_SYMBOL_LEN - 2:
        return text
    if re.fullmatch(r"\d*\*", label):
        return ""
    if label in ABBREVIATIONS:
        return ""
    atom_match = re.fullmatch(r"(?:\d+)?([A-Z][a-z]?)(?:@{1,2})?(?:H\d*)?(?:[+-]\d*)?", label)
    if atom_match and atom_match.group(1) in ELEMENT_SET:
        return ""
    aromatic_atom_match = re.fullmatch(r"([cnops])(?:H\d*)?(?:[+-]\d*)?", label)
    if aromatic_atom_match and aromatic_atom_match.group(1).upper() in ELEMENT_SET:
        return ""
    if re.fullmatch(r"R[A-Za-z0-9]*(?:'+)?", label):
        return ""
    if re.fullmatch(r"R'+", label):
        return ""
    if re.fullmatch(r"[A-Z][A-Za-z]?\d{1,2}(?:'+)?", label):
        return ""
    if re.fullmatch(r"[A-Z]'+", label):
        return ""
    if re.fullmatch(r"[A-Z]", label):
        return ""
    return text


class Decoder(nn.Module):
    """This class is a wrapper for different decoder architectures, and support multiple decoders."""

    def __init__(self, args, tokenizer):
        super(Decoder, self).__init__()
        self.args = args
        self.formats = args.formats
        self.tokenizer = tokenizer
        decoder = {}
        for format_ in args.formats:
            if format_ == 'edges':
                decoder['edges'] = GraphPredictor(args.dec_hidden_size, coords=args.continuous_coords)
            else:
                decoder[format_] = TransformerDecoderAR(args, tokenizer[format_])
        self.decoder = nn.ModuleDict(decoder)
        self.compute_confidence = args.compute_confidence

    def forward(self, encoder_out, hiddens, refs, logit_biases=None):
        """Training mode. Compute the logits with teacher forcing."""
        results = {}
        refs = to_device(refs, encoder_out.device)
        logit_biases = logit_biases or {}
        for format_ in self.formats:
            if format_ == 'edges':
                if 'atomtok_coords' in results:
                    dec_out = results['atomtok_coords'][2]
                    predictions = self.decoder['edges'](dec_out, indices=refs['atom_indices'][0])
                elif 'chartok_coords' in results:
                    dec_out = results['chartok_coords'][2]
                    predictions = self.decoder['edges'](dec_out, indices=refs['atom_indices'][0])
                else:
                    raise NotImplemented
                targets = {'edges': refs['edges']}
                if 'coords' in predictions:
                    targets['coords'] = refs['coords']
                results['edges'] = (predictions, targets)
            else:
                labels, label_lengths = refs[format_]
                results[format_] = self.decoder[format_](
                    encoder_out,
                    labels,
                    label_lengths,
                    logit_bias=logit_biases.get(format_),
                )
        return results

    def decode(
        self,
        encoder_out,
        hiddens=None,
        refs=None,
        beam_size=1,
        n_best=1,
        logit_biases=None,
        decode_constraints=None,
        return_atom_hidden=False,
        mask_y_coordinate_context=False,
        terminal_action_head=None,
        terminal_star_token_id=None,
        structured_terminal_dummy_edge=False,
        allow_sep: bool = False,
    ):
        """Inference mode. Call each decoder's decode method (if required), convert the output format (e.g. token to
        sequence). Beam search is not supported yet."""
        results = {}
        predictions = []
        logit_biases = logit_biases or {}
        decode_constraints = decode_constraints or {}
        for format_ in self.formats:
            if format_ in ['atomtok', 'atomtok_coords', 'chartok_coords']:
                max_len = FORMAT_INFO[format_]['max_len']
                results[format_] = self.decoder[format_].decode(
                    encoder_out,
                    beam_size,
                    n_best,
                    max_length=max_len,
                    logit_bias=logit_biases.get(format_),
                    decode_constraints=decode_constraints.get(format_),
                    mask_y_coordinate_context=bool(mask_y_coordinate_context),
                    terminal_action_head=(
                        terminal_action_head if format_ == 'chartok_coords' else None
                    ),
                    terminal_star_token_id=(
                        terminal_star_token_id if format_ == 'chartok_coords' else None
                    ),
                    allow_sep=(allow_sep if format_ == 'chartok_coords' else False),
                )
                outputs, scores, token_scores, *_ = results[format_]
                beam_preds = [[self.tokenizer[format_].sequence_to_smiles(x.tolist()) for x in pred]
                              for pred in outputs]
                predictions = []
                for pred in beam_preds:
                    if pred:
                        predictions.append({format_: pred[0]})
                    else:
                        predictions.append({
                            format_: {"smiles": "", "symbols": [], "coords": [], "indices": []},
                            "decode_quality_issue": "molnextr_decode_missing_eos",
                            "decode_atom_count": 0,
                        })
                for i, pred in enumerate(predictions):
                    sequence = outputs[i][0].tolist() if len(outputs[i]) else []
                    symbols = pred[format_].get('symbols') or []
                    atom_count = len(symbols)
                    pred['decode_atom_count'] = atom_count
                    if EOS_ID not in sequence:
                        pred['decode_quality_issue'] = 'molnextr_decode_missing_eos'
                    elif atom_count > MAX_DECODE_ATOMS:
                        pred['decode_quality_issue'] = f'molnextr_decode_atom_limit_exceeded:{atom_count}'
                    else:
                        bad_symbols = [symbol for symbol in symbols if invalid_decode_symbol(symbol)]
                        if bad_symbols:
                            pred['decode_quality_issue'] = (
                                'molnextr_decode_invalid_symbols:' + ','.join(bad_symbols[:3])
                            )
                if self.compute_confidence:
                    for i in range(len(predictions)):
                        sequence = (
                            outputs[i][0].tolist()
                            if i < len(outputs) and len(outputs[i])
                            else []
                        )
                        if (
                            predictions[i].get('decode_quality_issue')
                            or i >= len(token_scores)
                            or not token_scores[i]
                            or i >= len(scores)
                            or not scores[i]
                        ):
                            predictions[i][format_]['atom_scores'] = []
                            predictions[i][format_].pop('average_token_score', None)
                            continue
                        # -1: y score, -2: x score, -3: symbol score
                        indices = np.array(predictions[i][format_]['indices']) - 3
                        if format_ == 'chartok_coords':
                            atom_scores = []
                            for symbol, index in zip(predictions[i][format_]['symbols'], indices):
                                atom_score = (np.prod(token_scores[i][0][index - len(symbol) + 1:index + 1])
                                              ** (1 / len(symbol))).item()
                                atom_scores.append(atom_score)
                        else:
                            atom_scores = np.array(token_scores[i][0])[indices].tolist()
                        predictions[i][format_]['atom_scores'] = atom_scores
                        predictions[i][format_]['average_token_score'] = scores[i][0]
                        if allow_sep and getattr(self.tokenizer[format_], 'sep_id', None) in sequence:
                            sep_position = sequence.index(
                                self.tokenizer[format_].sep_id
                            )
                            extension_scores = []
                            for token_position in range(
                                sep_position + 1, len(sequence)
                            ):
                                if sequence[token_position] in (
                                    EOS_ID,
                                    PAD_ID,
                                    self.tokenizer[format_].sep_id,
                                ):
                                    break
                                if token_position < len(token_scores[i][0]):
                                    extension_scores.append(
                                        max(
                                            float(token_scores[i][0][token_position]),
                                            1.0e-12,
                                        )
                                    )
                            if extension_scores:
                                predictions[i][format_][
                                    'attachment_extension_score'
                                ] = float(
                                    math.exp(
                                        sum(math.log(value) for value in extension_scores)
                                        / len(extension_scores)
                                    )
                                )
            if format_ == 'edges':
                if 'atomtok_coords' in results:
                    atom_format = 'atomtok_coords'
                elif 'chartok_coords' in results:
                    atom_format = 'chartok_coords'
                else:
                    raise NotImplemented
                dec_out = results[atom_format][3]  # batch x n_best x len x dim
                for i in range(len(dec_out)):
                    if (
                        predictions[i].get('decode_quality_issue')
                        or i >= len(dec_out)
                        or len(dec_out[i]) == 0
                    ):
                        if not predictions[i].get('decode_quality_issue'):
                            predictions[i]['decode_quality_issue'] = 'molnextr_decode_missing_eos'
                        atom_count = len(predictions[i][atom_format].get('symbols') or [])
                        predictions[i]['edges'] = [[0] * atom_count for _ in range(atom_count)]
                        if self.compute_confidence:
                            predictions[i]['edge_scores'] = {}
                            predictions[i][atom_format].pop('average_token_score', None)
                        continue
                    hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                    indices = torch.LongTensor(predictions[i][atom_format]['indices']).unsqueeze(0)  # 1 * k
                    if return_atom_hidden:
                        gather_indices = indices.to(hidden.device).clamp(
                            min=0,
                            max=max(0, hidden.size(1) - 1),
                        )
                        predictions[i]["_decoder_atom_hidden"] = hidden[0].index_select(
                            0, gather_indices[0]
                        ).detach()
                    pred = self.decoder['edges'](hidden, indices)  # k * k
                    prob = F.softmax(pred['edges'].squeeze(0).permute(1, 2, 0), dim=2).detach().cpu().numpy()  # k * k * 7
                    edge_pred, edge_score = get_edge_prediction(prob)
                    if structured_terminal_dummy_edge:
                        (
                            edge_pred,
                            edge_score,
                            structured_edge_action,
                        ) = decode_terminal_dummy_single_bond_map(
                            prob,
                            edge_pred,
                            edge_score,
                            predictions[i][atom_format].get('symbols') or [],
                        )
                        if structured_edge_action is not None:
                            predictions[i][
                                'structured_terminal_edge_action'
                            ] = structured_edge_action
                    bond_count = len(edge_score)
                    bond_limit = decode_bond_limit(len(predictions[i][atom_format].get('symbols') or []))
                    if bond_count > bond_limit:
                        predictions[i]['decode_quality_issue'] = (
                            f'molnextr_decode_bond_limit_exceeded:{bond_count}>{bond_limit}'
                        )
                        atom_count = len(predictions[i][atom_format].get('symbols') or [])
                        predictions[i]['edges'] = [[0] * atom_count for _ in range(atom_count)]
                        if self.compute_confidence:
                            predictions[i]['edge_scores'] = {}
                            predictions[i][atom_format].pop('average_token_score', None)
                        del hidden, indices, pred, prob, edge_score, edge_pred
                        continue
                    predictions[i]['edges'] = edge_pred
                    if self.compute_confidence:
                        predictions[i]['edge_scores'] = edge_score
                        edge_score_values = list(edge_score.values())
                        predictions[i]['edge_score_product'] = (
                            np.sqrt(np.prod(edge_score_values)).item()
                            if edge_score_values else 1.0
                        )
                        predictions[i]['overall_score'] = predictions[i][atom_format]['average_token_score'] * \
                                                          predictions[i]['edge_score_product']
                        predictions[i][atom_format].pop('average_token_score')
                        predictions[i].pop('edge_score_product')
                    del hidden, indices, pred, prob, edge_score
                if allow_sep and atom_format == 'chartok_coords':
                    for prediction in predictions:
                        if prediction.get('decode_quality_issue'):
                            continue
                        atom_data = prediction.get(atom_format) or {}
                        extension_score = atom_data.get(
                            'attachment_extension_score'
                        )
                        try:
                            materialized_edges, extension_metadata = (
                                materialize_attachment_extension(
                                    atom_data,
                                    prediction.get('edges') or [],
                                    sep_count=int(
                                        atom_data.get(
                                            'extension_sep_count', 0
                                        )
                                    ),
                                    invalid_extension_tokens=bool(
                                        atom_data.get(
                                            'extension_has_invalid_tokens',
                                            False,
                                        )
                                    ),
                                    attachment_confidence=extension_score,
                                )
                            )
                        except AttachmentExtensionError as exc:
                            prediction['decode_quality_issue'] = (
                                'molnextr_invalid_attachment_extension:'
                                + str(exc)
                            )
                            continue
                        prediction['edges'] = materialized_edges
                        prediction['attachment_extension'] = extension_metadata
                        prediction['decode_atom_count'] = len(
                            atom_data.get('symbols') or []
                        )
                        if self.compute_confidence:
                            anchor_index = extension_metadata[
                                'anchor_atom_index'
                            ]
                            dummy_index = extension_metadata[
                                'dummy_atom_index'
                            ]
                            prediction.setdefault('edge_scores', {})[
                                (anchor_index, dummy_index)
                            ] = float(extension_score or 0.0)
                for decoder in self.decoder.values():
                    if hasattr(decoder, "decoder") and hasattr(decoder.decoder, "state"):
                        decoder.decoder.state["cache"] = None
                        decoder.decoder.state["src"] = None
                results.clear()
        return predictions
