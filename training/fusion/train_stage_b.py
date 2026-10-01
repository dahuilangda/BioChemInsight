#!/usr/bin/env python
"""Stage-B training: joint recognition adaptation over RoI features.

Recognition supervision comes from synthetic multi-structure pages whose
pasted crops carry known SMILES (self-labeled — no teacher calls). The
decoder consumes per-box RoI token grids via the delivered machinery:

  page -> (frozen) Swin -> detection head -> RoICollector -> per-box tokens
        -> expand_refs_for_boxes (labels per box, page-major order)
        -> reanchor_refs (RoI-frame supervision)
        -> MoEDecoder.training_forward (teacher-forced)

Coordinate supervision is symbol-only (mask_ratio=1 semantics: every atom
token followed by two MASK tokens) — synthetic pages have no atom coords.
Edges come from the RDKit graph of each SMILES (same construction as the
real-uspto complete shard). structure_labels route every box to expert0
(complete) since the crop pool is complete-compounds only.

Usage:
  python training/fusion/train_stage_b.py --pages 20000 --epochs 1 \
      --output experiments/fusion/stage_b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.fusion.fusion_model import FusionConfig, FusedStructureRecognizer
from utils.fusion.page_dataset import MG2PageDataset
from utils.fusion.synth_pages import SynthPageConfig, SyntheticMultiStructureDataset

PAD_ID = 0
MAX_LEN = 480


def build_tokenizer():
    """Match the production tokenizer wiring (tokenization.py:622)."""
    from utils.MolNexTR.tokenization import CharTokenizer

    return CharTokenizer(64, str(ROOT / "utils/MolNexTR/vocab/vocab_chars.json"),
                         sep_xy=True, continuous_coords=False)


class PageWithRefs(torch.utils.data.Dataset):
    """Wraps a page dataset; emits (image_tensor, gt_boxes, per-page refs).

    Per-structure refs are built the way ``TrainDataset`` does for
    coordinate-missing rows: chartok sequence with MASK coords, atom
    indices, RDKit edges, structure label 0 (complete).
    """

    def __init__(self, inner, tokenizer, size: int = 384) -> None:
        self.inner = inner
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self) -> int:
        return len(self.inner)

    def _structure_refs(self, smiles: str):
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        label, indices = self.tokenizer.smiles_to_sequence(smiles, coords=None, mask_ratio=1)
        if len(label) > MAX_LEN:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        n = mol.GetNumAtoms()
        edges = torch.zeros((n, n), dtype=torch.long)
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            code = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                    Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}.get(bond.GetBondType(), 1)
            edges[a, b] = code
            edges[b, a] = code
        return {
            "chartok_coords": torch.LongTensor(label),
            "atom_indices": torch.LongTensor([i for i in indices if i < MAX_LEN]),
            "edges": edges,
        }

    def __getitem__(self, index: int):
        import cv2

        item = self.inner[index]
        resized = cv2.resize(item["image"], (self.size, self.size),
                             interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gray = resized.mean(axis=2, keepdims=True)
        normalized = (gray - 0.447) / 0.226  # ImageNet gray, as Stage A
        image = np.repeat(normalized, 3, axis=2).transpose(2, 0, 1)
        structures = []
        boxes = []
        for box, smiles in zip(item["boxes"], item["smiles"]):
            refs = self._structure_refs(smiles)
            if refs is None:
                continue
            boxes.append(box)
            structures.append(refs)
        if not structures:  # degenerate page: fall back to full-page box
            structures = [self._structure_refs("C")]
            boxes = [np.array([0.02, 0.02, 0.98, 0.98], dtype=np.float32)]
        return {
            "image": torch.from_numpy(image),
            "boxes": torch.tensor(np.array(boxes, dtype=np.float32)),
            "structures": structures,
        }


def collate(batch):
    images = torch.stack([b["image"] for b in batch])
    gt_boxes = [b["boxes"] for b in batch]
    # page-concatenated structure rows (structure_counts axis)
    all_structures: list[dict] = []
    structure_counts = [len(b["structures"]) for b in batch]
    for b in batch:
        all_structures.extend(b["structures"])
    lengths = torch.tensor([[len(s["chartok_coords"])] for s in all_structures], dtype=torch.long)
    max_t = int(lengths.max())
    atom_idx_rows = [s["atom_indices"] for s in all_structures]
    atom_idx_lens = torch.tensor([[len(a)] for a in atom_idx_rows], dtype=torch.long)
    # The decoder sizes its edge logits from the token stream's atom count,
    # so the edge target must use the SAME N (not the RDKit N, which can
    # differ for bracket/charged spellings).
    max_n = max(int(atom_idx_lens.max()), max(int(s["edges"].shape[0]) for s in all_structures))
    labels = torch.full((len(all_structures), max_t), PAD_ID, dtype=torch.long)
    edge_mat = torch.full((len(all_structures), max_n, max_n), -100, dtype=torch.long)
    for row, s in enumerate(all_structures):
        seq = s["chartok_coords"]
        labels[row, : len(seq)] = seq
        n = s["edges"].shape[0]
        edge_mat[row, :n, :n] = s["edges"]
    from torch.nn.utils.rnn import pad_sequence

    refs = {
        "chartok_coords": [labels, lengths],
        "atom_indices": [pad_sequence(atom_idx_rows, batch_first=True, padding_value=PAD_ID),
                         atom_idx_lens.reshape(-1, 1)],
        "edges": edge_mat,
        "decoder_task_mask": torch.ones(len(all_structures), dtype=torch.bool),
    }
    structure_labels = torch.zeros(len(all_structures), dtype=torch.long)  # complete
    # ids are LOCAL per page ([0..count-1]); expand_refs_for_boxes applies
    # the page offset itself when structure_counts is provided
    box_structure_ids = [list(range(count)) for count in structure_counts]
    return images, gt_boxes, refs, structure_labels, box_structure_ids, structure_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synth", "mg2"], default="synth")
    parser.add_argument("--pages", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--unfreeze-interface", action="store_true",
                        help="B1: unfreeze enc_trans/LN/attachment projections (Bridging)")
    parser.add_argument("--unfreeze-decoder-last", type=int, default=0,
                        help="unfreeze the last N decoder cross-attention layers (0 = frozen)")
    parser.add_argument("--output", default="experiments/fusion/stage_b")
    parser.add_argument("--stage-a-heads", default="experiments/fusion/stage_a_p2/stage_a_heads.pth")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    from utils.structure_recognition import _load_molnextr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    mol = _load_molnextr(None, moe_config_path="experiments/moe/production/moe_config.json")
    model = FusedStructureRecognizer(mol.encoder, mol.moe_decoder,
                                     FusionConfig(stage="ab")).to(device)
    if Path(args.stage_a_heads).exists():
        state = torch.load(args.stage_a_heads, map_location=device)
        model._head.load_state_dict(state["head"])
        model._collector.load_state_dict(state["collector"])
        print(f"loaded Stage-A heads from {args.stage_a_heads}", flush=True)

    # freeze policy: encoder + decoder frozen; detection head + RoI projection
    # trainable; optionally unfreeze decoder last-N cross-attn layers
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    if args.unfreeze_decoder_last:
        cross_layers = [name for name, _ in model.decoder.named_parameters()
                        if "cross_attention" in name or "cross_attn" in name]
        chosen = cross_layers[-args.unfreeze_decoder_last:]
        for name, parameter in model.decoder.named_parameters():
            if name in chosen:
                parameter.requires_grad_(True)
    if args.unfreeze_interface:
        # Bridging-style (arXiv:2404.04624): keep frozen backbone weights but
        # train the interface — encoder projection, LayerNorms, and the
        # multiscale attachment projections. Zero-init residual would be the
        # full recipe; interface unfreeze is the quick B1 variant.
        interface_markers = ("enc_trans_layer", "enc_pos_emb",
                             "LayerNorm", "norm", "_project_level",
                             "_project_multiscale", "attachment_set_heads")
        for name, parameter in model.decoder.named_parameters():
            if any(marker in name for marker in interface_markers):
                parameter.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable tensors={len(trainable)} "
          f"(decoder unfrozen last {args.unfreeze_decoder_last})", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    tokenizer = build_tokenizer()
    if args.data == "synth":
        inner = SyntheticMultiStructureDataset(
            ROOT / "training/molnextr_markush/data/raw/molscribe/uspto_mol/train_200k.csv",
            ROOT / "training/molnextr_markush/data/raw/molscribe",
            pages=args.pages, config=SynthPageConfig(seed=23),
        )
    else:
        inner = MG2PageDataset(ROOT / "training/molnextr_markush/data/raw/markushgrapher2",
                               limit=args.pages)
    loader = DataLoader(PageWithRefs(inner, tokenizer), batch_size=args.batch_size,
                        shuffle=True, num_workers=4, collate_fn=collate)

    model.train()
    step, started = 0, time.time()
    running: dict[str, float] = {}
    for epoch in range(args.epochs):
        for images, gt_boxes, refs, structure_labels, box_ids, structure_counts in loader:
            images = images.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            for key, value in refs.items():
                if isinstance(value, list):
                    refs[key] = [v.to(device) for v in value]
                elif torch.is_tensor(value):
                    refs[key] = value.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, refs,
                           structure_labels=structure_labels.to(device),
                           gt_boxes=gt_boxes,
                           box_structure_ids=box_ids,
                           structure_counts=structure_counts)
            recognition = output.get("recognition")
            if recognition is None:
                step += 1
                continue
            # MoEDecoder.forward returns the RAW fwd payload; the training
            # losses are computed by compute_moe_loss (same split as the
            # production trainer in moe_trainer/entry.py).
            det_loss = output["losses"].get("loss_det")
            expanded = output.get('per_box_refs')
            box_labels = expanded.structure_labels if expanded is not None else structure_labels
            losses = model.decoder.compute_moe_loss(
                recognition, structure_labels=box_labels.to(device))
            loss = losses["loss"]
            if det_loss is not None:
                loss = loss + det_loss  # FOTS-style joint objective, lambda=1
            if loss is None or not torch.isfinite(loss):
                step += 1
                skipped = getattr(main, '_skipped', 0) + 1
                main._skipped = skipped
                if skipped <= 3:
                    val = 'None' if loss is None else float(loss)
                    print(f'step {step}: skip (loss={val})', flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            step += 1
            running["loss"] = running.get("loss", 0.0) * 0.95 + float(loss) * 0.05
            if step % args.log_every == 0:
                keys = list((losses.keys() if isinstance(losses, dict) else ["loss"]))[:6]
                print(f"epoch {epoch} step {step} loss={running['loss']:.4f} "
                      f"keys={keys} ({time.time() - started:.0f}s)", flush=True)

    save = {"head": model._head.state_dict(),
            "collector": model._collector.state_dict()}
    decoder_trainable = {n: p for n, p in model.decoder.named_parameters() if p.requires_grad}
    if decoder_trainable:
        save["decoder_interface"] = {n: p.detach().cpu() for n, p in decoder_trainable.items()}
    torch.save(save, out_dir / "stage_b_heads.pth")
    (out_dir / "summary.json").write_text(json.dumps({
        "steps": step, "epochs": args.epochs, "data": args.data,
        "pages": args.pages, "final_running_loss": running.get("loss"),
        "unfrozen_decoder_layers": args.unfreeze_decoder_last,
    }, indent=2))
    print(f"saved -> {out_dir / 'stage_b_heads.pth'}", flush=True)


if __name__ == "__main__":
    main()
