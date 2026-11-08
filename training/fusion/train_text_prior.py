"""Train the VTL text-prior projector on MG2 markush pages (frozen all else).

Two-phase protocol (MarkushGrapher-2): everything frozen except the text
projector. Teacher forcing via compute_moe_loss on the markush cxsmiles;
cells = the OCR text+bboxes shipped with MG2. Baseline-vs-trained A/B on a
held-out split measures the text-prior delta.
"""
import sys, json, time, random
from pathlib import Path

sys.path.insert(0, "/data/BioChemInsight")
import numpy as np
import torch

ROOT = Path("/data/BioChemInsight")

from utils.structure_recognition import _load_molnextr
from utils.fusion.text_prior import TextPriorProjector
from training.fusion.train_stage_b import build_tokenizer
from utils.fusion.page_dataset import MG2PageDataset
import pyarrow.parquet as pq


def page_cells(parquet_path: Path, want_row: int):
    pf = pq.ParquetFile(parquet_path)
    row = 0
    for batch in pf.iter_batches(columns=["cells"], batch_size=64):
        for cells in batch.column("cells").to_pylist():
            if row == want_row:
                return cells
            row += 1
    return []


class Mg2TextPriorDataset(torch.utils.data.Dataset):
    """(production-crop tensor, cells, chartok target) per page."""

    def __init__(self, tokenizer, limit=0):
        self.ds = MG2PageDataset(
            ROOT / "training/molnextr_markush/data/raw/markushgrapher2",
            limit=limit)
        self.tokenizer = tokenizer
        self.samples = []  # (index_in_ds, cells)
        for i in range(len(self.ds)):
            meta = self.ds.index[i]
            cells = page_cells(Path(meta["parquet"]), meta["row"])
            if cells:
                self.samples.append((i, cells))
        random.Random(7).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, cells = self.samples[idx]
        item = self.ds[i]
        gt = item["cxsmiles"].split("|")[0].strip() or "C"
        label, indices = self.tokenizer.smiles_to_sequence(gt, coords=None, mask_ratio=1)
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(gt)
        edges = torch.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=torch.long) if mol else None
        if mol is not None:
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                code = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                        Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}.get(
                    bond.GetBondType(), 1)
                edges[a, b] = code
                edges[b, a] = code
        import cv2
        from utils.fusion.fusion_model import CascadeStructureRecognizer
        bgr = item["image"][..., ::-1].copy()
        t = CascadeStructureRecognizer.production_crop_transform(bgr)
        n_stream = len([i for i in indices if i < 480])  # atoms in the target
        n = max(n_stream, edges.shape[0])
        padded = torch.full((n, n), -100, dtype=torch.long)
        padded[: edges.shape[0], : edges.shape[0]] = edges
        atom_ids = torch.tensor([i for i in indices if i < 480], dtype=torch.long)
        return {"image": t, "cells": cells,
                "target": torch.LongTensor(label[:300]),
                "edges": padded, "atom_ids": atom_ids}


def collate(batch):
    images = torch.stack([b["image"] for b in batch])
    targets = [b["target"] for b in batch]
    edges = [b["edges"] for b in batch]
    cells = [b["cells"] for b in batch]
    from torch.nn.utils.rnn import pad_sequence
    lengths = torch.tensor([[min(len(t), 300)] for t in targets], dtype=torch.long)
    labels = pad_sequence(targets, batch_first=True, padding_value=0)
    if labels.shape[1] < 300:
        labels = torch.nn.functional.pad(labels, (0, 300 - labels.shape[1]), value=0)
    max_n = max(e.shape[0] for e in edges)
    edge_mat = torch.full((len(edges), max_n, max_n), -100, dtype=torch.long)
    for r, e in enumerate(edges):
        edge_mat[r, : e.shape[0], : e.shape[0]] = e
    atom_ids = [b["atom_ids"] for b in batch]
    return images, cells, {"chartok_coords": [labels, lengths],
                           "atom_indices": [pad_sequence(
                               atom_ids, batch_first=True, padding_value=0),
                                             torch.tensor([[len(a)] for a in atom_ids], dtype=torch.long)],
                           "edges": edge_mat,
                           "decoder_task_mask": torch.ones(len(batch), dtype=torch.bool)}


def main():
    device = torch.device("cuda")
    mol = _load_molnextr(None, moe_config_path="experiments/moe/production/moe_config.json")
    for p in mol.encoder.parameters():
        p.requires_grad_(False)
    for p in mol.moe_decoder.parameters():
        p.requires_grad_(False)

    projector = TextPriorProjector(out_dim=1024).to(device)
    tokenizer = build_tokenizer()
    full = Mg2TextPriorDataset(tokenizer)
    split = int(len(full) * 0.9)
    train_set = torch.utils.data.Subset(full, range(split))
    eval_set = torch.utils.data.Subset(full, range(split, len(full)))
    loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True,
                                         num_workers=0, collate_fn=collate)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=3e-4)

    step = started = 0
    projector.train()
    for epoch in range(3):
        for images, cells, refs in loader:
            images = images.to(device)
            for k, v in refs.items():
                if isinstance(v, list):
                    refs[k] = [x.to(device) for x in v]
                elif torch.is_tensor(v):
                    refs[k] = v.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features, hiddens = mol.encoder(images)
                features = features.detach()
            # encoder frozen -> features are constants; projector adds the
            # trainable text residual and the loss backprop reaches ONLY it
            features = projector(features, cells[0])
            flat = [h if h.dim() == 3 else h.flatten(1, 2) for h in hiddens]
            fwd = mol.moe_decoder(features, refs, structure_labels=torch.ones(
                images.shape[0], dtype=torch.long, device=device),
                encoder_hiddens=flat)
            losses = mol.moe_decoder.compute_moe_loss(
                fwd, structure_labels=torch.ones(
                    images.shape[0], dtype=torch.long, device=device))
            loss = losses["loss"]
            if not torch.isfinite(loss):
                step += 1
                continue
            # re-enable grads through projector path only
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{epoch} step{step} loss={float(loss):.4f}", flush=True)
    torch.save({"projector": projector.state_dict()},
               ROOT / "experiments/fusion/text_prior/projector.pth")
    print(f"saved projector; steps={step}", flush=True)


if __name__ == "__main__":
    main()
