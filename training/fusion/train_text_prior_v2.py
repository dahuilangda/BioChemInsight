"""Text-prior projector v2: refs via the production TrainDataset path.

The hand-rolled collate was falsified at the decoder refs contract
(design §41); this version constructs refs exactly the way production
training does — TrainDataset with formats=['chartok_coords','edges'] over
a parquet — so the teacher-forced shapes are correct by construction.
"""
import sys, json, time, types, os, random
from pathlib import Path

sys.path.insert(0, "/data/BioChemInsight")
import torch

ROOT = Path("/data/BioChemInsight")

from utils.structure_recognition import _load_molnextr
from utils.fusion.text_prior import TextPriorProjector


def build_args():
    return types.SimpleNamespace(
        formats=['chartok_coords', 'edges'], input_size=384, test_file='',
        augment=False, data_path='/data/BioChemInsight', labelled=True,
        save_path='/tmp/tp_ds2', save_image=False, mask_ratio=1.0,
        continuous_coords=False, shuffle_nodes=False, dynamic_indigo=False,
        mol_augment=False, default_option=False, include_condensed=False,
        pseudo_coords=False, predict_coords=None, coords_file=None,
    )


def main():
    import pandas as pd
    from torch.utils.data import DataLoader
    from utils.MolNexTR.dataset import TrainDataset
    from utils.MolNexTR.tokenization import CharTokenizer

    device = torch.device("cuda")
    mol = _load_molnextr(None, moe_config_path="experiments/moe/production/moe_config.json")
    for p in mol.encoder.parameters():
        p.requires_grad_(False)
    for p in mol.moe_decoder.parameters():
        p.requires_grad_(False)

    os.makedirs('/tmp/tp_ds2', exist_ok=True)
    tok = {'chartok_coords': CharTokenizer(
        64, 'utils/MolNexTR/vocab/vocab_chars.json', sep_xy=True,
        continuous_coords=False)}

    # complete domain: real uspto crops (label 0); markush domain: mg2 v3
    df_c = pd.read_parquet(
        ROOT / "training/molnextr_markush/data/generated/real_uspto_complete/data.parquet").head(6000)
    df_c = df_c[df_c["node_coords"].astype(str) == ""].head(4000) if "node_coords" in df_c else df_c
    df_m = pd.read_parquet(
        ROOT / "training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v3/train/data.parquet").head(6000)
    df_m = df_m[df_m["node_coords"].astype(str) != ""].head(3000)  # pose rows carry real coords
    print(f"domains: complete={len(df_c)} markush={len(df_m)}", flush=True)

    def _collate(batches):
        return [b[0] for b in batches], torch.stack([b[1] for b in batches if b[1] is not None]) if all(b[1] is not None for b in batches) else None, [b[2] for b in batches]

    def make_loader(df):
        args = build_args()
        ds = TrainDataset(args, df, tok, split='train')
        return DataLoader(ds, batch_size=4, shuffle=True, num_workers=0,
                          collate_fn=_collate)

    loader_c = make_loader(df_c)
    loader_m = make_loader(df_m)

    projector = TextPriorProjector(out_dim=1024).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=3e-4)

    def run_domain(loader, name, epochs=1):
        projector.train()
        step = 0
        for epoch in range(epochs):
            for ids, images, ref_dicts in loader:
                if images is None or not ref_dicts:
                    continue
                keep = [i for i, d in zip(ids, ref_dicts) if d]
                ref_dicts = [d for d in ref_dicts if d]
                images = images.to(device)
                refs = {}
                from torch.nn.utils.rnn import pad_sequence
                from utils.MolNexTR.tokenization import PAD_ID
                fmt_keys = [k for k in ref_dicts[0].keys()]
                for key in fmt_keys:
                    vals = [d.get(key) for d in ref_dicts]
                    if any(v is None for v in vals):
                        continue
                    if key in ('chartok_coords',):
                        tensors = [torch.as_tensor(v, dtype=torch.long) for v in vals]
                        lens = torch.tensor([[len(v)] for v in tensors], dtype=torch.long)
                        pad = pad_sequence(tensors, batch_first=True, padding_value=PAD_ID)
                        refs[key] = [pad.to(device), lens.to(device)]
                    elif key == 'atom_indices':
                        tensors = [torch.as_tensor(v, dtype=torch.long) for v in vals]
                        lens = torch.tensor([[len(v)] for v in tensors], dtype=torch.long)
                        pad = pad_sequence(tensors, batch_first=True, padding_value=PAD_ID)
                        refs[key] = [pad.to(device), lens.to(device)]
                    elif key == 'edges':
                        maxn = max(v.shape[0] for v in vals)
                        mat = torch.full((len(vals), maxn, maxn), -100, dtype=torch.long)
                        for r, v in enumerate(vals):
                            mat[r, :v.shape[0], :v.shape[0]] = v
                        refs[key] = mat.to(device)
                    elif all(isinstance(v, (list, tuple)) or torch.is_tensor(v) for v in vals):
                        try:
                            tensors = [torch.as_tensor(v) for v in vals]
                            refs[key] = pad_sequence(tensors, batch_first=True).to(device)
                        except (ValueError, RuntimeError):
                            pass  # non-padable metadata key; skip
                refs['decoder_task_mask'] = torch.ones(len(ref_dicts), dtype=torch.bool, device=device)
                labels = torch.zeros(len(ref_dicts), dtype=torch.long, device=device) \
                    if name == 'complete' else torch.ones(len(ref_dicts), dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    features, hiddens = mol.encoder(images)
                text_cells = [{"bbox": [0.1, 0.1, 0.9, 0.9], "text": "R1"}]  # minimal smoke text
                features = projector(features, text_cells)
                flat = [h if h.dim() == 3 else h.flatten(1, 2) for h in hiddens]
                fwd = mol.moe_decoder(features, refs, structure_labels=labels,
                                      encoder_hiddens=flat)
                losses = mol.moe_decoder.compute_moe_loss(fwd, structure_labels=labels)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    step += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                optimizer.step()
                step += 1
                if step % 50 == 0:
                    print(f"[{name}] ep{epoch} step{step} loss={float(loss):.4f}", flush=True)
        return step

    sc = run_domain(loader_c, 'complete')
    sm = run_domain(loader_m, 'markush')
    out = ROOT / "experiments/fusion/text_prior"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"projector": projector.state_dict()}, out / "projector_v2.pth")
    print(f"DONE complete_steps={sc} markush_steps={sm}", flush=True)


if __name__ == "__main__":
    main()
