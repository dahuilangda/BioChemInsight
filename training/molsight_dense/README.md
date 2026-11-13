# MolSight visible graph branch

This adds pixel localization, visual bond reading and registered visible-group expansion to the existing MolSight checkpoint. Original encoder and AR decoder tensors remain frozen. No SELFIES. Complete molecules only.

See [audited results](../../docs/molsight_visible_root_fix_2026-11-12.md). Synthetic development gains do **not** establish real cyclic peptide generalization. The standalone visual branch is much weaker than original AR on real USPTO. `preserve_ar` remains the default; specialist policies caused real regressions.

Run from the repository root with the existing `/data/molsight_venv` environment:

```bash
export PYTHONPATH=.:training:training/molsight_runtime
export OMP_NUM_THREADS=2 NO_ALBUMENTATIONS_UPDATE=1
/data/molsight_venv/bin/python -m training.molsight_hybrid \
  --artifact /home/dahuilangda/molsight_work/molsight_dense_v1/artifact_visible_final_v1 \
  --inference-policy /home/dahuilangda/molsight_work/molsight_dense_v1/calibration_final_v1/policy.json \
  --image molecule.png --out prediction.json --device cuda:0 --policy preserve_ar
```

Output identifies the selected branch and preserves underlying predictions. `parse_success` means strict assembly succeeded; it does not mean the molecule is correct. Hidden atoms expanded from a group template have derived coordinates.

```bash
/data/molsight_venv/bin/python -m pytest -q \
  training/molsight_dense/tests training/molsight_visible/tests training/molsight_spatial/tests
```

Training uses `python -m training.molsight_dense.train --visible --preprocessing molsight_native` with explicit manifest, checkpoint, vocabulary and output paths. Run `--preflight` before a new data phase. Warmstarting preserves weights and explicitly resets optimizer state; `--resume` requires identical sources, configuration, provenance and checkpoint cursor. Never edit captured training or inference sources during active training.

## Root-loss experiments and resolution cascade

See [paired root-fix results](../../docs/molsight_root_loss_resolution_2026-11-12.md). On the same real 5607 images, 1500 additional updates gave 218 exact for the ordinary-loss control, 681 with weak-center/hard-atom supervision, and 910 with additional small-molecule coverage. These are **standalone graph results**, substantially below original AR accuracy.

The retained MolSight backbone and vocabulary are unchanged. `artifact_root_hard_v1` and `artifact_root_diverse_v1` embed threshold 0.5; the older artifact's calibration file must not be supplied to them.

Experimental cyclic specialist with a declared image-resolution cascade:

```bash
/data/molsight_venv/bin/python -m training.molsight_hybrid \
  --artifact /home/dahuilangda/molsight_work/molsight_dense_v1/artifact_root_diverse_v1 \
  --image molecule.png --out prediction.json --device cuda:0 \
  --policy cyclic_visible --graph-sizes 768 1024 512
```

The cascade preserves a strictly assembled primary graph; on failure it re-reads the original pixels at the next fixed resolution. It does not repair graphs or select using references. Output records the chosen branch, resolution and failed attempts. This adds inference work. Use `--policy preserve_ar` to retain the conservative AR-first default. Cyclic and linear peptide recognition remain experimental, with limited real macrocyclic evaluation coverage.

Training flags `--center-recall-weight .2 --hard-weight .1` add worst-center and worst-atom supervision. `--chemistry-weight .2` is an ablation option; it did not improve exact beyond the hard-loss group at the tested budget, so the retained hard model leaves it at zero. Raw averaged atom accuracy is not a release criterion.
