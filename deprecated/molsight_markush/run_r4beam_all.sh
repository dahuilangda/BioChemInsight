#!/bin/bash
# Round-4 (beam side-chain layout) driver: wait for the beam pool render ->
# assemble the mix -> train 2 epochs from the r3 winner (GPU-gated, OOM
# falls back to batch 3) -> full eval ladder (6 real figures bare decode,
# val26 / reg20 regression, markush200 + linear150 + MG2 capability probes,
# WildMol-384 identity vs the r3 baseline) -> acceptance summary.
# Runs unattended under nohup.
set -u
W=/data/BioChemInsight/training/molsight_markush
DATA=/data/molsight_realpatent_sft_r4beam
EV=$DATA/eval
PAIRS=/data/molsight_realpatent_sft/eval
R4PAIRS=/data/molsight_realpatent_sft_r4
MG2=/data/molsight_realpatent_sft_r4cdk/eval_mg2/pairs_mg2.json
BARE=$W/bare_decode_eval.py
WILDMOL=/tmp/eval_molsight_ft.py
PY=/data/molsight_venv/bin/python
R3_CKPT=/data/molsight_repo/runs/markush_v14_realpatent_sft_r3/ckpt_model/epoch_0.pth
EXP=markush_v14_realpatent_sft_r4beam
RUNS=/data/molsight_repo/runs/$EXP
mkdir -p $EV

log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# 1) wait for the beam pool render (multi-hour job — see render_beam_pool.py)
while [ ! -f $DATA/beam_synth/render_manifest.json ]; do
  log "waiting for beam render manifest ($(ls $DATA/beam_synth/images 2>/dev/null | wc -l) images so far)"
  sleep 600
done
log "render manifest present: rows=$(wc -l < $DATA/beam_synth/train_680k.csv 2>/dev/null || echo '?')"

# 2) assemble the mix
if [ ! -f $DATA/uspto_mol/train_680k.csv ]; then
  $PY -I $W/build_real_patent_sft_r4beam.py > $DATA/build_mix.log 2>&1 \
    || { log "BUILD_FAILED"; exit 1; }
fi
log "mix assembled: $(wc -l < $DATA/uspto_mol/train_680k.csv) rows"

# 3) train 2 epochs (gate-queued; OOM -> retry at batch 3)
TAG=r4beam
bash $W/train_run_r4beam.sh $DATA $EXP 2 $R3_CKPT \
  > /data/molsight_repo/runs/${EXP}_train.log 2>&1
if ! ls $RUNS/ckpt_model/epoch_1.pth >/dev/null 2>&1; then
  if grep -qi "out of memory" /data/molsight_repo/runs/${EXP}_train.log; then
    log "OOM detected — retrying with batch_size 3"
    rm -rf $RUNS/ckpt_model
    BS=3 bash $W/train_run_r4beam.sh $DATA $EXP 2 $R3_CKPT \
      > /data/molsight_repo/runs/${EXP}_train_bs3.log 2>&1
  fi
fi
[ -f $RUNS/ckpt_model/epoch_0.pth ] || { log "NO_CKPT"; exit 1; }
log "training done: $RUNS/ckpt_model/epoch_{0,1}.pth"

# 4) eval ladder for both epochs
for EP in 0 1; do
  CKPT=$RUNS/ckpt_model/epoch_$EP.pth
  [ -f $CKPT ] || continue
  for SET in 6fig val26 reg20; do
    log "EVAL ep$EP $SET"
    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
      $PY -I $BARE --checkpoint $CKPT --pairs $PAIRS/pairs_${SET}.json \
        --out $EV/${TAG}_ep$EP_${SET}.tsv --out_json $EV/${TAG}_ep$EP_${SET}.json \
        > $EV/${TAG}_ep$EP_${SET}.log 2>&1
  done
  log "EVAL ep$EP markush200"
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
    $PY -I $BARE --checkpoint $CKPT --pairs $R4PAIRS/markush_eval/pairs_markush200.json \
      --out $EV/${TAG}_ep$EP_markush200.tsv --out_json $EV/${TAG}_ep$EP_markush200.json \
      > $EV/${TAG}_ep$EP_markush200.log 2>&1
  log "EVAL ep$EP linear150"
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
    $PY -I $BARE --checkpoint $CKPT --pairs $R4PAIRS/linear_eval/pairs_linear150.json \
      --out $EV/${TAG}_ep$EP_linear150.tsv --out_json $EV/${TAG}_ep$EP_linear150.json \
      > $EV/${TAG}_ep$EP_linear150.log 2>&1
  log "EVAL ep$EP mg2"
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
    $PY -I $BARE --checkpoint $CKPT --pairs $MG2 \
      --out $EV/${TAG}_ep$EP_mg2.tsv --out_json $EV/${TAG}_ep$EP_mg2.json \
      > $EV/${TAG}_ep$EP_mg2.log 2>&1
  log "EVAL ep$EP wildmol384"
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
    $PY -I $WILDMOL $CKPT $EV/${TAG}_ep$EP_wildmol.json \
      > $EV/${TAG}_ep$EP_wildmol.log 2>&1
done
log "EVAL wildmol r3 baseline re-run"
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  $PY -I $WILDMOL $R3_CKPT $EV/r3_wildmol_rerun.json \
    > $EV/r3_wildmol_rerun.log 2>&1

# 5) acceptance summary
$PY -I - "$EV" <<'PYEOF'
import json, sys, glob, os
ev = sys.argv[1]
def mean(p):
    try:
        with open(p) as fh: return json.load(fh).get("mean_tanimoto")
    except Exception: return None
def wildmol(p):
    # /tmp/eval_molsight_ft.py prints "identity=H/384=r" to its log
    try:
        with open(p) as fh:
            for line in fh:
                if "identity=" in line:
                    return float(line.split("identity=")[1].split("/")[0]) / 384
    except Exception:
        return None
    return None
r3 = {"6fig": 0.9116, "val26": 0.9020, "reg20": 0.9405}
rows = {}
for ep in (0, 1):
    rows[f"ep{ep}"] = {
        "6fig": mean(os.path.join(ev, f"r4beam_ep{ep}_6fig.json")),
        "val26": mean(os.path.join(ev, f"r4beam_ep{ep}_val26.json")),
        "reg20": mean(os.path.join(ev, f"r4beam_ep{ep}_reg20.json")),
        "markush200": mean(os.path.join(ev, f"r4beam_ep{ep}_markush200.json")),
        "linear150": mean(os.path.join(ev, f"r4beam_ep{ep}_linear150.json")),
        "mg2": mean(os.path.join(ev, f"r4beam_ep{ep}_mg2.json")),
        "wildmol384": wildmol(os.path.join(ev, f"r4beam_ep{ep}_wildmol.log")),
    }
best_ep = max(rows, key=lambda k: (rows[k]["6fig"] or 0))
b = rows[best_ep]
gates = {
    "seq6fig_gt_0.90": (b["6fig"] or 0) > 0.90,
    "val26_no_regression": (b["val26"] or 0) >= r3["val26"] - 0.005,
    "reg20_no_regression": (b["reg20"] or 0) >= r3["reg20"] - 0.005,
    "wildmol_drop_le_0.005": (rows[best_ep]["wildmol384"] or 0) >= 0.1434,
}
verdict = ("PASS — recommend switching production to r4 "
           f"({best_ep})" if all(gates.values())
           else "NOT MET — stay on r3; see bottleneck notes")
summary = {
    "verdict": verdict,
    "baseline_r3": r3,
    "wildmol_baseline_r3": 0.1484, "wildmol_floor": 0.1434,
    "r4beam": rows, "best_epoch": best_ep, "gates": gates,
    "render_manifest": "/data/molsight_realpatent_sft_r4beam/beam_synth/render_manifest.json",
    "assembly": "/data/molsight_realpatent_sft_r4beam/assembly_summary_r4beam.json",
    "train_log": "/data/molsight_repo/runs/markush_v14_realpatent_sft_r4beam_train.log",
    "checkpoints": {
        "epoch_0": "/data/molsight_repo/runs/markush_v14_realpatent_sft_r4beam/ckpt_model/epoch_0.pth",
        "epoch_1": "/data/molsight_repo/runs/markush_v14_realpatent_sft_r4beam/ckpt_model/epoch_1.pth"},
}
with open(os.path.join(ev, "FINAL_SUMMARY_R4BEAM.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
PYEOF
log "ALL_EVALS_DONE — summary at $EV/FINAL_SUMMARY_R4BEAM.json"
