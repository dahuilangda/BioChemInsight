#!/bin/bash
# Round-4z driver: wait for the zig-zag render -> assemble the mix ->
# train (GPU-gated) -> full eval ladder -> acceptance summary.
# Runs unattended under nohup; ~1h render + ~20min training queue + evals.
set -u
W=/data/BioChemInsight/training/molsight_markush
DATA=/data/molsight_realpatent_sft_r4z
EV=$DATA/eval
PAIRS=/data/molsight_realpatent_sft/eval
R4PAIRS=/data/molsight_realpatent_sft_r4
BARE=$W/bare_decode_eval.py
WILDMOL=/tmp/eval_molsight_ft.py
PY=/data/molsight_venv/bin/python
R3_CKPT=/data/molsight_repo/runs/markush_v14_realpatent_sft_r3/ckpt_model/epoch_0.pth
EXP=markush_v14_realpatent_sft_r4z
mkdir -p $EV

log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# 1) wait for the zig-zag pool render
while [ ! -f $DATA/zigzag_synth/render_manifest.json ]; do
  log "waiting for zigzag render manifest"; sleep 120
done
log "render manifest present"

# 2) assemble the mix
if [ ! -f $DATA/uspto_mol/train_680k.csv ]; then
  $PY -I $W/build_real_patent_sft_r4z.py > $DATA/build_mix.log 2>&1 \
    || { log "BUILD_FAILED"; exit 1; }
fi
log "mix assembled: $(wc -l < $DATA/uspto_mol/train_680k.csv) rows"

# 3) train (gate-queued behind the study orchestrator)
bash $W/train_run_r4z.sh $DATA $EXP 1 $R3_CKPT \
  > /data/molsight_repo/runs/${EXP}_train.log 2>&1 \
  || { log "TRAIN_FAILED"; exit 1; }
CKPT=/data/molsight_repo/runs/$EXP/ckpt_model/epoch_0.pth
[ -f $CKPT ] || { log "NO_CKPT"; exit 1; }
log "training done: $CKPT"

# 4) eval ladder (GPU 1)
for SET in 6fig val26 reg20; do
  log "EVAL $SET"
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
    $PY -I $BARE --checkpoint $CKPT --pairs $PAIRS/pairs_${SET}.json \
      --out $EV/r4z_${SET}.tsv --out_json $EV/r4z_${SET}.json \
      > $EV/r4z_${SET}.log 2>&1
done
log "EVAL markush200"
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  $PY -I $BARE --checkpoint $CKPT --pairs $R4PAIRS/markush_eval/pairs_markush200.json \
    --out $EV/r4z_markush200.tsv --out_json $EV/r4z_markush200.json \
    > $EV/r4z_markush200.log 2>&1
log "EVAL linear150"
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  $PY -I $BARE --checkpoint $CKPT --pairs $R4PAIRS/linear_eval/pairs_linear150.json \
    --out $EV/r4z_linear150.tsv --out_json $EV/r4z_linear150.json \
    > $EV/r4z_linear150.log 2>&1
log "EVAL wildmol r4z + r3 baseline"
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  $PY -I $WILDMOL $CKPT $EV/r4z_wildmol.json > $EV/r4z_wildmol.log 2>&1
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  $PY -I $WILDMOL $R3_CKPT $EV/r3_wildmol.json > $EV/r3_wildmol.log 2>&1
log "ALL_EVALS_DONE"
