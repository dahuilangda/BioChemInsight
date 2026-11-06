#!/bin/bash
# Round-4 (CDK layout switch) eval ladder on GPU 1:
#   6 real patent figures (bare decode)  — baseline r3 epoch_0 mean 0.9116
#   val26 / reg20 regression             — baselines 0.9020 / 0.9405
#   MG2 official test split (48 pages)   — markush capability probe
#   WildMol-384 identity                 — baseline r3 0.1484, floor 0.1434
set -u
EV=/data/molsight_realpatent_sft_r4cdk/eval
PAIRS=/data/molsight_realpatent_sft/eval
BARE=/data/BioChemInsight/training/molsight_markush/bare_decode_eval.py
PY=/data/molsight_venv/bin/python
CKPT=$1          # e.g. /data/molsight_repo/runs/markush_v14_realpatent_sft_r4cdk/ckpt_model/epoch_1.pth
TAG=$2           # e.g. r4cdk_ep1
MG2=/data/molsight_realpatent_sft_r4cdk/eval_mg2/pairs_mg2.json
WILDMOL=/tmp/eval_molsight_ft.py
mkdir -p $EV

for SET in 6fig val26 reg20; do
  echo "=== ${TAG} ${SET} ==="
  CUDA_VISIBLE_DEVICES=1 $PY -I $BARE --checkpoint $CKPT --pairs $PAIRS/pairs_${SET}.json \
      --out $EV/${TAG}_${SET}.tsv --out_json $EV/${TAG}_${SET}.json \
      > $EV/${TAG}_${SET}.log 2>&1
  tail -1 $EV/${TAG}_${SET}.log
done

echo "=== ${TAG} mg2 ==="
CUDA_VISIBLE_DEVICES=1 $PY -I $BARE --checkpoint $CKPT --pairs $MG2 \
    --out $EV/${TAG}_mg2.tsv --out_json $EV/${TAG}_mg2.json \
    > $EV/${TAG}_mg2.log 2>&1
tail -1 $EV/${TAG}_mg2.log

echo "=== ${TAG} wildmol384 ==="
CUDA_VISIBLE_DEVICES=1 $PY -I $WILDMOL $CKPT /tmp/${TAG}_wildmol.txt \
    > $EV/${TAG}_wildmol.log 2>&1
tail -1 $EV/${TAG}_wildmol.log
echo "ALL_EVALS_DONE $TAG"
