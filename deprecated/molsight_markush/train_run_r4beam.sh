#!/bin/bash
# Round-4 (beam layout) training run: round-3 recipe verbatim
# (enc 2e-6 / dec 5e-6, cosine, warmup 0.02, accum 5, label_smoothing 0.1,
# AMP/fp16), GPU 1.  Batch size overridable via BS (the driver retries at
# BS=3 if the GPU runs out of memory).
#
# GPU gate: the host shares GPU 1 with other studies.  Wait while ANY other
# train.py process is running so the 16GB card is never double-booked.
# Usage: train_run_r4beam.sh <data_dir> <exp_name> <epochs> <load_path>
set -eu
DATA_DIR=$1
EXP=$2
EPOCHS=$3
LOAD_PATH=$4
SEED=${SEED:-42}
BS=${BS:-6}

export HF_HOME=/data/LLM_Models/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=1
PORT=$((29700 + RANDOM % 200))

while true; do
  BUSY=$(pgrep -af "train.py" | grep -v "grep" || true)
  if [ -z "$BUSY" ]; then break; fi
  echo "[gate] training active on GPU 1, waiting 60s: $(echo "$BUSY" | head -1 | cut -c1-100)"
  sleep 60
done

LOG_DIR=/data/molsight_repo/runs/$EXP
mkdir -p $LOG_DIR/ckpt_model

cd /data/molsight_repo
/data/molsight_venv/bin/torchrun --nproc_per_node=1 --master_port=$PORT train.py \
  --do_train --do_valid --amp --smiles_only \
  --seed $SEED --print_freq 100 \
  --encoder efficientvit --decoder transformer \
  --dec_n_layer 12 --dec_n_head 8 \
  --use_qknorm --use_swiglu --use_rmsnorm \
  --dropout 0.5 --embed_dim 512 \
  --data_path $DATA_DIR \
  --train_datasets uspto \
  --valid_file real/val.csv \
  --vocab_file vocab/vocab_chars.json \
  --formats char --num_workers 4 --input_size 512 \
  --augment --max_len 1024 \
  --epochs $EPOCHS --batch_size $BS \
  --encoder_lr 2e-6 --decoder_lr 5e-6 --predictor_lr 5e-6 \
  --weight_decay 1e-4 --max_grad_norm 1.0 \
  --scheduler cosine --warmup_ratio 0.02 --accum_freq 5 \
  --label_smoothing 0.1 --save_mode last \
  --resume --load_model_only \
  --load_path $LOAD_PATH \
  --log_base_dir /data/molsight_repo/runs/ --exp_name $EXP
echo "TRAIN_DONE $EXP"
