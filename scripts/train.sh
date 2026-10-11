#!/usr/bin/env bash
# Train the time-branch separator. Resumes automatically from the latest
# compatible checkpoint in ./ckpts (pass FRESH=1 to start over).
#   DATA_DIRS="D:/data/train1,D:/data/train2" DATA_WEIGHTS="3,1" ./scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/../main"
DATA_DIRS="${DATA_DIRS:-train1,train2}"
DATA_WEIGHTS="${DATA_WEIGHTS:-3,1}"
EXTRA=()
[[ "${FRESH:-0}" == "1" ]] && EXTRA+=(--fresh)
exec python mvsep.py --train \
  --data_dirs "$DATA_DIRS" --data_weights "$DATA_WEIGHTS" \
  --model_dim 256 --depth 16 --heads 8 \
  --segment_seconds 8 --inference_overlap_seconds 2 \
  --batch_size 1 --grad_accumulation 1 --num_workers 8 \
  --dataset_size 50000 --remix_probability 0.5 \
  --checkpoint_steps 8000 --weight_decay 0.0 --grad_clip 1.0 \
  --ema_decay 0.999 --precision bf16 --ckpt \
  "${EXTRA[@]}" "$@"
