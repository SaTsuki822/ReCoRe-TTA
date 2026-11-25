#!/usr/bin/env bash
# Train the VanillaMamba backbone on single-channel source images (e.g. M3FD).
set -e
cd "$(dirname "$0")/.."
exec python -m src.backbone.train \
  --dataroot /mnt/workspace/lyk/dataset/M3FD/ir \
  --scale 2 \
  --save_dir checkpoints/backbone_x2_v2 \
  --val_root datasets --val_every 1 \
  --epochs 100 --batch_size 1 --num_workers 2 \
  --lr 2e-4 --min_lr 1e-6 --clean_prob 0.2 --seed 3407 "$@"
