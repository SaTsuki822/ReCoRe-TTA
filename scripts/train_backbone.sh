#!/usr/bin/env bash
# Train the VanillaMamba backbone on single-channel source images (e.g. M3FD).
set -e
cd "$(dirname "$0")/.."
python -m src.backbone.train \
  --dataroot /path/to/M3FD \
  --scale 2 \
  --save_dir checkpoints/backbone_x2 \
  --epochs 100 --batch_size 16 --lr 2e-4 --seed 3407