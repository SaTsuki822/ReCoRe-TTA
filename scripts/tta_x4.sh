#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python main_tta.py \
  --scale 4 \
  --lr_dir /path/to/results-A/LR_bicubic/X4 \
  --hr_dir /path/to/results-A/HR \
  --backbone_ckpt checkpoints/VanillaMamba_x4.pth \
  --estimator_ckpt checkpoints/estimator/estimator_final.pt \
  --save_dir results/recore_x4 --metrics niqe