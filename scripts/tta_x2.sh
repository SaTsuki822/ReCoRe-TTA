#!/usr/bin/env bash
# ReCoRe TTA x2（全参考 Result-A 示例；请把路径替换为实际数据）
set -e
cd "$(dirname "$0")/.."
python main_tta.py \
  --scale 2 \
  --lr_dir /path/to/results-A/LR_bicubic/X2 \
  --hr_dir /path/to/results-A/HR \
  --backbone_ckpt checkpoints/VanillaMamba_x2.pth \
  --estimator_ckpt checkpoints/estimator/estimator_final.pt \
  --save_dir results/recore_x2 --metrics niqe