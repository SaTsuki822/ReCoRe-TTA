#!/usr/bin/env bash
# 预训练相对退化估计器 E_phi（M2）
set -e
cd "$(dirname "$0")/.."
# 用法：bash scripts/train_estimator.sh --img_dir /path/to/M3FD/ir
# 自动按源图留出 10% 验证、10% 校准；可另传 --val_dir / --cal_dir。
exec python -m src.estimator.train \
  --save_dir checkpoints/estimator \
  --crop_size 128 --length 20000 --epochs 30 --batch_size 16 \
  --lr 1e-3 --calibrate --seed 3407 "$@"
