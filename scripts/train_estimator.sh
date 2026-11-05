#!/usr/bin/env bash
# 预训练相对退化估计器 E_phi（M2）
set -e
cd "$(dirname "$0")/.."
# 无数据冒烟示例；有 M3FD 时把 --img_dir 指向 M3FD 源图目录
python -m src.estimator.train \
  --img_dir "" \
  --save_dir checkpoints/estimator \
  --crop_size 128 --length 2000 --epochs 20 --batch_size 16 \
  --lr 1e-3 --calibrate --seed 3407