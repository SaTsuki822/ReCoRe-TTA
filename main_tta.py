"""ReCoRe test-time adaptation entry point.

Usage (from ReCoRe-main):
    # full-reference (Result-A/C)
    python main_tta.py --scale 2 --lr_dir <LR> --hr_dir <HR> \
        --backbone_ckpt checkpoints/VanillaMamba_x2.pth --estimator_ckpt checkpoints/estimator_final.pt
    # no-reference (Drone Vehicle / DJIThermalSR)
    python main_tta.py --scale 2 --lr_dir <LR> \
        --backbone_ckpt checkpoints/VanillaMamba_x2.pth --estimator_ckpt checkpoints/estimator_final.pt
    # reconstruction only (no-adaptation baseline)
    python main_tta.py --scale 2 --lr_dir <LR> --backbone_ckpt checkpoints/VanillaMamba_x2.pth --no_tta
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from src.backbone import build_backbone, load_backbone_weights
from src.data import SingleImageDataset, PairedImageDataset
from src.metrics import calculate_psnr, calculate_ssim, calculate_gmsd, calculate_niqe, calculate_maniqa
from src.tta import ReCoRe, TTAConfig
from src.tta.recore import load_estimator
from src.utils import set_seed, write_img, tensor_to_img


def _align_size(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def build_parser():
    p = argparse.ArgumentParser(description="ReCoRe test-time adaptation for infrared SR")
    # data / model
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("--lr_dir", type=str, required=True, help="test LR image directory")
    p.add_argument("--hr_dir", type=str, default=None, help="optional HR directory (full-reference evaluation)")
    p.add_argument("--backbone_ckpt", type=str, default=None, help="VanillaMamba pretrained weights")
    p.add_argument("--estimator_ckpt", type=str, default=None, help="frozen E_phi estimator weights")
    p.add_argument("--save_dir", type=str, default="./results/tta")
    p.add_argument("--device", type=str, default="cuda")
    # TTA hyper-parameters
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_up", type=float, default=1.0)
    p.add_argument("--lambda_anc", type=float, default=0.1)
    p.add_argument("--region_size", type=int, default=8)
    p.add_argument("--tau_A", type=float, default=0.75)
    p.add_argument("--T_A", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--delta_A", type=float, default=0.05)
    p.add_argument("--eps_A", type=float, default=0.10)
    p.add_argument("--tunable", type=str, default="out_proj",
                   help="tunable linear-layer name substrings (comma-separated)")
    # evaluation
    p.add_argument("--crop_border", type=int, default=0)
    p.add_argument("--metrics", type=str, default="niqe",
                   help="no-reference metric subset (comma-separated): niqe,maniqa")
    p.add_argument("--no_tta", action="store_true", help="reconstruct without adaptation (baseline)")
    p.add_argument("--warmup", type=int, default=0, help="warmup images before timing")
    p.add_argument("--timeit", action="store_true", help="measure per-image end-to-end GPU time")
    p.add_argument("--seed", type=int, default=3407)
    return p


def main(args):
    set_seed(args.seed)
    device = args.device

    # 1) backbone
    backbone = build_backbone(scale=args.scale, in_chans=1)
    if args.backbone_ckpt:
        backbone, skipped, unexpected = load_backbone_weights(backbone, args.backbone_ckpt, strict=True)
        if skipped:
            print(f"[backbone] {len(skipped)} keys skipped (channel-mismatch) e.g. {skipped[:3]}")
    backbone.to(device)
    backbone.eval()

    # 2) estimator (optional; not needed for --no_tta)
    estimator = None
    if not args.no_tta:
        if not args.estimator_ckpt:
            raise SystemExit("--estimator_ckpt is required for TTA (or pass --no_tta)")
        estimator = load_estimator(args.estimator_ckpt, device)

    # 3) ReCoRe
    cfg = TTAConfig(
        iterations=args.iterations, lr=args.lr,
        lambda_up=args.lambda_up, lambda_anc=args.lambda_anc,
        region_size=args.region_size, tau_A=args.tau_A, T_A=args.T_A,
        beta=args.beta, delta_A=args.delta_A, eps_A=args.eps_A,
        tunable_keywords=tuple(k for k in args.tunable.split(",") if k),
        seed=args.seed)
    recore = ReCoRe(backbone, estimator, cfg, device=device) if estimator is not None else None

    # 4) data
    if args.hr_dir:
        ds = PairedImageDataset(args.lr_dir, args.hr_dir, scale=args.scale)
        paired = True
    else:
        ds = SingleImageDataset(args.lr_dir)
        paired = False

    os.makedirs(args.save_dir, exist_ok=True)
    metric_keys = [k for k in args.metrics.split(",") if k]

    # warmup for timing
    if args.timeit and args.warmup > 0:
        for i in range(min(args.warmup, len(ds))):
            item = ds[i]
            lr = item[0].to(device)
            if args.no_tta:
                recore.reconstruct(lr) if recore else backbone(lr)
            else:
                recore.adapt(lr)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    acc = {}
    times = []
    for i in range(len(ds)):
        item = ds[i]
        lr = item[0].to(device)
        hr = item[1] if paired else None
        name = item[-1]

        t0 = time.perf_counter()
        if args.no_tta:
            sr = recore.reconstruct(lr) if recore is not None else backbone(lr)
        else:
            sr = recore.adapt(lr)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)

        write_img(os.path.join(args.save_dir, name), sr)
        sr_np = tensor_to_img(sr)

        row = {"img": name}
        if hr is not None:
            hr_np = hr.squeeze().numpy()
            a, b = _align_size(sr_np, hr_np)
            row["PSNR"] = calculate_psnr(a, b, crop_border=args.crop_border)
            row["SSIM"] = calculate_ssim(a, b, crop_border=args.crop_border)
            row["GMSD"] = calculate_gmsd(a, b, crop_border=args.crop_border)
        if "niqe" in metric_keys:
            row["NIQE"] = calculate_niqe(sr_np, crop_border=args.crop_border)
        if "maniqa" in metric_keys:
            row["MANIQA"] = calculate_maniqa(sr, device)
        for k, v in row.items():
            if isinstance(v, float):
                acc.setdefault(k, []).append(v)

        print(" | ".join(f"{k}={v if isinstance(v, str) else format(v, '.4f')}" for k, v in row.items()))

    # summary
    if acc:
        print("--- summary ---")
        for k, vs in acc.items():
            print(f"{k}: {sum(vs)/len(vs):.4f}")
    if args.timeit and times:
        print(f"avg GPU time per image: {sum(times)/len(times)*1000:.1f} ms")


if __name__ == "__main__":
    main(build_parser().parse_args())

