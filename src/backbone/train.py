"""Train the VanillaMamba backbone for infrared super-resolution.

Synthesizes LR-HR pairs from single-channel source images using the full degradation
model (blur -> downsampling -> non-uniformity/noise -> bad pixels -> output chain),
then trains the backbone with a pixel L1 loss.

Usage:
    python -m src.backbone.train --dataroot <M3FD_dir> --scale 2 --save_dir checkpoints/backbone_x2
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .vanilla_mamba import build_backbone
from .validation import list_images, build_validation_sets, evaluate
from ..degradation import full_degradation
from ..utils.image_io import read_img_gray


def _require_finite(tensors, context):
    bad = [name for name, value in tensors.items() if not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError("Non-finite %s at %s" % (", ".join(bad), context))


def _backward_step(loss, model, optimizer, max_grad_norm, context):
    _require_finite({"loss": loss}, context)
    loss.backward()
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise RuntimeError("Non-finite gradients at %s; optimizer step not applied" % context) from exc
    optimizer.step()
    return grad_norm


def _list_images(dir_path):
    return list_images(dir_path)


def _read_single(path):
    return read_img_gray(path)


def _resize_min(img, size):
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    h, w = img.shape
    factor = max(size / h, size / w)
    t = F.interpolate(t, size=(math.ceil(h * factor), math.ceil(w * factor)),
                      mode="bicubic", align_corners=False).clamp(0, 1)
    return t[0, 0].numpy()


class PairedSRDataset(Dataset):
    """Epoch/index-local streams; recreate workers after set_epoch (no persistent workers)."""

    def __init__(self, img_paths, scale=2, gt_size=128, length=None, seed=0, clean_prob=0.2):
        self.img_paths = list(img_paths)
        if not self.img_paths:
            raise ValueError("Training image list must not be empty")
        if scale not in (2, 4) or gt_size % scale or gt_size < 16 * scale:
            raise ValueError("gt_size must be divisible by scale and provide LR patches >=16x16")
        if not 0 <= clean_prob <= 1:
            raise ValueError("clean_prob must be in [0,1]")
        self.scale = scale
        self.gt_size = gt_size
        self.length = len(self.img_paths) if length is None else length
        if self.length < 1:
            raise ValueError("Dataset length must be positive")
        self.seed = seed
        self.epoch = 0
        self.clean_prob = clean_prob

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = ("%d:%d:%d" % (self.seed, self.epoch, idx)).encode("ascii")
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % (2 ** 63)
        rng = np.random.default_rng(seed)
        path = self.img_paths[int(rng.integers(0, len(self.img_paths)))]
        img = _read_single(path)
        h, w = img.shape
        s = self.gt_size
        if h < s or w < s:
            img = _resize_min(img, s)
            h, w = img.shape
        ih = int(rng.integers(0, h - s + 1))
        iw = int(rng.integers(0, w - s + 1))
        hr = img[ih:ih + s, iw:iw + s]
        # random flips and 90-degree rotations
        if rng.random() < 0.5:
            hr = hr[:, ::-1]
        if rng.random() < 0.5:
            hr = hr[::-1, :]
        k = int(rng.integers(0, 4))
        if k:
            hr = np.rot90(hr, k)
        hr = np.ascontiguousarray(hr)
        hr = torch.from_numpy(hr).float().unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
        if rng.random() < self.clean_prob:
            lr = F.interpolate(hr, scale_factor=1 / self.scale, mode="bicubic",
                               align_corners=False, antialias=True).clamp(0, 1)
        else:
            gen = torch.Generator().manual_seed(seed)
            lr, _, _, _ = full_degradation(hr, scale=self.scale, rng=gen)
        # Operators require a temporary batch dimension; Dataset returns CHW,
        # and DataLoader adds the actual training batch dimension.
        return lr.squeeze(0), hr.squeeze(0)


def train(args):
    if min(args.epochs, args.batch_size, args.save_every, args.val_every, args.images_per_epoch) < 1:
        raise ValueError("epochs, batch_size, save_every, val_every, images_per_epoch must be positive")
    if args.num_workers < 0 or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("num_workers >=0 and finite max_grad_norm >0 are required")
    if not (math.isfinite(args.lr) and math.isfinite(args.min_lr)
            and 0 < args.lr and 0 <= args.min_lr <= args.lr):
        raise ValueError("Require 0 <= min_lr <= lr and lr >0")
    if args.crop_border is None:
        args.crop_border = args.scale
    if args.crop_border < 0:
        raise ValueError("crop_border must be nonnegative")
    if args.save_dir is None:
        args.save_dir = "checkpoints/backbone_x%d_v2" % args.scale
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    img_paths = _list_images(args.dataroot)
    if len(img_paths) == 0:
        raise SystemExit("No images found in %s" % args.dataroot)
    validations = build_validation_sets(args.val_root, args.scale, args.crop_border)
    validation_identities = set()
    for ds in validations.values():
        for pair in ds.pairs:
            for path in pair:
                info = os.stat(path)
                validation_identities.add((info.st_dev, info.st_ino))
    for path in img_paths:
        info = os.stat(path)
        if (info.st_dev, info.st_ino) in validation_identities:
            raise ValueError("Training image overlaps validation: %s" % path)
    dataset = PairedSRDataset(img_paths, scale=args.scale, gt_size=args.gt_size,
                              length=args.images_per_epoch, seed=args.seed, clean_prob=args.clean_prob)
    model = build_backbone(scale=args.scale, in_chans=1).to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"),
                        generator=torch.Generator().manual_seed(args.seed))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)
    os.makedirs(args.save_dir, exist_ok=True)
    manifest = {"training_recipe": "source_mixed_clean_v2", "args": vars(args),
                "train_sources": img_paths,
                "validation_pairs": {name: ds.pairs for name, ds in validations.items()},
                "validation_protocol": {"channel": "grayscale [0,1]", "sr_clamp": True,
                    "crop_border_hr": args.crop_border, "ssim": "11x11 Gaussian sigma=1.5, valid centers",
                    "selection": "equal-weight mean of Result-A and Result-C image-mean PSNR"}}
    with open(os.path.join(args.save_dir, "data_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print("train: %d sources, %d pairs/epoch; clean probability=%.2f" %
          (len(img_paths), len(dataset), args.clean_prob), flush=True)
    for name, ds in validations.items():
        print("%s: %d native x%d validation pairs" % (name, len(ds), args.scale), flush=True)
    best_score, best_epoch = -float("inf"), None
    best_path = os.path.join(args.save_dir, "backbone_x%d_best.pt" % args.scale)
    last_path = os.path.join(args.save_dir, "backbone_x%d_last.pt" % args.scale)
    with open(os.path.join(args.save_dir, "metrics.jsonl"), "w") as history:
        for epoch in range(args.epochs):
            dataset.set_epoch(epoch)
            model.train()
            running, count = 0.0, 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for batch_idx, (lr, hr) in enumerate(loader):
                lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                context = "epoch %d batch %d" % (epoch + 1, batch_idx + 1)
                _require_finite({"lr": lr, "hr": hr}, context)
                optimizer.zero_grad(set_to_none=True)
                sr = model(lr)
                if sr.shape != hr.shape:
                    raise RuntimeError("Training SR/HR shape mismatch at " + context)
                _require_finite({"sr": sr}, context)
                loss = F.l1_loss(sr, hr)
                _backward_step(loss, model, optimizer, args.max_grad_norm, context)
                running += loss.item() * len(lr)
                count += len(lr)
            record = {"epoch": epoch + 1, "train_l1": running / count, "lr": learning_rate}
            print("[epoch %d/%d] L1=%.5f lr=%.3g" %
                  (epoch + 1, args.epochs, record["train_l1"], learning_rate), flush=True)
            # Always validate the final epoch, even when it is off the interval.
            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
                validation = evaluate(model, validations, device, args.crop_border)
                record["val"] = validation
                score = validation["mean"]["psnr"]
                if score > best_score:
                    best_score, best_epoch = score, epoch + 1
                    torch.save({"state_dict": model.state_dict(), "epoch": best_epoch,
                                "val_metrics": validation, "training_recipe": "source_mixed_clean_v2",
                                "scale": args.scale, "crop_border": args.crop_border}, best_path)
                    print("saved best epoch %d: mean PSNR=%.4f" % (best_epoch, best_score), flush=True)
            scheduler.step()
            record["next_lr"] = optimizer.param_groups[0]["lr"]
            history.write(json.dumps(record) + "\n")
            history.flush()
            if (epoch + 1) % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir,
                           "backbone_x%d_epoch%d.pt" % (args.scale, epoch + 1)))

    torch.save(model.state_dict(), last_path)
    # Final public checkpoint is the validation winner, not an unselected last epoch.
    best = torch.load(best_path, map_location="cpu")
    final_path = os.path.join(args.save_dir, "backbone_x%d.pt" % args.scale)
    torch.save(best["state_dict"], final_path)
    with open(os.path.join(args.save_dir, "training_summary.json"), "w") as handle:
        json.dump({"best_epoch": best_epoch, "best_mean_psnr": best_score,
                   "best_val_metrics": best["val_metrics"], "final_checkpoint": final_path,
                   "last_checkpoint": last_path, "training_recipe": "source_mixed_clean_v2"}, handle, indent=2)
    print("saved best epoch %d to %s (last epoch saved separately)" % (best_epoch, final_path), flush=True)


def make_parser():
    p = argparse.ArgumentParser(description="Train the VanillaMamba backbone for infrared SR")
    p.add_argument("--dataroot", type=str, required=True, help="source image directory (e.g. M3FD)")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("--gt_size", type=int, default=128, help="HR patch size (a multiple of scale)")
    p.add_argument("--images_per_epoch", type=int, default=1000, help="synthetic samples per epoch")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6, help="cosine schedule minimum learning rate")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--clean_prob", type=float, default=0.2,
                   help="probability of clean antialiased bicubic LR; 0 uses mixed degradation only")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default=None, help="default: checkpoints/backbone_x{scale}_v2")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--val_root", type=str, default=str(Path(__file__).resolve().parents[2] / "datasets"),
                   help="contains results-A/results-C, each with HR and LR_bicubic/X{scale}")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--crop_border", type=int, default=None, help="HR border shaved for metrics; default: scale")
    return p


def main():
    train(make_parser().parse_args())


if __name__ == "__main__":
    main()
