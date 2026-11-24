"""Pretrain E_phi with source-disjoint validation and temperature calibration.

    python -m src.estimator.train --img_dir <IR_dir> --calibrate
    python -m src.estimator.train --synthetic --length 8 --val_length 4 --epochs 2
"""

import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader

from .network import DegradationEstimator
from .losses import compute_losses
from .data import DegradationPairDataset, discover_images
from .calibrate import temperature_scale
from ..degradation.params import DEGRADATIONS, PARAM_SLICES, activation_mask


def build_datasets(args):
    """Split sources before patch generation; supplied split dirs must be disjoint."""
    if args.synthetic:
        if args.img_dir is not None or args.val_dir is not None or args.cal_dir is not None:
            raise ValueError("--synthetic cannot be combined with image directories")
        sources = {"train": None, "val": None, "cal": None}
    else:
        if not args.img_dir:
            raise ValueError("Provide --img_dir for real training, or --synthetic for smoke tests")
        if args.cal_dir and not args.calibrate:
            raise ValueError("--cal_dir requires --calibrate")
        remaining = discover_images(args.img_dir)
        val = discover_images(args.val_dir) if args.val_dir is not None else None
        cal = discover_images(args.cal_dir) if args.cal_dir is not None else None
        n = len(remaining)
        n_val = 0
        n_cal = 0
        if val is None:
            if not 0 < args.val_fraction < 1:
                raise ValueError("--val_fraction must be between 0 and 1")
            n_val = max(1, int(n * args.val_fraction))
        if args.calibrate and cal is None:
            if not 0 < args.cal_fraction < 1:
                raise ValueError("--cal_fraction must be between 0 and 1")
            n_cal = max(1, int(n * args.cal_fraction))
        if n_val + n_cal >= n:
            raise ValueError("Not enough source images for nonempty train/val/cal splits")
        random.Random(args.seed).shuffle(remaining)
        if val is None:
            val = remaining[:n_val]
        if args.calibrate and cal is None:
            cal = remaining[n_val:n_val + n_cal]
        sources = {"train": remaining[n_val + n_cal:], "val": val,
                   "cal": cal if args.calibrate else []}
        # inode checks also catch symlinks and hardlinks across supplied directories.
        identities = {}
        for split, paths in sources.items():
            for path in paths:
                info = os.stat(path)
                identity = (info.st_dev, info.st_ino)
                if identity in identities and identities[identity] != split:
                    raise ValueError("Source image overlaps %s and %s: %s" %
                                     (identities[identity], split, path))
                identities[identity] = split

    datasets = {}
    for i, (split, length) in enumerate((("train", args.length), ("val", args.val_length),
                                       ("cal", args.cal_length))):
        if split == "cal" and not args.calibrate:
            continue
        datasets[split] = DegradationPairDataset(
            img_paths=sources[split], crop_size=args.crop_size, length=length,
            augment=(split == "train" and not args.no_augment),
            seed=args.seed + i * (2 ** 48), fixed=(split != "train"))
    return datasets, sources


@torch.no_grad()
def evaluate(model, loader, device):
    """Deterministic held-out loss, multi-label F1, and active-parameter MAE."""
    model.eval()
    totals = {k: 0.0 for k in ("loss", "L_cls", "L_reg", "L_syn", "L_conf")}
    tp, fp, fn = (torch.zeros(9, device=device) for _ in range(3))
    abs_error = torch.zeros(18, device=device)
    param_count = torch.zeros(18, device=device)
    count = 0
    for batch_idx, batch in enumerate(loader):
        x, xp, active, norm = (t.to(device) for t in batch)
        logits, phi, confidence = model(x, xp)
        loss, parts = compute_losses(logits, phi, confidence, x, xp, active, norm,
                                     seed=20000 + batch_idx)
        b = x.shape[0]
        count += b
        totals["loss"] += loss.item() * b
        for key, value in parts.items():
            totals[key] += value.item() * b
        pred, target = logits >= 0, active.bool()
        tp += (pred & target).sum(0)
        fp += (pred & ~target).sum(0)
        fn += (~pred & target).sum(0)
        mask = activation_mask(active)
        abs_error += ((phi - norm).abs() * mask).sum(0)
        param_count += mask.sum(0)
    metrics = {key: value / count for key, value in totals.items()}
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    metrics["macro_f1"] = f1.mean().item()
    metrics["active_param_mae"] = (abs_error.sum() / param_count.sum().clamp_min(1)).item()
    metrics["per_class"] = {
        spec.name: {"f1": f1[i].item(), "support": int((tp[i] + fn[i]).item())}
        for i, spec in enumerate(DEGRADATIONS)}
    metrics["per_param_mae"] = {
        spec.name + "." + param.name:
            ((abs_error[j] / param_count[j]).item() if param_count[j] > 0 else None)
        for spec in DEGRADATIONS
        for j, param in zip(range(PARAM_SLICES[spec.index].start,
                                  PARAM_SLICES[spec.index].stop), spec.params)}
    return metrics


def train(args):
    if min(args.epochs, args.batch_size, args.save_every) < 1 or args.num_workers < 0:
        raise ValueError("epochs, batch_size, save_every must be positive; num_workers >= 0")
    datasets, sources = build_datasets(args)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    torch.manual_seed(args.seed)
    model = DegradationEstimator(
        n_classes=9, n_params=18, in_ch=1, descriptor_dim=args.descriptor_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)
    manifest = {"mode": "synthetic" if args.synthetic else "real", "args": vars(args),
                "sources": sources,
                "split_seeds": {k: v.seed for k, v in datasets.items()}}
    with open(os.path.join(args.save_dir, "data_split.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    for split, dataset in datasets.items():
        print("%s: %d source images, %d pairs%s" %
              (split, len(dataset.img_paths), len(dataset),
               " (SYNTHETIC SMOKE TEST)" if args.synthetic else ""))

    best_loss = float("inf")
    best_epoch = None
    best_path = os.path.join(args.save_dir, "estimator_best.pt")
    with open(os.path.join(args.save_dir, "metrics.jsonl"), "w") as history:
        for epoch in range(args.epochs):
            datasets["train"].set_epoch(epoch)
            model.train()
            totals = {k: 0.0 for k in ("loss", "L_cls", "L_reg", "L_syn", "L_conf")}
            count = 0
            for batch in loader:
                x, xp, active, norm = (t.to(device) for t in batch)
                optimizer.zero_grad()
                logits, phi, confidence = model(x, xp)
                loss, parts = compute_losses(logits, phi, confidence, x, xp, active, norm)
                loss.backward()
                optimizer.step()
                b = x.shape[0]
                count += b
                totals["loss"] += loss.item() * b
                for key, value in parts.items():
                    totals[key] += value.item() * b
            training = {key: value / count for key, value in totals.items()}
            validation = evaluate(model, val_loader, device)
            if not all(torch.isfinite(torch.tensor(v)) for v in
                       (training["loss"], validation["loss"])):
                raise RuntimeError("Non-finite training/validation loss")
            print("[epoch %d] train=%.4f val=%.4f macro_F1=%.4f active_MAE=%.4f" %
                  (epoch + 1, training["loss"], validation["loss"],
                   validation["macro_f1"], validation["active_param_mae"]))
            history.write(json.dumps({"epoch": epoch + 1, "train": training,
                                      "val": validation}) + "\n")
            history.flush()
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch + 1
                torch.save({"state_dict": model.state_dict(), "temperature": 1.0,
                            "epoch": best_epoch, "val_metrics": validation}, best_path)
            if (epoch + 1) % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.save_dir, "estimator_epoch%d.pt" % (epoch + 1)))

    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["state_dict"])
    T = 1.0
    if args.calibrate:
        cal_loader = DataLoader(datasets["cal"], batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        T = temperature_scale(model, cal_loader)
        print("calibrated temperature = %.4f (held-out calibration split)" % T)
    best["temperature"] = T
    final_path = os.path.join(args.save_dir, "estimator_final.pt")
    torch.save(best, final_path)
    print("saved best epoch %d to %s" % (best_epoch, final_path))


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=None, help="training/source IR image directory")
    parser.add_argument("--val_dir", type=str, default=None, help="optional separate validation image directory")
    parser.add_argument("--cal_dir", type=str, default=None, help="optional separate calibration image directory")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="source fraction reserved when val_dir is absent")
    parser.add_argument("--cal_fraction", type=float, default=0.1, help="source fraction reserved when calibrating without cal_dir")
    parser.add_argument("--synthetic", action="store_true", help="explicit synthetic smoke test (no real image training)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/estimator")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--length", type=int, default=1000, help="training pairs sampled per epoch, including real-image mode")
    parser.add_argument("--val_length", type=int, default=1000, help="fixed validation pairs")
    parser.add_argument("--cal_length", type=int, default=1000, help="fixed calibration pairs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--descriptor_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    train(make_parser().parse_args())


if __name__ == "__main__":
    main()
