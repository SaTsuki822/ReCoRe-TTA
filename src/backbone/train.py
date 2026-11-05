"""Train the VanillaMamba backbone for infrared super-resolution.

Synthesizes LR-HR pairs from single-channel source images using the full degradation
model (blur -> downsampling -> non-uniformity/noise -> bad pixels -> output chain),
then trains the backbone with a pixel L1 loss.

Usage:
    python -m src.backbone.train --dataroot <M3FD_dir> --scale 2 --save_dir checkpoints/backbone_x2
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .vanilla_mamba import build_backbone
from ..degradation import full_degradation

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def _list_images(dir_path):
    paths = []
    for ext in _IMG_EXTS:
        paths += sorted(glob.glob(os.path.join(dir_path, ext)))
        paths += sorted(glob.glob(os.path.join(dir_path, ext.upper())))
    return sorted(set(paths))


def _read_single(path):
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img[:, :, :3].astype(np.float32)
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.ascontiguousarray(img)


def _resize_min(img, size):
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


class PairedSRDataset(Dataset):
    """Synthesize LR-HR pairs from single-channel source images."""

    def __init__(self, img_paths, scale=2, gt_size=128, length=None, seed=0):
        self.img_paths = img_paths
        self.scale = scale
        self.gt_size = gt_size
        self.length = length or len(img_paths)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.img_paths[int(self.rng.integers(0, len(self.img_paths)))]
        img = _read_single(path)
        h, w = img.shape
        s = self.gt_size
        if h < s or w < s:
            img = _resize_min(img, s)
            h, w = img.shape
        ih = int(self.rng.integers(0, h - s + 1))
        iw = int(self.rng.integers(0, w - s + 1))
        hr = img[ih:ih + s, iw:iw + s]
        # random flips and 90-degree rotations
        if self.rng.random() < 0.5:
            hr = hr[:, ::-1]
        if self.rng.random() < 0.5:
            hr = hr[::-1, :]
        k = int(self.rng.integers(0, 4))
        if k:
            hr = np.rot90(hr, k)
        hr = np.ascontiguousarray(hr)
        hr = torch.from_numpy(hr).float().unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
        gen = torch.Generator().manual_seed(self.seed + idx)
        lr, _, _, _ = full_degradation(hr, scale=self.scale, rng=gen)
        return lr, hr


def train(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    img_paths = _list_images(args.dataroot)
    if len(img_paths) == 0:
        raise SystemExit("No images found in %s" % args.dataroot)

    model = build_backbone(scale=args.scale, in_chans=1).to(device)
    dataset = PairedSRDataset(img_paths, scale=args.scale, gt_size=args.gt_size,
                              length=args.images_per_epoch, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = F.l1_loss(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / max(len(loader), 1)
        print("[epoch %d/%d] L1=%.5f" % (epoch, args.epochs, avg))
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "backbone_x%d_epoch%d.pt" % (args.scale, epoch)))

    final_path = os.path.join(args.save_dir, "backbone_x%d.pt" % args.scale)
    torch.save(model.state_dict(), final_path)
    print("saved to %s" % final_path)


def main():
    p = argparse.ArgumentParser(description="Train the VanillaMamba backbone for infrared SR")
    p.add_argument("--dataroot", type=str, required=True, help="source image directory (e.g. M3FD)")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("--gt_size", type=int, default=128, help="HR patch size (a multiple of scale)")
    p.add_argument("--images_per_epoch", type=int, default=1000, help="synthetic samples per epoch")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default=None, help="default: checkpoints/backbone_x{scale}")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join("checkpoints", "backbone_x%d" % args.scale)
    train(args)


if __name__ == "__main__":
    main()