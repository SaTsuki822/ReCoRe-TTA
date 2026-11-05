"""Pretraining data: synthesize degradation pairs (x, x') from source images.

Source image -> single-channel intensity -> [0,1] -> random 128x128 crop ->
horizontal/vertical flip and 90-degree rotation -> sample a degradation pair
(both views share the same crop, so they stay content-aligned).
"""

import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..degradation import sample_degradation, apply_all

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


def _to_single_gray(img):
    """Convert a numpy image (HWC or single channel, uint8/float) to single-channel [0,1]."""
    if img.ndim == 3:
        img = img[:, :, :3]
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if img.dtype == np.uint8 or float(img.max()) > 1.0:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return img


class DegradationPairDataset(Dataset):
    """Synthesize degradation pairs (x, x_prime, active, norm) from source images.

    If img_dir is None, random smooth synthetic scenes are used as the source
    (useful for unit tests without real data).
    """

    def __init__(self, img_dir=None, crop_size=128, length=1000, augment=True, seed=0):
        super().__init__()
        self.crop_size = crop_size
        self.length = length
        self.augment = augment
        self.synthetic = img_dir is None
        self.img_paths = []
        if img_dir is not None:
            self.img_paths = (sorted(glob.glob(os.path.join(img_dir, "*.png"))) +
                              sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) +
                              sorted(glob.glob(os.path.join(img_dir, "*.bmp"))))
            self.synthetic = len(self.img_paths) == 0
        self.rng = random.Random(seed)

    def __len__(self):
        return self.length

    def _source(self):
        if self.synthetic:
            return self._synthetic_image()
        path = self.rng.choice(self.img_paths)
        return _to_single_gray(imageio.imread(path))

    def _synthetic_image(self):
        """Random low-frequency smooth scene, slightly larger than the crop size."""
        H = W = int(self.crop_size * 1.25)
        gh = gw = 10
        low = torch.rand(1, 1, gh, gw)
        img = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.squeeze(0).squeeze(0).numpy()  # (H, W)

    def _crop_augment(self, img):
        h, w = img.shape
        if h < self.crop_size or w < self.crop_size:
            img = F.interpolate(
                torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0),
                size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0).numpy()
            h, w = img.shape
        ih = self.rng.randint(0, h - self.crop_size)
        iw = self.rng.randint(0, w - self.crop_size)
        img = img[ih:ih + self.crop_size, iw:iw + self.crop_size]
        if self.augment:
            if self.rng.random() < 0.5:
                img = img[:, ::-1]
            if self.rng.random() < 0.5:
                img = img[::-1, :]
            k = self.rng.randint(0, 4)
            if k:
                img = np.rot90(img, k)
        return img

    def __getitem__(self, idx):
        img = self._source()
        x = self._crop_augment(np.ascontiguousarray(img))
        x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        active, raw, norm = sample_degradation()
        x_prime = apply_all(x, norm, active)
        # drop the batch dim so DataLoader stacks items into (B, C, H, W)
        return x.squeeze(0), x_prime.squeeze(0), active, norm