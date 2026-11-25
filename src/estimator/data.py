"""Pretraining data: synthesize degradation pairs (x, x') from source images.

Source image -> single-channel intensity -> [0,1] -> random 128x128 crop ->
horizontal/vertical flip and 90-degree rotation -> sample a degradation pair
(both views share the same crop, so they stay content-aligned).
"""

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..degradation import sample_degradation, apply_all

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


def discover_images(img_dir):
    """Find direct image children; invalid/empty directories must not use fake data."""
    root = Path(img_dir)
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: %s" % root)
    paths = sorted({str(p.resolve()) for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}})
    if not paths:
        raise FileNotFoundError(
            "No supported images directly inside %s (subdirectories are not searched)." % root)
    return paths


def _to_single_gray(img):
    """Convert a numpy image (HWC or single channel, uint8/float) to single-channel [0,1]."""
    # Preserve the source dtype before RGB conversion promotes uint8 to float.
    # Otherwise a dark RGB image with max==1 is incorrectly treated as [0,1].
    source_dtype = img.dtype
    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[..., 0]
        else:
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if source_dtype in (np.dtype("uint8"), np.dtype("uint16")):
        img = img.astype(np.float32) / np.iinfo(source_dtype).max
    elif float(img.max()) > 1.0:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return img


class DegradationPairDataset(Dataset):
    """Synthesize degradation pairs (x, x_prime, active, norm) from source images.

    If neither img_dir nor img_paths is given, use synthetic scenes for tests.
    Sample streams depend on (seed, epoch, index), independent of worker count.
    Call set_epoch before each training epoch; fixed=True keeps held-out pairs fixed.
    """

    def __init__(self, img_dir=None, crop_size=128, length=1000, augment=True, seed=0,
                 img_paths=None, fixed=False):
        super().__init__()
        if crop_size < 16 or length < 1:
            raise ValueError("crop_size must be >= 16 and length must be positive")
        if img_dir is not None and img_paths is not None:
            raise ValueError("Pass img_dir or img_paths, not both")
        self.crop_size = crop_size
        self.length = length
        self.augment = augment
        self.synthetic = img_dir is None and img_paths is None
        self.img_paths = discover_images(img_dir) if img_dir is not None else list(img_paths or [])
        if not self.synthetic and not self.img_paths:
            raise ValueError("Real-image dataset cannot have an empty image list")
        for path in self.img_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        self.seed = seed
        self.epoch = 0
        self.fixed = fixed

    def set_epoch(self, epoch):
        """Refresh training pairs each epoch; held-out pairs stay fixed."""
        self.epoch = 0 if self.fixed else epoch

    def __len__(self):
        return self.length

    def _source(self, idx, rng, generator):
        if self.synthetic:
            return self._synthetic_image(generator)
        # Held-out datasets cycle through sources to ensure coverage.
        path = self.img_paths[idx % len(self.img_paths)] if self.fixed else rng.choice(self.img_paths)
        return _to_single_gray(imageio.imread(path))

    def _synthetic_image(self, generator):
        """Random low-frequency smooth scene, slightly larger than the crop size."""
        H = W = int(self.crop_size * 1.25)
        gh = gw = 10
        low = torch.rand(1, 1, gh, gw, generator=generator)
        img = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.squeeze(0).squeeze(0).numpy()  # (H, W)

    def _crop_augment(self, img, rng):
        h, w = img.shape
        if h < self.crop_size or w < self.crop_size:
            img = F.interpolate(
                torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0),
                size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0).numpy()
            h, w = img.shape
        ih = rng.randint(0, h - self.crop_size)
        iw = rng.randint(0, w - self.crop_size)
        img = img[ih:ih + self.crop_size, iw:iw + self.crop_size]
        if self.augment:
            if rng.random() < 0.5:
                img = img[:, ::-1]
            if rng.random() < 0.5:
                img = img[::-1, :]
            k = rng.randrange(4)
            if k:
                img = np.rot90(img, k)
        return img

    def __getitem__(self, idx):
        # Local streams make samples independent of worker scheduling and model RNG.
        # Mix the full split seed: some RNG implementations use only its low bits.
        key = ("%d:%d:%d" % (self.seed, self.epoch, idx)).encode("ascii")
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % (2 ** 63)
        rng = random.Random(seed)
        generator = torch.Generator().manual_seed(seed)
        img = self._source(idx, rng, generator)
        x = self._crop_augment(np.ascontiguousarray(img), rng)
        x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        active, raw, norm = sample_degradation(generator)
        noise_seed = int(torch.randint(0, 2 ** 31, (1,), generator=generator).item())
        x_prime = apply_all(x, norm, active, seed=noise_seed)
        # drop the batch dim so DataLoader stacks items into (B, C, H, W)
        return x.squeeze(0), x_prime.squeeze(0), active, norm
