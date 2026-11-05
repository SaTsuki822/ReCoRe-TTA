"""Single-channel [0,1] dataset loading for full-reference and no-reference evaluation.

- M3FD: source-domain pretraining (backbone and estimator).
- Result-A / Result-C: full-reference evaluation (LR + HR).
- Drone Vehicle / DJIThermalSR: no-reference real-world evaluation (LR only).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def _list_images(dir_path):
    paths = []
    for ext in _IMG_EXTS:
        paths += sorted(glob.glob(os.path.join(dir_path, ext)))
        paths += sorted(glob.glob(os.path.join(dir_path, ext.upper())))
    return sorted(set(paths))


def load_single_channel(path) -> np.ndarray:
    """Read an image and convert it to single-channel [0,1] float."""
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img[:, :, :3]
        img = 0.299 * img[..., 0].astype(np.float32) + \
              0.587 * img[..., 1].astype(np.float32) + \
              0.114 * img[..., 2].astype(np.float32)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """numpy (H,W) [0,1] -> (1,1,H,W) tensor."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


class SingleImageDataset(Dataset):
    """No-reference / TTA input dataset (LR images only)."""

    def __init__(self, lr_dir):
        self.paths = _list_images(lr_dir)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {lr_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = load_single_channel(path)
        return img_to_tensor(img), os.path.basename(path)


class PairedImageDataset(Dataset):
    """Full-reference dataset with matched LR-HR pairs (Result-A / Result-C).

    LR and HR files are matched by stem, optionally stripping a '{scale}x' suffix
    from the HR filename.
    """

    def __init__(self, lr_dir, hr_dir, scale: int = 2):
        self.scale = scale
        self.lr_paths = _list_images(lr_dir)
        self.hr_paths = _list_images(hr_dir)
        if len(self.lr_paths) == 0 or len(self.hr_paths) == 0:
            raise FileNotFoundError(f"Empty paired dataset: lr={lr_dir} hr={hr_dir}")
        self.pairs = self._match_pairs(self.lr_paths, self.hr_paths, scale)

    @staticmethod
    def _match_pairs(lr_paths, hr_paths, scale):
        hr_by_stem = {}
        for p in hr_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            hr_by_stem[stem] = p
            if stem.endswith(f"x{scale}"):
                hr_by_stem[stem[:-len(f"x{scale}")]] = p
        pairs = []
        for lr in lr_paths:
            stem = os.path.splitext(os.path.basename(lr))[0]
            hr = hr_by_stem.get(stem)
            if hr is None and stem.endswith(f"x{scale}"):
                hr = hr_by_stem.get(stem[:-len(f"x{scale}")])
            if hr is not None:
                pairs.append((lr, hr))
        if len(pairs) == 0:
            raise FileNotFoundError("No matched LR-HR pairs; check filename convention.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        lr = img_to_tensor(load_single_channel(lr_path))
        hr = img_to_tensor(load_single_channel(hr_path))
        return lr, hr, os.path.basename(lr_path)

