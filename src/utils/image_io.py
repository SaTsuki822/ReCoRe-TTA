"""Image I/O and seeding utilities (single channel, [0,1])."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio


def set_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_img_gray(path) -> np.ndarray:
    """Read an image as single-channel [0,1] float (H,W)."""
    img = imageio.imread(path)
    source_dtype = img.dtype
    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[..., 0]
        elif img.shape[-1] in (3, 4):
            img = img[:, :, :3].astype(np.float32)
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        else:
            raise ValueError("Unsupported image channels in %s" % path)
    if img.ndim != 2 or img.size == 0 or not np.isfinite(img).all():
        raise ValueError("Expected a finite nonempty grayscale/RGB image: %s" % path)
    img = img.astype(np.float32)
    if source_dtype in (np.dtype("uint8"), np.dtype("uint16")):
        img /= np.iinfo(source_dtype).max
    elif source_dtype == np.dtype("int32") and img.min() >= 0 and img.max() <= 65535:
        # Some imageio/Pillow versions decode 16-bit PNGs as int32.
        img /= 65535.0
    elif img.max() > 1.0:
        img = img / 255.0
    if img.min() < 0 or img.max() > 1.00001:
        raise ValueError("Image cannot be normalized to [0,1]: %s" % path)
    return np.ascontiguousarray(img)


def write_img(path: str, img):
    """Write an image. img: numpy (H,W) [0,1]/uint8 or a tensor."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0) * 255.0
        img = np.round(img).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.imwrite(path, img)


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """(1,1,H,W)/(H,W) tensor -> numpy (H,W) [0,1] float."""
    return t.detach().cpu().squeeze().numpy()


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """numpy (H,W) [0,1] -> (1,1,H,W) tensor."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

