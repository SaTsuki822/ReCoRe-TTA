"""No-reference MANIQA metric via pyiqa (optional dependency).

Requires RGB input, so single-channel infrared images are replicated to 3 channels.
Raises an ImportError with an install hint when pyiqa is missing.
"""

from __future__ import annotations

import numpy as np
import torch

_metric_cache = {}


def _get_metric(device):
    if device not in _metric_cache:
        try:
            import pyiqa  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "MANIQA requires pyiqa; install with: pip install pyiqa") from e
        _metric_cache[device] = pyiqa.create_metric("maniqa", device=device)
    return _metric_cache[device]


def calculate_maniqa(img, device: str = "cuda") -> float:
    """Compute MANIQA (higher is better). img: numpy (H,W) [0,1] or tensor (1,1,H,W)."""
    metric = _get_metric(device)
    if isinstance(img, np.ndarray):
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        t = torch.from_numpy(img).unsqueeze(0)          # (1,H,W)
    else:
        t = img.detach().float().squeeze(0)              # (1,H,W)
    if t.dim() == 3:
        t = t.unsqueeze(1)                               # (1,1,H,W)
    t = t.repeat(1, 3, 1, 1).to(device)                  # replicate to RGB
    score = metric(t)
    return float(score.item())

