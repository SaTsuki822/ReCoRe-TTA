"""Degradation composition, safe projection and the full degradation model.

- sample_degradation: sample active degradations and their parameters.
- safe_project: clip parameters to valid ranges and drop discarded components.
- apply_retain: apply retained operators to build a same-scale harder view.
- apply_discard: apply discarded operators (synthesis only).
- full_degradation: HR -> blur -> downsampling -> non-uniformity/noise/bad pixels -> output chain -> LR.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import operators as ops
from . import params as P


def _next_seed(rng):
    """Derive an integer sub-seed from the main rng for reproducibility."""
    if rng is None:
        return None
    return int(torch.randint(0, 2 ** 31, (1,), generator=rng).item())


def sample_degradation(rng: Optional[torch.Generator] = None, device=None,
                       dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one degradation. Returns (active, raw, norm).

    active: (9,) 0/1 multi-label (at least one active)
    raw:    (18,) raw parameters (inactive classes zeroed)
    norm:   (18,) parameters normalized to [0,1] (regression target)
    """
    if rng is None:
        rng = torch.Generator(device=device)
    probs = torch.tensor([d.prob for d in P.DEGRADATIONS], dtype=dtype, device=device)
    active = torch.bernoulli(probs, generator=rng)
    if active.sum().item() == 0:
        active[0] = 1.0  # keep at least one active
    low = torch.tensor(P.PARAM_LOWS, dtype=dtype, device=device)
    high = torch.tensor(P.PARAM_HIGHS, dtype=dtype, device=device)
    raw = low + (high - low) * torch.rand(P.N_PARAMS, generator=rng, dtype=dtype, device=device)
    raw = raw * P.activation_mask(active)  # zero out inactive parameters
    norm = P.normalize_params(raw)
    return active, raw, norm


def safe_project(norm: torch.Tensor) -> torch.Tensor:
    """Clip normalized parameters to [0,1] and zero out discarded components."""
    out = norm.clone().clamp(0.0, 1.0)
    for i in P.DISCARD_INDICES:
        out[..., P.PARAM_SLICES[i]] = 0.0
    return out


def apply_retain(x: torch.Tensor, norm_params: torch.Tensor, active=None,
                 seed: Optional[int] = None) -> torch.Tensor:
    """Apply retained operators to build a same-scale harder view.

    norm_params: (...,18) normalized parameters (usually already safe-projected)
    active:      (9,) activation labels; None applies all retained operators
    """
    raw = P.denormalize_params(norm_params)
    segs = P.split_params(raw)
    for i in P.RETAIN_ORDER:
        if active is not None and float(active[i]) <= 0.5:
            continue
        x = ops.apply_single(i, x, segs[i], seed)
    return x


def apply_discard(x: torch.Tensor, norm_params: torch.Tensor, active=None,
                  seed: Optional[int] = None) -> torch.Tensor:
    """Apply discarded operators (synthesis only)."""
    raw = P.denormalize_params(norm_params)
    segs = P.split_params(raw)
    for i in P.DISCARD_INDICES:
        if active is not None and float(active[i]) <= 0.5:
            continue
        x = ops.apply_single(i, x, segs[i], seed)
    return x


def apply_all(x: torch.Tensor, norm_params: torch.Tensor, active,
              seed: Optional[int] = None, retain_only: bool = False) -> torch.Tensor:
    """Apply all active operators (retain then discard) in a fixed order."""
    x = apply_retain(x, norm_params, active, seed)
    if not retain_only:
        x = apply_discard(x, norm_params, active, seed)
    return x


def full_degradation(x: torch.Tensor, scale: int, rng: Optional[torch.Generator] = None,
                     device=None, dtype=torch.float32):
    """Full degradation model: HR -> blur -> downsampling -> non-uniformity/noise -> bad pixels -> output chain -> LR.

    Returns (lr, active, raw, norm); lr has size H//scale x W//scale (same channels as x).
    """
    active, raw, norm = sample_degradation(rng, device, dtype)
    segs = P.split_params(raw)

    def on(i):
        return float(active[i]) > 0.5

    # optical blur (only if the degradation is active)
    if on(0):
        x = ops.apply_blur(x, segs[0][..., 0], segs[0][..., 1], segs[0][..., 2])
    # detector sampling (average pooling, always applied)
    x = F.avg_pool2d(x, kernel_size=scale, stride=scale)
    # non-uniformity + noise + stripe (gated by their activation flags)
    if on(1):
        x = ops.apply_gain_nonuniformity(x, segs[1][..., 0], segs[1][..., 1], seed=_next_seed(rng))
    if on(2):
        x = ops.apply_offset_nonuniformity(x, segs[2][..., 0], segs[2][..., 1], seed=_next_seed(rng))
    if on(3):
        x = ops.apply_readout_noise(x, segs[3][..., 0], seed=_next_seed(rng))
    if on(4):
        x = ops.apply_signal_dependent_noise(x, segs[4][..., 0], seed=_next_seed(rng))
    if on(5):
        x = ops.apply_stripe(x, segs[5][..., 0], segs[5][..., 1], segs[5][..., 2], seed=_next_seed(rng))
    # bad pixels (gated)
    if on(8):
        x = ops.apply_bad_pixels(x, segs[8][..., 0], segs[8][..., 1], seed=_next_seed(rng))
    # output chain: quantization/compression + saturation/AGC (gated)
    if on(6):
        x = ops.apply_jpeg(ops.apply_quantization(x, segs[6][..., 0]), segs[6][..., 1])
    if on(7):
        x = ops.apply_saturation_agc(x, segs[7][..., 0], segs[7][..., 1])
    x = x.clamp(0.0, 1.0)
    return x, active, raw, norm
