"""Degradation parameter space.

Defines the 9 infrared degradation families, their parameter names, sampling ranges
and normalization, plus retain/discard flags. The 18-dim continuous parameter vector
(3+2+2+1+1+3+2+2+2) is aligned with the estimator's regression head, and the 9 labels
with its classification head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class ParamSpec:
    """A single continuous parameter, normalized to [0,1] (raw = low + norm * (high-low))."""
    name: str
    low: float
    high: float
    discrete: bool = False  # discrete parameter, rounded to the nearest valid value when applied


@dataclass(frozen=True)
class DegradationSpec:
    """Specification of one degradation family."""
    index: int    # 0-based degradation index
    name: str
    prob: float   # activation probability
    retain: bool  # whether it can be re-applied at test time
    params: Tuple[ParamSpec, ...]


# 9 degradations (order defines the 18-dim parameter and 9-label layout, do not reorder)
DEGRADATIONS: Tuple[DegradationSpec, ...] = (
    DegradationSpec(0, "blur", 0.70, True, (
        ParamSpec("sigma_x", 0.2, 2.5),
        ParamSpec("sigma_y", 0.2, 2.5),
        ParamSpec("theta", 0.0, 3.141592653589793),  # [0, pi)
    )),
    DegradationSpec(1, "gain_nonuniformity", 0.50, True, (
        ParamSpec("a_g", 0.0, 0.12),
        ParamSpec("l_g", 16.0, 96.0),
    )),
    DegradationSpec(2, "offset_nonuniformity", 0.50, True, (
        ParamSpec("a_o", 0.0, 0.08),
        ParamSpec("l_o", 16.0, 96.0),
    )),
    DegradationSpec(3, "readout_noise", 0.80, True, (
        ParamSpec("sigma_r", 0.0, 0.06),
    )),
    DegradationSpec(4, "signal_dependent_noise", 0.50, True, (
        ParamSpec("sigma_s", 0.0, 0.04),
    )),
    DegradationSpec(5, "stripe", 0.35, True, (
        ParamSpec("a_s", 0.0, 0.06),
        ParamSpec("w_s", 1.0, 4.0),
        ParamSpec("d_s", 0.0, 1.0, discrete=True),  # 0=horizontal, 1=vertical
    )),
    DegradationSpec(6, "quantization", 0.40, False, (
        ParamSpec("b", 6.0, 8.0, discrete=True),    # bit depth
        ParamSpec("Q", 35.0, 100.0),                # JPEG quality
    )),
    DegradationSpec(7, "saturation_agc", 0.25, False, (
        ParamSpec("p_clip", 0.0, 0.04),
        ParamSpec("gamma", 0.7, 1.5),
    )),
    DegradationSpec(8, "bad_pixels", 0.15, False, (
        ParamSpec("p_bad", 0.0, 0.01),
        ParamSpec("v_bad", 0.0, 1.0, discrete=True),  # bad pixel value
    )),
)

N_CLASSES = len(DEGRADATIONS)                          # 9
N_PARAMS = sum(len(d.params) for d in DEGRADATIONS)    # 18
RETAIN_INDICES: Tuple[int, ...] = tuple(d.index for d in DEGRADATIONS if d.retain)
DISCARD_INDICES: Tuple[int, ...] = tuple(d.index for d in DEGRADATIONS if not d.retain)
RETAIN_ORDER: Tuple[int, ...] = RETAIN_INDICES          # application order

# per-class slice in the 18-dim vector
PARAM_SLICES: dict = {}
_off = 0
for _d in DEGRADATIONS:
    PARAM_SLICES[_d.index] = slice(_off, _off + len(_d.params))
    _off += len(_d.params)

# per-dim low/high (for normalization) and discrete flags
PARAM_LOWS: List[float] = [p.low for d in DEGRADATIONS for p in d.params]
PARAM_HIGHS: List[float] = [p.high for d in DEGRADATIONS for p in d.params]
PARAM_DISCRETE: List[bool] = [p.discrete for d in DEGRADATIONS for p in d.params]


def _low_tensor(dtype, device):
    return torch.tensor(PARAM_LOWS, dtype=dtype, device=device)


def _high_tensor(dtype, device):
    return torch.tensor(PARAM_HIGHS, dtype=dtype, device=device)


def normalize_params(raw: torch.Tensor) -> torch.Tensor:
    """Min-max normalize raw params to [0,1]."""
    low = _low_tensor(raw.dtype, raw.device)
    high = _high_tensor(raw.dtype, raw.device)
    return ((raw - low) / (high - low)).clamp(0.0, 1.0)


def denormalize_params(norm: torch.Tensor) -> torch.Tensor:
    """Map normalized params back to their raw ranges."""
    low = _low_tensor(norm.dtype, norm.device)
    high = _high_tensor(norm.dtype, norm.device)
    return norm.clamp(0.0, 1.0) * (high - low) + low


def split_params(params: torch.Tensor) -> List[torch.Tensor]:
    """Split (...,18) into 9 per-class segments for the operators."""
    return [params[..., PARAM_SLICES[i]] for i in range(N_CLASSES)]


def activation_mask(active: torch.Tensor) -> torch.Tensor:
    """Expand (...,9) activation labels into a (...,18) per-parameter mask."""
    counts = torch.tensor([len(d.params) for d in DEGRADATIONS], device=active.device)
    return active.repeat_interleave(counts, dim=-1)
