"""Differentiable infrared degradation operators, safe projection and synthesis.

Main interface:
- sample_degradation / safe_project / apply_retain / apply_discard / apply_all / full_degradation
- low-level operators apply_*
- parameter space DEGRADATIONS / normalize_params / denormalize_params / split_params
"""

from .params import (
    DEGRADATIONS, N_CLASSES, N_PARAMS, RETAIN_INDICES, DISCARD_INDICES,
    RETAIN_ORDER, PARAM_SLICES, PARAM_LOWS, PARAM_HIGHS, PARAM_DISCRETE,
    normalize_params, denormalize_params, split_params, activation_mask,
)
from .operators import (
    apply_blur, apply_gain_nonuniformity, apply_offset_nonuniformity,
    apply_readout_noise, apply_signal_dependent_noise, apply_stripe,
    apply_quantization, apply_jpeg, apply_saturation_agc, apply_bad_pixels,
    apply_single,
)
from .core import (
    sample_degradation, safe_project, apply_retain, apply_discard,
    apply_all, full_degradation,
)

__all__ = [
    "DEGRADATIONS", "N_CLASSES", "N_PARAMS", "RETAIN_INDICES", "DISCARD_INDICES",
    "RETAIN_ORDER", "PARAM_SLICES", "PARAM_LOWS", "PARAM_HIGHS", "PARAM_DISCRETE",
    "normalize_params", "denormalize_params", "split_params", "activation_mask",
    "apply_blur", "apply_gain_nonuniformity", "apply_offset_nonuniformity",
    "apply_readout_noise", "apply_signal_dependent_noise", "apply_stripe",
    "apply_quantization", "apply_jpeg", "apply_saturation_agc", "apply_bad_pixels",
    "apply_single",
    "sample_degradation", "safe_project", "apply_retain", "apply_discard",
    "apply_all", "full_degradation",
]
