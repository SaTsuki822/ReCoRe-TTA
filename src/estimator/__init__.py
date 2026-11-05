"""Paired relative degradation estimator.

Given a same-scale image pair (x, x'), predict 9 degradation labels, an 18-dim
normalized continuous parameter vector, and a calibrated confidence. Trained offline
and frozen at test time.

Main interface:
- DegradationEstimator / DescriptorEncoder (network)
- compute_losses (L_cls/L_reg/L_syn/L_conf -> L_E)
- DegradationPairDataset (synthetic degradation pairs)
- temperature_scale (confidence temperature calibration)
"""

from .network import DegradationEstimator, DescriptorEncoder, ResidualBlock
from .losses import compute_losses, masked_smooth_l1
from .data import DegradationPairDataset, _to_single_gray
from .calibrate import temperature_scale

__all__ = [
    "DegradationEstimator", "DescriptorEncoder", "ResidualBlock",
    "compute_losses", "masked_smooth_l1",
    "DegradationPairDataset", "_to_single_gray",
    "temperature_scale",
]