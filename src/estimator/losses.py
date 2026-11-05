"""Training losses for the degradation estimator.

L_E = L_cls + lambda_reg * L_reg + lambda_syn * L_syn + lambda_conf * L_conf,
where L_cls is 9-way multi-label BCE, L_reg is masked Smooth-L1 over active
parameters, L_syn is a pseudo-Huber synthesis-consistency term, and L_conf is
confidence BCE with target exp(-e_syn / tau_c).
"""

import torch
import torch.nn.functional as F

from ..degradation import apply_retain, activation_mask


def masked_smooth_l1(pred, target, mask, beta=1.0):
    """Masked Smooth-L1, normalized by the number of active parameters."""
    diff = (pred - target) * mask
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * abs_diff ** 2 / beta, abs_diff - 0.5 * beta)
    denom = mask.sum() + 1e-8
    return loss.sum() / denom


def _apply_retain_batch(x, phi, active, seed=None):
    """Apply the retained operators per sample and stack into a batch."""
    outs = []
    for b in range(x.shape[0]):
        outs.append(apply_retain(x[b:b + 1], phi[b], active[b], seed))
    return torch.cat(outs, dim=0)


def compute_losses(q_logit, phi, c_logit, x, x_prime, active, norm_target, seed=None,
                   lambda_reg=1.0, lambda_syn=0.5, lambda_conf=0.1,
                   eps_E=1e-3, tau_c=0.05):
    """Compute the total loss and its components.

    Args:
        q_logit/phi/c_logit: network outputs (q_logit, c_logit are raw logits; phi is sigmoid).
        x / x_prime: same-scale reference / degraded image (B,1,H,W).
        active: (B,9) 0/1 multi-label.
        norm_target: (B,18) normalized parameter regression target.
    Returns:
        (L_E, {"L_cls":..., "L_reg":..., "L_syn":..., "L_conf":...}).
    """
    # classification loss
    L_cls = F.binary_cross_entropy_with_logits(q_logit, active)

    # masked regression loss (active parameters only)
    mask = activation_mask(active)  # (B,18)
    L_reg = masked_smooth_l1(phi, norm_target, mask)

    # synthesis consistency (pseudo-Huber)
    x_tilde = _apply_retain_batch(x, phi, active, seed)  # (B,1,H,W)
    sq_error = ((x_tilde - x_prime) ** 2).reshape(x.shape[0], -1).mean(dim=1)  # (B,)
    L_syn = (torch.sqrt(sq_error + eps_E ** 2) - eps_E).mean()

    # confidence loss (target = exp(-e_syn/tau_c), detached)
    with torch.no_grad():
        c_star = torch.exp(-sq_error / tau_c).unsqueeze(1).clamp(1e-4, 1.0)  # (B,1)
    L_conf = F.binary_cross_entropy_with_logits(c_logit, c_star)

    L_E = L_cls + lambda_reg * L_reg + lambda_syn * L_syn + lambda_conf * L_conf
    return L_E, {"L_cls": L_cls, "L_reg": L_reg, "L_syn": L_syn, "L_conf": L_conf}