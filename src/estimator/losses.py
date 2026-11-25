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
        sample_seed = None if seed is None else (int(seed) + 1009 * b) % (2 ** 63)
        outs.append(apply_retain(x[b:b + 1], phi[b], active[b], sample_seed))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def confidence_targets(x, x_prime, q_logit, phi, seed=None, tau_c=0.05):
    """Observable reconstruction reliability with predicted labels, as at TTA.

    Unlike L_syn, this compares the retained reconstruction with the full
    observation: irreversible effects should lower the confidence gate.
    """
    active = (q_logit > 0).to(x.dtype)
    # Match the harder-view fallback when no retained family passes the threshold.
    missing = active[:, :6].sum(1) == 0
    top = q_logit[:, :6].argmax(1)
    active[missing, top[missing]] = 1
    reconstruction = _apply_retain_batch(x, phi, active, seed)
    error = (reconstruction - x_prime).square().flatten(1).mean(1)
    return torch.exp(-error / tau_c).unsqueeze(1).clamp(1e-4, 1.0)


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

    # Replay retained GT with common random numbers. Comparing to x_prime would
    # penalize irreversible effects and unrelated noise draws, even at exact GT,
    # encouraging the estimator to underestimate stochastic noise amplitudes.
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31, (1,)).item())
    x_tilde = _apply_retain_batch(x, phi, active, seed)  # (B,1,H,W)
    with torch.no_grad():
        retained_target = _apply_retain_batch(x, norm_target, active, seed)
    sq_error = ((x_tilde - retained_target) ** 2).reshape(x.shape[0], -1).mean(dim=1)
    L_syn = (torch.sqrt(sq_error + eps_E ** 2) - eps_E).mean()

    # confidence loss (target = exp(-e_syn/tau_c), detached)
    c_star = confidence_targets(x, x_prime, q_logit, phi, seed, tau_c)
    L_conf = F.binary_cross_entropy_with_logits(c_logit, c_star)

    L_E = L_cls + lambda_reg * L_reg + lambda_syn * L_syn + lambda_conf * L_conf
    return L_E, {"L_cls": L_cls, "L_reg": L_reg, "L_syn": L_syn, "L_conf": L_conf}
