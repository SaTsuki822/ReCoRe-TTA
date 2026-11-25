"""Confidence temperature calibration.

Fit a single scalar temperature T so that c_hat = sigmoid(c_logit / T), on a
calibration split after training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import confidence_targets


def temperature_scale(model, loader, tau_c=0.05, epochs=2, lr=0.05, return_metrics=False):
    """Fit a scalar temperature T on the calibration set (model weights frozen).

    c_hat = sigmoid(c_logit / T), target c* = exp(-e_syn / tau_c), where e_syn is the
    detached synthesis error. Returns T (float).
    """
    device = next(model.parameters()).device
    model.eval()
    # The frozen network and fixed calibration pairs need only one forward pass.
    cached = []
    with torch.no_grad():
        for batch_idx, (x, x_prime, active, norm) in enumerate(loader):
            x, x_prime = x.to(device), x_prime.to(device)
            q_logit, phi, c_logit = model(x, x_prime)
            target = confidence_targets(x, x_prime, q_logit, phi,
                                        seed=10000 + batch_idx, tau_c=tau_c)
            if not torch.isfinite(c_logit).all() or not torch.isfinite(target).all():
                raise RuntimeError("Non-finite calibration logits/targets")
            cached.append((c_logit.detach(), target))
    if not cached:
        raise ValueError("Calibration loader must not be empty")
    all_logits = torch.cat([p[0] for p in cached])
    all_targets = torch.cat([p[1] for p in cached])
    before = F.binary_cross_entropy_with_logits(all_logits, all_targets).item()
    best_loss, best_T = before, 1.0
    log_T = nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.Adam([log_T], lr=lr)
    for _ in range(epochs):
        for c_logit, c_star in cached:
            opt.zero_grad()
            T = log_T.clamp(-5.0, 5.0).exp()
            loss = F.binary_cross_entropy_with_logits(c_logit / T, c_star)
            loss.backward()
            opt.step()
        with torch.no_grad():
            T = log_T.clamp(-5.0, 5.0).exp()
            current = F.binary_cross_entropy_with_logits(all_logits / T, all_targets).item()
            if current < best_loss:
                best_loss, best_T = current, T.item()
    metrics = {"bce_before": before, "bce_after": best_loss,
               "temperature": best_T, "samples": len(all_logits)}
    return (best_T, metrics) if return_metrics else float(best_T)
