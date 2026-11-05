"""Confidence temperature calibration.

Fit a single scalar temperature T so that c_hat = sigmoid(c_logit / T), on a
calibration split after training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import _apply_retain_batch


def temperature_scale(model, loader, tau_c=0.05, epochs=2, lr=0.05):
    """Fit a scalar temperature T on the calibration set (model weights frozen).

    c_hat = sigmoid(c_logit / T), target c* = exp(-e_syn / tau_c), where e_syn is the
    detached synthesis error. Returns T (float).
    """
    device = next(model.parameters()).device
    model.eval()
    T = nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.Adam([T], lr=lr)
    for _ in range(epochs):
        for x, x_prime, active, norm in loader:
            x = x.to(device)
            x_prime = x_prime.to(device)
            with torch.no_grad():
                q_logit, phi, c_logit = model(x, x_prime)
                pred_active = (torch.sigmoid(q_logit) > 0.5).float()
                x_tilde = _apply_retain_batch(x, phi, pred_active)
                sq_error = ((x_tilde - x_prime) ** 2).reshape(x.shape[0], -1).mean(dim=1)
                c_star = torch.exp(-sq_error / tau_c).unsqueeze(1).clamp(1e-4, 1.0)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(c_logit / T, c_star)
            loss.backward()
            opt.step()
    return float(T.item())