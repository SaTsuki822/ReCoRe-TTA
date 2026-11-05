"""Degradation-estimator unit tests (self-contained, no pytest).

Run: python tests/test_estimator.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader

from src.degradation import sample_degradation, apply_all
from src.estimator import (
    DegradationEstimator, DescriptorEncoder, compute_losses, masked_smooth_l1,
    DegradationPairDataset, temperature_scale,
)

_PASS = 0


def check(cond, msg):
    global _PASS
    assert cond, msg
    _PASS += 1


def test_encoder_output_dim():
    torch.manual_seed(0)
    enc = DescriptorEncoder(in_ch=1, descriptor_dim=128)
    x = torch.rand(2, 1, 128, 128)
    h = enc(x)
    check(h.shape == (2, 128), "encoder descriptor shape (B,128)")


def test_estimator_forward():
    torch.manual_seed(0)
    model = DegradationEstimator()
    x = torch.rand(2, 1, 128, 128)
    xp = torch.rand(2, 1, 128, 128)
    q_logit, phi, c_logit = model(x, xp)
    check(q_logit.shape == (2, 9), "q_logit shape (B,9)")
    check(phi.shape == (2, 18), "phi shape (B,18)")
    check(bool(torch.all(phi >= 0)) and bool(torch.all(phi <= 1)), "phi in [0,1]")
    check(c_logit.shape == (2, 1), "c_logit shape (B,1)")
    q_hat, phi2, c_hat = model.predict(x, xp)
    check(bool(torch.all(q_hat >= 0)) and bool(torch.all(q_hat <= 1)), "q_hat in [0,1]")
    check(bool(torch.all(c_hat >= 0)) and bool(torch.all(c_hat <= 1)), "c_hat in [0,1]")


def test_masked_smooth_l1():
    pred = torch.tensor([[0.2, 0.9, 0.5]])
    target = torch.tensor([[0.3, 0.1, 0.5]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss = masked_smooth_l1(pred, target, mask)
    check(loss.ndim == 0, "masked_smooth_l1 returns scalar")
    expected = (0.5 * (0.2 - 0.3) ** 2 + 0.5 * (0.5 - 0.5) ** 2) / 2.0
    check(abs(loss.item() - expected) < 1e-6, "masked smooth l1 value correct")


def test_compute_losses():
    torch.manual_seed(0)
    model = DegradationEstimator()
    B = 2
    x = torch.rand(B, 1, 64, 64)
    xp_list, active_list, norm_list = [], [], []
    for b in range(B):
        a, raw, n = sample_degradation()
        xp_list.append(apply_all(x[b:b + 1], n, a))
        active_list.append(a)
        norm_list.append(n)
    x_prime = torch.cat(xp_list, dim=0)
    active = torch.stack(active_list, dim=0)
    norm = torch.stack(norm_list, dim=0)
    q_logit, phi, c_logit = model(x, x_prime)
    L_E, losses = compute_losses(q_logit, phi, c_logit, x, x_prime, active, norm, seed=1)
    check(L_E.ndim == 0, "L_E scalar")
    check(losses["L_cls"].ndim == 0 and losses["L_reg"].ndim == 0, "sub-losses scalar")
    L_E.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check(len(grads) > 0, "gradients flow into model")


def test_dataset():
    ds = DegradationPairDataset(img_dir=None, crop_size=64, length=4, seed=0)
    check(len(ds) == 4, "dataset length")
    x, xp, active, norm = ds[0]
    check(x.shape == (1, 64, 64), "x shape (C,H,W)=(1,64,64)")
    check(xp.shape == x.shape, "x_prime shape matches x")
    check(active.shape == (9,), "active shape (9,)")
    check(norm.shape == (18,), "norm shape (18,)")
    check(not torch.allclose(x, xp), "x_prime differs from x (degradation applied)")


def test_temperature_scale():
    torch.manual_seed(0)
    model = DegradationEstimator()
    ds = DegradationPairDataset(img_dir=None, crop_size=64, length=8, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    T = temperature_scale(model, loader, epochs=1)
    check(isinstance(T, float), "temperature returns float")
    check(T > 0, "temperature positive")


def main():
    global _PASS
    tests = [test_encoder_output_dim, test_estimator_forward, test_masked_smooth_l1,
             test_compute_losses, test_dataset, test_temperature_scale]
    for t in tests:
        t()
        print("  [PASS] %s" % t.__name__)
    print("\nAll %d checks passed (%d test functions)." % (_PASS, len(tests)))


if __name__ == "__main__":
    main()