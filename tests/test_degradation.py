"""Degradation operator unit tests (self-contained, no pytest).

Run: python tests/test_degradation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from src.degradation import (
    N_CLASSES, N_PARAMS, RETAIN_INDICES, DISCARD_INDICES,
    normalize_params, denormalize_params, split_params,
    apply_blur, apply_gain_nonuniformity, apply_offset_nonuniformity,
    apply_readout_noise, apply_signal_dependent_noise, apply_stripe,
    apply_quantization, apply_jpeg, apply_saturation_agc, apply_bad_pixels,
    sample_degradation, safe_project, apply_retain, apply_discard, apply_all,
    full_degradation,
)
import src.degradation.params as P


_PASS = 0


def check(cond, msg):
    global _PASS
    assert cond, msg
    _PASS += 1


def test_params():
    check(N_CLASSES == 9, "N_CLASSES should be 9")
    check(N_PARAMS == 18, "N_PARAMS should be 18")
    check(len(RETAIN_INDICES) == 6, "6 retain classes")
    check(len(DISCARD_INDICES) == 3, "3 discard classes")
    low = torch.tensor(P.PARAM_LOWS)
    high = torch.tensor(P.PARAM_HIGHS)
    raw = low + (high - low) * torch.rand(18)
    norm = normalize_params(raw)
    check(bool(torch.all(norm >= 0)) and bool(torch.all(norm <= 1)), "norm in [0,1]")
    back = denormalize_params(norm)
    check(torch.allclose(back, raw, atol=1e-6), "denormalize(normalize(x)) == x")
    segs = split_params(raw)
    check(len(segs) == 9, "split into 9 segments")
    check(sum(s.numel() for s in segs) == 18, "split total length 18")


def test_operator_shapes_and_range():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64)
    outs = [
        apply_blur(x, 1.0, 1.0, 0.0),
        apply_gain_nonuniformity(x, 0.05, 32.0, seed=1),
        apply_offset_nonuniformity(x, 0.05, 32.0, seed=1),
        apply_readout_noise(x, 0.03, seed=1),
        apply_signal_dependent_noise(x, 0.02, seed=1),
        apply_stripe(x, 0.03, 2.0, 0.0, seed=1),
        apply_quantization(x, 8.0),
        apply_jpeg(x, 80.0),
        apply_saturation_agc(x, 0.02, 1.0),
        apply_bad_pixels(x, 0.005, 0.0, seed=1),
    ]
    for o in outs:
        check(o.shape == x.shape, "operator preserves shape")
        check(bool(torch.isfinite(o).all()), "operator output finite")
    # JPEG on a non-multiple-of-8 size (50x50)
    y = apply_jpeg(torch.rand(1, 1, 50, 50), 60.0)
    check(y.shape == (1, 1, 50, 50), "jpeg on non-multiple-of-8 size")


def test_determinism():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 32, 32)
    a = apply_offset_nonuniformity(x, 0.05, 20.0, seed=7)
    b = apply_offset_nonuniformity(x, 0.05, 20.0, seed=7)
    check(torch.allclose(a, b), "same seed -> same output")


def test_differentiability():
    x = torch.rand(1, 1, 32, 32)
    sx = torch.tensor(1.0, requires_grad=True)
    sy = torch.tensor(1.2, requires_grad=True)
    th = torch.tensor(0.3, requires_grad=True)
    y = apply_blur(x, sx, sy, th)
    y.sum().backward()
    check(sx.grad is not None and sy.grad is not None and th.grad is not None, "blur grads exist")
    sr = torch.tensor(0.03, requires_grad=True)
    y = apply_readout_noise(x, sr, seed=3)
    y.sum().backward()
    check(sr.grad is not None, "readout noise grad exists")
    ag = torch.tensor(0.05, requires_grad=True)
    y = apply_gain_nonuniformity(x, ag, 32.0, seed=3)
    y.sum().backward()
    check(ag.grad is not None, "gain amplitude grad exists")


def test_sample_degradation():
    rng = torch.Generator().manual_seed(0)
    active, raw, norm = sample_degradation(rng)
    check(active.shape == (9,), "active shape (9,)")
    check(float(active.sum()) >= 1.0, "at least one active")
    check(raw.shape == (18,) and norm.shape == (18,), "raw/norm shape (18,)")
    low = torch.tensor(P.PARAM_LOWS)
    high = torch.tensor(P.PARAM_HIGHS)
    mask = P.activation_mask(active)
    check(torch.all(raw[mask.bool()] >= low[mask.bool()] - 1e-6), "active raw >= low")
    check(torch.all(raw[mask.bool()] <= high[mask.bool()] + 1e-6), "active raw <= high")
    check(torch.all(raw[~mask.bool()] == 0), "inactive raw == 0")


def test_safe_project():
    norm = torch.rand(18)
    safe = safe_project(norm)
    check(bool(torch.all(safe >= 0)) and bool(torch.all(safe <= 1)), "safe in [0,1]")
    for i in DISCARD_INDICES:
        check(torch.all(safe[P.PARAM_SLICES[i]] == 0), "discard class zeroed")
    for i in RETAIN_INDICES:
        check(torch.allclose(safe[P.PARAM_SLICES[i]], norm[P.PARAM_SLICES[i]]), "retain class kept")


def test_apply_retain_and_discard():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 48, 48)
    active = torch.zeros(9)
    active[0] = 1.0  # blur
    active[3] = 1.0  # readout noise
    active[6] = 1.0  # quantization (discard)
    norm = torch.rand(18)
    y_retain = apply_retain(x, norm, active, seed=5)
    check(y_retain.shape == x.shape, "apply_retain preserves shape (no downsampling)")
    check(not torch.allclose(y_retain, x), "apply_retain actually degraded input")
    y_all = apply_all(x, norm, active, seed=5)
    check(y_all.shape == x.shape, "apply_all preserves shape")


def test_full_degradation():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 128, 128)
    rng = torch.Generator().manual_seed(0)
    lr, active, raw, norm = full_degradation(x, scale=2, rng=rng)
    check(lr.shape == (1, 1, 64, 64), "LR size = H//2 x W//2")
    check(bool(torch.all(lr >= 0)) and bool(torch.all(lr <= 1)), "LR in [0,1]")
    rng2 = torch.Generator().manual_seed(0)
    lr2, _, _, _ = full_degradation(x, scale=2, rng=rng2)
    check(torch.allclose(lr, lr2), "full_degradation reproducible")


def main():
    global _PASS
    tests = [
        test_params, test_operator_shapes_and_range, test_determinism,
        test_differentiability, test_sample_degradation, test_safe_project,
        test_apply_retain_and_discard, test_full_degradation,
    ]
    for t in tests:
        t()
        print("  [PASS] %s" % t.__name__)
    print("\nAll %d checks passed (%d test functions)." % (_PASS, len(tests)))


if __name__ == "__main__":
    main()
