"""Region-affinity unit tests (self-contained, no pytest).

Run: python tests/test_affinity.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from src.affinity import RegionAffinity

_PASS = 0


def check(cond, msg):
    global _PASS
    assert cond, msg
    _PASS += 1


def test_prepare_and_affinity():
    torch.manual_seed(0)
    ra = RegionAffinity(region_size=8)
    y = torch.rand(1, 1, 64, 64)
    ra.prepare(y)
    Z = torch.rand(1, 180, 64, 64)
    A = ra.affinity(Z)
    check(A.ndim == 2, "A is (B, nEdges)")
    check(A.shape[0] == 1, "batch dim 1")
    check(A.shape[1] == ra.n_edges_total, "n_edges consistent")
    check(bool((A >= 0).all()) and bool((A <= 1).all()), "affinity in [0,1]")


def test_anchor_loss_zero_within_tolerance():
    ra = RegionAffinity(region_size=8, eps_A=0.10)
    y = torch.rand(1, 1, 64, 64)
    ra.prepare(y)
    A_T = torch.rand(1, ra.n_edges_total) * 0.5 + 0.25
    A_S = A_T.clone()
    L = ra.anchor_loss(A_S, A_T)
    check(abs(L.item()) < 1e-6, "L_anc ~ 0 when |A_S-A_T| <= eps_A")


def test_anchor_loss_positive_beyond_tolerance():
    ra = RegionAffinity(region_size=8, eps_A=0.10)
    y = torch.rand(1, 1, 64, 64)
    ra.prepare(y)
    A_T = torch.zeros(1, ra.n_edges_total)
    A_S = torch.ones(1, ra.n_edges_total)
    L = ra.anchor_loss(A_S, A_T)
    check(L.item() > 0, "L_anc > 0 when affinity far apart")


def test_recovery_loss_grad_only_for_low_A():
    ra = RegionAffinity(region_size=8)
    y = torch.rand(1, 1, 64, 64)
    ra.prepare(y)
    A_T = torch.full((1, ra.n_edges_total), 0.8)
    r = torch.full((1, ra.n_edges_total), 0.5)
    A_S = torch.full((1, ra.n_edges_total), 0.2, requires_grad=True)
    L = ra.recovery_loss(A_S, A_T, r)
    L.backward()
    check(A_S.grad is not None, "grad flows into A_S")
    check(bool((A_S.grad <= 0).all()), "grad pushes A_S up (negative grad)")


def test_gate_monotonic_in_confidence():
    ra = RegionAffinity(region_size=8, tau_A=0.75, T_A=0.1, beta=2.0)
    y = torch.zeros(1, 1, 64, 64)  # B = 0 everywhere
    ra.prepare(y)
    A_T = torch.full((1, ra.n_edges_total), 0.9)
    r_high = ra.gate(A_T, 1.0)
    r_low = ra.gate(A_T, 0.1)
    check(bool((r_high > r_low).all()), "gate increases with confidence")
    check(bool((r_high >= 0).all()) and bool((r_high <= 1).all()), "gate in [0,1]")


def main():
    global _PASS
    tests = [test_prepare_and_affinity, test_anchor_loss_zero_within_tolerance,
             test_anchor_loss_positive_beyond_tolerance,
             test_recovery_loss_grad_only_for_low_A, test_gate_monotonic_in_confidence]
    for t in tests:
        t()
        print("  [PASS] %s" % t.__name__)
    print("\nAll %d checks passed (%d test functions)." % (_PASS, len(tests)))


if __name__ == "__main__":
    main()

