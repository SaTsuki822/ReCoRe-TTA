"""Backbone unit tests (self-contained, no pytest).

Uses tiny depths/embed_dim so it runs quickly with the pure-PyTorch selective-scan fallback.
Run: python tests/test_backbone.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from src.backbone import build_backbone, configure_model, collect_params

_PASS = 0


def check(cond, msg):
    global _PASS
    assert cond, msg
    _PASS += 1


def _tiny_backbone(scale=2):
    return build_backbone(scale=scale, in_chans=1, embed_dim=32, d_state=8,
                          depths=(1, 1, 1, 1, 1, 1), img_size=16)


def test_forward_shape():
    torch.manual_seed(0)
    model = _tiny_backbone(scale=2).eval()
    x = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    check(y.shape == (1, 1, 32, 32), "x2 SR shape correct")


def test_extract_features():
    torch.manual_seed(0)
    model = _tiny_backbone(scale=2).eval()
    x = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        Z = model.extract_features(x)
    check(Z.shape == (1, 32, 16, 16), "deep feature shape (B, embed_dim, H, W)")


def test_configure_model():
    model = _tiny_backbone(scale=2)
    configure_model(model, ("out_proj",))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    check(len(trainable) > 0, "has trainable params")
    check(all("out_proj" in n for n in trainable), "only out_proj trainable")
    check(len(frozen) > 0, "has frozen params")
    params, names = collect_params(model)
    check(len(params) == len(trainable), "collect_params matches trainable")


def test_gradient_flows_to_tunable():
    model = _tiny_backbone(scale=2)
    configure_model(model, ("out_proj",))
    model.eval()
    x = torch.rand(1, 1, 16, 16)
    Z = model.extract_features(x)
    Z.sum().backward()
    tunable_grad = [p for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    check(len(tunable_grad) > 0, "gradient flows into tunable linear layers")


def main():
    global _PASS
    tests = [test_forward_shape, test_extract_features, test_configure_model,
             test_gradient_flows_to_tunable]
    for t in tests:
        t()
        print("  [PASS] %s" % t.__name__)
    print("\nAll %d checks passed (%d test functions)." % (_PASS, len(tests)))


if __name__ == "__main__":
    main()

