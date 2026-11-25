"""Semantic regressions: JPEG intensity, stochastic synthesis and confidence.

Run: python tests/test_estimator_objectives.py
"""

import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch

from src.degradation import apply_all, apply_retain
from src.degradation.operators import _dct_basis, apply_jpeg
from src.estimator.losses import compute_losses


class ObjectiveTests(unittest.TestCase):
    def test_dct_is_orthonormal(self):
        basis = _dct_basis()
        torch.testing.assert_close(basis @ basis.t(), torch.eye(8), atol=1e-6, rtol=1e-6)

    def test_jpeg_preserves_constant_brightness(self):
        for value in (0., .25, .5, .75, 1.):
            for quality in (35, 50, 95, 100):
                x = torch.full((1, 1, 17, 23), value)
                y = apply_jpeg(x, quality)
                self.assertEqual(x.shape, y.shape)
                self.assertLess((y - x).abs().max().item(), .02)
                self.assertTrue(((y >= 0) & (y <= 1)).all())

    def test_jpeg_agrees_with_pillow_on_grayscale_blocks(self):
        # Integer pixels, complete blocks: both implementations use the luminance table.
        torch.manual_seed(7)
        pixels = torch.randint(0, 256, (32, 40), dtype=torch.uint8).numpy()
        x = torch.from_numpy(pixels.astype(np.float32) / 255)[None, None]
        for quality in (35, 50, 80, 95, 100):
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            reference = torch.from_numpy(np.asarray(Image.open(buffer)).astype(np.float32) / 255)
            result = apply_jpeg(x, quality)[0, 0]
            self.assertLess((result - reference).abs().mean().item(), .01)

    def test_oracle_parameters_have_zero_retained_synthesis_error(self):
        torch.manual_seed(9)
        x = torch.rand(2, 1, 32, 32)
        active = torch.ones(2, 9)  # discarded effects must not penalize retained parameters
        norm = torch.full((2, 18), .4)
        xp = torch.cat([apply_all(x[b:b+1], norm[b], active[b], seed=17 + b)
                        for b in range(2)])
        pred = norm.clone().requires_grad_()
        _, parts = compute_losses(torch.full_like(active, 10), pred,
                                  torch.zeros(2, 1), x, xp, active, norm, seed=42)
        self.assertLess(abs(parts["L_syn"].item()), 1e-7)
        parts["L_syn"].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_noise_strength_is_not_driven_down_at_ground_truth(self):
        x = torch.full((1, 1, 32, 32), .5)
        active = torch.zeros(1, 9)
        active[:, 3] = 1
        norm = torch.full((1, 18), .7)
        xp = apply_retain(x, norm[0], active[0], seed=10)
        pred = norm.clone().requires_grad_()
        _, parts = compute_losses(torch.full_like(active, 10), pred,
                                  torch.zeros(1, 1), x, xp, active, norm, seed=99)
        parts["L_syn"].backward()
        self.assertLess(abs(pred.grad[0, 7].item()), 1e-7)

    def test_different_noise_families_use_independent_streams(self):
        x = torch.full((1, 1, 48, 48), .5)
        norm = torch.full((18,), .5)
        active_r, active_s = torch.zeros(9), torch.zeros(9)
        active_r[3], active_s[4] = 1, 1
        r = (apply_retain(x, norm, active_r, seed=3) - x).flatten()
        s = (apply_retain(x, norm, active_s, seed=3) - x).flatten()
        correlation = torch.corrcoef(torch.stack((r, s)))[0, 1]
        self.assertLess(abs(correlation.item()), .1)

    def test_confidence_uses_predicted_labels_instead_of_training_labels(self):
        x = torch.full((1, 1, 32, 32), .5)
        norm = torch.full((1, 18), .7)
        predicted = torch.zeros(1, 9)
        predicted[:, 3] = 1
        xp = apply_retain(x, norm[0], predicted[0], seed=42)
        q = torch.full((1, 9), -10.)
        q[:, 3] = 10
        # Deliberately contradictory training labels: the confidence target
        # must still describe the reconstruction selected by q, as at inference.
        _, parts = compute_losses(q, norm, torch.full((1, 1), 2.), x, xp,
                                  torch.zeros_like(predicted), norm, seed=42)
        expected = torch.nn.functional.softplus(torch.tensor(-2.)).item()
        self.assertAlmostEqual(parts["L_conf"].item(), expected, places=6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
