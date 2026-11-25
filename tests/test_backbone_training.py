"""Sampling, native validation metrics and checkpoint-selection regressions.

Run: python tests/test_backbone_training.py
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.backbone.train import PairedSRDataset, _backward_step, make_parser, train
from src.backbone.validation import ValidationPairs, build_validation_sets, evaluate
from src.metrics.psnr_ssim import _gaussian_window, calculate_psnr, calculate_ssim
from src.utils.image_io import read_img_gray


class Interpolator(torch.nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.gain = torch.nn.Parameter(torch.tensor(.8))

    def forward(self, x):
        return self.gain * F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)


class BackboneTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / "train"
        self.sources.mkdir()
        rng = np.random.default_rng(3)
        for i in range(6):
            imageio.imwrite(self.sources / ("%d.png" % i), rng.integers(0, 256, (72, 88), dtype=np.uint8))
        self.paths = sorted(str(p) for p in self.sources.glob("*.png"))
        self.validation = self.root / "datasets"
        for folder in ("results-A", "results-C"):
            base = self.validation / folder
            (base / "HR").mkdir(parents=True)
            for i in range(2):
                hr = rng.integers(0, 256, (64, 72), dtype=np.uint8)
                imageio.imwrite(base / "HR" / ("image%d.png" % i), hr)
                for scale in (2, 4):
                    lr_dir = base / "LR_bicubic" / ("X%d" % scale)
                    lr_dir.mkdir(parents=True, exist_ok=True)
                    lr = F.interpolate(torch.from_numpy(hr.astype(np.float32))[None, None],
                                       scale_factor=1/scale, mode="bicubic", align_corners=False,
                                       antialias=True).clamp(0, 255)[0, 0].numpy().round().astype(np.uint8)
                    imageio.imwrite(lr_dir / ("image%dx%d.png" % (i, scale)), lr)

    def args(self, *extra):
        return make_parser().parse_args([
            "--dataroot", str(self.sources), "--val_root", str(self.validation),
            "--save_dir", str(self.root / "output"), "--gt_size", "32",
            "--images_per_epoch", "3", "--batch_size", "2", "--epochs", "3",
            "--save_every", "1", "--cpu", *extra])

    def test_streams_refresh_each_epoch_and_are_worker_independent(self):
        ds = PairedSRDataset(self.paths, gt_size=32, length=8, seed=3407)
        single = list(DataLoader(ds, batch_size=1, num_workers=0))
        parallel = list(DataLoader(ds, batch_size=1, num_workers=2))
        repeat = list(DataLoader(ds, batch_size=1, num_workers=2))
        self.assertTrue(all(torch.equal(a, b) for x, y in zip(single, parallel) for a, b in zip(x, y)))
        self.assertTrue(all(torch.equal(a, b) for x, y in zip(parallel, repeat) for a, b in zip(x, y)))
        self.assertTrue(all(not torch.equal(parallel[i][1], parallel[i+1][1]) for i in (0, 2, 4, 6)))
        ds.set_epoch(1)
        new = list(DataLoader(ds, batch_size=1, num_workers=2))
        self.assertTrue(all(not torch.equal(x[1], y[1]) for x, y in zip(single, new)))
        other = PairedSRDataset(self.paths, gt_size=32, length=8, seed=3407)
        other.set_epoch(1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(ds[0], other[0])))

    def test_clean_branch_and_input_validation(self):
        ds = PairedSRDataset(self.paths, gt_size=32, clean_prob=1)
        lr, hr = ds[0]
        expected = F.interpolate(hr[None], scale_factor=.5, mode="bicubic",
                                 align_corners=False, antialias=True).clamp(0, 1)[0]
        torch.testing.assert_close(lr, expected)
        for kwargs in ({"gt_size": 33}, {"clean_prob": -1}, {"length": 0}):
            with self.assertRaises(ValueError):
                PairedSRDataset(self.paths, **kwargs)

    def test_native_pair_matching_for_x2_and_x4(self):
        for scale in (2, 4):
            sets = build_validation_sets(self.validation, scale, scale)
            self.assertEqual(set(sets), {"Result-A", "Result-C"})
            for ds in sets.values():
                self.assertEqual(len(ds), 2)
                lr, hr, name = ds[0]
                self.assertEqual(hr.shape[-2:], (lr.shape[-2] * scale, lr.shape[-1] * scale))
                self.assertIn("x%d" % scale, name)
        root = self.validation / "results-A"
        imageio.imwrite(root / "LR_bicubic/X2/image0x2.png", np.zeros((16, 16), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "scale mismatch"):
            ValidationPairs(root, 2)

    def test_missing_pairs_are_rejected(self):
        (self.validation / "results-A/LR_bicubic/X2/image0x2.png").unlink()
        with self.assertRaisesRegex(ValueError, "without matching LR"):
            build_validation_sets(self.validation, 2, 2)

    def test_training_validation_aliases_are_rejected(self):
        os.link(self.validation / "results-A/HR/image0.png", self.sources / "alias.png")
        with self.assertRaisesRegex(ValueError, "overlaps validation"):
            train(self.args())

    def test_evaluation_is_deterministic_frozen_and_restores_training_mode(self):
        sets = build_validation_sets(self.validation, 2, 2)
        model = Interpolator().train()
        before = model.gain.detach().clone()
        a = evaluate(model, sets, "cpu", 2)
        b = evaluate(model, sets, "cpu", 2)
        self.assertEqual(a, b)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.gain, before))
        self.assertIsNone(model.gain.grad)
        self.assertEqual(a["Result-A"]["images"], 2)
        self.assertAlmostEqual(a["mean"]["psnr"], (a["Result-A"]["psnr"]+a["Result-C"]["psnr"])/2)

    @unittest.skipUnless(importlib.util.find_spec("skimage"), "optional reference library unavailable")
    def test_ssim_agrees_with_independent_reference(self):
        from skimage.metrics import structural_similarity
        rng = np.random.default_rng(5)
        a = rng.random((64, 72))
        b = (a + rng.normal(0, .02, a.shape)).clip(0, 1)
        win = _gaussian_window()
        self.assertEqual(win.shape, (11, 11))
        self.assertTrue(np.allclose(win, win[::-1, ::-1]))
        for border in (0, 2, 4):
            x = a[border:-border, border:-border] if border else a
            y = b[border:-border, border:-border] if border else b
            expected = structural_similarity(x, y, data_range=1., gaussian_weights=True,
                                               sigma=1.5, use_sample_covariance=False)
            self.assertAlmostEqual(calculate_ssim(a, b, border), expected, places=10)
            self.assertAlmostEqual(calculate_psnr(a, b, border), -10*np.log10(np.mean((x-y)**2)), places=10)

    def test_dark_and_uint16_images_are_normalized(self):
        dark = self.root / "dark.png"
        imageio.imwrite(dark, np.ones((16, 16, 3), dtype=np.uint8))
        self.assertTrue(np.allclose(read_img_gray(dark), 1 / 255))
        raw = self.root / "raw16.png"
        imageio.imwrite(raw, np.full((16, 16), 32768, dtype=np.uint16))
        self.assertTrue(np.allclose(read_img_gray(raw), 32768 / 65535))

    def test_training_selects_validation_winner_not_last_and_decays_lr(self):
        args = self.args()
        scores = iter((30., 20., 10.))

        def controlled(model, datasets, device, crop_border):
            result = evaluate(model, datasets, device, crop_border)
            result["mean"]["psnr"] = next(scores)
            return result

        with patch("src.backbone.train.build_backbone", side_effect=lambda **kw: Interpolator(kw["scale"])), \
             patch("src.backbone.train.evaluate", side_effect=controlled):
            train(args)
        p = Path(args.save_dir)
        best = torch.load(p / "backbone_x2_best.pt", weights_only=True)
        final = torch.load(p / "backbone_x2.pt", weights_only=True)
        last = torch.load(p / "backbone_x2_last.pt", weights_only=True)
        history = [json.loads(line) for line in (p / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(best["epoch"], 1)
        self.assertTrue(all(torch.equal(v, final[k]) for k, v in best["state_dict"].items()))
        self.assertFalse(torch.equal(final["gain"], last["gain"]))
        self.assertGreater(history[0]["lr"], history[-1]["lr"])
        self.assertAlmostEqual(history[-1]["next_lr"], args.min_lr)
        self.assertTrue((p / "backbone_x2_epoch3.pt").is_file())
        self.assertEqual(json.loads((p / "training_summary.json").read_text())["best_epoch"], 1)

    def test_final_epoch_is_validated_off_interval(self):
        args = self.args("--val_every", "2")
        with patch("src.backbone.train.build_backbone", side_effect=lambda **kw: Interpolator(kw["scale"])):
            train(args)
        history = [json.loads(line) for line in (Path(args.save_dir) / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(["val" in row for row in history], [False, True, True])

    def test_nonfinite_gradients_are_rejected_before_optimizer_update(self):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(RuntimeError, "optimizer step not applied"):
            _backward_step(model.weight.sqrt().sum(), model, optimizer, 1., "test batch")
        self.assertEqual(len(optimizer.state), 0)
        torch.testing.assert_close(model.weight, torch.zeros_like(model.weight))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
