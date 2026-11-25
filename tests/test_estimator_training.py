"""Regression tests for sampling, source splits, and checkpoint selection.

Run: python tests/test_estimator_training.py
"""

import json
import math
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
from torch.utils.data import DataLoader

from src.degradation import sample_degradation, apply_readout_noise, apply_bad_pixels, apply_all
from src.estimator import DegradationEstimator, DegradationPairDataset, temperature_scale, compute_losses
from src.estimator.data import discover_images, _to_single_gray
from src.estimator.train import build_datasets, evaluate, make_parser, train, _backward_step


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for i in range(10):
            img = (np.arange(48 * 48).reshape(48, 48) + i * 17) % 256
            imageio.imwrite(self.images / ("%02d.png" % i), img.astype(np.uint8))

    def args(self, *extra):
        return make_parser().parse_args([
            "--img_dir", str(self.images), "--crop_size", "32",
            "--length", "8", "--val_length", "4", "--cal_length", "4",
            "--epochs", "2", "--batch_size", "2", "--lr", "0.0001",
            "--save_every", "1", "--save_dir", str(self.root / "output"),
            "--cpu", *extra])

    def test_default_rng_advances_and_reseeds(self):
        torch.manual_seed(12)
        first = [sample_degradation() for _ in range(12)]
        self.assertGreater(len({tuple(s[0].tolist()) for s in first}), 1)
        self.assertFalse(torch.equal(first[0][2], first[1][2]))
        torch.manual_seed(12)
        second = [sample_degradation() for _ in range(12)]
        self.assertTrue(all(torch.equal(a, b) for sa, sb in zip(first, second)
                            for a, b in zip(sa, sb)))
        a = torch.Generator().manual_seed(5)
        b = torch.Generator().manual_seed(5)
        self.assertTrue(all(torch.equal(x, y) for x, y in
                            zip(sample_degradation(a), sample_degradation(b))))
        x = torch.zeros(1, 1, 32, 32)
        for op, kwargs in ((apply_readout_noise, {"sigma_r": .03}),
                           (apply_bad_pixels, {"p_bad": .3, "v_bad": 1})):
            self.assertFalse(torch.equal(op(x, **kwargs), op(x, **kwargs)))
            self.assertTrue(torch.equal(op(x, seed=7, **kwargs), op(x, seed=7, **kwargs)))

    def test_dark_rgb_and_uint16_normalization(self):
        dark = np.ones((8, 8, 3), dtype=np.uint8)
        self.assertTrue(np.allclose(_to_single_gray(dark), 1 / 255))
        infrared = np.full((8, 8), 32768, dtype=np.uint16)
        self.assertTrue(np.allclose(_to_single_gray(infrared), 32768 / 65535))
        self.assertEqual(_to_single_gray(dark[..., :1]).shape, (8, 8))

    def test_empty_or_missing_real_data_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        for path in (empty, self.root / "missing"):
            with self.assertRaises(FileNotFoundError):
                DegradationPairDataset(img_dir=path)
        with self.assertRaises(ValueError):
            build_datasets(make_parser().parse_args([]))
        with self.assertRaises(ValueError):
            build_datasets(self.args("--synthetic"))
        # Case-insensitive extensions; nested files are deliberately not included.
        imageio.imwrite(empty / "test.PNG", np.zeros((32, 32), dtype=np.uint8))
        (empty / "nested").mkdir()
        self.assertEqual(len(discover_images(empty)), 1)

    def test_source_splits_are_disjoint_and_reproducible(self):
        args = self.args("--calibrate")
        datasets, sources = build_datasets(args)
        _, repeat = build_datasets(args)
        self.assertEqual(sources, repeat)
        self.assertEqual([len(sources[k]) for k in ("train", "val", "cal")], [8, 1, 1])
        self.assertEqual(len(set(sum(sources.values(), []))), 10)
        self.assertFalse(datasets["train"].fixed)
        self.assertTrue(datasets["val"].fixed and datasets["cal"].fixed)
        self.assertFalse(datasets["val"].augment)

    def test_supplied_splits_and_alias_overlap(self):
        val = self.root / "val"
        cal = self.root / "cal"
        val.mkdir()
        cal.mkdir()
        for folder in (val, cal):
            imageio.imwrite(folder / "new.png", np.zeros((32, 32), dtype=np.uint8))
        _, sources = build_datasets(self.args("--calibrate", "--val_dir", str(val),
                                              "--cal_dir", str(cal)))
        self.assertEqual([len(sources[k]) for k in ("train", "val", "cal")], [10, 1, 1])
        for extra in (("--val_dir", str(val)), ("--cal_dir", str(cal))):
            _, splits = build_datasets(self.args("--calibrate", *extra))
            self.assertEqual(len(set(sum(splits.values(), []))), 11)
        os.link(self.images / "00.png", val / "alias.png")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_datasets(self.args("--val_dir", str(val), "--cal_dir", str(cal), "--calibrate"))

    def test_pairs_refresh_by_epoch_but_held_out_pairs_stay_fixed(self):
        datasets, _ = build_datasets(self.args("--calibrate"))
        training = datasets["train"]
        first = training[0]
        training.set_epoch(1)
        self.assertFalse(torch.equal(first[3], training[0][3]))
        repeat, _ = build_datasets(self.args("--calibrate"))
        repeat["train"].set_epoch(1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(training[0], repeat["train"][0])))
        for split in ("val", "cal"):
            dataset = datasets[split]
            before = dataset[0]
            torch.manual_seed(999)
            dataset.set_epoch(8)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, dataset[0])))
        # Worker count must not change the generated data or duplicate worker streams.
        for dataset in (training, datasets["val"]):
            single = list(DataLoader(dataset, batch_size=2, num_workers=0))
            workers = list(DataLoader(dataset, batch_size=2, num_workers=2))
            self.assertTrue(all(torch.equal(a, b) for sa, sb in zip(single, workers)
                                for a, b in zip(sa, sb)))

    def test_evaluation_is_fixed_and_does_not_update_model(self):
        datasets, _ = build_datasets(self.args())
        model = DegradationEstimator()
        loader = DataLoader(datasets["val"], batch_size=2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        first = evaluate(model, loader, "cpu")
        torch.manual_seed(12345)
        second = evaluate(model, loader, "cpu")
        self.assertEqual(first, second)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertEqual(len(first["per_class"]), 9)
        self.assertEqual(len(first["per_param_mae"]), 18)
        self.assertIn("retained_macro_f1", first)
        self.assertIn("recall", first["per_class"]["stripe"])
        self.assertIn("all_positive_retained_macro_f1", first["baselines"])

    def test_calibration_is_cached_frozen_and_never_worse_on_its_split(self):
        datasets, _ = build_datasets(self.args("--calibrate"))
        model = DegradationEstimator()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        loader = DataLoader(datasets["cal"], batch_size=2)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            T, metrics = temperature_scale(model, loader, epochs=3, return_metrics=True)
        self.assertEqual(forward.call_count, len(loader))
        self.assertEqual(metrics["samples"], 4)
        self.assertLessEqual(metrics["bce_after"], metrics["bce_before"])
        self.assertGreater(T, 0)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_reported_baselines_match_an_actual_constant_predictor(self):
        class ConstantPredictor(torch.nn.Module):
            def forward(self, x, xp):
                return (x.new_ones((len(x), 9)), x.new_full((len(x), 18), .5),
                        x.new_zeros((len(x), 1)))

        datasets, _ = build_datasets(self.args())
        metrics = evaluate(ConstantPredictor(), DataLoader(datasets["val"], batch_size=2), "cpu")
        baseline = metrics["baselines"]
        self.assertAlmostEqual(metrics["macro_f1"], baseline["all_positive_macro_f1"])
        self.assertAlmostEqual(metrics["retained_macro_f1"], baseline["all_positive_retained_macro_f1"])
        self.assertAlmostEqual(metrics["active_param_mae"], baseline["midpoint_active_param_mae"])
        self.assertAlmostEqual(metrics["retained_active_param_mae"],
                               baseline["midpoint_retained_active_param_mae"])

    def test_real_image_training_selects_best_and_calibrates(self):
        args = self.args("--calibrate")
        # Force the first epoch to win, so saving the last epoch cannot pass.
        scheduled_losses = iter((.1, .9))

        def controlled_evaluation(*items):
            result = evaluate(*items)
            result["loss"] = next(scheduled_losses)
            return result

        with patch("src.estimator.train.evaluate", side_effect=controlled_evaluation), \
             patch("src.estimator.train.temperature_scale", wraps=temperature_scale) as calibrator:
            train(args)
        self.assertEqual(calibrator.call_count, 1)
        calibration_dataset = calibrator.call_args[0][1].dataset
        output = Path(args.save_dir)
        manifest = json.loads((output / "data_split.json").read_text())
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        best = torch.load(output / "estimator_best.pt", map_location="cpu")
        final = torch.load(output / "estimator_final.pt", map_location="cpu")
        self.assertEqual(manifest["mode"], "real")
        self.assertEqual(calibration_dataset.img_paths, manifest["sources"]["cal"])
        self.assertEqual(len(records), 2)
        self.assertEqual(final["epoch"], 1)
        self.assertEqual(final["epoch"], min(records, key=lambda r: r["val"]["loss"])["epoch"])
        self.assertTrue(all(torch.equal(v, final["state_dict"][k]) for k, v in best["state_dict"].items()))
        self.assertTrue(math.isfinite(final["temperature"]) and final["temperature"] > 0)
        DegradationEstimator().load_state_dict(final["state_dict"], strict=True)
        self.assertTrue((output / "estimator_epoch2.pt").is_file())

    def test_synthetic_mode_is_explicit_and_has_fixed_holdouts(self):
        args = make_parser().parse_args(["--synthetic", "--calibrate", "--length", "2"])
        datasets, sources = build_datasets(args)
        self.assertTrue(all(ds.synthetic for ds in datasets.values()))
        self.assertTrue(all(paths is None for paths in sources.values()))
        self.assertFalse(torch.equal(datasets["train"][0][0], datasets["val"][0][0]))
        self.assertFalse(torch.equal(datasets["train"][0][3], datasets["val"][0][3]))
        self.assertFalse(torch.equal(datasets["val"][0][3], datasets["cal"][0][3]))


    def test_black_regions_keep_synthesis_gradients_finite(self):
        torch.manual_seed(3407)
        model = DegradationEstimator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.zeros(2, 1, 32, 32)
        x[:, :, 12:20, 12:20] = .5
        active = torch.zeros(2, 9)
        active[:, 0] = active[:, 4] = 1  # blur -> signal noise, with black borders
        norm = torch.full((2, 18), .4)
        xp = torch.cat([apply_all(x[b:b+1], norm[b], active[b], seed=3)
                        for b in range(2)])
        for step in range(3):
            optimizer.zero_grad()
            logits, phi, conf = model(x, xp)
            loss, _ = compute_losses(logits, phi, conf, x, xp, active, norm, seed=3)
            grad_norm = _backward_step(loss, model, optimizer, 1.0, "black region step %d" % step)
            self.assertTrue(torch.isfinite(grad_norm))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_invalid_gradients_do_not_reach_optimizer(self):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Forward loss is finite; its backward derivative is infinite.
        loss = model.weight.sqrt().sum()
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(RuntimeError, "Non-finite gradients at test batch"):
            _backward_step(loss, model, optimizer, 1.0, "test batch")
        self.assertTrue(torch.equal(model.weight, torch.zeros_like(model.weight)))
        self.assertEqual(len(optimizer.state), 0)
        optimizer.zero_grad()
        with self.assertRaisesRegex(RuntimeError, "Non-finite loss at test loss"):
            _backward_step(model.weight.sum() * float("nan"), model, optimizer, 1.0, "test loss")
        self.assertEqual(len(optimizer.state), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
