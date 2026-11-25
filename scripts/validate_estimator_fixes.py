"""Compare estimator semantics against a local Git revision, on held-out M3FD crops.

python scripts/validate_estimator_fixes.py --baseline_ref <old-commit> \
    --split checkpoints/estimator/data_split.json --output <report.json>
No training weights are modified. The baseline source is loaded from local Git.
"""

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch

import src.degradation.core as core
from src.degradation import operators
from src.estimator.data import DegradationPairDataset
from src.estimator.losses import compute_losses


def old_module(revision, path, package):
    code = subprocess.check_output(["git", "show", revision + ":" + path], text=True)
    module = types.ModuleType("baseline_" + Path(path).stem)
    module.__package__ = package
    exec(compile(code, revision + ":" + path, "exec"), module.__dict__)
    return module


def noise_recovery(loss_fn):
    """Optimize one known readout-noise parameter, keeping the image/labels fixed."""
    x = torch.full((1, 1, 64, 64), .5)
    active = torch.zeros(1, 9)
    active[:, 3] = 1
    norm = torch.full((1, 18), .5)
    norm[:, 7] = .7
    observed = core.apply_retain(x, norm[0], active[0], seed=10)
    estimate = torch.nn.Parameter(torch.tensor(.8))
    optimizer = torch.optim.Adam([estimate], lr=.02)
    q = torch.full((1, 9), -10.)
    q[:, 3] = 10
    for _ in range(100):
        phi = norm.clone()
        phi[:, 7] = estimate.clamp(0, 1)
        _, parts = loss_fn(q, phi, torch.zeros(1, 1), x, observed, active, norm, seed=99)
        optimizer.zero_grad()
        parts["L_syn"].backward()
        optimizer.step()
    return {"ground_truth": .7, "initial": .8, "recovered": estimate.clamp(0, 1).item()}


def run_variant(dataset, ops, loss_fn):
    jpeg_mae, jpeg_pillow_mae, zero_fraction, oracle_error = [], [], [], []
    for i in range(min(16, len(dataset))):
        x = dataset[i][0][None]
        jpeg = ops.apply_jpeg(x, 50)
        pixels = (x[0, 0].numpy() * 255).round().astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        reference = torch.from_numpy(np.asarray(Image.open(buffer)).astype(np.float32) / 255)
        jpeg_mae.append((jpeg - x).abs().mean().item())
        jpeg_pillow_mae.append((jpeg[0, 0] - reference).abs().mean().item())
        zero_fraction.append((jpeg == 0).float().mean().item())
        active, norm = torch.ones(1, 9), torch.full((1, 18), .4)
        observed = core.apply_all(x, norm[0], active[0], seed=10 + i)
        _, parts = loss_fn(torch.full((1, 9), 10.), norm, torch.zeros(1, 1),
                           x, observed, active, norm, seed=99 + i)
        oracle_error.append(parts["L_syn"].item())
    mean = lambda values: sum(values) / len(values)
    return {"samples": len(jpeg_mae), "jpeg_q50_input_mae": mean(jpeg_mae),
            "jpeg_q50_pillow_mae": mean(jpeg_pillow_mae),
            "jpeg_zero_fraction": mean(zero_fraction),
            "oracle_retained_synthesis_loss": mean(oracle_error),
            "noise_strength_recovery": noise_recovery(loss_fn)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ref", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    meta = json.loads(Path(args.split).read_text())
    dataset = DegradationPairDataset(img_paths=meta["sources"]["val"], length=16,
                                     crop_size=128, augment=False, fixed=True,
                                     seed=meta["split_seeds"]["val"])
    old_ops = old_module(args.baseline_ref, "src/degradation/operators.py", "src.degradation")
    old_losses = old_module(args.baseline_ref, "src/estimator/losses.py", "src.estimator")
    old_core = old_module(args.baseline_ref, "src/degradation/core.py", "src.degradation")
    old_core.ops = old_ops
    # The old loss's imported function resolves through its original composition.
    old_losses.apply_retain = old_core.apply_retain
    with patch.object(core, "ops", old_ops), \
         patch.object(core, "apply_retain", old_core.apply_retain), \
         patch.object(core, "apply_all", old_core.apply_all):
        baseline = run_variant(dataset, old_ops, old_losses.compute_losses)
    fixed = run_variant(dataset, operators, compute_losses)
    report = {"baseline_ref": args.baseline_ref, "split": args.split,
              "baseline": baseline, "fixed": fixed}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
