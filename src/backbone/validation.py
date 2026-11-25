"""Strict, full-image Result-A / Result-C validation for source pretraining."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ..metrics.psnr_ssim import calculate_psnr, calculate_ssim
from ..utils.image_io import read_img_gray


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: %s" % root)
    return sorted(str(p.resolve()) for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


class ValidationPairs(Dataset):
    """Pair LR_bicubic/X{scale}/*x{scale} with HR/*, with no resizing/crops."""

    def __init__(self, root, scale, crop_border=0):
        self.root = Path(root).resolve()
        self.scale = scale
        lr_paths = list_images(self.root / "LR_bicubic" / ("X%d" % scale))
        hr_paths = list_images(self.root / "HR")
        if not lr_paths or not hr_paths:
            raise ValueError("Empty validation LR/HR directory: %s" % self.root)
        hr_map = {}
        for path in hr_paths:
            stem = Path(path).stem
            if stem in hr_map:
                raise ValueError("Duplicate validation HR stem: %s" % stem)
            hr_map[stem] = path
        used = set()
        self.pairs = []
        for lr_path in lr_paths:
            stem = Path(lr_path).stem
            suffix = "x%d" % scale
            if stem.lower().endswith(suffix):
                stem = stem[:-len(suffix)]
            if stem not in hr_map or stem in used:
                raise ValueError("Unmatched/duplicate validation LR: %s" % lr_path)
            hr_path = hr_map[stem]
            with Image.open(lr_path) as lr, Image.open(hr_path) as hr:
                expected = (lr.width * scale, lr.height * scale)
                if hr.size != expected:
                    raise ValueError("Validation scale mismatch: %s -> %s; expected HR %s, got %s" %
                                     (lr_path, hr_path, expected, hr.size))
                if min(hr.size) - 2 * crop_border < 11:
                    raise ValueError("Validation image too small for SSIM after border crop: %s" % hr_path)
            used.add(stem)
            self.pairs.append((lr_path, hr_path))
        if used != set(hr_map):
            raise ValueError("Validation HR files without matching LR in %s: %s" %
                             (self.root, sorted(set(hr_map) - used)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lr, hr = self.pairs[index]
        return (torch.from_numpy(read_img_gray(lr)).unsqueeze(0),
                torch.from_numpy(read_img_gray(hr)).unsqueeze(0), Path(lr).name)


def build_validation_sets(root, scale, crop_border):
    return {name: ValidationPairs(Path(root) / folder, scale, crop_border)
            for name, folder in (("Result-A", "results-A"), ("Result-C", "results-C"))}


@torch.inference_mode()
def evaluate(model, datasets, device, crop_border):
    """Average image metrics within each set, then give the two sets equal weight.

    Single-channel [0,1], clamped SR; PSNR/11x11 Gaussian SSIM on native full
    images after shaving crop_border HR pixels. No test-time adaptation.
    """
    was_training = model.training
    model.eval()
    results = {}
    try:
        for name, dataset in datasets.items():
            rows = []
            for index in range(len(dataset)):
                lr, hr, filename = dataset[index]
                lr, hr = lr.unsqueeze(0).to(device), hr.unsqueeze(0).to(device)
                if not torch.isfinite(lr).all() or not torch.isfinite(hr).all():
                    raise RuntimeError("Non-finite validation input: %s/%s" % (name, filename))
                sr = model(lr)
                if sr.shape != hr.shape:
                    raise RuntimeError("Validation SR/HR shape mismatch: %s/%s: %s vs %s" %
                                       (name, filename, tuple(sr.shape), tuple(hr.shape)))
                if not torch.isfinite(sr).all():
                    raise RuntimeError("Non-finite validation output: %s/%s" % (name, filename))
                sr, hr = sr.clamp(0, 1)[0, 0].cpu().numpy(), hr[0, 0].cpu().numpy()
                psnr = calculate_psnr(sr, hr, crop_border=crop_border)
                ssim = calculate_ssim(sr, hr, crop_border=crop_border)
                if np.isnan(psnr) or not np.isfinite(ssim):
                    raise RuntimeError("Invalid validation metrics: %s/%s" % (name, filename))
                rows.append({"image": filename, "psnr": psnr, "ssim": ssim})
            results[name] = {"images": len(rows),
                             "psnr": float(np.mean([r["psnr"] for r in rows])),
                             "ssim": float(np.mean([r["ssim"] for r in rows])), "per_image": rows}
            print("[val %s] images=%d PSNR=%.4f SSIM=%.6f" %
                  (name, len(rows), results[name]["psnr"], results[name]["ssim"]), flush=True)
        results["mean"] = {key: float(np.mean([r[key] for r in results.values()]))
                           for key in ("psnr", "ssim")}
    finally:
        model.train(was_training)
    return results
