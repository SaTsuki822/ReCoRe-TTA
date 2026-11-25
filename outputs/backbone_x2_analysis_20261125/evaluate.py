"""Small x2 checkpoint comparison on external HR crops, with fixed synthesized LR.

Existing local LR folders are x4, so they are deliberately not used here.
Run from the repository: CUDA_VISIBLE_DEVICES=1 python outputs/backbone_x2_analysis_20261125/evaluate.py
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F

from src.backbone.vanilla_mamba import build_backbone
from src.degradation import full_degradation
from src.estimator.data import _to_single_gray


def metrics(prediction, target):
    a, b = prediction[2:-2, 2:-2], target[2:-2, 2:-2]
    error = a.astype(np.float64) - b.astype(np.float64)
    return {"psnr": float(-10 * np.log10(max(float(np.mean(error ** 2)), 1e-12))),
            "ssim": float(structural_similarity(a, b, data_range=1., win_size=11,
                           gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
            "mae": float(np.abs(error).mean())}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    output = Path(__file__).resolve().parent
    sources = []
    for name in ("FLIR-IISR", "Result-A"):
        files = sorted((Path('/mnt/workspace/lyk/dataset') / name / 'testset/HR').glob('*.png'))
        selected = np.linspace(0, len(files) - 1, min(8, len(files)), dtype=int)
        sources.extend((name, files[i]) for i in selected)
    cases = []
    for i, (dataset, path) in enumerate(sources):
        image = _to_single_gray(imageio.imread(path))
        h, w = image.shape
        top, left = (h - 128) // 2, (w - 128) // 2
        hr = torch.from_numpy(image[top:top+128, left:left+128].copy())[None, None]
        mixed, active, raw, norm = full_degradation(hr, scale=2,
                                 rng=torch.Generator().manual_seed(80000 + i))
        views = {'average_x2': F.avg_pool2d(hr, 2),
                 'bicubic_x2': F.interpolate(hr, scale_factor=.5, mode='bicubic',
                                            align_corners=False, antialias=True).clamp(0, 1),
                 'mixed_v2_x2': mixed}
        for condition, lr in views.items():
            cases.append({'dataset': dataset, 'path': str(path), 'crop': [top, left, 128, 128],
                          'condition': condition, 'hr': hr, 'lr': lr,
                          'active': active.tolist() if condition == 'mixed_v2_x2' else None})
    results = []
    for case in cases:
        up = F.interpolate(case['lr'], scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        results.append({k: case[k] for k in ('dataset', 'path', 'crop', 'condition', 'active')} |
                       {'model': 'bicubic', **metrics(up[0, 0].numpy(), case['hr'][0, 0].numpy())})
    model = build_backbone(scale=2, in_chans=1).cuda().eval()
    checks = {}
    with torch.inference_mode():
        for tag in ('epoch9', 'epoch49', 'final'):
            filename = 'backbone_x2.pt' if tag == 'final' else 'backbone_x2_' + tag + '.pt'
            state = torch.load(ROOT / 'checkpoints/backbone_x2' / filename,
                               map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            checks[tag] = {'finite_weights': all(torch.isfinite(v).all().item() for v in state.values()),
                           'strict_load': True}
            start = time.monotonic()
            for index, case in enumerate(cases):
                sr = model(case['lr'].cuda())
                if not torch.isfinite(sr).all():
                    raise RuntimeError('Non-finite output: ' + tag + str(index))
                array = sr.clamp(0, 1)[0, 0].cpu().numpy()
                results.append({k: case[k] for k in ('dataset', 'path', 'crop', 'condition', 'active')} |
                               {'model': tag, **metrics(array, case['hr'][0, 0].numpy())})
                if tag == 'final' and index < 3:
                    folder = output / 'example'
                    folder.mkdir(exist_ok=True)
                    bicubic = F.interpolate(case['lr'], scale_factor=2, mode='bicubic',
                                            align_corners=False).clamp(0, 1)[0, 0].numpy()
                    for name, im in [('hr', case['hr'][0, 0].numpy()), ('sr', array), ('bicubic', bicubic)]:
                        Image.fromarray((im * 255).round().astype(np.uint8)).save(
                            folder / (case['condition'] + '_' + name + '.png'))
            print(tag, 'finished', len(cases), 'forwards; seconds', round(time.monotonic()-start, 2), flush=True)
    summary = []
    for condition in ('average_x2', 'bicubic_x2', 'mixed_v2_x2'):
        for tag in ('bicubic', 'epoch9', 'epoch49', 'final'):
            rows = [r for r in results if r['condition'] == condition and r['model'] == tag]
            summary.append({'condition': condition, 'model': tag, 'n': len(rows),
                            **{key: float(np.mean([r[key] for r in rows])) for key in ('psnr', 'ssim', 'mae')}})
    report = {'protocol': '16 external HR center crops (8 FLIR + 8 Result-A), HR128/LR64, x2; '
                         'three deterministic conditions; crop_border=2; outputs clamped to [0,1]. '
                         'This is a diagnostic crop comparison, not a full benchmark or a training validation split.',
              'checkpoint_checks': checks, 'summary': summary, 'per_case': results}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
