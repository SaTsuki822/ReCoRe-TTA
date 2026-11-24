"""Backbone dataset/batching regression tests. Run: python tests/test_backbone_data.py"""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.backbone import build_backbone
from src.backbone.train import PairedSRDataset
from src.backbone.vanilla_mamba import SS2D, _selective_scan_ref


class BackboneDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "ir.png"
        image = (np.arange(128 * 144).reshape(128, 144) % 256).astype(np.uint8)
        imageio.imwrite(path, image)
        self.paths = [str(path)]

    def test_sample_and_loader_shapes(self):
        for scale in (2, 4):
            with self.subTest(scale=scale):
                dataset = PairedSRDataset(self.paths, scale=scale, gt_size=128, length=3)
                lr, hr = dataset[0]
                self.assertEqual(lr.shape, (1, 128 // scale, 128 // scale))
                self.assertEqual(hr.shape, (1, 128, 128))
                # Include the final batch of one: neither C nor B may be squeezed away.
                for (lr, hr), batch_size in zip(DataLoader(dataset, batch_size=2), (2, 1)):
                    self.assertEqual(lr.shape, (batch_size, 1, 128 // scale, 128 // scale))
                    self.assertEqual(hr.shape, (batch_size, 1, 128, 128))

    def test_dataset_to_backbone_loss_and_backward(self):
        for scale in (2, 4):
            with self.subTest(scale=scale):
                torch.manual_seed(3407)
                dataset = PairedSRDataset(self.paths, scale=scale, gt_size=16 * scale, length=2)
                lr, hr = next(iter(DataLoader(dataset, batch_size=2)))
                model = build_backbone(scale=scale, in_chans=1, embed_dim=32,
                                       depths=(1,), d_state=4, img_size=16,
                                       drop_path_rate=0.)
                # Explicit CPU reference backend; CUDA extensions may be installed.
                for module in model.modules():
                    if isinstance(module, SS2D):
                        module.selective_scan = _selective_scan_ref
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                sr = model(lr)
                self.assertEqual(sr.shape, hr.shape)
                loss = F.l1_loss(sr, hr)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.conv_first.weight.grad)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                optimizer.step()
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
