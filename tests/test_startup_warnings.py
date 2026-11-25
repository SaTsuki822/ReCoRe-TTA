"""Regression coverage for cache deprecation and zero-channel initialization.

Run: python tests/test_startup_warnings.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src import _configure_huggingface_cache
from src.backbone import build_backbone
from src.backbone.vanilla_mamba import CAB, ChannelAttention, ChannelAttentionModified


class StartupWarningTests(unittest.TestCase):
    def tiny(self, channels=1):
        return build_backbone(scale=2, in_chans=channels, embed_dim=32, depths=(1,),
                              d_state=4, img_size=16, drop_path_rate=0).eval()

    def test_cache_migration_preserves_legacy_only_path(self):
        env = {"TRANSFORMERS_CACHE": "/existing/cache", "HF_HOME": "/existing/home"}
        _configure_huggingface_cache(env)
        self.assertEqual(env, {"HF_HUB_CACHE": "/existing/cache", "HF_HOME": "/existing/home"})
        again = env.copy()
        _configure_huggingface_cache(env)
        self.assertEqual(env, again)

    def test_explicit_modern_cache_wins(self):
        env = {"TRANSFORMERS_CACHE": "/old", "HF_HUB_CACHE": "/chosen", "HF_HOME": "/home"}
        _configure_huggingface_cache(env)
        self.assertNotIn("TRANSFORMERS_CACHE", env)
        self.assertEqual(env["HF_HUB_CACHE"], "/chosen")
        self.assertEqual(env["HF_HOME"], "/home")

    def test_fresh_process_import_and_model_have_no_target_warnings(self):
        code = '''
import json, os, warnings
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    from src.backbone import build_backbone
    from src.backbone.vanilla_mamba import _HAS_MAMBA
    model = build_backbone(scale=2, in_chans=1, embed_dim=32, depths=(1,), d_state=4)
messages = [str(w.message) for w in captured]
assert not any("TRANSFORMERS_CACHE" in m or "zero-element tensors" in m for m in messages), messages
assert "TRANSFORMERS_CACHE" not in os.environ
print(json.dumps({"cache": os.environ["HF_HUB_CACHE"], "mamba_available": _HAS_MAMBA}))
'''
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, TRANSFORMERS_CACHE=str(Path(folder) / "old"),
                       HF_HOME=str(Path(folder) / "hf"), HF_HUB_CACHE=str(Path(folder) / "chosen"),
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(json.loads(result.stdout)["cache"], str(Path(folder) / "chosen"))

    def test_small_channel_attention_is_valid_and_differentiable(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for channels in (1, 3, 16, 32):
                model = ChannelAttentionModified(channels)
                out = model(torch.rand(2, channels, 8, 8))
                self.assertEqual(out.shape, (2, channels, 1, 1))
                out.sum().backward()
                self.assertTrue(all(p.numel() > 0 and p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
            for model in (ChannelAttention(1), CAB(1)):
                out = model(torch.rand(2, 1, 8, 8))
                self.assertEqual(out.shape, (2, 1, 8, 8))
        self.assertFalse(any("zero-element tensors" in str(w.message) for w in captured))

    def test_default_backbone_has_no_empty_parameters(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = build_backbone(scale=2, in_chans=1)
        self.assertFalse(any("zero-element tensors" in str(w.message) for w in captured))
        self.assertTrue(all(p.numel() > 0 for p in model.parameters()))

    def test_old_wavelet_checkpoint_shapes_load_strictly_without_changing_output(self):
        for channels in (1, 3):
            original = self.tiny(channels)
            legacy = {k: v.clone() for k, v in original.state_dict().items()}
            prefix = "featureFusionmodule.wavelet_attention.channel_attention."
            legacy[prefix + "fc1.weight"] = torch.empty(0, channels, 1, 1)
            legacy[prefix + "fc2.weight"] = torch.rand(channels, channels, 1, 1)
            restored = self.tiny(channels)
            result = restored.load_state_dict(legacy, strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            x = torch.rand(1, channels, 8, 10)
            with torch.no_grad():
                torch.testing.assert_close(original(x), restored(x), atol=0, rtol=0)

    def test_active_weight_shape_errors_still_fail_strict_loading(self):
        model = self.tiny()
        state = model.state_dict()
        state["conv_first.weight"] = torch.empty(1, 1, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "size mismatch for conv_first.weight"):
            model.load_state_dict(state, strict=True)

    def test_one_cpu_tta_iteration_has_no_initialization_warning(self):
        from src.estimator import DegradationEstimator
        from src.tta import ReCoRe, TTAConfig
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = self.tiny()
            tta = ReCoRe(model, DegradationEstimator(), TTAConfig(iterations=1), device="cpu")
            output = tta.adapt(torch.rand(1, 1, 16, 16))
        self.assertEqual(output.shape, (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(any("zero-element tensors" in str(w.message) or "TRANSFORMERS_CACHE" in str(w.message)
                             for w in captured))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
