"""Differentiable operators for the 9 infrared degradations.

Every operator maps x in [0,1] (B,C,H,W) to a same-shape tensor (stride-1, no
downsampling/cropping), so an input image and its harder view stay pixel-aligned.

Retained operators (blur/gain/offset/readout/signal/stripe) are differentiable in
their continuous parameters; discarded operators (quantization+JPEG, saturation/AGC,
bad pixels) are only used for synthesizing training pairs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---- internal helpers ----

def _as_tensor(v, dtype, device):
    """Cast a scalar/tensor to the target dtype/device."""
    if isinstance(v, torch.Tensor):
        return v.to(dtype=dtype, device=device)
    return torch.tensor(v, dtype=dtype, device=device)


def _gaussian_kernel(sigma_x, sigma_y, theta, ks):
    """Build an anisotropic rotated Gaussian kernel, differentiable in sigma_x/sigma_y/theta."""
    ax = torch.linspace(-(ks - 1) / 2.0, (ks - 1) / 2.0, ks,
                        dtype=sigma_x.dtype, device=sigma_x.device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    xr = cos_t * xx + sin_t * yy
    yr = -sin_t * xx + cos_t * yy
    kernel = torch.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return kernel / (kernel.sum() + 1e-12)


def _smooth_field_2d(h, w, corr_len, seed, dtype, device):
    """Zero-mean smooth 2D random field normalized to [-1,1]; corr_len controls roughness."""
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    gh = max(1, int(math.ceil(h / max(float(corr_len), 1.0))))
    gw = max(1, int(math.ceil(w / max(float(corr_len), 1.0))))
    z = torch.randn(1, 1, gh, gw, generator=gen, dtype=dtype, device=device)
    field = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
    field = field - field.mean()
    return field / (field.abs().max() + 1e-8)


def _smooth_field_1d(n, corr_len, seed, dtype, device):
    """Zero-mean smooth 1D random field normalized to [-1,1]."""
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    gn = max(1, int(math.ceil(n / max(float(corr_len), 1.0))))
    z = torch.randn(1, 1, gn, generator=gen, dtype=dtype, device=device)
    field = F.interpolate(z, size=(n,), mode="linear", align_corners=False)
    field = field - field.mean()
    return field / (field.abs().max() + 1e-8)


# ---- degradation operators ----

def apply_blur(x, sigma_x, sigma_y, theta, kernel_size=None):
    """Optical/defocus blur: stride-1 anisotropic Gaussian convolution with reflection padding."""
    sx = _as_tensor(sigma_x, x.dtype, x.device)
    sy = _as_tensor(sigma_y, x.dtype, x.device)
    th = _as_tensor(theta, x.dtype, x.device)
    if kernel_size is None:
        kernel_size = 2 * int(math.ceil(3.0 * max(float(sx), float(sy)))) + 1
        kernel_size = max(kernel_size, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
    kernel = _gaussian_kernel(sx, sy, th, kernel_size)  # (ks, ks)
    C = x.shape[1]
    k = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
    pad = kernel_size // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(xp, k, groups=C)


def apply_gain_nonuniformity(x, a_g, l_g, seed=None):
    """Gain non-uniformity: smooth multiplicative field (gain = 1 + a_g * field)."""
    a = _as_tensor(a_g, x.dtype, x.device)
    field = _smooth_field_2d(x.shape[-2], x.shape[-1], l_g, seed, x.dtype, x.device)
    return x * (1.0 + a * field)


def apply_offset_nonuniformity(x, a_o, l_o, seed=None):
    """Offset non-uniformity: smooth additive field."""
    a = _as_tensor(a_o, x.dtype, x.device)
    field = _smooth_field_2d(x.shape[-2], x.shape[-1], l_o, seed, x.dtype, x.device)
    return x + a * field


def apply_readout_noise(x, sigma_r, seed=None):
    """Readout noise: zero-mean, signal-independent Gaussian (reparameterized)."""
    s = _as_tensor(sigma_r, x.dtype, x.device)
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    noise = torch.randn(x.size(), generator=gen, dtype=x.dtype, device=x.device) * s
    return x + noise


def apply_signal_dependent_noise(x, sigma_s, seed=None):
    """Signal-dependent noise: variance proportional to local intensity."""
    s = _as_tensor(sigma_s, x.dtype, x.device)
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    std = s * torch.sqrt(x.clamp(min=0.0))
    return x + torch.randn(x.size(), generator=gen, dtype=x.dtype, device=x.device) * std


def apply_stripe(x, a_s, w_s, d_s, seed=None):
    """Stripe interference: additive response variation along rows (d_s=0) or columns (d_s=1)."""
    a = _as_tensor(a_s, x.dtype, x.device)
    d = int(round(float(d_s)))
    if d == 0:
        field = _smooth_field_1d(x.shape[-2], w_s, seed, x.dtype, x.device)
        field = field.view(1, 1, -1, 1)
    else:
        field = _smooth_field_1d(x.shape[-1], w_s, seed, x.dtype, x.device)
        field = field.view(1, 1, 1, -1)
    return x + a * field


def apply_quantization(x, b):
    """Quantization to b bits (b in {6,7,8}); non-differentiable, synthesis only."""
    b = int(round(float(b)))
    levels = float(2 ** b - 1)
    return torch.round(x * levels) / levels


def _dct_basis(n=8):
    """n x n orthonormal DCT-II basis matrix."""
    i = torch.arange(n, dtype=torch.float32).view(n, 1)
    j = torch.arange(n, dtype=torch.float32).view(1, n)
    basis = torch.cos(math.pi * (2 * i + 1) * j / (2 * n))
    basis[0, :] /= math.sqrt(2.0)
    return basis * math.sqrt(2.0 / n)


_JPEG_QTABLE = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)


def apply_jpeg(x, quality):
    """JPEG compression: 8x8 block DCT + quality-based quantization + IDCT."""
    B, C, H, W = x.shape
    n = 8
    ph, pw = (n - H % n) % n, (n - W % n) % n
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
    Hp, Wp = xp.shape[-2], xp.shape[-1]
    basis = _dct_basis(n).to(x.device)  # (8,8)
    q = float(quality)
    s = 5000.0 / q if q < 50.0 else 200.0 - 2.0 * q
    qstep = torch.clamp((_JPEG_QTABLE.to(x.device) * s + 50.0) / 100.0, min=1.0)
    xflat = xp.reshape(B * C, 1, Hp, Wp)
    blocks = xflat.reshape(B * C, Hp // n, n, Wp // n, n).permute(0, 1, 3, 2, 4).reshape(-1, n, n)
    dct = torch.einsum("ij,bjk,kl->bil", basis, blocks, basis.t())
    qdct = torch.round(dct / qstep) * qstep
    idct = torch.einsum("ij,bjk,kl->bil", basis.t(), qdct, basis)
    out = idct.reshape(B * C, Hp // n, Wp // n, n, n).permute(0, 1, 3, 2, 4).reshape(B * C, 1, Hp, Wp)
    out = out.reshape(B, C, Hp, Wp)
    return out[:, :, :H, :W]


def apply_saturation_agc(x, p_clip, gamma):
    """Saturation/AGC: clip to [p_clip, 1-p_clip] then gamma correction."""
    lo = _as_tensor(p_clip, x.dtype, x.device)
    g = _as_tensor(gamma, x.dtype, x.device)
    hi = 1.0 - lo
    xc = x.clamp(lo, hi)
    xc = (xc - lo) / (hi - lo + 1e-8)
    return (xc ** g).clamp(0.0, 1.0)


def apply_bad_pixels(x, p_bad, v_bad, seed=None):
    """Fixed bad pixels: replace random positions with v_bad in {0,1}."""
    p = _as_tensor(p_bad, x.dtype, x.device)
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    mask = (torch.rand(x.shape, generator=gen, device=x.device) < p).to(x.dtype)
    v = float(int(round(float(v_bad))))
    return x * (1.0 - mask) + v * mask


# ---- dispatch ----

def _apply_quantization_jpeg(x, p, seed):
    return apply_jpeg(apply_quantization(x, p[..., 0]), p[..., 1])


_APPLY = {
    0: lambda x, p, seed: apply_blur(x, p[..., 0], p[..., 1], p[..., 2]),
    1: lambda x, p, seed: apply_gain_nonuniformity(x, p[..., 0], p[..., 1], seed),
    2: lambda x, p, seed: apply_offset_nonuniformity(x, p[..., 0], p[..., 1], seed),
    3: lambda x, p, seed: apply_readout_noise(x, p[..., 0], seed),
    4: lambda x, p, seed: apply_signal_dependent_noise(x, p[..., 0], seed),
    5: lambda x, p, seed: apply_stripe(x, p[..., 0], p[..., 1], p[..., 2], seed),
    6: _apply_quantization_jpeg,
    7: lambda x, p, seed: apply_saturation_agc(x, p[..., 0], p[..., 1]),
    8: lambda x, p, seed: apply_bad_pixels(x, p[..., 0], p[..., 1], seed),
}


def apply_single(index: int, x: torch.Tensor, params: torch.Tensor, seed: Optional[int] = None):
    """Apply a single degradation operator; params is its raw parameter segment."""
    return _APPLY[index](x, params, seed)
