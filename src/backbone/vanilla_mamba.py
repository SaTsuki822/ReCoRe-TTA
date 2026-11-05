"""Mamba-based infrared super-resolution backbone.

FeatureFusionModule -> conv_first -> deep feature extraction (ResidualGroup x N, each
containing VSSBlocks) -> conv_after_body -> PixelShuffle -> conv_last. Only the last
linear layer of each VSSM (SS2D.out_proj) is finetuned during test-time adaptation.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# optional mamba-ssm dependency and pure-PyTorch selective-scan fallback
# ----------------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    _HAS_MAMBA = True
except Exception:  # pragma: no cover - environment-dependent
    _HAS_MAMBA = False
    selective_scan_fn = None


def _selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                        delta_softplus=False, return_last_state=False):
    """Pure-PyTorch reference S6 selective scan (slow but correct, used without mamba-ssm).

    Follows the mamba_ssm.selective_scan_fn convention:
        u:      (B, D, L)
        delta:  (B, D, L)
        A:      (D, N)
        B, C:   (B, N, L) or (B, G, N, L) with G groups over the D channels
        D:      (D,)
        z:      (B, D, L) or None
    """
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    dtype = u.dtype
    uf = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    A = A.float()
    deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, dim, 1, dstate))  # (B, D, L, N)

    Bf, Cf = B.float(), C.float()
    if Bf.dim() == 4:  # grouped (B, G, N, L) -> one group per channel chunk
        groups = Bf.shape[1]
        Bf = Bf.repeat_interleave(dim // groups, dim=1)                 # (B, D, N, L)
        Cf = Cf.repeat_interleave(dim // groups, dim=1)
        Bt = Bf.transpose(2, 3)                                         # (B, D, L, N)
        Ct = Cf.transpose(2, 3)
    else:              # ungrouped (B, N, L)
        Bt = Bf.transpose(1, 2).unsqueeze(1)                            # (B, 1, L, N)
        Ct = Cf.transpose(1, 2).unsqueeze(1)

    deltaB_u = delta.unsqueeze(-1) * uf.unsqueeze(-1) * Bt              # (B, D, L, N)
    x = torch.zeros(batch, dim, dstate, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(seqlen):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]                    # (B, D, N)
        y = (x * Ct[:, :, i]).sum(-1)                                  # (B, D)
        ys.append(y)
    y = torch.stack(ys, dim=2)                                         # (B, D, L)
    if D is not None:
        y = y + uf * D.float().view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z.float())
    y = y.to(dtype)
    return (y, x) if return_last_state else y


def _selective_scan_dispatch():
    return selective_scan_fn if selective_scan_fn is not None else _selective_scan_ref


# ----------------------------------------------------------------------------------
# minimal local replacements for timm
# ----------------------------------------------------------------------------------
def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""

    def _norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = _norm_cdf((a - mean) / std)
        u = _norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=mean - 2 * std, max=mean + 2 * std)
    return tensor


class _DropPath(nn.Module):
    """DropPath (stochastic depth)."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ----------------------------------------------------------------------------------
# channel attention / CAB (used by VSSBlock)
# ----------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        if is_light_sr:
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        return self.cab(x)


# ----------------------------------------------------------------------------------
# SS2D / VSSBlock / BasicLayer
# ----------------------------------------------------------------------------------
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2., dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2,
                                **factory_kwargs)
        self.act = nn.SiLU()

        x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))

        dt_projs = (
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
                          dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
                          dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
                          dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
                          dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = _selective_scan_dispatch()

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0)
        A = A.expand(d_inner, d_state).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).expand(copies, d_inner, d_state).contiguous()
            if merge:
                A_log = A_log.reshape(copies * d_inner, d_state)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).expand(copies, d_inner).contiguous()
            if merge:
                D = D.reshape(copies * d_inner)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L),
             torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, C, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias,
            delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable = nn.LayerNorm,
                 attn_drop_rate: float = 0, d_state: int = 16, expand: float = 2.,
                 is_light_sr: bool = False, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand,
                                   dropout=attn_drop_rate, **kwargs)
        self.drop_path = _DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, drop_path=0., d_state=16, mlp_ratio=2.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, is_light_sr=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution, is_light_sr=is_light_sr))

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ----------------------------------------------------------------------------------
# wavelet feature fusion module (VanillaMamba-specific)
# ----------------------------------------------------------------------------------
def _dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))
    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        return _dwt_init(x)


class ChannelAttentionModified(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)


class WaveletAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.dwt = DWT()
        self.channel_attention = ChannelAttentionModified(in_channels, reduction_ratio)

    def forward(self, x):
        cA, cH, cV, cD = self.dwt(x)
        cA = F.interpolate(cA, scale_factor=2, mode='bicubic', align_corners=False)
        cH = F.interpolate(cH, scale_factor=2, mode='bicubic', align_corners=False)
        cV = F.interpolate(cV, scale_factor=2, mode='bicubic', align_corners=False)
        cD = F.interpolate(cD, scale_factor=2, mode='bicubic', align_corners=False)
        return cA, cH, cV, cD


class FeatureMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.silu(self.conv1(x))
        return self.conv2(x)


class FeatureModulation(nn.Module):
    def __init__(self, in_channels, wavelet_channels, scale_factor=1):
        super().__init__()
        self.mapping = FeatureMapping(in_channels * 4, wavelet_channels * scale_factor)
        self.scale_factor = scale_factor

    def forward(self, large_feature_map, detail_feature_map):
        modulation_params = self.mapping(detail_feature_map)
        if self.scale_factor > 1:
            modulation_params = F.interpolate(modulation_params, scale_factor=self.scale_factor,
                                              mode='bilinear')
        desired_size = (large_feature_map.size(2), large_feature_map.size(3))
        modulation_params = F.interpolate(modulation_params, size=desired_size,
                                          mode='bilinear', align_corners=False)
        return large_feature_map * modulation_params


class SmallScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv3x3(x)


class LargeScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwconv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.dwconv7x7(x))


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, wavelet_channels):
        super().__init__()
        mid_channels = in_channels
        wavelet_channels = in_channels
        self.small_scale_extractor = SmallScaleFeatureExtractor(in_channels, mid_channels)
        self.large_scale_extractor = LargeScaleFeatureExtractor(in_channels, in_channels)
        self.wavelet_attention = WaveletAttention(mid_channels)
        self.feature_modulation = FeatureModulation(mid_channels, wavelet_channels, scale_factor=1)
        self.fusion_conv = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        small_scale_features = self.small_scale_extractor(x)
        large_scale_features = self.large_scale_extractor(x)
        cA, attn_cH, attn_cV, attn_cD = self.wavelet_attention(small_scale_features)
        recombined_features = torch.cat((cA, attn_cH, attn_cV, attn_cD), dim=1)
        modulated_large_scale_features = self.feature_modulation(large_scale_features, recombined_features)
        combined_features = torch.cat([modulated_large_scale_features, small_scale_features], dim=1)
        return self.fusion_conv(combined_features)


# ----------------------------------------------------------------------------------
# backbone network
# ----------------------------------------------------------------------------------
class VanillaMamba(nn.Module):
    """Mamba-based infrared super-resolution backbone (source model f_theta0).

    This is the plain backbone without any test-time optimization; ReCoRe adapts it
    by finetuning only its SS2D.out_proj linear layers.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=1, embed_dim=180,
                 depths=(8, 8, 8, 8, 8, 8), drop_rate=0., d_state=16, mlp_ratio=2.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='pixelshuffle',
                 resi_connection='1conv', **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            self.mean = torch.Tensor((0.3317, 0.3317, 0.3317)).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 1. shallow feature extraction
        self.featureFusionmodule = FeatureFusionModule(num_in_ch, num_out_ch, upscale, num_in_ch)
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # 2. deep feature extraction
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=embed_dim, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.patches_resolution = self.patch_embed.patches_resolution

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=embed_dim, embed_dim=embed_dim,
                                          norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = (upsampler == 'pixelshuffledirect')
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                depth=depths[i_layer], d_state=d_state, mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size,
                resi_connection=resi_connection, is_light_sr=self.is_light_sr)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # 3. high-quality reconstruction
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def extract_features(self, x: torch.Tensor):
        """Return the deep feature map (B, embed_dim, H, W) before reconstruction."""
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == 'pixelshuffle':
            x = self.featureFusionmodule(x)
        x = self.conv_first(x)
        x = self.forward_features(x)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.featureFusionmodule(x)
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x


class ResidualGroup(nn.Module):
    def __init__(self, dim, input_resolution, depth, d_state=16, mlp_ratio=4., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=None, patch_size=None, resi_connection='1conv', is_light_sr=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth, d_state=d_state,
            mlp_ratio=mlp_ratio, drop_path=drop_path, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint, is_light_sr=is_light_sr)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = [nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1), nn.PixelShuffle(scale)]
        super().__init__(*m)


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super().__init__(*m)


# ----------------------------------------------------------------------------------
# TTA helpers: tunable linear layers / weight loading
# ----------------------------------------------------------------------------------
def configure_model(model: nn.Module, tunable_keywords=("out_proj",)):
    """Freeze the whole model, unfreezing only parameters whose name matches a keyword."""
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in tunable_keywords)
    return model


def collect_params(model: nn.Module):
    """Collect trainable weight/bias parameters and their names."""
    params, names = [], []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if p.requires_grad and np in ('weight', 'bias'):
                params.append(p)
                names.append(f"{nm}.{np}" if nm else np)
    return params, names


def _reduce_channel(v: torch.Tensor, target_shape):
    """Average over a single mismatched channel dim (e.g. 3 -> 1) to match target_shape."""
    if v.ndim != len(target_shape):
        return None
    diff_dims = [d for d in range(v.ndim) if v.shape[d] != target_shape[d]]
    if len(diff_dims) == 0:
        return v
    if len(diff_dims) == 1 and target_shape[diff_dims[0]] == 1:
        return v.mean(dim=diff_dims[0], keepdim=True)
    return None


def load_backbone_weights(model: nn.Module, ckpt_path: str, strict: bool = True):
    """Load backbone weights, averaging channels when migrating 3-channel -> single-channel."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'params' in ckpt:
            ckpt = ckpt['params']
        elif 'params_ema' in ckpt:
            ckpt = ckpt['params_ema']
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unrecognized checkpoint format: {type(ckpt)}")

    model_sd = model.state_dict()
    ckpt = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ckpt.items()}

    if strict:
        try:
            model.load_state_dict(ckpt, strict=True)
            return model, [], []
        except RuntimeError:
            pass

    aligned, skipped = {}, []
    for k, v in ckpt.items():
        if k not in model_sd:
            skipped.append(k)
            continue
        target = model_sd[k]
        if v.shape == target.shape:
            aligned[k] = v
        else:
            v2 = _reduce_channel(v, target.shape)
            if v2 is not None and tuple(v2.shape) == tuple(target.shape):
                aligned[k] = v2
            else:
                skipped.append(k)

    result = model.load_state_dict(aligned, strict=False)
    return model, result.missing_keys + skipped, result.unexpected_keys


def build_backbone(scale: int = 2, in_chans: int = 1, embed_dim: int = 180,
                   depths=(8, 8, 8, 8, 8, 8), d_state: int = 16, mlp_ratio: float = 2.,
                   img_size: int = 64, upsampler: str = 'pixelshuffle', **kwargs) -> VanillaMamba:
    """Build a VanillaMamba backbone with the default hyper-parameters."""
    return VanillaMamba(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim,
                     depths=depths, d_state=d_state, mlp_ratio=mlp_ratio,
                     upscale=scale, upsampler=upsampler, **kwargs)

