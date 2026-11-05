"""Paired relative degradation estimator.

Takes a same-scale image pair (x, x'), where x is the content-aligned reference and
x' the degraded observation. A weight-shared CNN encoder extracts two descriptors,
which are concatenated into z_pair = [h; h'; h'-h; |h'-h|] and passed through a
two-layer MLP followed by classification / regression / confidence heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block (Conv-SiLU-Conv + shortcut), with optional channel change and stride."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        return identity + self.conv2(self.act(self.conv1(x)))


class DescriptorEncoder(nn.Module):
    """Weight-shared CNN encoder mapping an image to a descriptor vector.

    stem 3x3 conv(32ch) -> 4 stages {32,64,128,256} (each with 2 residual blocks,
    first block of stages 2-4 uses stride 2) -> per-stage GAP -> concat(480) ->
    Linear(480 -> descriptor_dim) -> descriptor.
    """

    def __init__(self, in_ch: int = 1, stage_channels=(32, 64, 128, 256),
                 descriptor_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        prev = 32
        for i, ch in enumerate(stage_channels):
            blocks = [ResidualBlock(prev, ch, stride=2 if i > 0 else 1),
                      ResidualBlock(ch, ch, stride=1)]
            self.stages.append(nn.Sequential(*blocks))
            prev = ch
        self.neck = nn.Linear(sum(stage_channels), descriptor_dim)
        self.descriptor_dim = descriptor_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        gaps = []
        for stage in self.stages:
            x = stage(x)
            gaps.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        return self.neck(torch.cat(gaps, dim=1))


class DegradationEstimator(nn.Module):
    """Paired relative degradation estimator.

    Forward returns (q_logit, phi, c_logit):
    - q_logit (B,9)  classification logits (sigmoid -> multi-label probabilities)
    - phi      (B,18) normalized continuous parameters (sigmoid -> [0,1])
    - c_logit  (B,1)  confidence logit (sigmoid -> calibrated confidence)
    """

    def __init__(self, n_classes: int = 9, n_params: int = 18, in_ch: int = 1,
                 descriptor_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = DescriptorEncoder(in_ch=in_ch, descriptor_dim=descriptor_dim)
        # two-layer MLP with 512 and 256 hidden units
        self.mlp = nn.Sequential(
            nn.Linear(4 * descriptor_dim, 4 * descriptor_dim),
            nn.SiLU(inplace=True),
            nn.Linear(4 * descriptor_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head_cls = nn.Linear(hidden_dim, n_classes)
        self.head_reg = nn.Linear(hidden_dim, n_params)
        self.head_conf = nn.Linear(hidden_dim, 1)
        self.n_classes = n_classes
        self.n_params = n_params
        self.temperature = 1.0  # confidence temperature, overwritten after calibration

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor):
        h = self.encoder(x)
        hp = self.encoder(x_prime)
        z = torch.cat([h, hp, hp - h, torch.abs(hp - h)], dim=-1)  # pair descriptor
        feat = self.mlp(z)
        q_logit = self.head_cls(feat)
        phi = torch.sigmoid(self.head_reg(feat))
        c_logit = self.head_conf(feat)
        return q_logit, phi, c_logit

    @torch.no_grad()
    def predict(self, x: torch.Tensor, x_prime: torch.Tensor):
        """Return activated (q_hat, phi, c_hat); c_hat is temperature-calibrated."""
        q_logit, phi, c_logit = self.forward(x, x_prime)
        T = getattr(self, 'temperature', 1.0)
        return torch.sigmoid(q_logit), phi, torch.sigmoid(c_logit / T)