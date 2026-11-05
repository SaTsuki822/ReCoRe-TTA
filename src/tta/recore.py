"""ReCoRe test-time adaptation loop.

For each test image: cache the teacher prediction and region affinities, then for T
steps build a dynamic teacher-student reference, estimate the relative degradation,
construct a harder view, and update the student's linear layers with
L_total = L_redeg + lambda_up * L_up + lambda_anc * L_anc.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import optim

from ..backbone import configure_model, collect_params
from ..affinity import RegionAffinity
from ..degradation import safe_project, apply_retain, RETAIN_INDICES


@dataclass
class TTAConfig:
    """Hyper-parameters for ReCoRe test-time adaptation."""
    iterations: int = 10                 # updates per test image
    lr: float = 1e-4                     # learning rate
    betas: Tuple[float, float] = (0.9, 0.999)
    omega_1: float = 0.0                 # initial teacher weight
    kappa: float = 0.1                   # dynamic-weight update rate
    eps_omega: float = 1e-8              # division guard
    lambda_up: float = 1.0
    lambda_anc: float = 0.1
    eps_f: float = 1e-6                  # stability term
    region_size: int = 8
    tau_A: float = 0.75                  # teacher affinity threshold
    T_A: float = 0.10                    # gate temperature
    beta: float = 2.0                    # boundary suppression
    delta_A: float = 0.05                # max one-step affinity increase
    eps_A: float = 0.10                  # teacher anchor tolerance
    tunable_keywords: Tuple[str, ...] = ("out_proj",)
    student_view: str = 'harder'         # student affinities on the harder view
    seed: int = 3407
    clamp_ref: bool = True               # clamp the reference to [0,1]


class ReCoRe:
    """ReCoRe test-time adaptation wrapper."""

    def __init__(self, backbone, estimator, config: Optional[TTAConfig] = None,
                 device: str = 'cuda'):
        self.config = config or TTAConfig()
        self.device = device

        # student (tunable linear layers) and frozen teacher (both copied from the source model)
        self.student = backbone
        self.teacher = copy.deepcopy(backbone)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # freeze the estimator
        self.estimator = estimator
        for p in self.estimator.parameters():
            p.requires_grad = False
        self.estimator.eval()

        # finetune only the student's linear layers; eval mode disables DropPath/Dropout
        configure_model(self.student, self.config.tunable_keywords)
        self.student.eval()
        params, self.param_names = collect_params(self.student)
        if len(params) == 0:
            raise ValueError(
                f"No tunable parameters matched keywords {self.config.tunable_keywords}; "
                "check backbone linear-layer names (default: SS2D.out_proj).")
        self.optimizer = optim.Adam(params, lr=self.config.lr, betas=self.config.betas)

        self.affinity = RegionAffinity(
            region_size=self.config.region_size, tau_A=self.config.tau_A,
            T_A=self.config.T_A, beta=self.config.beta, delta_A=self.config.delta_A,
            eps_A=self.config.eps_A)

    # ---- helpers ----
    def _downsample(self, sr: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample a scale-x SR image back to the observation scale (average pooling)."""
        if scale == 1:
            return sr
        return F.avg_pool2d(sr, kernel_size=scale, stride=scale)

    def _harder_view(self, y: torch.Tensor, phi_hat: torch.Tensor, q_hat: torch.Tensor,
                     seed: Optional[int] = None) -> torch.Tensor:
        """Build the same-scale harder view by applying retained operators after safe projection."""
        phi_safe = safe_project(phi_hat)          # clip and drop discarded components
        active = (q_hat[0] > 0.5).float()
        retain_idx = torch.tensor(RETAIN_INDICES, device=active.device)
        if active[retain_idx].sum() == 0:
            # fallback: activate the most probable retained class
            top = q_hat[0][retain_idx].argmax()
            active[retain_idx[top]] = 1.0
        return apply_retain(y, phi_safe, active=active, seed=seed)

    # ---- main loop ----
    @torch.no_grad()
    def _cache_teacher(self, y: torch.Tensor):
        """Cache the teacher prediction and region affinities (once per image)."""
        sr_teacher = self.teacher(y).detach()
        Z_T = self.teacher.extract_features(y).detach()
        self.affinity.prepare(y)
        A_T = self.affinity.affinity(Z_T).detach()
        return sr_teacher, A_T

    def adapt(self, y: torch.Tensor) -> torch.Tensor:
        """Adapt to a single test image y (1,1,H,W) in [0,1]; return x_hat (1,1,sH,sW)."""
        cfg = self.config
        scale = self.student.upscale

        sr_teacher, A_T = self._cache_teacher(y)
        omega = cfg.omega_1
        L_prev = None

        for t in range(cfg.iterations):
            # 1) dynamic reference D_t (stop-gradient)
            with torch.no_grad():
                sr_student = self.student(y)  # f_theta{t-1}(y)
            fused = omega * sr_teacher + (1.0 - omega) * sr_student.detach()
            D_t = self._downsample(fused, scale)
            if cfg.clamp_ref:
                D_t = D_t.clamp(0.0, 1.0)

            # 2) estimate degradation + safe projection + harder view
            with torch.no_grad():
                q_hat, phi_hat, c_hat = self.estimator.predict(D_t, y)
                y_tilde = self._harder_view(y, phi_hat, q_hat, seed=cfg.seed + t)

            # 3) student features (with grad)
            Z_y = self.student.extract_features(y)
            Z_ytilde = self.student.extract_features(y_tilde)

            # 4) re-degradation consistency loss
            L_redeg = ((Z_y - Z_ytilde) ** 2).mean() + cfg.eps_f

            # 5) affinity recovery (student affinity on the harder view)
            if cfg.student_view == 'harder':
                A_S = self.affinity.affinity(Z_ytilde)
            else:
                A_S = self.affinity.affinity(Z_y)
            L_anc, L_up, r = self.affinity.losses(A_S, A_T, c_hat)

            # 6) total loss + Adam update
            L_total = L_redeg + cfg.lambda_up * L_up + cfg.lambda_anc * L_anc
            self.optimizer.zero_grad()
            L_total.backward()
            self.optimizer.step()

            # 7) dynamic weight update
            if L_prev is not None:
                delta = (L_total.item() - L_prev) / (L_prev + cfg.eps_omega)
                omega = float(torch.clamp(torch.tensor(omega + cfg.kappa * delta), 0.0, 1.0))
            L_prev = L_total.item()

        with torch.no_grad():
            x_hat = self.student(y)
        return x_hat

    def reconstruct(self, y: torch.Tensor) -> torch.Tensor:
        """Reconstruct without adaptation (source-model output)."""
        with torch.no_grad():
            return self.student(y)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return self.adapt(y)


def load_estimator(path: str, device: str = 'cpu'):
    """Load the frozen estimator checkpoint (with temperature calibration)."""
    from ..estimator import DegradationEstimator
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        temperature = float(ckpt.get('temperature', 1.0))
    else:
        state, temperature = ckpt, 1.0
    model = DegradationEstimator(n_classes=9, n_params=18, in_ch=1)
    model.load_state_dict(state, strict=True)
    model.temperature = temperature
    model.to(device).eval()
    return model

