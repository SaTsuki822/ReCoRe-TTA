"""Region-affinity recovery under shifted dual partitions.

Splits a feature map under two complementary grids (a regular grid and one shifted by
half a region), tokenizes each region by GAP + L2 normalization, measures 4-connected
affinities, and computes a teacher-anchored tolerance loss plus a gated one-sided
recovery loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class RegionAffinity:
    """Shifted dual-partition region affinities, reliable gating and recovery losses."""

    def __init__(self, region_size: int = 8, tau_A: float = 0.75, T_A: float = 0.10,
                 beta: float = 2.0, delta_A: float = 0.05, eps_A: float = 0.10,
                 eps: float = 1e-6):
        self.R = int(region_size)
        self.half = self.R // 2
        self.tau_A = tau_A
        self.T_A = T_A
        self.beta = beta
        self.delta_A = delta_A
        self.eps_A = eps_A
        self.eps = eps
        self._prepared = False

    # ---- grid geometry ----
    def _grid_n(self, H, W, shift):
        if shift == 0:
            return max(1, math.ceil(H / self.R)), max(1, math.ceil(W / self.R))
        return max(1, math.ceil((H - self.half) / self.R)), max(1, math.ceil((W - self.half) / self.R))

    def _valid_mask(self, H, W, nH, nW, shift):
        valid = torch.ones(nH, nW, dtype=torch.bool)
        for i in range(nH):
            if i * self.R + shift + self.R > H:
                valid[i, :] = False
        for j in range(nW):
            if j * self.R + shift + self.R > W:
                valid[:, j] = False
        return valid

    def _partition_tokens(self, Z, shift, nH, nW):
        H, W = Z.shape[-2], Z.shape[-1]
        if shift == 0:
            ph = nH * self.R - H
            pw = nW * self.R - W
            Zp = F.pad(Z, (0, pw, 0, ph), mode='reflect')
            return F.avg_pool2d(Zp, kernel_size=self.R, stride=self.R)
        ph = nH * self.R + self.half - H
        pw = nW * self.R + self.half - W
        Zp = F.pad(Z, (0, pw, 0, ph), mode='reflect')
        Zs = Zp[:, :, self.half:, self.half:]
        return F.avg_pool2d(Zs, kernel_size=self.R, stride=self.R)

    @staticmethod
    def _normalize_tokens(tokens, eps=1e-6):
        # L2-normalize region tokens along the channel dim
        return tokens / (tokens.norm(p=2, dim=1, keepdim=True) + eps)

    @staticmethod
    def _neighbor_affinity(tokens):
        # tokens: (B,C,nH,nW), L2-normalized; return right/down affinities
        Ar = (tokens[:, :, :, :-1] * tokens[:, :, :, 1:]).sum(dim=1)
        Ar = (1.0 + Ar) / 2.0
        Ad = (tokens[:, :, :-1, :] * tokens[:, :, 1:, :]).sum(dim=1)
        Ad = (1.0 + Ad) / 2.0
        return Ar, Ad

    @staticmethod
    def _boundary_gradient(y, R, half, shift, nH, nW):
        # y: (H,W) single channel in [0,1]; return right/down boundary gradients
        H, W = y.shape
        B_right = torch.zeros(nH, max(1, nW - 1), device=y.device, dtype=y.dtype)
        B_down = torch.zeros(max(1, nH - 1), nW, device=y.device, dtype=y.dtype)
        for i in range(nH):
            r0 = i * R + shift
            r1 = min(r0 + R, H)
            if r1 <= r0:
                continue
            for j in range(nW - 1):
                b = (j + 1) * R + shift
                if b >= W:
                    continue
                B_right[i, j] = (y[r0:r1, b] - y[r0:r1, b - 1]).abs().mean()
        for j in range(nW):
            c0 = j * R + shift
            c1 = min(c0 + R, W)
            if c1 <= c0:
                continue
            for i in range(nH - 1):
                b = (i + 1) * R + shift
                if b >= H:
                    continue
                B_down[i, j] = (y[b, c0:c1] - y[b - 1, c0:c1]).abs().mean()
        return B_right, B_down

    # ---- public interface ----
    def prepare(self, y: torch.Tensor):
        """Precompute the dual-grid geometry, valid masks and boundary gradients.

        y: (1,1,H,W) or (B,1,H,W) in [0,1]. Cached for reuse by affinity/gate.
        """
        H, W = y.shape[-2], y.shape[-1]
        y2 = y.reshape(H, W) if y.ndim == 4 else y.reshape(H, W)
        self.H, self.W = int(H), int(W)
        self.valid_all = []
        self.B_all = []
        for shift in (0, self.half):
            nH, nW = self._grid_n(H, W, shift)
            v = self._valid_mask(H, W, nH, nW, shift)
            v_right = v[:, :-1] & v[:, 1:] if nW > 1 else torch.zeros(nH, 0, dtype=torch.bool)
            v_down = v[:-1, :] & v[1:, :] if nH > 1 else torch.zeros(0, nW, dtype=torch.bool)
            B_right, B_down = self._boundary_gradient(y2, self.R, self.half, shift, nH, nW)
            B_right = B_right[:nH, :max(0, nW - 1)]
            B_down = B_down[:max(0, nH - 1), :nW]
            self.valid_all.append(torch.cat([v_right.reshape(-1), v_down.reshape(-1)], dim=0))
            self.B_all.append(torch.cat([B_right.reshape(-1), B_down.reshape(-1)], dim=0))
        self.valid_all = torch.cat(self.valid_all, dim=0).to(y.device)   # (nEdges,)
        self.B_all = torch.cat(self.B_all, dim=0).to(y.device)           # (nEdges,)
        self.n_edges_total = self.valid_all.numel()
        self._prepared = True
        return self

    def affinity(self, Z: torch.Tensor):
        """Compute 4-connected region affinities of Z under both grids.

        Z: (B,C,H,W). Returns A: (B, n_edges_total). Requires prepare() first.
        """
        B = Z.shape[0]
        As = []
        for shift in (0, self.half):
            nH, nW = self._grid_n(self.H, self.W, shift)
            tokens = self._partition_tokens(Z, shift, nH, nW)
            tokens = self._normalize_tokens(tokens, self.eps)
            Ar, Ad = self._neighbor_affinity(tokens)
            As.append(torch.cat([Ar.reshape(B, -1), Ad.reshape(B, -1)], dim=1))
        return torch.cat(As, dim=1)  # (B, n_edges_total)

    def gate(self, A_T: torch.Tensor, c_hat):
        """Reliable-neighbor gate r (stop-gradient, clipped to [0,1]).

        A_T: (B, nEdges) detached teacher affinity; c_hat: scalar or (B,1) confidence.
        Returns r: (B, nEdges).
        """
        c = float(c_hat) if not isinstance(c_hat, torch.Tensor) else c_hat
        if isinstance(c, torch.Tensor) and c.numel() == 1:
            c = c.reshape(())
        Bv = self.B_all.to(A_T.device)  # (nEdges,)
        r = c * (A_T - self.tau_A) / self.T_A * torch.exp(-self.beta * Bv)
        return r.clamp(0.0, 1.0).detach()  # stop-gradient

    def anchor_loss(self, A_S: torch.Tensor, A_T: torch.Tensor):
        """Tolerance-band anchor loss, averaged over valid edges."""
        valid = self.valid_all.to(A_S.device)
        if not bool(valid.any()):
            return A_S.sum() * 0.0
        diff = (A_S - A_T).abs() - self.eps_A
        diff = diff.clamp(min=0.0)
        rho = torch.sqrt(diff ** 2 + self.eps ** 2) - self.eps
        return rho[:, valid].mean()

    def recovery_loss(self, A_S: torch.Tensor, A_T: torch.Tensor, r: torch.Tensor):
        """One-sided recovery loss, strengthening only reliable, compatible relations."""
        valid = self.valid_all.to(A_S.device)
        if not bool(valid.any()):
            return A_S.sum() * 0.0
        A_S_det = A_S.detach()  # detached student affinity (pre-update)
        M = torch.minimum(
            torch.minimum(torch.ones_like(A_T), A_T + self.eps_A),
            A_S_det + self.delta_A * r)
        term = r * (M - A_S).clamp(min=0.0) ** 2
        Z = r[:, valid].sum() + self.eps
        return term[:, valid].sum() / Z

    def losses(self, A_S: torch.Tensor, A_T: torch.Tensor, c_hat):
        """Return (L_anc, L_up, r) in one call for the adaptation loop."""
        r = self.gate(A_T, c_hat)
        L_anc = self.anchor_loss(A_S, A_T)
        L_up = self.recovery_loss(A_S, A_T, r)
        return L_anc, L_up, r

