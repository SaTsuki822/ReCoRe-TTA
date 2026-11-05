"""No-reference NIQE metric (self-contained).

Uses niqe_pris_params.npz in the same directory for the pristine MVG model.
Input is single channel in [0,1]; internally converted to [0,255].
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

# vectorized math.gamma and cv2.filter2D (avoid scipy compiled extensions)
_gamma = np.vectorize(math.gamma, otypes=[np.float64])
# gam / r_gam are constants, computed once at import
_GAM = np.arange(0.2, 10.001, 0.001)
_GAM_RECIP = np.reciprocal(_GAM)
_R_GAM = np.square(_gamma(_GAM_RECIP * 2)) / (_gamma(_GAM_RECIP) * _gamma(_GAM_RECIP * 3))


def _conv2d_nearest(img, kernel):
    return cv2.filter2D(img.astype(np.float32), -1, kernel.astype(np.float32),
                        borderType=cv2.BORDER_REPLICATE)


_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "niqe_pris_params.npz")


def _estimate_aggd_param(block):
    block = block.flatten()
    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block ** 2)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
    array_position = np.argmin((_R_GAM - rhatnorm) ** 2)

    alpha = _GAM[array_position]
    beta_l = left_std * np.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    beta_r = right_std * np.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    return alpha, beta_l, beta_r


def _compute_feature(block):
    feat = []
    alpha, beta_l, beta_r = _estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for sh in shifts:
        shifted_block = np.roll(block, sh, axis=(0, 1))
        alpha, beta_l, beta_r = _estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (_gamma(2 / alpha) / _gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def _niqe(img, mu_pris, cov_pris, gaussian_window, block_size_h=96, block_size_w=96):
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []
    for scale in (1, 2):
        mu = _conv2d_nearest(img, gaussian_window)
        sigma = np.sqrt(np.abs(_conv2d_nearest(np.square(img), gaussian_window) - np.square(mu)))
        img_normalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_normalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                      idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(_compute_feature(block))
        distparam.append(np.array(feat))

        if scale == 1:
            # 0.5 downsampling (bicubic)
            img = _imresize_half(img)

    distparam = np.concatenate(distparam, axis=1)
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    invcov_param = np.linalg.pinv((cov_pris + cov_distparam) / 2)
    quality = np.matmul(np.matmul((mu_pris - mu_distparam), invcov_param),
                        np.transpose((mu_pris - mu_distparam)))
    return float(np.squeeze(np.sqrt(quality)))


def _imresize_half(img):
    """0.5x bicubic downsampling."""
    try:
        import cv2
        return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    except ImportError:  # pragma: no cover
        from scipy.ndimage import zoom
        return zoom(img, 0.5, order=3)


def calculate_niqe(img, crop_border: int = 0) -> float:
    """Compute NIQE. img: single channel (H,W) in [0,1]."""
    data = np.load(_NPZ)
    mu_pris = data["mu_pris_param"]
    cov_pris = data["cov_pris_param"]
    gaussian_window = data["gaussian_window"]

    img = img.astype(np.float64)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    img = np.clip(img, 0.0, 1.0) * 255.0
    img = np.round(img)
    return _niqe(img, mu_pris, cov_pris, gaussian_window)

