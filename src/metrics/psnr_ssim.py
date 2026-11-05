"""Full-reference metrics: PSNR / SSIM / GMSD (numpy, single channel [0,1])."""

from __future__ import annotations

import numpy as np
import cv2


def _conv2d(img, kernel, border="reflect"):
    """2D convolution via cv2.filter2D (reflect or nearest border)."""
    btype = cv2.BORDER_REFLECT_101 if border == "reflect" else cv2.BORDER_REPLICATE
    return cv2.filter2D(img.astype(np.float32), -1, kernel.astype(np.float32), borderType=btype)


def _crop(img, border):
    if border > 0:
        return img[border:-border, border:-border]
    return img


def calculate_psnr(img1, img2, crop_border: int = 0, data_range: float = 1.0) -> float:
    """PSNR (single channel, [0,1])."""
    img1 = _crop(img1.astype(np.float64), crop_border)
    img2 = _crop(img2.astype(np.float64), crop_border)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _gaussian_window(size: int = 11, sigma: float = 1.5):
    ax = np.arange(-size // 2, size // 2 + 1, dtype=np.float64)
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def calculate_ssim(img1, img2, crop_border: int = 0, data_range: float = 1.0) -> float:
    """SSIM with an 11x11 Gaussian window (sigma=1.5)."""
    img1 = _crop(img1.astype(np.float64), crop_border)
    img2 = _crop(img2.astype(np.float64), crop_border)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win = _gaussian_window()
    mu1 = _conv2d(img1, win, "reflect")
    mu2 = _conv2d(img2, win, "reflect")
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = _conv2d(img1 * img1, win, "reflect") - mu1_sq
    sigma2_sq = _conv2d(img2 * img2, win, "reflect") - mu2_sq
    sigma12 = _conv2d(img1 * img2, win, "reflect") - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


_PREWITT_X = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]) / 3.0
_PREWITT_Y = _PREWITT_X.T


def calculate_gmsd(img1, img2, crop_border: int = 0, data_range: float = 1.0) -> float:
    """GMSD (Gradient Magnitude Similarity Deviation)."""
    img1 = _crop(img1.astype(np.float64), crop_border)
    img2 = _crop(img2.astype(np.float64), crop_border)

    def ave2(x):
        h, w = x.shape[0] - x.shape[0] % 2, x.shape[1] - x.shape[1] % 2
        x = x[:h, :w]
        return (x[::2, ::2] + x[1::2, ::2] + x[::2, 1::2] + x[1::2, 1::2]) / 4.0

    y1, y2 = ave2(img1), ave2(img2)

    def grad(x):
        gx = _conv2d(x, _PREWITT_X, "reflect")
        gy = _conv2d(x, _PREWITT_Y, "reflect")
        return np.sqrt(gx ** 2 + gy ** 2)

    g1, g2 = grad(y1), grad(y2)
    c = 170.0 * (data_range / 255.0) ** 2  # scale the constant with data_range
    gms = (2.0 * g1 * g2 + c) / (g1 ** 2 + g2 ** 2 + c)
    return float(gms.std())

