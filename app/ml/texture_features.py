"""Текстура + геометрия предмета для сети 1/2."""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError:  # pragma: no cover
    graycomatrix = None
    graycoprops = None
    local_binary_pattern = None

TEXTURE_SIZE = 224
LBP_POINTS = 8
LBP_RADIUS = 1
GLCM_LEVELS = 32
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
GEOMETRY_DIM = 2  # aspect_ratio (elongation), solidity


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    if local_binary_pattern is None:
        return np.zeros(10, dtype=np.float32)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    n_bins = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    if len(hist) < 10:
        hist = np.pad(hist, (0, 10 - len(hist)))
    return hist[:10].astype(np.float32)


def _glcm_features(gray: np.ndarray) -> np.ndarray:
    if graycomatrix is None or graycoprops is None:
        return np.zeros(len(GLCM_PROPS), dtype=np.float32)
    quantized = (gray.astype(np.float32) / 256.0 * (GLCM_LEVELS - 1)).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return np.array(
        [float(graycoprops(glcm, prop).mean()) for prop in GLCM_PROPS],
        dtype=np.float32,
    )


def extract_geometry_features(pil: Image.Image) -> np.ndarray:
    """Пропорции bbox и заполненность маски (нож vs наконечник).

    aspect_ratio = max(side)/min(side)  (инвариантно к повороту кадра)
    solidity = площадь маски / площадь bbox
    """
    from app.ml.preprocess import _foreground_mask

    mask = _foreground_mask(pil)
    fg = int(mask.sum())
    if fg < 40:
        return np.array([1.0, 0.5], dtype=np.float32)

    ys, xs = np.where(mask)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    short = max(min(bw, bh), 1)
    long = max(bw, bh)
    aspect = float(long / short)
    # нормируем к ~[0, 1]: типичные 1..5 → /5
    aspect_n = min(aspect / 5.0, 1.0)
    solidity = float(fg / max(bw * bh, 1))
    solidity = min(max(solidity, 0.0), 1.0)
    return np.array([aspect_n, solidity], dtype=np.float32)


def extract_texture_vector(pil: Image.Image, size: int = TEXTURE_SIZE) -> np.ndarray:
    """15 текстурных + 2 геометрических признака."""
    gray = np.array(pil.convert("L").resize((size, size), Image.LANCZOS), dtype=np.uint8)
    lbp = _lbp_histogram(gray)
    glcm = _glcm_features(gray)
    geom = extract_geometry_features(pil)
    return np.concatenate([lbp, glcm, geom]).astype(np.float32)


def texture_dim() -> int:
    return 10 + len(GLCM_PROPS) + GEOMETRY_DIM
