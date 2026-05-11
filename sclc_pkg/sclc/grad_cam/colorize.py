from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


def _jet(x: np.ndarray) -> np.ndarray:
    """Jet-like colormap: blue (low) -> cyan -> green -> yellow -> red (high).
    Input in [0, 1]; returns float array in [0, 1] with trailing channel dim of 3.
    """
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def colorize_heatmap(cam: np.ndarray) -> np.ndarray:
    """Scalar CAM volume -> RGB uint8 volume with a jet colormap."""
    return (_jet(cam) * 255.0).astype(np.uint8)


def colorize_overlay(
    ct_gray: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    threshold: float = 0.15,
) -> np.ndarray:
    """Blend a jet-colored CAM on top of grayscale CT. CT stays visible where
    the CAM is below ``threshold``; the colored CAM fades in (weighted by
    magnitude * ``alpha``) where activation is high. Returns (H, W, 3) uint8.
    """
    ct = np.clip(ct_gray.astype(np.float32), 0.0, 1.0)
    c = np.clip(cam.astype(np.float32), 0.0, 1.0)

    base = np.stack([ct, ct, ct], axis=-1)
    heat = _jet(c)

    w = np.where(c >= threshold, (c - threshold) / max(1e-6, 1.0 - threshold), 0.0)
    w = (alpha * w).astype(np.float32)[..., None]

    out = (1.0 - w) * base + w * heat
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
