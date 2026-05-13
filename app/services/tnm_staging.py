"""TNM staging (IASLC 9th edition, effective 2025) from model outputs.

T category is estimated from the tumour's largest cross-sectional diameter
derived from the per-instance segmentation masks (preferred) or the averaged
2-D mask, falling back to the normalised bounding box from the detection head.
The in-plane pixel spacing is approximated at 1.0 mm/px (see note below).

N is reported as Nx — nodal status cannot be determined without a dedicated
lymph-node segmentation head or PET-CT correlation.

M is reported as Mx — distant metastases cannot be determined from a single
chest CT alone.

Pixel-spacing note
------------------
The preprocessing pipeline resamples to 1x1 mm in-plane (Spacingd) and then
resizes to 384x384.  The resize step makes effective spacing = original_FOV /
384 ≈ 350-400 mm / 384 ≈ 0.91-1.04 mm/px for a typical chest CT.  1.0 mm/px
is used as a calibration constant; size estimates carry ±10-15% error from
this approximation alone.

References
----------
Nicholson AG et al. (2024). The Proposed Ninth Edition TNM Classification of
Lung Cancer. CHEST, 166(5), 1006-1023.
https://doi.org/10.1016/j.chest.2024.07.062
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# In-plane pixel calibration (mm per pixel) — see module docstring
_PIXEL_SPACING_MM: float = 1.0

# Segmentation probability threshold for binary mask
_SEG_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# T-category size boundaries (solid component, mm) — IASLC 9th edition
# ---------------------------------------------------------------------------
# (upper_bound_exclusive, label)
_T_SIZE_RULES: list[tuple[float, str]] = [
    (10.0,  "T1a"),
    (20.0,  "T1b"),
    (30.0,  "T1c"),
    (40.0,  "T2a"),
    (50.0,  "T2b"),
    (70.0,  "T3"),
]
# Anything > 70 mm → T4


# ---------------------------------------------------------------------------
# Stage grouping — IASLC 9th edition
# N2 is subdivided into N2a (single-station) and N2b (multi-station).
# Because we report Nx we only expose the N0 best-case group here.
# ---------------------------------------------------------------------------

def _stage_group(t: str, n: str, m: str) -> str:
    """Return the 9th-edition stage group string, or '' if undetermined."""
    if "x" in t.lower() or "x" in n.lower() or "x" in m.lower():
        return ""

    m_upper = m.upper()
    if m_upper == "M1C2" or m_upper == "M1C1":
        return "IVB"
    if m_upper in ("M1A", "M1B"):
        return "IVA"

    n_upper = n.upper()
    t_upper = t.upper()

    if n_upper == "N3":
        return "IIIC"

    if t_upper == "T4":
        if n_upper in ("N0", "N1", "N2A", "N2B", "N2"):
            return "IIIB"

    if t_upper == "T3":
        if n_upper == "N0" or n_upper == "N1":
            return "IIIA"
        if n_upper == "N2A":
            return "IIIA"
        if n_upper in ("N2B", "N2"):
            return "IIIB"

    if t_upper in ("T2A", "T2B"):
        if n_upper == "N2B" or n_upper == "N2":
            return "IIIB"
        if n_upper == "N2A":
            return "IIIA" if t_upper == "T2A" else "IIIA"
        if n_upper == "N1":
            return "IIA" if t_upper == "T2A" else "IIB"
        if n_upper == "N0":
            return "IB" if t_upper == "T2A" else "IIA"

    if t_upper in ("T1A", "T1B", "T1C"):
        if n_upper in ("N2B", "N2"):
            return "IIIA"
        if n_upper == "N2A":
            return "IIB"
        if n_upper == "N1":
            return "IIB" if t_upper == "T1C" else "IB"
        if n_upper == "N0":
            if t_upper == "T1A":
                return "IA1"
            if t_upper == "T1B":
                return "IA2"
            return "IA3"

    return ""


# ---------------------------------------------------------------------------
# Size estimation from segmentation outputs
# ---------------------------------------------------------------------------

def _equivalent_diameter_mm(binary_mask: np.ndarray) -> float:
    """Approximate largest-dimension diameter from a binary 2-D mask (pixels → mm)."""
    area_px = float(binary_mask.sum())
    if area_px <= 0:
        return 0.0
    # Equivalent circle diameter
    diameter_px = 2.0 * math.sqrt(area_px / math.pi)
    return diameter_px * _PIXEL_SPACING_MM


def _diameter_from_seg_per_instance(seg_per_instance: np.ndarray) -> tuple[float, str]:
    """Max equivalent diameter across all bag instances (N, H, W) → (mm, method)."""
    max_d = 0.0
    for i in range(seg_per_instance.shape[0]):
        mask = seg_per_instance[i] >= _SEG_THRESHOLD
        d = _equivalent_diameter_mm(mask)
        if d > max_d:
            max_d = d
    return max_d, "seg_per_instance"


def _diameter_from_seg_mask(seg_mask: np.ndarray) -> tuple[float, str]:
    """Equivalent diameter from averaged 2-D segmentation mask → (mm, method)."""
    mask = seg_mask >= _SEG_THRESHOLD
    return _equivalent_diameter_mm(mask), "seg_mask_avg"


def _diameter_from_bbox(bbox: np.ndarray) -> tuple[float, str]:
    """Approximate diameter from normalised detection-head bbox → (mm, method)."""
    if len(bbox) < 4:
        return 0.0, "bbox"
    w_mm = abs(float(bbox[2]) - float(bbox[0])) * 384.0 * _PIXEL_SPACING_MM
    h_mm = abs(float(bbox[3]) - float(bbox[1])) * 384.0 * _PIXEL_SPACING_MM
    return max(w_mm, h_mm), "bbox"


def _estimate_diameter(
    seg_per_instance: np.ndarray | None,
    seg_mask: np.ndarray | None,
    bbox: np.ndarray | None,
) -> tuple[float | None, str]:
    """Return (diameter_mm, method) using the best available source."""
    if seg_per_instance is not None and seg_per_instance.ndim == 3 and seg_per_instance.shape[0] > 0:
        d, method = _diameter_from_seg_per_instance(seg_per_instance)
        if d > 0:
            return d, method

    if seg_mask is not None and seg_mask.ndim == 2:
        d, method = _diameter_from_seg_mask(seg_mask)
        if d > 0:
            return d, method

    if bbox is not None and len(bbox) >= 4:
        d, method = _diameter_from_bbox(bbox)
        if d > 0:
            return d, method

    return None, "none"


# ---------------------------------------------------------------------------
# T assignment
# ---------------------------------------------------------------------------

def _assign_t(diameter_mm: float | None) -> str:
    if diameter_mm is None or diameter_mm <= 0:
        return "Tx"
    for upper, label in _T_SIZE_RULES:
        if diameter_mm <= upper:
            return label
    return "T4"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_tnm(
    predicted_type: str,
    confidence: float,
    seg_per_instance: np.ndarray | None,
    seg_mask: np.ndarray | None,
    bbox: np.ndarray | None,
) -> dict[str, Any]:
    """Estimate TNM staging (IASLC 9th edition) from model inference outputs.

    Returns a dict with keys:
      t, n, m, tnm_string, best_case_stage, estimated_diameter_mm,
      size_method, caveats, edition
    """
    diameter_mm, size_method = _estimate_diameter(seg_per_instance, seg_mask, bbox)

    t = _assign_t(diameter_mm)
    n = "Nx"
    m = "Mx"

    tnm_string = f"{t} {n} {m}"

    # Best-case stage: assume N0 M0 (no nodal/distant spread detected)
    best_case_stage = _stage_group(t, "N0", "M0")
    best_case_label = f"Stage {best_case_stage} (N0 M0 assumed)" if best_case_stage else ""

    caveats: list[str] = [
        "T is estimated from imaging size only; invasion features (visceral pleural, bronchial, "
        "chest-wall, vascular) that may upgrade the T category cannot be assessed by the current model.",
        "N is Nx — lymph node involvement requires PET-CT or mediastinoscopy; "
        "a future model version with a lymph-node segmentation head could determine N.",
        "M is Mx — distant metastases cannot be assessed from a single chest CT.",
    ]
    if size_method == "bbox":
        caveats.append(
            "Tumour size derived from detection-head bounding box (segmentation mask was absent); "
            "accuracy may be lower."
        )
    if size_method == "none" or t == "Tx":
        caveats.append("Tumour size could not be estimated; T category is indeterminate.")

    return {
        "t": t,
        "n": n,
        "m": m,
        "tnm_string": tnm_string,
        "best_case_stage": best_case_label,
        "estimated_diameter_mm": round(diameter_mm, 1) if diameter_mm is not None else None,
        "size_method": size_method,
        "caveats": caveats,
        "edition": "IASLC 9th edition (2024/2025)",
    }
