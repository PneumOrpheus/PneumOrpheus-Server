"""TNM staging (IASLC 9th edition, effective 2025) from model outputs.

T category is estimated from the tumour's largest axial cross-sectional
diameter, measured directly off the CT_Tumor segmentation mask —
a real per-voxel mask on a known physical grid (1x1x2mm, see
segmentation_runtime.py).

N is reported as Nx — nodal status cannot be determined without a dedicated
lymph-node segmentation head or PET-CT correlation.

M is reported as Mx — distant metastases cannot be determined from a single
chest CT alone.

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

# In-plane spacing (mm/px) of the tumour mask grid, which matches the
# Spacingd(pixdim=(1.0, 1.0, 2.0)) resample in segmentation_runtime.py.
_IN_PLANE_SPACING_MM: float = 1.0


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
# Size estimation from the tumour segmentation mask
# ---------------------------------------------------------------------------

def _equivalent_diameter_mm(binary_slice: np.ndarray) -> float:
    """Equivalent-circle diameter of a binary 2-D axial slice (mm)."""
    area_px = float(binary_slice.sum())
    if area_px <= 0:
        return 0.0
    diameter_px = 2.0 * math.sqrt(area_px / math.pi)
    return diameter_px * _IN_PLANE_SPACING_MM


def _largest_diameter_mm(tumor_mask_3d: np.ndarray) -> float:
    """Largest single-slice equivalent diameter across the volume (mm).

    Mirrors how the greatest tumour dimension is conventionally read off
    axial CT for T-staging: the largest single-slice extent, not a 3-D
    volume-equivalent size.
    """
    if tumor_mask_3d.ndim != 3:
        return 0.0

    max_d = 0.0
    for z in range(tumor_mask_3d.shape[2]):
        d = _equivalent_diameter_mm(tumor_mask_3d[:, :, z] > 0)
        if d > max_d:
            max_d = d
    return max_d


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
    tumor_mask_3d: np.ndarray | None,
) -> dict[str, Any]:
    """Estimate TNM staging (IASLC 9th edition) from the tumour segmentation mask.

    Returns a dict with keys:
      t, n, m, tnm_string, best_case_stage, estimated_diameter_mm,
      size_method, caveats, edition
    """
    diameter_mm: float | None = None
    if tumor_mask_3d is not None:
        d = _largest_diameter_mm(tumor_mask_3d)
        if d > 0:
            diameter_mm = d

    t = _assign_t(diameter_mm)
    n = "Nx"
    m = "Mx"

    tnm_string = f"{t} {n} {m}"

    # Best-case stage: assume N0 M0 (no nodal/distant spread detected)
    best_case_stage = _stage_group(t, "N0", "M0")
    best_case_label = f"Stage {best_case_stage} (N0 M0 assumed)" if best_case_stage else ""

    caveats: list[str] = [
        "T is estimated from tumour segmentation size only; invasion features (visceral pleural, "
        "bronchial, chest-wall, vascular) that may upgrade the T category cannot be assessed by the "
        "current model.",
        "N is Nx — lymph node involvement requires PET-CT or mediastinoscopy; "
        "a future model version with a lymph-node segmentation head could determine N.",
        "M is Mx — distant metastases cannot be assessed from a single chest CT.",
    ]
    if t == "Tx":
        caveats.append("Tumour segmentation was empty or unavailable; T category is indeterminate.")

    return {
        "t": t,
        "n": n,
        "m": m,
        "tnm_string": tnm_string,
        "best_case_stage": best_case_label,
        "estimated_diameter_mm": round(diameter_mm, 1) if diameter_mm is not None else None,
        "size_method": "tumor_mask" if diameter_mm is not None else "none",
        "caveats": caveats,
        "edition": "IASLC 9th edition (2024/2025)",
    }
