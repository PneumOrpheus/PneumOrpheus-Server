"""SCLC preprocessing adapter for pneumorpheus-server inference.

Converts raw CT bytes into a model-ready tensor using the MONAI pipeline used
for SCLC model training.  Dispatches on ``metadata["pipeline"]``:

  * "mil" → (1, bag_size, 1, img_size, img_size)
  * "2d"  → (1, 1, img_size, img_size)
  * "3d"  → (1, 1, img_size, img_size, depth_size)

When a ``tumor_mask`` (binary HxWxZ numpy array in RAS / 1x1x2 mm space) is
supplied to the MIL path, axial slices are selected from tumour-positive
positions (mirroring the training-time ``TumorPositiveBagSelectd`` transform)
instead of being evenly spaced across the full volume.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

_GZIP_MAGIC = b"\x1f\x8b"


def preprocess_for_inference(
    file_bytes: bytes,
    modality: str,
    metadata: dict[str, Any],
    tumor_mask: np.ndarray | None = None,
) -> tuple["torch.Tensor", dict[str, Any]]:
    """Entry point called by ModelRuntime._make_input_tensor.

    tumor_mask : optional binary 3-D numpy array (HxWxZ) in RAS / 1x1x2 mm
                 space.  When provided, MIL bag slices are selected from
                 tumour-positive axial positions instead of evenly spaced.

    Returns (tensor, extras); extras has ct_volume/ct_affine/mil_indices for
    the "mil" pipeline, else {}.
    """
    pipeline = str(metadata.get("pipeline", "mil")).lower()
    img_size = int(metadata.get("img_size", 256))

    suffix = ".nii.gz" if file_bytes[:2] == _GZIP_MAGIC else ".nii"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(file_bytes)
        if pipeline == "mil":
            return _preprocess_mil(
                tmp_path, img_size, int(metadata.get("bag_size", 16)), tumor_mask
            )
        if pipeline == "2d":
            return _preprocess_2d(tmp_path, img_size), {}
        return _preprocess_3d(tmp_path, img_size, int(metadata.get("depth_size", 64))), {}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_volume(tmp_path: str, img_size: int) -> "torch.Tensor":
    """Load NIfTI, standardize to RAS orientation + 1×1×2 mm spacing, scale HU → [0,1].

    Returns a MetaTensor of shape (1, H, W, Z).
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
        ToTensord,
    )
    from sclc.data.transforms import LoadNiftiWithRGBSupportd

    result = Compose([
        LoadNiftiWithRGBSupportd(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 2.0), mode=["bilinear"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True),
        ToTensord(keys=["image"]),
    ])({"image": tmp_path})

    return result["image"]  # (1, H, W, Z)


def _select_mil_indices(
    Z: int,
    bag_size: int,
    tumor_mask: np.ndarray | None,
) -> "torch.Tensor":
    """Return a 1-D LongTensor of bag_size z-indices.

    When tumor_mask is provided (HxWxZ_mask), z-indices are sampled from
    tumour-positive slices.  The mask's Z dimension is mapped proportionally
    onto [0, Z-1] to handle spacing differences between the segmentation and
    the preprocessed CT.  Falls back to evenly-spaced sampling when fewer
    than bag_size tumour slices exist or when no mask is provided.
    """
    import torch

    upper = max(Z - 1, 0)

    if tumor_mask is not None and tumor_mask.ndim == 3:
        Z_mask = tumor_mask.shape[2]
        tumor_present = np.any(tumor_mask > 0, axis=(0, 1))  # (Z_mask,)
        tumor_z_mask = np.where(tumor_present)[0]

        if len(tumor_z_mask) >= bag_size:
            # Map mask z-indices → preprocessed CT z-indices (proportional)
            scale = upper / max(Z_mask - 1, 1)
            tumor_z_ct = np.round(tumor_z_mask * scale).astype(int)
            tumor_z_ct = np.clip(tumor_z_ct, 0, upper)
            # Evenly sample bag_size indices from the tumour extent
            positions = np.linspace(0, len(tumor_z_ct) - 1, bag_size)
            idxs_np = tumor_z_ct[np.round(positions).astype(int)]
            return torch.from_numpy(idxs_np).long()

    # Fallback: evenly spaced across the full volume
    return torch.linspace(0, upper, bag_size).round().long().clamp(0, upper)


def _preprocess_mil(
    tmp_path: str,
    img_size: int,
    bag_size: int,
    tumor_mask: np.ndarray | None = None,
) -> tuple["torch.Tensor", dict[str, Any]]:
    """MIL bag: tumour-positive (or evenly-spaced) axial slices.

    Returns ((1, bag_size, 1, img_size, img_size), extras).
    """
    import torch
    from monai.transforms import NormalizeIntensity, Resize

    full_res = _load_volume(tmp_path, img_size)  # (1, H, W, Z), native res, .affine attached
    ct_affine = np.asarray(full_res.affine.cpu() if hasattr(full_res.affine, "cpu") else full_res.affine)
    ct_volume = full_res[0].detach().cpu().numpy()  # (H, W, Z)

    img = Resize(spatial_size=(img_size, img_size, -1), mode="trilinear")(full_res)  # (1, H, W, Z)

    Z = img.shape[-1]
    idxs = _select_mil_indices(Z, bag_size, tumor_mask)

    # (1, H, W, Z) → select N slices along Z → permute to (N, 1, H, W)
    bag = torch.index_select(img, dim=-1, index=idxs).permute(3, 0, 1, 2).contiguous()

    norm = NormalizeIntensity(nonzero=True, channel_wise=True)
    bag = torch.stack([norm(bag[i]) for i in range(bag.shape[0])], dim=0)

    extras = {
        "ct_volume": ct_volume,
        "ct_affine": ct_affine,
        "mil_indices": idxs.cpu().numpy(),
    }
    return bag.unsqueeze(0), extras  # (1, N, 1, H, W)


def _preprocess_2d(tmp_path: str, img_size: int) -> "torch.Tensor":
    """Single middle axial slice → (1, 1, img_size, img_size)."""
    from monai.transforms import NormalizeIntensity, Resize

    img = _load_volume(tmp_path, img_size)
    img = Resize(spatial_size=(img_size, img_size, -1), mode="trilinear")(img)  # (1, H, W, Z)
    mid = img.shape[-1] // 2
    slc = img[..., mid]  # (1, H, W)
    slc = NormalizeIntensity(nonzero=True, channel_wise=True)(slc)
    return slc.unsqueeze(0)  # (1, 1, H, W)


def _preprocess_3d(tmp_path: str, img_size: int, depth_size: int) -> "torch.Tensor":
    """Full 3-D volume → (1, 1, img_size, img_size, depth_size)."""
    from monai.transforms import NormalizeIntensity, Resize

    img = _load_volume(tmp_path, img_size)
    img = Resize(spatial_size=(img_size, img_size, depth_size), mode="trilinear")(img)  # (1, H, W, D)
    img = NormalizeIntensity(nonzero=True, channel_wise=True)(img)
    return img.unsqueeze(0)  # (1, 1, H, W, D)
