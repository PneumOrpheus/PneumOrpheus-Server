"""SCLC preprocessing adapter for pneumorpheus-server inference.

Converts raw CT bytes into a model-ready tensor using the MONAI pipeline used for SCLC model training.  Dispatches on ``metadata["pipeline"]``:

  * "mil" → (1, bag_size, 1, img_size, img_size)
  * "2d"  → (1, 1, img_size, img_size)
  * "3d"  → (1, 1, img_size, img_size, depth_size)
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

_GZIP_MAGIC = b"\x1f\x8b"


def preprocess_for_inference(
    file_bytes: bytes,
    modality: str,
    metadata: dict[str, Any],
) -> "torch.Tensor":
    """Entry point called by ModelRuntime._make_input_tensor."""
    pipeline = str(metadata.get("pipeline", "mil")).lower()
    img_size = int(metadata.get("img_size", 384))

    suffix = ".nii.gz" if file_bytes[:2] == _GZIP_MAGIC else ".nii"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(file_bytes)
        if pipeline == "mil":
            return _preprocess_mil(tmp_path, img_size, int(metadata.get("bag_size", 16)))
        if pipeline == "2d":
            return _preprocess_2d(tmp_path, img_size)
        return _preprocess_3d(tmp_path, img_size, int(metadata.get("depth_size", 64)))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_volume(tmp_path: str, img_size: int) -> "torch.Tensor":
    """Load NIfTI, standardize to RAS orientation + 1x1x2 mm spacing, scale HU → [0,1].

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

    return result["image"] # (1, H, W, Z)


def _preprocess_mil(tmp_path: str, img_size: int, bag_size: int) -> "torch.Tensor":
    """MIL bag: evenly-spaced axial slices → (1, bag_size, 1, img_size, img_size)."""
    import torch
    from monai.transforms import NormalizeIntensity, Resize

    img = _load_volume(tmp_path, img_size)
    img = Resize(spatial_size=(img_size, img_size, -1), mode="trilinear")(img) # (1, H, W, Z)

    Z = img.shape[-1]
    upper = max(Z - 1, 0)
    idxs = torch.linspace(0, upper, bag_size).round().long().clamp(0, upper)
    # (1, H, W, Z) → select N slices along Z → permute to (N, 1, H, W)
    bag = torch.index_select(img, dim=-1, index=idxs).permute(3, 0, 1, 2).contiguous()

    norm = NormalizeIntensity(nonzero=True, channel_wise=True)
    bag = torch.stack([norm(bag[i]) for i in range(bag.shape[0])], dim=0)

    return bag.unsqueeze(0) # (1, N, 1, H, W)


def _preprocess_2d(tmp_path: str, img_size: int) -> "torch.Tensor":
    """Single middle axial slice → (1, 1, img_size, img_size)."""
    from monai.transforms import NormalizeIntensity, Resize

    img = _load_volume(tmp_path, img_size)
    img = Resize(spatial_size=(img_size, img_size, -1), mode="trilinear")(img) # (1, H, W, Z)
    mid = img.shape[-1] // 2
    slc = img[..., mid] # (1, H, W)
    slc = NormalizeIntensity(nonzero=True, channel_wise=True)(slc)
    return slc.unsqueeze(0)  # (1, 1, H, W)


def _preprocess_3d(tmp_path: str, img_size: int, depth_size: int) -> "torch.Tensor":
    """Full 3-D volume → (1, 1, img_size, img_size, depth_size)."""
    from monai.transforms import NormalizeIntensity, Resize

    img = _load_volume(tmp_path, img_size)
    img = Resize(spatial_size=(img_size, img_size, depth_size), mode="trilinear")(img) # (1, H, W, D)
    img = NormalizeIntensity(nonzero=True, channel_wise=True)(img)
    return img.unsqueeze(0) # (1, 1, H, W, D)
