from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Any, cast

import nibabel as nib
import numpy as np
from PIL import Image


MAX_RENDERED_SLICES = 28
RENDER_SIZE = (320, 320)
_SEG_COLOR = np.array([255.0, 45.0, 45.0], dtype=np.float32)
_SEG_ALPHA = 0.34


@dataclass
class VolumeData:
    image: np.ndarray
    mask: np.ndarray | None


def build_nifti_visualization(file_name: str, file_bytes: bytes, runtime_result: dict[str, Any]) -> dict[str, Any] | None:
    if not _looks_like_nifti(file_name):
        return None

    try:
        volume = _load_volume(file_name=file_name, file_bytes=file_bytes, runtime_result=runtime_result)
    except Exception:
        return None

    if volume.image.ndim != 3 or min(volume.image.shape) <= 1:
        return None

    slice_indices = _select_slice_indices(volume.image, volume.mask)
    if not slice_indices:
        return None

    slices: list[dict[str, Any]] = []
    for slice_index in slice_indices:
        image = volume.image[:, :, slice_index]
        mask_slice = volume.mask[:, :, slice_index] if volume.mask is not None else None

        rendered, mask_coverage = _render_slice(image, mask_slice)
        slices.append(
            {
                "sliceIndex": int(slice_index),
                "imageDataUrl": rendered,
                "hasMask": bool(mask_slice is not None and np.any(mask_slice > 0)),
                "maskCoverage": round(mask_coverage, 6),
            }
        )

    default_slice = _choose_default_slice(slices)

    return {
        "format": "slice-overlay-v1",
        "imageFormat": "image/jpeg",
        "orientation": "axial",
        "totalSlices": int(volume.image.shape[2]),
        "defaultSliceIndex": int(default_slice),
        "slices": slices,
    }


def _looks_like_nifti(file_name: str) -> bool:
    lower = file_name.lower().strip()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _load_volume(file_name: str, file_bytes: bytes, runtime_result: dict[str, Any]) -> VolumeData:
    suffix = ".nii.gz" if file_name.lower().endswith(".nii.gz") else ".nii"
    with NamedTemporaryFile(suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()

        nifti = cast(nib.Nifti1Image, nib.load(temp_file.name))
        image = np.asarray(nifti.get_fdata(dtype=np.float32), dtype=np.float32)

    image = _ensure_3d(image)

    mask = _extract_mask(runtime_result=runtime_result, expected_shape=image.shape)

    return VolumeData(image=image, mask=mask)


def _extract_mask(runtime_result: dict[str, Any], expected_shape: tuple[int, int, int]) -> np.ndarray | None:
    candidates = [
        runtime_result.get("mask_volume"),
        runtime_result.get("maskVolume"),
        runtime_result.get("segmentation_mask"),
        runtime_result.get("segmentationMask"),
    ]

    segmentation_data = runtime_result.get("segmentation_data")
    if isinstance(segmentation_data, dict):
        candidates.extend(
            [
                segmentation_data.get("mask_volume"),
                segmentation_data.get("maskVolume"),
                segmentation_data.get("segmentation_mask"),
                segmentation_data.get("segmentationMask"),
            ]
        )

    for candidate in candidates:
        if candidate is None:
            continue

        try:
            parsed = np.asarray(candidate, dtype=np.float32)
        except Exception:
            continue

        parsed = _ensure_3d(parsed)
        if parsed.shape != expected_shape:
            continue

        return (parsed > 0).astype(np.uint8)

    return None


def _ensure_3d(image: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(image)
    if squeezed.ndim == 4:
        squeezed = squeezed[..., 0]
    return np.asarray(squeezed, dtype=np.float32)


def _select_slice_indices(image: np.ndarray, mask: np.ndarray | None) -> list[int]:
    depth = image.shape[2]
    if depth <= MAX_RENDERED_SLICES:
        return list(range(depth))

    if mask is not None:
        mask_presence = np.any(mask > 0, axis=(0, 1))
        mask_slices = np.flatnonzero(mask_presence)
        if mask_slices.size > 0:
            return _sample_evenly(mask_slices.tolist(), MAX_RENDERED_SLICES)

    return _sample_evenly(list(range(depth)), MAX_RENDERED_SLICES)


def _sample_evenly(indices: list[int], target_count: int) -> list[int]:
    if len(indices) <= target_count:
        return indices

    samples = np.linspace(0, len(indices) - 1, num=target_count)
    return sorted({indices[int(round(position))] for position in samples})


def _render_slice(image_slice: np.ndarray, mask_slice: np.ndarray | None) -> tuple[str, float]:
    normalized = _normalize_slice(image_slice)
    base_rgb = np.stack([normalized, normalized, normalized], axis=-1)

    mask_coverage = 0.0
    if mask_slice is not None and np.any(mask_slice > 0):
        mask = mask_slice > 0
        mask_coverage = float(np.mean(mask))
        overlay_color = np.array([255.0, 45.0, 45.0], dtype=np.float32)
        alpha = 0.34
        base_rgb = base_rgb.astype(np.float32)
        base_rgb[mask] = (1.0 - alpha) * base_rgb[mask] + alpha * overlay_color

    image = Image.fromarray(base_rgb.astype(np.uint8), mode="RGB")
    image = image.resize(RENDER_SIZE, resample=Image.Resampling.BILINEAR)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=84)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", mask_coverage


def _normalize_slice(image_slice: np.ndarray) -> np.ndarray:
    finite_values = image_slice[np.isfinite(image_slice)]
    if finite_values.size == 0:
        return np.zeros_like(image_slice, dtype=np.uint8)

    low = float(np.percentile(finite_values, 1.0))
    high = float(np.percentile(finite_values, 99.0))
    if high <= low:
        high = low + 1e-6

    clipped = np.clip(image_slice, low, high)
    scaled = ((clipped - low) / (high - low)) * 255.0
    return scaled.astype(np.uint8)


def _choose_default_slice(slices: list[dict[str, Any]]) -> int:
    with_mask = [slice_item for slice_item in slices if slice_item.get("hasMask")]
    source = with_mask if with_mask else slices
    middle_index = len(source) // 2
    return int(source[middle_index]["sliceIndex"])


# ---------------------------------------------------------------------------
# Three-panel MIL visualization (plain CT / seg overlay / GradCAM overlay)
# ---------------------------------------------------------------------------

def build_three_visualizations(
    input_tensor: Any,
    seg_per_instance: np.ndarray | None,
    gradcam_data: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any] | None:
    """Build three slice-overlay-v1 dicts from preprocessed MIL bag slices.

    input_tensor : (1, N, 1, H, W) PyTorch tensor or None
    seg_per_instance : (N, H, W) float32 sigmoid probabilities or None
    gradcam_data : (cam_np (N,H,W), att_np (N,)) or None
    """
    if input_tensor is None:
        return None

    try:
        bag = _extract_bag_numpy(input_tensor)  # (N, H, W) float32
    except Exception:
        return None

    N = bag.shape[0]

    cam_np, att_np = (None, None)
    if gradcam_data is not None:
        try:
            cam_np, att_np = gradcam_data
            if cam_np.shape[0] != N:
                cam_np, att_np = None, None
        except Exception:
            cam_np, att_np = None, None

    if seg_per_instance is not None and seg_per_instance.shape[0] != N:
        seg_per_instance = None

    plain_slices: list[dict[str, Any]] = []
    seg_slices: list[dict[str, Any]] = []
    cam_slices: list[dict[str, Any]] = []

    for i in range(N):
        ct_slice = bag[i]  # (H, W)

        # --- plain CT ---
        rendered_plain, _ = _render_slice(ct_slice, None)
        plain_slices.append({
            "sliceIndex": i,
            "imageDataUrl": rendered_plain,
            "hasMask": False,
            "maskCoverage": 0.0,
        })

        # --- segmentation overlay ---
        seg_slice = seg_per_instance[i] if seg_per_instance is not None else None
        seg_mask_bin = (seg_slice > 0.5).astype(np.uint8) if seg_slice is not None else None
        rendered_seg, seg_cov = _render_slice(ct_slice, seg_mask_bin)
        seg_slices.append({
            "sliceIndex": i,
            "imageDataUrl": rendered_seg,
            "hasMask": bool(seg_mask_bin is not None and np.any(seg_mask_bin > 0)),
            "maskCoverage": round(seg_cov, 6),
        })

        # --- GradCAM overlay ---
        cam_slice = cam_np[i] if cam_np is not None else None
        rendered_cam, cam_cov = _render_gradcam_slice(ct_slice, cam_slice)
        cam_slices.append({
            "sliceIndex": i,
            "imageDataUrl": rendered_cam,
            "hasMask": cam_slice is not None,
            "maskCoverage": round(cam_cov, 6),
        })

    # Default slice: highest attention weight for cam/seg; middle for plain
    if att_np is not None and len(att_np) == N:
        default_cam = int(np.argmax(att_np))
        default_seg = default_cam
    else:
        default_cam = N // 2
        default_seg = N // 2
    default_plain = N // 2

    def _vis(slices, default):
        return {
            "format": "slice-overlay-v1",
            "imageFormat": "image/jpeg",
            "orientation": "axial",
            "totalSlices": N,
            "defaultSliceIndex": default,
            "slices": slices,
        }

    return {
        "plainCt": _vis(plain_slices, default_plain),
        "segmentationOverlay": _vis(seg_slices, default_seg),
        "gradCamOverlay": _vis(cam_slices, default_cam),
    }


def _extract_bag_numpy(input_tensor: Any) -> np.ndarray:
    """Extract (N, H, W) float32 from (1, N, 1, H, W) tensor or ndarray."""
    try:
        arr = input_tensor.detach().cpu().numpy()
    except AttributeError:
        arr = np.asarray(input_tensor, dtype=np.float32)
    # (1, N, 1, H, W) → (N, H, W)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        # (N, 1, H, W) → (N, H, W)
        arr = arr[:, 0, :, :]
    return arr.astype(np.float32)


def _render_gradcam_slice(ct_slice: np.ndarray, cam_slice: np.ndarray | None) -> tuple[str, float]:
    """Render a CT slice with a jet-colored GradCAM overlay."""
    from sclc.grad_cam.colorize import colorize_overlay

    ct_norm = _normalize_slice(ct_slice).astype(np.float32) / 255.0  # [0, 1]

    if cam_slice is not None and np.any(cam_slice > 0):
        cam_coverage = float(np.mean(cam_slice))
        rgb = colorize_overlay(ct_norm, cam_slice, alpha=0.55, threshold=0.10)
    else:
        cam_coverage = 0.0
        gray = (ct_norm * 255.0).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize(RENDER_SIZE, resample=Image.Resampling.BILINEAR)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=84)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", cam_coverage


def build_nifti_file_payload(
    array: np.ndarray, affine: np.ndarray, filename: str, dtype: Any = np.float32
) -> dict[str, Any]:
    """Serialize a (H,W,Z) numpy volume + affine into a downloadable NIfTI blob."""
    with NamedTemporaryFile(suffix=".nii.gz") as tmp:
        nib.save(nib.Nifti1Image(array.astype(dtype), affine), tmp.name)
        tmp.seek(0)
        raw = tmp.read()
    return {
        "filename": filename,
        "mimeType": "application/gzip",
        "sizeBytes": len(raw),
        "base64Data": base64.b64encode(raw).decode("ascii"),
    }


_RGB_DTYPE = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])


def build_rgb_nifti_file_payload(rgb: np.ndarray, affine: np.ndarray, filename: str) -> dict[str, Any]:
    """Serialize a (H,W,Z,3) uint8 volume + affine into an RGB24 NIfTI blob."""
    packed = np.ascontiguousarray(rgb).view(dtype=_RGB_DTYPE).reshape(rgb.shape[:3])
    img = nib.Nifti1Image(packed, affine)
    img.header.set_data_dtype(_RGB_DTYPE)
    with NamedTemporaryFile(suffix=".nii.gz") as tmp:
        nib.save(img, tmp.name)
        tmp.seek(0)
        raw = tmp.read()
    return {
        "filename": filename,
        "mimeType": "application/gzip",
        "sizeBytes": len(raw),
        "base64Data": base64.b64encode(raw).decode("ascii"),
    }


def build_segmentation_overlay_volume(
    ct_volume: np.ndarray, mask_volume: np.ndarray, color: tuple[int, int, int] = (244, 63, 94), alpha: float = 0.55
) -> np.ndarray:
    """CT grayscale with a flat-colored overlay baked in where mask > 0. Returns (H,W,Z,3) uint8."""
    ct = np.clip(ct_volume.astype(np.float32), 0.0, 1.0)
    base = np.stack([ct, ct, ct], axis=-1)
    heat = np.array(color, dtype=np.float32) / 255.0
    w = (alpha * (mask_volume > 0)).astype(np.float32)[..., None]
    out = (1.0 - w) * base + w * heat
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def build_gradcam_overlay_volume(
    ct_volume: np.ndarray, cam_np: np.ndarray, mil_indices: np.ndarray
) -> np.ndarray:
    """CT grayscale with a jet-colored Grad-CAM overlay baked in on the sampled MIL
    slices; other slices fall back to plain CT (colorize_overlay with cam=0 is a
    no-op blend). Returns (H,W,Z,3) uint8.
    """
    import torch
    import torch.nn.functional as F
    from sclc.grad_cam.colorize import colorize_overlay

    H, W, Z = ct_volume.shape
    cam_full = np.zeros((H, W, Z), dtype=np.float32)
    cam_t = torch.from_numpy(cam_np).float().unsqueeze(1)  # (N, 1, h, w)
    resized = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()
    for i, z in enumerate(mil_indices.tolist()):
        cam_full[:, :, int(z)] = resized[i]

    out = np.empty((H, W, Z, 3), dtype=np.uint8)
    ct = np.clip(ct_volume.astype(np.float32), 0.0, 1.0)
    for z in range(Z):
        out[:, :, z, :] = colorize_overlay(ct[:, :, z], cam_full[:, :, z])
    return out
