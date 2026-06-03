"""Lung tumour segmentation service.

Uses a MONAI-based 3-D segmentation model (UNet / AttentionUNet / SwinUNETR)
trained with the segmentation-torch framework.  Returns a binary mask in the
same RAS / 1×1×2 mm space used by the SCLC preprocessor so the MIL bag
builder can select tumour-positive axial slices.

Configuration (env vars / .env):
  SEGMENTATION_MODEL_PATH         Path to a PyTorch or Lightning checkpoint.
  SEGMENTATION_MODEL_CONFIG_JSON  JSON with model + preprocessing params.
  SEGMENTATION_DEVICE             "cpu" | "cuda" | "cuda:0" etc.

Model config JSON fields:
  model_type            "AttentionUNet" | "UNet" | "SwinUNETR"
  nb_classes            int (default 2)
  network_shape         [H, W, D, C]  — spatial dims + input channels
  features              list[int]  — UNet/AttentionUNet channel sizes
  normalization_layer   "instance" | "batch" | "layer"
  dropout_rate          float
  roi_size              list[int]  — sliding-window patch (default [128,128,128])
  sw_overlap            float (default 0.5)
  output_spacing        list[float]  — mm, default [1.0, 1.0, 2.0]
  intensity_normalization  "znormalization" | "scaleintensity" | null

Falls back to None (no segmentation) when unconfigured or on any error.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings


class SegmentationRuntime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._cfg: dict[str, Any] = {}
        self._torch = None

    def _is_configured(self) -> bool:
        p = self.settings.segmentation_model_path
        return bool(p and Path(p).exists())

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self._is_configured():
            return False
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        import torch
        try:
            self._cfg = json.loads(self.settings.segmentation_model_config_json or "{}")
        except json.JSONDecodeError:
            return False
        try:
            self._model = self._build_model()
            self._load_checkpoint(self._model)
            self._model.to(torch.device(self.settings.segmentation_device))
            self._model.eval()
            self._torch = torch
        except Exception:
            self._model = None
            return False
        return True

    def _build_model(self):
        model_type = self._cfg.get("model_type", "AttentionUNet")
        nb_classes = int(self._cfg.get("nb_classes", 2))
        shape = self._cfg.get("network_shape", [128, 128, 128, 1])
        features = self._cfg.get("features", [16, 32, 64, 128, 256])
        norm = self._cfg.get("normalization_layer", "instance")
        dropout = float(self._cfg.get("dropout_rate", 0.0))
        spatial_dims = len(shape) - 1
        in_ch = shape[-1]

        if model_type == "AttentionUNet":
            from monai.networks.nets import AttentionUnet
            return AttentionUnet(
                spatial_dims=spatial_dims, in_channels=in_ch, out_channels=nb_classes,
                channels=features, strides=[2] * (len(features) - 1), dropout=dropout,
            )
        if model_type == "UNet":
            from monai.networks.nets import UNet
            return UNet(
                spatial_dims=spatial_dims, in_channels=in_ch, out_channels=nb_classes,
                channels=features, strides=[2] * (len(features) - 1),
                num_res_units=2, norm=norm, dropout=dropout,
            )
        if model_type == "SwinUNETR":
            from monai.networks.nets import SwinUNETR
            return SwinUNETR(
                img_size=shape[:-1], in_channels=in_ch, out_channels=nb_classes,
                feature_size=features[0] if features else 48,
                norm_name=norm, spatial_dims=spatial_dims,
            )
        raise ValueError(f"Unknown segmentation model_type: {model_type!r}")

    def _load_checkpoint(self, model) -> None:
        import torch
        ckpt = torch.load(
            str(self.settings.segmentation_model_path), map_location="cpu", weights_only=False
        )
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        for prefix in ("model.", "net.", ""):
            stripped = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
            res = model.load_state_dict(stripped, strict=False)
            if len(res.missing_keys) < len(model.state_dict()) // 2:
                return
        model.load_state_dict(sd, strict=False)

    def segment(self, file_bytes: bytes) -> np.ndarray | None:
        """Return binary 3-D mask (H×W×Z) in RAS / 1×1×2 mm space, or None."""
        if not self._ensure_loaded():
            return None
        _GZIP = b"\x1f\x8b"
        suffix = ".nii.gz" if file_bytes[:2] == _GZIP else ".nii"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(file_bytes)
            return self._run_inference(tmp_path)
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_inference(self, tmp_path: str) -> np.ndarray | None:
        import torch
        from monai.inferers import SlidingWindowInferer
        from monai.transforms import (
            Compose, EnsureChannelFirstd, Orientationd,
            ScaleIntensityRanged, Spacingd, ToTensord,
        )
        from sclc.data.transforms import LoadNiftiWithRGBSupportd

        spacing = self._cfg.get("output_spacing", [1.0, 1.0, 2.0])
        norm_method = self._cfg.get("intensity_normalization", "znormalization")
        roi_size = self._cfg.get("roi_size", [128, 128, 128])
        overlap = float(self._cfg.get("sw_overlap", 0.5))

        tfms = [
            LoadNiftiWithRGBSupportd(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode=["bilinear"]),
        ]
        if norm_method == "znormalization":
            from monai.transforms import NormalizeIntensityd
            tfms.append(NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True))
        elif norm_method == "scaleintensity":
            tfms.append(ScaleIntensityRanged(
                keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True
            ))
        tfms.append(ToTensord(keys=["image"]))

        img = Compose(tfms)({"image": tmp_path})["image"].unsqueeze(0)
        device = torch.device(self.settings.segmentation_device)
        img = img.to(device)

        inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=overlap, mode="gaussian")
        with torch.no_grad():
            logits = inferer(img, self._model)

        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
