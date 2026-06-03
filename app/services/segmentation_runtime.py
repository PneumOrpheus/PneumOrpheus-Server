"""Lung tumour segmentation service.

Uses a MONAI-based 3-D segmentation model (UNet / AttentionUNet / SwinUNETR).
Returns a binary mask in the same RAS / 1×1×2 mm space used by the SCLC
preprocessor so the MIL bag builder can select tumour-positive axial slices.

Blob layout (mirrors classification model layout):
  {AZURE_BLOB_PREFIX}/{SEGMENTATION_MODEL_NAME}/{SEGMENTATION_MODEL_VERSION}/
      checkpoint.pth          ← weights  (filename set by SEGMENTATION_MODEL_FILE)
      segmentation_config.json ← architecture + preprocessing params (see below)

When MODEL_SOURCE is not "azure_blob", set SEGMENTATION_MODEL_PATH to an
explicit local checkpoint path and supply SEGMENTATION_MODEL_CONFIG_JSON as a
fallback for the architecture params.

segmentation_config.json fields:
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
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from app.config import get_settings

_SEG_CONFIG_FILE = "segmentation_config.json"


class SegmentationRuntime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._cfg: dict[str, Any] = {}
        self._torch = None
        self._resolved_path: Path | None = None

    # ------------------------------------------------------------------
    # Path resolution + blob download
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> Path | None:
        """Return the local checkpoint path, downloading from blob when needed.

        Also populates self._cfg from segmentation_config.json if found.
        """
        if self._resolved_path is not None:
            return self._resolved_path

        explicit = self.settings.segmentation_model_path
        if explicit:
            p = Path(explicit)
            if p.exists():
                self._resolved_path = p
            return self._resolved_path

        if self.settings.model_source.lower() == "azure_blob":
            try:
                model_dir = self._download_folder_from_blob()
                cfg_file = model_dir / _SEG_CONFIG_FILE
                if cfg_file.exists():
                    self._cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                checkpoint = model_dir / self.settings.segmentation_model_file
                if checkpoint.exists():
                    self._resolved_path = checkpoint
            except Exception:
                pass

        return self._resolved_path

    def _download_folder_from_blob(self) -> Path:
        """Download all blobs under the segmentation model prefix into the local cache."""
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        if not self.settings.azure_storage_account_url:
            raise ValueError("AZURE_STORAGE_ACCOUNT_URL is required for blob segmentation model.")
        if not self.settings.azure_blob_container:
            raise ValueError("AZURE_BLOB_CONTAINER is required for blob segmentation model.")

        name = self.settings.segmentation_model_name
        version = self.settings.segmentation_model_version
        prefix_parts = [self.settings.azure_blob_prefix.strip("/"), name.strip("/"), version.strip("/")]
        blob_prefix = "/".join(p for p in prefix_parts if p)

        target_dir = Path(self.settings.model_cache_dir) / name / version
        target_dir.mkdir(parents=True, exist_ok=True)

        credential = DefaultAzureCredential()
        service = BlobServiceClient(
            account_url=self.settings.azure_storage_account_url,
            credential=credential,
        )
        container = service.get_container_client(self.settings.azure_blob_container)

        blobs = list(container.list_blobs(name_starts_with=blob_prefix))
        if not blobs:
            raise FileNotFoundError(
                f"No segmentation artifacts found in blob container "
                f"'{self.settings.azure_blob_container}' for prefix '{blob_prefix}'."
            )

        for blob in blobs:
            relative = blob.name[len(blob_prefix):].lstrip("/")
            if not relative:
                continue
            dest = target_dir / PurePosixPath(relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                container.get_blob_client(blob.name).download_blob().readinto(fh)

        return target_dir

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True

        model_path = self._resolve_model_path()
        if model_path is None:
            return False

        # _cfg may already be set from segmentation_config.json in blob;
        # fall back to the env var if not.
        if not self._cfg:
            try:
                self._cfg = json.loads(self.settings.segmentation_model_config_json or "{}")
            except json.JSONDecodeError:
                pass

        try:
            import torch  # noqa: F401
        except ImportError:
            return False

        import torch

        try:
            self._model = self._build_model()
            self._load_checkpoint(self._model, model_path)
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

    def _load_checkpoint(self, model, path: Path) -> None:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        for prefix in ("model.", "net.", ""):
            stripped = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
            res = model.load_state_dict(stripped, strict=False)
            if len(res.missing_keys) < len(model.state_dict()) // 2:
                return
        model.load_state_dict(sd, strict=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
                keys=["image"], a_min=-1024, a_max=3071, b_min=0, b_max=1, clip=True,
            ))
        tfms.append(ToTensord(keys=["image"]))

        img = Compose(tfms)({"image": tmp_path})["image"].unsqueeze(0)
        device = torch.device(self.settings.segmentation_device)
        img = img.to(device)

        inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=1, overlap=overlap, mode="gaussian",
        )
        with torch.no_grad():
            logits = inferer(img, self._model)

        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
