"""Mediastinal tumour segmentation via Raidionics' pretrained CT_Tumor model.

Runs raidionics_rads_lib's "Model selection" -> CT_Tumor pipeline (which
internally also runs CT_Lungs as a required preprocessing/cropping step) on
the uploaded CT, then resamples the resulting tumour mask into the same
RAS / 1x1x2mm grid the SCLC preprocessor uses, so it can bias MIL slice
selection and back the segmentation-ROI NIfTI overlay.

Model bundles are downloaded once (lazily) from the public Raidionics-models
GitHub releases into MODEL_CACHE_DIR/raidionics/{CT_Lungs,CT_Tumor}/hr/.

raidionicsrads.compute.run_rads() never raises and uses a process-wide
config singleton, so a lock serializes calls and output-file existence is
the correctness signal (matching this class's existing fail-soft contract:
falls back to None on any error so classification never breaks).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings

_MODEL_URLS = {
    "CT_Lungs": "https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Lungs-v13.zip",
    "CT_Tumor": "https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Tumor-v13.zip",
}

_PIPELINE_JSON = {
    "1": {
        "task": "Model selection",
        "model": "CT_Tumor",
        "timestamp": 0,
        "format": "thresholding",
        "description": "Tumor segmentation model selection",
    }
}

_GZIP_MAGIC = b"\x1f\x8b"
_run_rads_lock = threading.Lock()


class SegmentationRuntime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._last_affine: Any = None
        self._last_ct_volume: Any = None
        self._models_ready: bool | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_last_affine(self) -> np.ndarray | None:
        return self._last_affine

    def get_last_ct_volume(self) -> np.ndarray | None:
        return self._last_ct_volume

    def segment(self, file_bytes: bytes) -> np.ndarray | None:
        """Return binary 3-D tumour mask (H×W×Z) in RAS / 1×1×2mm space, or None."""
        try:
            ct_volume, ct_affine = self._resample_ct_background(file_bytes)
            self._last_ct_volume = ct_volume
            self._last_affine = ct_affine
        except Exception:
            return None

        model_folder = Path(self.settings.model_cache_dir) / "raidionics"
        if not self._ensure_models_downloaded(model_folder):
            return None

        try:
            mask_path = self._run_raidionics(file_bytes, model_folder)
            if mask_path is None:
                return None
            return self._resample_mask_to_ct_grid(mask_path, ct_volume.shape, ct_affine)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Model download
    # ------------------------------------------------------------------

    def _ensure_models_downloaded(self, model_folder: Path) -> bool:
        if self._models_ready:
            return True
        try:
            for name, url in _MODEL_URLS.items():
                if (model_folder / name / "hr" / "pipeline.json").exists():
                    continue
                model_folder.mkdir(parents=True, exist_ok=True)
                zip_path = model_folder / f"{name}.zip"
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(model_folder)
                zip_path.unlink(missing_ok=True)
            self._models_ready = True
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # CT background (independent of raidionics; needed for overlay compositing
    # and because this runs before ModelRuntime's own resampled CT is ready)
    # ------------------------------------------------------------------

    def _resample_ct_background(self, file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, Spacingd
        from sclc.data.transforms import LoadNiftiWithRGBSupportd

        suffix = ".nii.gz" if file_bytes[:2] == _GZIP_MAGIC else ".nii"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(file_bytes)
            load_tfms = Compose([
                LoadNiftiWithRGBSupportd(keys=["image"]),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 2.0), mode=["bilinear"]),
            ])
            raw_img = load_tfms({"image": tmp_path})["image"]
            affine = np.asarray(raw_img.affine.cpu() if hasattr(raw_img.affine, "cpu") else raw_img.affine)
            volume = np.clip((raw_img[0].detach().cpu().numpy() + 1024.0) / (3071.0 + 1024.0), 0.0, 1.0)
            return volume, affine
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Raidionics pipeline
    # ------------------------------------------------------------------

    def _run_raidionics(self, file_bytes: bytes, model_folder: Path) -> Path | None:
        import configparser

        from raidionicsrads.compute import run_rads

        suffix = ".nii.gz" if file_bytes[:2] == _GZIP_MAGIC else ".nii"
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            input_dir = tmp_dir / "input" / "0"
            input_dir.mkdir(parents=True)
            output_dir = tmp_dir / "output"
            output_dir.mkdir()
            (input_dir / f"study{suffix}").write_bytes(file_bytes)

            pipeline_path = tmp_dir / "pipeline.json"
            pipeline_path.write_text(json.dumps(_PIPELINE_JSON), encoding="utf-8")

            # gpu_id=-1 combined with acceleration=torch crashes inside
            # raidionicsseg (tries an invalid "cuda:-1" torch device even
            # with no GPU requested) — gpu_id=0/acceleration=torch is the
            # only combination verified to work on this hardware; the ONNX
            # model itself still falls back to its CPU execution provider
            # if no CUDA-enabled onnxruntime build is installed.
            has_cuda = "cuda" in self.settings.segmentation_device.lower()
            gpu_id, acceleration = ("0", "torch") if has_cuda else ("-1", "cpu")

            config = configparser.ConfigParser()
            config["Default"] = {"task": "mediastinum_diagnosis", "trace": "False", "caller": ""}
            config["System"] = {
                "gpu_id": gpu_id,
                "acceleration": acceleration,
                "input_folder": str(tmp_dir / "input"),
                "output_folder": str(output_dir),
                "model_folder": str(model_folder),
                "pipeline_filename": str(pipeline_path),
            }
            config["Runtime"] = {
                "reconstruction_method": "thresholding",
                "reconstruction_order": "resample_first",
                "use_stripped_data": "False",
                "use_registered_data": "False",
            }
            config_path = tmp_dir / "config.ini"
            with config_path.open("w", encoding="utf-8") as fh:
                config.write(fh)

            with _run_rads_lock:
                run_rads(str(config_path))

            matches = list(output_dir.rglob("*_annotation-Tumor.nii.gz"))
            if not matches:
                return None

            # Copy out of the temp dir before it's cleaned up by the context manager.
            persisted = Path(tempfile.mkstemp(suffix=".nii.gz")[1])
            persisted.write_bytes(matches[0].read_bytes())
            return persisted

    def _resample_mask_to_ct_grid(
        self, mask_path: Path, target_shape: tuple[int, ...], target_affine: np.ndarray
    ) -> np.ndarray:
        import nibabel as nib
        from nibabel.processing import resample_from_to

        try:
            mask_img = nib.load(str(mask_path))
            resampled = resample_from_to(mask_img, (target_shape, target_affine), order=0)
            return (resampled.get_fdata() > 0.5).astype(np.uint8)
        finally:
            try:
                mask_path.unlink()
            except OSError:
                pass
