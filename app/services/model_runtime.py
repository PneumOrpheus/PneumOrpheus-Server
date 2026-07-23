from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.config import (
    get_settings,
    parse_class_labels,
    parse_factory_kwargs,
    parse_model_input_shape,
)


class ModelRuntimeError(RuntimeError):
    pass


class ModelRuntime:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.settings = get_settings()
        self._torch = None
        self._model = None
        self._metadata: dict[str, Any] = {}
        self._last_input_tensor: Any = None
        self._is_sclc_factory_model = False
        self._last_ct_volume: Any = None
        self._last_ct_affine: Any = None
        self._last_mil_indices: Any = None

    def _import_from_path(self, target: str) -> Any:
        if ":" not in target:
            raise ModelRuntimeError(f"Invalid import path '{target}'. Expected format module.submodule:object")
        module_name, object_name = target.split(":", 1)
        module = import_module(module_name)
        return getattr(module, object_name)

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.model_path / "model_config.json"
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ModelRuntimeError(f"Invalid JSON in metadata file {metadata_path}") from error

    def _resolve_model_file(self, metadata: dict[str, Any]) -> Path:
        configured_file = self.settings.model_file_name or metadata.get("model_file")
        if configured_file:
            candidate = self.model_path / configured_file
            if not candidate.exists():
                raise ModelRuntimeError(f"Configured model file not found: {candidate}")
            return candidate

        candidates = sorted(
            list(self.model_path.rglob("*.pth"))
            + list(self.model_path.rglob("*.pt"))
            + list(self.model_path.rglob("*.ckpt"))
        )
        if not candidates:
            raise ModelRuntimeError(f"No .pth/.pt/.ckpt model files found in {self.model_path}")
        return candidates[0]

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._torch is not None:
            return

        try:
            import torch  # type: ignore
        except Exception as error:
            raise ModelRuntimeError(
                "PyTorch is not installed. Add torch (and optional monai) to runtime dependencies."
            ) from error

        self._torch = torch
        self._metadata = self._load_metadata()
        model_file = self._resolve_model_file(self._metadata)

        device = self.settings.model_device
        map_location = torch.device(device)

        is_torchscript = bool(self._metadata.get("torchscript", False))
        if is_torchscript or model_file.suffix == ".pt":
            model = torch.jit.load(str(model_file), map_location=map_location)
            model.eval()
            self._model = model
            return

        model_factory_path = self.settings.model_factory_path or self._metadata.get("model_factory")

        if model_factory_path:
            factory = self._import_from_path(model_factory_path)
            if not callable(factory):
                raise ModelRuntimeError(f"Model factory '{model_factory_path}' is not callable.")

            factory_kwargs = parse_factory_kwargs(self.settings)
            metadata_kwargs = self._metadata.get("model_factory_kwargs")
            if isinstance(metadata_kwargs, dict):
                factory_kwargs.update(metadata_kwargs)

            model = factory(**factory_kwargs)

            checkpoint = torch.load(str(model_file), map_location=map_location, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            model.to(map_location)
            model.eval()
            self._model = model
            self._is_sclc_factory_model = True
            return

        checkpoint = torch.load(str(model_file), map_location=map_location, weights_only=False)
        if hasattr(checkpoint, "eval"):
            checkpoint.eval()
            self._model = checkpoint
            return

        raise ModelRuntimeError(
            "Unable to initialize model. Provide model_config.json with 'model_factory' or MODEL_FACTORY_PATH."
        )

    def _make_input_tensor(self, file_bytes: bytes, modality: str, tumor_mask=None):
        torch = self._torch
        assert torch is not None

        preprocessor_path = self.settings.model_preprocessor_factory_path or self._metadata.get("preprocessor_factory")
        if preprocessor_path:
            preprocessor = self._import_from_path(preprocessor_path)
            if not callable(preprocessor):
                raise ModelRuntimeError(f"Preprocessor '{preprocessor_path}' is not callable.")
            return preprocessor(
                file_bytes=file_bytes,
                modality=modality,
                metadata=self._metadata,
                tumor_mask=tumor_mask,
            )

        shape = parse_model_input_shape(self.settings)
        return torch.zeros(shape), {}

    def _extract_prediction(self, output: Any) -> tuple[int, float]:
        """Legacy single-output extraction for non-SCLC models."""
        torch = self._torch
        assert torch is not None

        if isinstance(output, (tuple, list)) and output:
            tensor = output[0]
        else:
            tensor = output

        if not hasattr(tensor, "shape"):
            raise ModelRuntimeError("Model output is not tensor-like.")

        if tensor.dim() > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        probs = torch.softmax(tensor, dim=1)
        confidence, prediction_index = torch.max(probs, dim=1)
        return int(prediction_index.item()), float(confidence.item())

    def _extract_sclc_outputs(self, output: Any, class_labels: list[str]) -> dict[str, Any]:
        """Parse the (cls_logits, [seg_logits], [box_pred]) tuple from SCLC models.

        MIL with FPN returns a 3-tuple when return_segmentation=True:
          cls_logits  : (B, num_classes)
          seg_logits  : (B, N, 1, H, W)  — per-instance sigmoid-able masks
          box_pred    : (B, N, 4)         — per-instance normalised bbox [x0,y0,x1,y1]

        Non-FPN path returns a 2-tuple: (cls_logits, seg).
        """
        torch = self._torch
        assert torch is not None

        if isinstance(output, (tuple, list)):
            cls_logits = output[0]
            seg_logits = output[1] if len(output) > 1 else None
            box_pred   = output[2] if len(output) > 2 else None
        else:
            cls_logits = output
            seg_logits = None
            box_pred   = None

        # ---- classification ----
        probs = torch.softmax(cls_logits, dim=1)  # (B, C)
        confidence, pred_idx = torch.max(probs, dim=1)
        predicted_index = int(pred_idx[0].item())
        confidence_val  = float(confidence[0].item())

        predicted_type = (
            class_labels[predicted_index]
            if 0 <= predicted_index < len(class_labels)
            else (class_labels[0] if class_labels else "Unknown")
        )
        all_class_probs = {
            label: float(probs[0, i].item())
            for i, label in enumerate(class_labels)
        }

        # ---- segmentation mask (averaged over bag instances) ----
        seg_mask_np: np.ndarray | None = None
        seg_per_instance_np: np.ndarray | None = None
        if seg_logits is not None:
            sig = torch.sigmoid(seg_logits)
            if sig.ndim == 5:  # (B, N, 1, H, W) — MIL
                seg_per_instance_np = sig[0, :, 0].cpu().numpy()  # (N, H, W)
                avg = sig[0].mean(dim=0)  # (1, H, W)
            elif sig.ndim == 4:  # (B, 1, H, W) — 2D / 3D
                avg = sig[0]
            else:
                avg = sig[0]
            seg_mask_np = avg[0].cpu().numpy()  # (H, W)

        # ---- bounding box (averaged over bag instances) ----
        bbox_np: np.ndarray | None = None
        if box_pred is not None:
            if box_pred.ndim == 3:
                bbox_np = box_pred[0].mean(dim=0).cpu().numpy()
            elif box_pred.ndim == 2:
                bbox_np = box_pred[0].cpu().numpy()

        return {
            "predicted_type":   predicted_type,
            "predicted_index":  predicted_index,
            "confidence":       confidence_val,
            "all_class_probs":  all_class_probs,
            "seg_mask":         seg_mask_np,
            "seg_per_instance": seg_per_instance_np,
            "bbox":             bbox_np,
            "tnm":              self.settings.model_default_tnm_stage,
            "reasoning":        "Prediction generated by SCLC-Diagnostic multi-head model.",
        }

    def run(self, file_bytes: bytes, modality: str, tumor_mask=None) -> dict:
        self._ensure_loaded()
        torch = self._torch
        model = self._model
        assert torch is not None and model is not None
        input_tensor, extras = self._make_input_tensor(
            file_bytes=file_bytes, modality=modality, tumor_mask=tumor_mask
        )
        self._last_ct_volume = extras.get("ct_volume")
        self._last_ct_affine = extras.get("ct_affine")
        self._last_mil_indices = extras.get("mil_indices")
        device = torch.device(self.settings.model_device)
        input_tensor = input_tensor.to(device)
        self._last_input_tensor = input_tensor.detach().cpu()
        all_heads = bool(self._metadata.get("all_heads_active", False))
        with torch.no_grad():
            if all_heads:
                output = model(input_tensor, return_segmentation=True)
            else:
                output = model(input_tensor)
        class_labels = parse_class_labels(self.settings)
        if self._is_sclc_factory_model:
            return self._extract_sclc_outputs(output, class_labels)
        predicted_index, confidence = self._extract_prediction(output)
        predicted_type = (
            class_labels[predicted_index]
            if 0 <= predicted_index < len(class_labels)
            else class_labels[0] if class_labels else "Unknown"
        )
        return {
            "predicted_type":  predicted_type,
            "predicted_index": predicted_index,
            "confidence":      confidence,
            "tnm":             self.settings.model_default_tnm_stage,
            "reasoning":       "Prediction generated using loaded model artifact.",
        }

    def compute_gradcam(self, class_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self._last_input_tensor is None or self._model is None or self._torch is None:
            return None
        try:
            from app.services.sclc_gradcam import compute_gradcam_pp
            device = self._torch.device(self.settings.model_device)
            tensor = self._last_input_tensor.to(device)
            return compute_gradcam_pp(self._model, tensor, class_idx)
        except Exception:
            return None