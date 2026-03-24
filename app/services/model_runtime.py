from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
from typing import Any, Callable

from app.config import (
    get_settings,
    parse_class_labels,
    parse_factory_kwargs,
    parse_model_input_shape,
)


class ModelRuntimeError(RuntimeError):
    pass


class ModelRuntime:
    """Hook point for your real DL inference code.

    Replace `run` with model loading/inference (PyTorch, MONAI, nnU-Net, etc.).
    Keep the return structure aligned with `app/schemas.py` so `pneumorpheus-app`
    can consume responses without changes.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.settings = get_settings()
        self._torch = None
        self._model = None
        self._metadata: dict[str, Any] = {}

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

            checkpoint = torch.load(str(model_file), map_location=map_location)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            model.to(map_location)
            model.eval()
            self._model = model
            return

        checkpoint = torch.load(str(model_file), map_location=map_location)
        if hasattr(checkpoint, "eval"):
            checkpoint.eval()
            self._model = checkpoint
            return

        raise ModelRuntimeError(
            "Unable to initialize model. Provide model_config.json with 'model_factory' or MODEL_FACTORY_PATH."
        )

    def _make_input_tensor(self, file_bytes: bytes, modality: str):
        torch = self._torch
        assert torch is not None

        preprocessor_path = self.settings.model_preprocessor_factory_path or self._metadata.get("preprocessor_factory")
        if preprocessor_path:
            preprocessor = self._import_from_path(preprocessor_path)
            if not callable(preprocessor):
                raise ModelRuntimeError(f"Preprocessor '{preprocessor_path}' is not callable.")
            return preprocessor(file_bytes=file_bytes, modality=modality, metadata=self._metadata)

        shape = parse_model_input_shape(self.settings)
        return torch.zeros(shape)

    def _extract_prediction(self, output: Any) -> tuple[int, float]:
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

    def run(self, file_bytes: bytes, modality: str) -> dict:
        self._ensure_loaded()

        torch = self._torch
        model = self._model
        assert torch is not None and model is not None

        input_tensor = self._make_input_tensor(file_bytes=file_bytes, modality=modality)
        device = torch.device(self.settings.model_device)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        predicted_index, confidence = self._extract_prediction(output)
        class_labels = parse_class_labels(self.settings)
        predicted_type = (
            class_labels[predicted_index]
            if 0 <= predicted_index < len(class_labels)
            else class_labels[0]
            if class_labels
            else "Unknown"
        )

        return {
            "predicted_type": predicted_type,
            "confidence": confidence,
            "tnm": self.settings.model_default_tnm_stage,
            "reasoning": "Prediction generated using loaded model artifact and runtime forward pass.",
        }
