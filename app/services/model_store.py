from __future__ import annotations

from pathlib import Path

from app.config import get_settings


class ModelStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_path(self, model_name: str, model_version: str) -> Path:
        model_source = self.settings.model_source.lower()

        if model_source == "local":
            return self._resolve_local_model_path(model_name=model_name, model_version=model_version)

        if model_source == "azure_blob":
            return self._resolve_blob_mounted_model_path(model_name=model_name, model_version=model_version)

        raise ValueError(f"Unsupported MODEL_SOURCE '{self.settings.model_source}'. Use 'local' or 'azure_blob'.")

    def _resolve_local_model_path(self, model_name: str, model_version: str) -> Path:
        model_dir = Path(self.settings.model_local_dir) / model_name / model_version
        return model_dir

    def _resolve_blob_mounted_model_path(self, model_name: str, model_version: str) -> Path:
        model_dir = self.cache_dir / model_name / model_version
        return model_dir
