from __future__ import annotations

from pathlib import PurePosixPath
from pathlib import Path
import shutil

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from app.config import get_settings


class ModelStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_path(self, model_name: str, model_version: str) -> Path:
        model_source = self.settings.model_source.lower()

        if model_source == "local":
            model_dir = self._resolve_local_model_path(model_name=model_name, model_version=model_version)
            if not model_dir.exists():
                raise FileNotFoundError(f"Local model directory not found: {model_dir}")
            return model_dir

        if model_source == "azure_blob":
            return self._sync_model_from_blob(model_name=model_name, model_version=model_version)

        raise ValueError(f"Unsupported MODEL_SOURCE '{self.settings.model_source}'. Use 'local' or 'azure_blob'.")

    def _resolve_local_model_path(self, model_name: str, model_version: str) -> Path:
        model_dir = Path(self.settings.model_local_dir) / model_name / model_version
        return model_dir

    def _resolve_blob_mounted_model_path(self, model_name: str, model_version: str) -> Path:
        model_dir = self.cache_dir / model_name / model_version
        return model_dir

    def _sync_model_from_blob(self, model_name: str, model_version: str) -> Path:
        if not self.settings.azure_storage_account_url:
            raise ValueError("AZURE_STORAGE_ACCOUNT_URL is required when MODEL_SOURCE=azure_blob.")

        if not self.settings.azure_blob_container:
            raise ValueError("AZURE_BLOB_CONTAINER is required when MODEL_SOURCE=azure_blob.")

        target_dir = self._resolve_blob_mounted_model_path(model_name=model_name, model_version=model_version)
        target_dir.mkdir(parents=True, exist_ok=True)

        prefix_parts = [
            self.settings.azure_blob_prefix.strip("/"),
            model_name.strip("/"),
            model_version.strip("/"),
        ]
        blob_prefix = "/".join([part for part in prefix_parts if part])

        credential = DefaultAzureCredential()
        service = BlobServiceClient(account_url=self.settings.azure_storage_account_url, credential=credential)
        container_client = service.get_container_client(self.settings.azure_blob_container)

        blobs = list(container_client.list_blobs(name_starts_with=blob_prefix))
        if not blobs:
            raise FileNotFoundError(
                f"No model artifacts found in blob container '{self.settings.azure_blob_container}' "
                f"for prefix '{blob_prefix}'."
            )

        existing_paths = {path for path in target_dir.rglob("*") if path.is_file()}
        downloaded_paths: set[Path] = set()

        for blob in blobs:
            blob_name = blob.name
            relative_blob_path = blob_name[len(blob_prefix) :].lstrip("/")
            if not relative_blob_path:
                continue

            destination = target_dir / PurePosixPath(relative_blob_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

            blob_client = container_client.get_blob_client(blob_name)
            with destination.open("wb") as file_obj:
                stream = blob_client.download_blob()
                stream.readinto(file_obj)

            downloaded_paths.add(destination)

        stale_paths = existing_paths - downloaded_paths
        for stale_file in stale_paths:
            stale_file.unlink(missing_ok=True)

        for path in sorted(target_dir.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                shutil.rmtree(path, ignore_errors=True)

        return target_dir
