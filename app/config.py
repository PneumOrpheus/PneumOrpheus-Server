from functools import lru_cache
import json

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "PneumOrpheus Inference Server"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    app_log_level: str = "INFO"

    inference_api_key: str | None = Field(default=None, alias="INFERENCE_API_KEY")

    default_model_name: str = "pneumorpheus-primary"
    default_model_version: str = "v1"
    model_file_name: str | None = None

    model_source: str = Field(default="local", description="local|azure_blob")
    model_local_dir: str = "./models"

    azure_storage_account_url: str | None = None
    azure_blob_container: str | None = None
    azure_blob_prefix: str = "models"

    model_cache_dir: str = "./model-cache"
    model_device: str = "cpu"
    model_input_shape: str = "1,1,96,96,96"
    model_class_labels: str = "Adenocarcinoma,Small Cell Carcinoma,Squamous Cell Carcinoma"
    model_default_tnm_stage: str = "T1N0M0"
    model_factory_path: str | None = None
    model_preprocessor_factory_path: str | None = None
    model_factory_kwargs_json: str = "{}"

    # Segmentation model (optional — used for tumor-positive MIL slice selection)
    segmentation_model_path: str | None = None
    segmentation_model_config_json: str = "{}"
    segmentation_device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def parse_model_input_shape(settings: Settings) -> tuple[int, ...]:
    parts = [part.strip() for part in settings.model_input_shape.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def parse_class_labels(settings: Settings) -> list[str]:
    return [label.strip() for label in settings.model_class_labels.split(",") if label.strip()]


def parse_factory_kwargs(settings: Settings) -> dict:
    try:
        data = json.loads(settings.model_factory_kwargs_json or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("Invalid MODEL_FACTORY_KWARGS_JSON value.") from error

    if not isinstance(data, dict):
        raise ValueError("MODEL_FACTORY_KWARGS_JSON must decode to an object.")

    return data
