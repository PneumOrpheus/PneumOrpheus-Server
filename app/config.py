from functools import lru_cache

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

    model_source: str = Field(default="local", description="local|azure_blob")
    model_local_dir: str = "./models"

    azure_storage_account_url: str | None = None
    azure_blob_container: str | None = None
    azure_blob_prefix: str = "models"

    model_cache_dir: str = "./model-cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
