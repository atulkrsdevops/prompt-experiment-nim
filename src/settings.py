from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    chat_model: str = "meta/llama-3.1-8b-instruct"
    temperature: float = 0.1
    max_tokens: int = 512
    results_dir: str = "results"

    def require_key(self) -> str:
        if not self.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set. Get a free key at https://build.nvidia.com")
        return self.nvidia_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
