"""Minimal settings loader — secrets from environment only."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    voyage_api_key: str | None = None
    qwen_api_key: str | None = None
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.7-flash"

    @property
    def voyage_configured(self) -> bool:
        return bool(self.voyage_api_key)

    @property
    def qwen_configured(self) -> bool:
        return bool(self.qwen_api_key)


def get_settings() -> Settings:
    return Settings()
