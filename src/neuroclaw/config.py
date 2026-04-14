"""Application settings: HF tokens, memory watermark, hemodynamic offset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Shell env overrides .env (precedence: environment > .env file)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    HF_TOKEN: str | None = Field(default=None, description="Hugging Face token (gated models)")
    HUGGINGFACEHUB_API_TOKEN: str | None = Field(
        default=None,
        description="Legacy fallback for HF token",
    )

    MAX_MEMORY_GB: float = Field(default=13.5, description="RSS watermark on 16GB Mac")
    HEMODYNAMIC_OFFSET_S: float = Field(default=5.0, description="TRIBE fMRI offset vs stimulus")
    SEED: int = Field(default=42)

    @property
    def hf_token(self) -> str | None:
        """Primary token: shell env wins over .env via pydantic-settings."""
        return self.HF_TOKEN or self.HUGGINGFACEHUB_API_TOKEN

    def require_hf_token(self) -> str:
        """Raise if no Hugging Face token (required for gated LLaMA weights)."""
        t = self.hf_token
        if not t or not str(t).strip():
            msg = (
                "HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is required for gated model weights. "
                "Set in shell or .env"
            )
            raise RuntimeError(msg)
        return str(t).strip()


def load_settings(**overrides: Any) -> Settings:
    """Load settings; optional overrides for tests."""
    return Settings(**overrides)


def project_root() -> Path:
    """Best-effort repo root (cwd)."""
    return Path.cwd()
