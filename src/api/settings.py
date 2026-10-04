"""Typed runtime configuration loaded from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime variables RoadGuard reads at startup.

    Field names match .env.example and are case-insensitive; pydantic-settings
    maps them to attributes with lowercase names. Unknown env vars are
    ignored so the same binary can read a VM env file that carries extras.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    agent_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    roadguard_local_llm_model: str = ""
    roadguard_local_llm_ctx: int = 1024
    roadguard_local_llm_threads: int = 2
    roadguard_local_llm_max_tokens: int = 180

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_bucket: str = "roadguard-hazards"
    roadguard_image_ttl_hours: int = 48

    roadguard_passphrase_hash: str = ""
    roadguard_allowed_emails: str = ""
    google_oauth_client_id: str = ""
    roadguard_jwt_secret: str = ""
    roadguard_jwt_ttl: int = 604800

    roadguard_limit_per_min: str = "30/minute"
    roadguard_limit_per_day: str = "500/day"
    roadguard_user_limit: str = "20/minute"

    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_address: str = ""
    email_app_password: str = ""
    report_recipient: str = ""

    roadguard_db_path: str = "./roadguard.db"
    roadguard_rag_path: str = "./chroma"

    def allowed_emails_set(self) -> set[str]:
        return {
            e.strip().lower()
            for e in self.roadguard_allowed_emails.split(",")
            if e.strip()
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Clear the lru_cache so tests can override environment between runs."""
    get_settings.cache_clear()
