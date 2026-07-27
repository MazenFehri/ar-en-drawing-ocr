from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openrouter_api_key: str = ""   # Must be set in .env for LLM correction to work
    openrouter_model: str = "google/gemma-4-26b-a4b-it:free"
    # Comma-separated fallback model IDs (env: OPENROUTER_FALLBACK_MODELS), tried in
    # order when the primary is rate-limited, retired, or rejected. Kept as a plain
    # str field on purpose: a bare list[str] field makes pydantic-settings try to
    # JSON-parse the env var, which is a common footgun for a simple CSV list. Parse
    # it via the openrouter_fallback_model_list property below.
    openrouter_fallback_models: str = "google/gemma-4-31b-it:free,nvidia/nemotron-nano-12b-v2-vl:free"
    confidence_threshold: float = 0.75
    label_shapes: bool = True
    database_url: str = "postgresql://ocr:ocr@localhost:5432/ocr_db"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")

    @property
    def openrouter_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.openrouter_fallback_models.split(",") if m.strip()]

settings = Settings()
