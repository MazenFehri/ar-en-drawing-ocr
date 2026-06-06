from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openrouter_api_key: str = ""   # Must be set in .env for LLM correction to work
    openrouter_model: str = "qwen/qwen2-vl-7b-instruct:free"
    confidence_threshold: float = 0.75
    label_shapes: bool = True
    database_url: str = "postgresql://ocr:ocr@localhost:5432/ocr_db"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
