from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # The key, under whichever name the configured provider uses. Any OpenAI-compatible
    # endpoint works — the client is the OpenAI SDK pointed at llm_base_url below.
    #
    # The alias list is load-bearing, not a convenience: pydantic-settings forbids unknown
    # fields, so an .env carrying NVIDIA_API_KEY and no OPENROUTER_API_KEY did not merely
    # disable correction, it raised at `Settings()` construction and took the whole service
    # down at import. `extra="ignore"` below closes that failure class for good.
    openrouter_api_key: str = Field("", validation_alias=AliasChoices(
        "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "LLM_API_KEY",
    ))
    # OpenAI-compatible base URL. OpenRouter by default; NVIDIA's hosted models are
    # https://integrate.api.nvidia.com/v1 and behave the same way through the SDK.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemma-4-26b-a4b-it:free"
    # Comma-separated fallback model IDs (env: OPENROUTER_FALLBACK_MODELS), tried in
    # order when the primary is rate-limited, retired, or rejected. Kept as a plain
    # str field on purpose: a bare list[str] field makes pydantic-settings try to
    # JSON-parse the env var, which is a common footgun for a simple CSV list. Parse
    # it via the openrouter_fallback_model_list property below.
    openrouter_fallback_models: str = "google/gemma-4-31b-it:free,nvidia/nemotron-nano-12b-v2-vl:free"
    confidence_threshold: float = 0.75
    label_shapes: bool = True
    # Include confident sibling OCR text (same detected shape, e.g. same UML class
    # box) as context in a flagged word's correction prompt. Default OFF: measured
    # on class-diagram.png this only reaches 1-2 of 4 flagged words (most flagged
    # tokens there are relationship-cardinality labels that live outside any box by
    # design, not inside one), and Kanerva et al. 2025 found added prompt context
    # helps 27B/70B vision models but makes 8B-class ones *worse* — this repo's
    # configured model (nvidia/nemotron-nano-12b-v2-vl:free) sits right in that
    # smaller tier. Flip on only once a model upgrade or a bigger measured benefit
    # justifies the risk.
    shape_context_enabled: bool = False
    # Include the confident lines either side of a flagged one, in reading order, as context
    # in its correction prompt. Distinct from shape_context_enabled above: that one is
    # *spatial* (same detected box) and is empty on a prose page with no shapes, which is
    # exactly where a misread word is most recoverable from the sentence around it.
    #
    # Same risk though, and the same reason it is a flag rather than unconditional: Kanerva
    # et al. 2025 found added prompt context helps 27B/70B vision models and makes 8B-class
    # ones worse, and the configured free model sits in the smaller tier. Measure on
    # tools/eval.py --llm before turning this on for a given model.
    sentence_context_enabled: bool = False
    database_url: str = "postgresql://ocr:ocr@localhost:5432/ocr_db"
    log_level: str = "INFO"

    # extra="ignore": an unrecognised key in .env must not be able to stop the service
    # starting. It could, and did — see the alias comment above. The cost is that a typo'd
    # setting name is silently inert instead of loud, which is the better trade for a
    # component another system calls over HTTP.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def openrouter_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.openrouter_fallback_models.split(",") if m.strip()]

settings = Settings()
