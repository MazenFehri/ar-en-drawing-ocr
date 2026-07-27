import pytest
from unittest.mock import patch, MagicMock
from models.elements import TextElement, BBox, LLMCorrection
from pipeline.llm_corrector import apply_corrections, _build_prompt


def test_unchanged_word_is_not_highlighted():
    """The model echoing a word back unchanged is a confirmation, not a correction."""
    from unittest.mock import patch as _patch, MagicMock as _MagicMock
    from pipeline.llm_corrector import apply_corrections as _apply
    from models.elements import TextElement as _TE, BBox as _BBox

    el = _TE(id="t0", bbox=_BBox(x=0.0, y=0.0, w=0.1, h=0.02),
             content="GROUND", language="english", confidence=0.92)
    resp = _MagicMock()
    resp.choices = [_MagicMock(message=_MagicMock(
        content='[{"original": "GROUND", "corrected": "GROUND", "certainty": 1.0}]'))]
    client = _MagicMock()
    client.chat.completions.create.return_value = resp

    with _patch("pipeline.llm_corrector._get_client", return_value=client), \
         _patch("app.config.settings.openrouter_api_key", "test-key"):
        out, status = _apply([el], b"img", confidence_threshold=0.99)

    assert out[0].highlight is None
    assert out[0].llm_correction is None
    assert out[0].content == "GROUND"
    assert status["state"] == "success"


def make_text_el(id_, text, conf):
    return TextElement(
        id=id_,
        bbox=BBox(x=0.0, y=0.0, w=0.1, h=0.02),
        content=text,
        language="english" if text.isascii() else "arabic",
        confidence=conf,
    )


MOCK_LLM_RESPONSE = '[{"original": "entrnce", "corrected": "entrance", "certainty": 0.97}]'
LOW_CERTAINTY_RESPONSE = '[{"original": "entrnce", "corrected": "entrance", "certainty": 0.4}]'


def _mock_client(content):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_corrects_flagged_word(mock_client_fn):
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)

    elements = [
        make_text_el("t0", "غرفة", 0.91),
        make_text_el("t1", "entrnce", 0.48),
    ]
    result, status = apply_corrections(elements, b"fake_image", confidence_threshold=0.75)
    corrected = [e for e in result if e.llm_correction is not None]
    assert len(corrected) == 1
    assert corrected[0].llm_correction.corrected == "entrance"
    assert corrected[0].llm_correction.certainty == pytest.approx(0.97)
    assert status["state"] == "success"
    assert status["reason"] is None


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_high_certainty_gets_yellow(mock_client_fn):
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)

    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75)
    assert result[0].highlight == "yellow"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_low_certainty_gets_red(mock_client_fn):
    mock_client_fn.return_value = _mock_client(LOW_CERTAINTY_RESPONSE)

    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75)
    assert result[0].highlight == "red"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_empty_correction_is_not_applied(mock_client_fn):
    """A model that fails to answer and returns an empty 'corrected' string must
    not blank out real OCR text — that's not a correction, it's a bad response."""
    mock_client_fn.return_value = _mock_client(
        '[{"original": "entrnce", "corrected": "", "certainty": 0.0}]'
    )

    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75)
    assert result[0].content == "entrnce"
    assert result[0].llm_correction is None
    assert status["state"] == "success"


@patch("pipeline.llm_corrector._get_client")
def test_high_confidence_not_sent(mock_client_fn):
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    result, status = apply_corrections([make_text_el("t0", "entrance", 0.95)], b"fake", confidence_threshold=0.75)
    mock_client.chat.completions.create.assert_not_called()
    assert status == {"state": "not_attempted", "reason": "no_flagged_words", "model": None}


@patch("app.config.settings.openrouter_api_key", "")
@patch("pipeline.llm_corrector._get_client")
def test_missing_api_key_reports_not_configured_without_calling(mock_client_fn):
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75)
    mock_client_fn.assert_not_called()
    assert result[0].content == "entrnce"
    assert status == {"state": "not_attempted", "reason": "not_configured", "model": None}


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector._get_client")
def test_permanent_error_does_not_retry(mock_client_fn):
    """A 400 (invalid request / model) is not worth retrying — it will fail the
    same way every time. It should be tried exactly once, then reported failed."""
    from openai import BadRequestError
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": "bad model"})
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = BadRequestError(
        "bad request", response=response, body={"error": "bad model"}
    )
    mock_client_fn.return_value = mock_client

    with patch("pipeline.llm_corrector.time.sleep") as mock_sleep:
        result, status = apply_corrections(
            [make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75
        )

    assert mock_client.chat.completions.create.call_count == 1
    mock_sleep.assert_not_called()
    assert result[0].content == "entrnce"
    assert status["state"] == "failed"
    assert status["reason"] == "invalid_model"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "fallback/model:free")
@patch("pipeline.llm_corrector._get_client")
def test_rate_limit_retries_then_falls_back_to_next_model(mock_client_fn):
    """A 429 is retried (with backoff) against the primary model up to the retry
    budget; once that's exhausted, the next configured fallback model is tried
    and its success is what gets reported."""
    from openai import RateLimitError
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={}, json={"error": "rate limited"})
    rate_limit_exc = RateLimitError("rate limited", response=response, body={"error": "rate limited"})

    ok_choice = MagicMock()
    ok_choice.message.content = MOCK_LLM_RESPONSE
    ok_resp = MagicMock()
    ok_resp.choices = [ok_choice]

    mock_client = MagicMock()
    # Primary model: fails MAX_RETRY_ATTEMPTS times with 429. Fallback model: succeeds.
    mock_client.chat.completions.create.side_effect = [
        rate_limit_exc, rate_limit_exc, rate_limit_exc, ok_resp,
    ]
    mock_client_fn.return_value = mock_client

    with patch("pipeline.llm_corrector.time.sleep") as mock_sleep:
        result, status = apply_corrections(
            [make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75
        )

    assert mock_client.chat.completions.create.call_count == 4
    assert mock_sleep.call_count == 2  # 2 retries against the primary before falling back
    assert status["state"] == "success"
    assert status["model"] == "fallback/model:free"
    corrected = [e for e in result if e.llm_correction is not None]
    assert corrected[0].llm_correction.corrected == "entrance"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector._get_client")
def test_all_models_exhausted_reports_failed_with_reason(mock_client_fn):
    from openai import RateLimitError
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={}, json={"error": "rate limited"})
    rate_limit_exc = RateLimitError("rate limited", response=response, body={"error": "rate limited"})

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = rate_limit_exc
    mock_client_fn.return_value = mock_client

    with patch("pipeline.llm_corrector.time.sleep"):
        result, status = apply_corrections(
            [make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75
        )

    assert result[0].content == "entrnce"  # graceful degradation: original text kept
    assert status["state"] == "failed"
    assert status["reason"] == "rate_limited"


def _rate_limit_exc():
    from openai import RateLimitError
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={}, json={"error": "rate limited"})
    return RateLimitError("rate limited", response=response, body={"error": "rate limited"})


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector.TOTAL_LLM_BUDGET_SECONDS", 0.0)
@patch("pipeline.llm_corrector._get_client")
def test_exhausted_budget_gives_up_without_calling_provider(mock_client_fn):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _rate_limit_exc()
    mock_client_fn.return_value = mock_client

    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75
    )

    assert mock_client.chat.completions.create.call_count == 0
    assert status["state"] == "failed"
    assert status["reason"] == "timed_out"
    assert result[0].content == "entrnce"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "model-b:free,model-c:free")
@patch("pipeline.llm_corrector.TOTAL_LLM_BUDGET_SECONDS", 0.5)
@patch("pipeline.llm_corrector._get_client")
def test_budget_caps_wall_clock_across_whole_fallback_chain(mock_client_fn):
    """Real sleeps, deliberately not patched — the point is the wall clock."""
    import time as _time

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _rate_limit_exc()
    mock_client_fn.return_value = mock_client

    start = _time.monotonic()
    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75
    )
    elapsed = _time.monotonic() - start

    # Without the budget this is 3 models x 3 attempts x (1s + 2s) backoff = ~18s.
    assert elapsed < 3.0, f"budget not enforced, took {elapsed:.1f}s"
    assert status["state"] == "failed"
    assert result[0].content == "entrnce"


def test_build_prompt_contains_words():
    elements = [make_text_el("t0", "entrnce", 0.48)]
    prompt = _build_prompt(elements)
    assert "entrnce" in prompt
    assert "0.48" in prompt
