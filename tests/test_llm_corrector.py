import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from models.elements import TextElement, BBox, LLMCorrection
from pipeline.llm_corrector import apply_corrections, _build_prompt


def _fake_page_bytes(w=200, h=200):
    """A real, decodable JPEG so cv2.imdecode in apply_corrections succeeds.
    Content doesn't matter for these tests — the LLM call itself is mocked —
    only that it's a valid image cv2 can crop out of."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


FAKE_PAGE = _fake_page_bytes()


def make_text_el(id_, text, conf, x=0.0, y=0.0, w=0.1, h=0.02):
    return TextElement(
        id=id_,
        bbox=BBox(x=x, y=y, w=w, h=h),
        content=text,
        language="english" if text.isascii() else "arabic",
        confidence=conf,
    )


def _mock_client(content):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


def _sent_content(mock_client):
    """The user-message content list actually handed to the chat completion."""
    _, kwargs = mock_client.chat.completions.create.call_args
    return kwargs["messages"][1]["content"]


def test_unchanged_word_is_not_highlighted():
    """The model echoing a word back unchanged is a confirmation, not a correction."""
    el = make_text_el("t0", "GROUND", 0.92)
    resp = '[{"index": 0, "corrected": "GROUND", "certainty": 1.0}]'
    client = _mock_client(resp)

    with patch("pipeline.llm_corrector._get_client", return_value=client), \
         patch("app.config.settings.openrouter_api_key", "test-key"):
        out, status = apply_corrections([el], FAKE_PAGE, confidence_threshold=0.99)

    assert out[0].highlight is None
    assert out[0].llm_correction is None
    assert out[0].content == "GROUND"
    assert status["state"] == "success"


MOCK_LLM_RESPONSE = '[{"index": 0, "corrected": "entrance", "certainty": 0.97}]'
LOW_CERTAINTY_RESPONSE = '[{"index": 0, "corrected": "entrance", "certainty": 0.4}]'


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_corrects_flagged_word(mock_client_fn):
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)

    elements = [
        make_text_el("t0", "غرفة", 0.91),
        make_text_el("t1", "entrnce", 0.48),
    ]
    result, status = apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)
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

    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75)
    assert result[0].highlight == "yellow"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_low_certainty_gets_red(mock_client_fn):
    mock_client_fn.return_value = _mock_client(LOW_CERTAINTY_RESPONSE)

    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75)
    assert result[0].highlight == "red"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_empty_correction_is_not_applied(mock_client_fn):
    """A model that fails to answer and returns an empty 'corrected' string must
    not blank out real OCR text — that's not a correction, it's a bad response."""
    mock_client_fn.return_value = _mock_client(
        '[{"index": 0, "corrected": "", "certainty": 0.0}]'
    )

    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75)
    assert result[0].content == "entrnce"
    assert result[0].llm_correction is None
    assert status["state"] == "success"


@patch("pipeline.llm_corrector._get_client")
def test_high_confidence_not_sent(mock_client_fn):
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    result, status = apply_corrections([make_text_el("t0", "entrance", 0.95)], FAKE_PAGE, confidence_threshold=0.75)
    mock_client.chat.completions.create.assert_not_called()
    assert status == {"state": "not_attempted", "reason": "no_flagged_words", "model": None}


@patch("app.config.settings.openrouter_api_key", "")
@patch("pipeline.llm_corrector._get_client")
def test_missing_api_key_reports_not_configured_without_calling(mock_client_fn):
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    result, status = apply_corrections([make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75)
    mock_client_fn.assert_not_called()
    assert result[0].content == "entrnce"
    assert status == {"state": "not_attempted", "reason": "not_configured", "model": None}


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_undecodable_image_reports_parse_error(mock_client_fn):
    """Garbage image bytes can't be cropped at all — this must degrade exactly
    like any other LLM failure, never raise."""
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], b"not a real image", confidence_threshold=0.75
    )
    mock_client.chat.completions.create.assert_not_called()
    assert result[0].content == "entrnce"
    assert status["state"] == "failed"
    assert status["reason"] == "parse_error"


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
            [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
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
            [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
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
            [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
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
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
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
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
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


# --- Crop-based sending (the actual fix under test) -------------------------

@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_crops_are_generated_and_sent(mock_client_fn):
    """The whole point of this change: send one cropped image per flagged word
    (plus one trailing text part), not the whole page."""
    page = _fake_page_bytes(400, 300)
    elements = [
        make_text_el("t0", "entrnce", 0.4, x=0.1, y=0.1, w=0.2, h=0.05),
        make_text_el("t1", "غرفة", 0.3, x=0.5, y=0.6, w=0.2, h=0.05),
        make_text_el("t2", "kitcen", 0.5, x=0.05, y=0.8, w=0.15, h=0.05),
    ]
    resp = (
        '[{"index": 0, "corrected": "entrance", "certainty": 0.9}, '
        '{"index": 1, "corrected": "غرفة", "certainty": 0.8}, '
        '{"index": 2, "corrected": "kitchen", "certainty": 0.7}]'
    )
    mock_client = _mock_client(resp)
    mock_client_fn.return_value = mock_client

    result, status = apply_corrections(elements, page, confidence_threshold=0.75)

    content = _sent_content(mock_client)
    image_parts = [c for c in content if c["type"] == "image_url"]
    text_parts = [c for c in content if c["type"] == "text"]
    assert len(image_parts) == 3  # one crop per flagged word, no whole-page image
    assert len(text_parts) == 1
    for part in image_parts:
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert status["state"] == "success"
    by_id = {e.id: e for e in result}
    assert by_id["t2"].llm_correction.corrected == "kitchen"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_index_keyed_application_works_out_of_order(mock_client_fn):
    """The model is free to answer in any order — indices, not list position or
    string matching, must decide which element each answer belongs to."""
    page = _fake_page_bytes(300, 200)
    elements = [
        make_text_el("t0", "entrnce", 0.4, x=0.1, y=0.1, w=0.2, h=0.05),
        make_text_el("t1", "kitcen", 0.3, x=0.5, y=0.5, w=0.2, h=0.05),
    ]
    # Response lists index 1 before index 0.
    resp = (
        '[{"index": 1, "corrected": "kitchen", "certainty": 0.9}, '
        '{"index": 0, "corrected": "entrance", "certainty": 0.85}]'
    )
    mock_client_fn.return_value = _mock_client(resp)

    result, status = apply_corrections(elements, page, confidence_threshold=0.75)

    by_id = {e.id: e for e in result}
    assert by_id["t0"].llm_correction.corrected == "entrance"
    assert by_id["t1"].llm_correction.corrected == "kitchen"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_duplicate_word_different_confidence_corrected_independently(mock_client_fn):
    """The same OCR text appearing twice at different confidences (and different
    positions) must not collapse into a single correction keyed by string."""
    page = _fake_page_bytes(300, 200)
    elements = [
        make_text_el("t0", "entrnce", 0.4, x=0.1, y=0.1, w=0.2, h=0.05),
        make_text_el("t1", "entrnce", 0.6, x=0.5, y=0.6, w=0.2, h=0.05),
    ]
    resp = (
        '[{"index": 0, "corrected": "entrance", "certainty": 0.9}, '
        '{"index": 1, "corrected": "entrance hall", "certainty": 0.65}]'
    )
    mock_client_fn.return_value = _mock_client(resp)

    result, status = apply_corrections(elements, page, confidence_threshold=0.75)

    by_id = {e.id: e for e in result}
    assert by_id["t0"].llm_correction.corrected == "entrance"
    assert by_id["t0"].llm_correction.certainty == pytest.approx(0.9)
    assert by_id["t1"].llm_correction.corrected == "entrance hall"
    assert by_id["t1"].llm_correction.certainty == pytest.approx(0.65)
    assert status["state"] == "success"
