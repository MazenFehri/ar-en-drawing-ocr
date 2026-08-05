import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from models.elements import TextElement, BBox, LLMCorrection, ComplexShapeElement
from pipeline.llm_corrector import (
    apply_corrections, _build_prompt, label_complex_shapes, _call_with_retry,
    _build_system_prompt, _chunk_language, _introduces_unseen_script,
)

# Same range utils.bidi.is_arabic/detect_language use to identify Arabic-block
# characters. Tests assert against this range directly (not against specific
# glossary words) so a test can't pass by accident while some other Arabic
# term leaks through.
_ARABIC_BLOCK = ("؀", "ۿ")


def _has_any_arabic_char(text: str) -> bool:
    return any(_ARABIC_BLOCK[0] <= c <= _ARABIC_BLOCK[1] for c in text)


def _fake_page_bytes(w=200, h=200):
    """A real, decodable JPEG so cv2.imdecode in apply_corrections succeeds.
    Content doesn't matter for these tests — the LLM call itself is mocked —
    only that it's a valid image cv2 can crop out of."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


FAKE_PAGE = _fake_page_bytes()


def make_text_el(id_, text, conf, x=0.0, y=0.0, w=0.1, h=0.02, container_shape_id=None,
                 margin_column=False):
    return TextElement(
        id=id_,
        bbox=BBox(x=x, y=y, w=w, h=h),
        content=text,
        language="english" if text.isascii() else "arabic",
        confidence=conf,
        container_shape_id=container_shape_id,
        margin_column=margin_column,
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


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector.TOTAL_LLM_BUDGET_SECONDS", 1.0)
@patch("pipeline.llm_corrector._get_client")
def test_hard_timeout_caps_wall_clock_even_if_call_never_returns(mock_client_fn):
    """The actual bug from the live Docker run: a call already in flight can't
    be interrupted by a deadline check that only runs *between* attempts, and
    REQUEST_TIMEOUT_SECONDS is a read timeout that a provider trickling
    keep-alive bytes can hold open indefinitely. Real elapsed time, real sleep
    in the mocked call (not mocked time.sleep) — this is the only way to prove
    a wall-clock guarantee rather than just asserting a constant exists.
    """
    import time as _time

    def _blocks_far_past_the_budget(*args, **kwargs):
        _time.sleep(4)
        raise AssertionError("must never complete before the hard timeout fires")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _blocks_far_past_the_budget
    mock_client_fn.return_value = mock_client

    start = _time.monotonic()
    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
    )
    elapsed = _time.monotonic() - start

    assert elapsed < 3.0, f"hard ceiling not enforced: took {elapsed:.1f}s against a 1s budget / 4s block"
    assert status["state"] == "failed"
    assert status["reason"] == "timed_out"
    assert result[0].content == "entrnce"  # graceful degradation: original text kept


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector.TOTAL_LLM_BUDGET_SECONDS", 0.5)
@patch("pipeline.llm_corrector.MAX_WORDS_PER_CALL", 1)
@patch("pipeline.llm_corrector._get_client")
def test_budget_is_shared_across_chunks_not_reset_per_chunk(mock_client_fn):
    """apply_corrections can split flagged words into several chunks; each
    used to get its own fresh TOTAL_LLM_BUDGET_SECONDS deadline, so a page
    with N chunks could spend up to N x the documented budget (this is
    plausibly what turned a 60s budget into the >120s hang seen in the live
    Docker run). One deadline computed once and threaded through every chunk
    closes that gap: only the first chunk should pay for the hard-timeout
    wait; the rest should see the shared deadline already gone and give up
    immediately without attempting a call.

    Real sleep in the mocked call (not mocked time.sleep) is required to prove
    this: a mock that fails instantly can't distinguish "one shared deadline"
    from "one fresh deadline per chunk," since both would finish fast.
    """
    import time as _time

    def _blocks_past_the_budget(*args, **kwargs):
        _time.sleep(0.6)  # longer than the 0.5s budget
        raise _rate_limit_exc()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _blocks_past_the_budget
    mock_client_fn.return_value = mock_client

    # MAX_WORDS_PER_CALL=1 forces 3 flagged words into 3 separate chunks.
    elements = [
        make_text_el("t0", "entrnce", 0.4),
        make_text_el("t1", "kitcen", 0.3),
        make_text_el("t2", "غرفة", 0.2),
    ]

    start = _time.monotonic()
    result, status = apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)
    elapsed = _time.monotonic() - start

    # One chunk's hard-timeout wait (~0.5s) plus two chunks that see the
    # already-expired shared deadline and skip the call entirely. Without the
    # fix each of the 3 chunks would pay its own ~0.5s wait: ~1.5s total.
    assert elapsed < 1.0, f"budget reset per chunk: took {elapsed:.1f}s for 3 chunks on a 0.5s budget"
    assert status["state"] == "failed"
    assert status["reason"] == "timed_out"
    assert [e.content for e in result] == ["entrnce", "kitcen", "غرفة"]


# --- Fix 1: one shared deadline across both LLM stages ---------------------

@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector._get_client")
def test_shared_deadline_is_respected_by_both_functions(mock_client_fn):
    """pipeline/__init__.py computes ONE deadline before stage 6 and passes the
    same value into both apply_corrections and label_complex_shapes. If
    label_complex_shapes quietly computed its own fresh TOTAL_LLM_BUDGET_SECONDS
    instead of honouring the deadline it was handed, it would get a full budget
    of its own even though the shared one is already gone -- this is exactly
    the 60s -> 120s doubling Fix 1 closes. Real sleep, not mocked time.sleep:
    a mock that fails instantly can't distinguish "shared deadline" from
    "fresh deadline per call".
    """
    import time as _time

    def _blocks_past_the_budget(*args, **kwargs):
        _time.sleep(0.6)  # longer than the 0.5s shared budget below
        raise _rate_limit_exc()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _blocks_past_the_budget
    mock_client_fn.return_value = mock_client

    shared_deadline = _time.monotonic() + 0.5

    # First call burns (most of) the shared deadline on its one hard-timeout wait.
    text_result, text_status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75,
        deadline=shared_deadline,
    )
    assert text_status["state"] == "failed"
    mock_client.chat.completions.create.assert_called_once()

    # Second call gets the SAME (now-expired) deadline -- it must give up
    # immediately rather than computing a fresh full budget of its own.
    start = _time.monotonic()
    shape_result, shape_status = label_complex_shapes(
        [make_shape_el("s0")], {"s0": b"fake-png-bytes"}, deadline=shared_deadline,
    )
    elapsed = _time.monotonic() - start

    assert elapsed < 0.3, f"label_complex_shapes got a fresh budget: took {elapsed:.1f}s"
    assert shape_status["state"] == "failed"
    assert shape_status["reason"] == "timed_out"
    # Still only the one call from the text-correction stage -- shape labelling
    # never even reached the provider.
    mock_client.chat.completions.create.assert_called_once()


# --- Fix 2: a single attempt is capped below the whole remaining budget ----

@patch("pipeline.llm_corrector.PER_ATTEMPT_CAP_SECONDS", 0.3)
def test_call_with_retry_never_exceeds_per_attempt_cap():
    """Even with a huge remaining budget, one attempt must not run past
    PER_ATTEMPT_CAP_SECONDS -- otherwise a single hanging model call eats the
    whole shared deadline and the 3-model fallback chain never gets a turn.
    Real sleep in the mocked call, not mocked time.sleep -- proves the actual
    wall-clock ceiling, not just that a constant exists.
    """
    import time as _time

    def _blocks_forever(*args, **kwargs):
        _time.sleep(5)
        raise AssertionError("must never complete before the per-attempt cap fires")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _blocks_forever

    huge_deadline = _time.monotonic() + 1000  # budget far larger than the cap

    start = _time.monotonic()
    with pytest.raises(TimeoutError):
        _call_with_retry(
            mock_client, "some/model", "sys prompt", [{"type": "text", "text": "hi"}],
            max_attempts=1, deadline=huge_deadline,
        )
    elapsed = _time.monotonic() - start

    assert elapsed < 1.0, f"attempt ran past PER_ATTEMPT_CAP_SECONDS: took {elapsed:.1f}s"


# --- Fix 3: a model that already failed permanently isn't retried per chunk -

@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector.MAX_WORDS_PER_CALL", 1)
@patch("pipeline.llm_corrector._get_client")
def test_permanently_failed_model_not_retried_within_call_but_fresh_call_tries_again(mock_client_fn):
    """A model that fails with a permanent reason (404 'no endpoints found
    that support image input' -> invalid_model) fails the exact same way on
    every chunk, so it must be tried once per call, not once per chunk --
    shape labelling can have up to 6 chunks, so this is real waste otherwise.

    The blacklist must be scoped to a single call, not module-level state, or
    one bad 404 would permanently disable a model for every future request --
    the second half of this test proves a fresh call tries the model again.
    """
    from openai import NotFoundError
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        404, request=request, json={"error": "no endpoints found that support image input"}
    )
    not_found_exc = NotFoundError(
        "no endpoints found that support image input", response=response,
        body={"error": "no endpoints found that support image input"},
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = not_found_exc
    mock_client_fn.return_value = mock_client

    # 3 flagged words, MAX_WORDS_PER_CALL=1 forces 3 chunks in this one call.
    elements = [
        make_text_el("t0", "entrnce", 0.4),
        make_text_el("t1", "kitcen", 0.3),
        make_text_el("t2", "غرفة", 0.2),
    ]
    result, status = apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    # Only the first chunk actually calls the provider; chunks 2 and 3 see the
    # model already blacklisted for this call and skip it without calling.
    assert mock_client.chat.completions.create.call_count == 1
    assert status["state"] == "failed"
    assert status["reason"] == "invalid_model"
    assert [e.content for e in result] == ["entrnce", "kitcen", "غرفة"]

    # A brand new call, same client mock: the model must be tried again --
    # proof the "already failed" set didn't leak past the call that built it.
    mock_client.chat.completions.create.reset_mock()
    mock_client.chat.completions.create.side_effect = not_found_exc
    result2, status2 = apply_corrections(
        [make_text_el("t0", "entrnce", 0.4)], FAKE_PAGE, confidence_threshold=0.75
    )
    assert mock_client.chat.completions.create.call_count == 1
    assert status2["reason"] == "invalid_model"


# --- Malformed / empty completion responses (the 'choices' TypeError bug) ---
# OpenRouter's gateway can return HTTP 200 with `choices: null` and an `error`
# body when an upstream provider fails but the gateway itself doesn't -- the
# production log evidence this reproduces was `TypeError: 'NoneType' object is
# not subscriptable` from indexing straight into response.choices[0].

@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "fallback/model:free")
@patch("pipeline.llm_corrector._get_client")
def test_none_choices_treated_as_model_failure_and_fallback_succeeds(mock_client_fn):
    """A None `choices` must not raise a TypeError out of apply_corrections --
    it should fail just this model and let the fallback chain proceed, exactly
    like any other classified failure."""
    bad_resp = MagicMock()
    bad_resp.choices = None
    bad_resp.model_extra = {}

    ok_choice = MagicMock()
    ok_choice.message.content = MOCK_LLM_RESPONSE
    ok_resp = MagicMock()
    ok_resp.choices = [ok_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [bad_resp, ok_resp]
    mock_client_fn.return_value = mock_client

    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
    )

    assert mock_client.chat.completions.create.call_count == 2
    assert status["state"] == "success"
    assert status["model"] == "fallback/model:free"
    corrected = [e for e in result if e.llm_correction is not None]
    assert corrected[0].llm_correction.corrected == "entrance"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector._get_client")
def test_empty_choices_list_reports_empty_response(mock_client_fn):
    """An empty list is a distinct case from None -- same TypeError-shaped bug
    (`choices[0]` on an empty list is an IndexError, not a TypeError, but
    equally unguarded), must be handled the same way."""
    bad_resp = MagicMock()
    bad_resp.choices = []
    bad_resp.model_extra = {}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = bad_resp
    mock_client_fn.return_value = mock_client

    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
    )

    assert result[0].content == "entrnce"
    assert status["state"] == "failed"
    assert status["reason"] == "empty_response"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector._get_client")
def test_none_message_content_reports_empty_response(mock_client_fn):
    """choices is present and non-empty but message.content is None -- the
    third distinct malformed-response shape."""
    choice = MagicMock()
    choice.message.content = None
    bad_resp = MagicMock()
    bad_resp.choices = [choice]
    bad_resp.model_extra = {}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = bad_resp
    mock_client_fn.return_value = mock_client

    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
    )

    assert result[0].content == "entrnce"
    assert status["state"] == "failed"
    assert status["reason"] == "empty_response"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector._get_client")
def test_error_payload_is_logged_not_discarded(mock_client_fn, caplog):
    """When OpenRouter's 200-with-error body carries the real explanation
    (provider name, upstream message), that must end up in the log instead of
    being thrown away in favour of a bare TypeError."""
    bad_resp = MagicMock()
    bad_resp.choices = None
    bad_resp.model_extra = {
        "error": {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": "google/gemma-4-31b-it:free is temporarily rate-limited upstream",
                "provider_name": "Google AI Studio",
            },
        }
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = bad_resp
    mock_client_fn.return_value = mock_client

    with caplog.at_level("WARNING", logger="pipeline.llm_corrector"):
        result, status = apply_corrections(
            [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
        )

    assert status["reason"] == "empty_response"
    assert result[0].content == "entrnce"
    assert "Provider returned error" in caplog.text
    assert "Google AI Studio" in caplog.text


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.openrouter_fallback_models", "")
@patch("pipeline.llm_corrector._get_client")
def test_real_sdk_object_with_null_choices_is_handled(mock_client_fn):
    """Not a MagicMock artifact: build an actual openai ChatCompletion the same
    way the SDK's own base client builds one from a JSON body (construct_type,
    not model_validate -- see openai._base_client._process_response). This is
    the real shape _call_with_hard_timeout receives from a live call when
    OpenRouter sends `choices: null`; model_validate would reject that body
    with a pydantic ValidationError instead, which is not the exception the
    production log shows, so this confirms construct_type is really what runs.
    """
    from openai._models import construct_type
    from openai.types.chat import ChatCompletion

    body = {
        "id": "gen-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": None,
        "error": {
            "message": "Provider returned error", "code": 429,
            "metadata": {"raw": "rate-limited upstream", "provider_name": "Google AI Studio"},
        },
    }
    bad_resp = construct_type(type_=ChatCompletion, value=body)
    assert bad_resp.choices is None  # sanity: construct_type didn't coerce/validate this away

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = bad_resp
    mock_client_fn.return_value = mock_client

    result, status = apply_corrections(
        [make_text_el("t1", "entrnce", 0.48)], FAKE_PAGE, confidence_threshold=0.75
    )

    assert status["reason"] == "empty_response"
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


# --- label_complex_shapes: visual dedup of repeated symbol crops -----------

def make_shape_el(id_, x=0.0, y=0.0, w=0.1, h=0.1):
    return ComplexShapeElement(id=id_, bbox=BBox(x=x, y=y, w=w, h=h))


def _shape_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _door_swing_crop(size=160, dx=0, dy=0):
    """A door-swing-like symbol: a corner box plus a quarter-arc, drawn with
    anti-aliasing so a small (dx, dy) shift produces the same kind of 1-2px
    crop-bound jitter real independently-detected crops of the same symbol
    would have -- not random pixel noise, which isn't representative of clean
    vector line art. Sized close to a real symbol crop (see the sample-
    drawing measurement backing SHAPE_HASH_MAX_HAMMING): at this scale a
    couple of pixels of jitter is a small fraction of the shape, exactly like
    the real measurement, whereas a tiny crop would make the same pixel
    jitter proportionally huge and is not representative."""
    img = np.full((size, size), 255, dtype=np.uint8)
    cv2.rectangle(img, (15 + dx, 15 + dy), (size - 15 + dx, size - 15 + dy), 0, 4, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (15 + dx, 15 + dy), (size - 45, size - 45), 0, 0, 90, 0, 2, lineType=cv2.LINE_AA)
    return img


def _window_crop(size=160):
    """A visually distinct symbol (a plain rectangle bisected by a line) --
    same rough aspect ratio as the door swing, but a different pattern, so
    only the hash (not the aspect-ratio gate) is what must keep it separate."""
    img = np.full((size, size), 255, dtype=np.uint8)
    cv2.rectangle(img, (15, 15), (size - 15, size - 15), 0, 4, lineType=cv2.LINE_AA)
    cv2.line(img, (15, size // 2), (size - 15, size // 2), 0, 2, lineType=cv2.LINE_AA)
    return img


def _triangle_crop(size=160):
    img = np.full((size, size), 255, dtype=np.uint8)
    pts = np.array([[size // 2, 15], [15, size - 15], [size - 15, size - 15]], dtype=np.int32)
    cv2.polylines(img, [pts], True, 0, 4, lineType=cv2.LINE_AA)
    return img


def _shape_content_image_count(mock_client) -> int:
    _, kwargs = mock_client.chat.completions.create.call_args
    return sum(1 for c in kwargs["messages"][1]["content"] if c["type"] == "image_url")


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_identical_crops_send_one_image_but_label_every_element(mock_client_fn):
    """The core optimisation: N crops of the same symbol must produce exactly
    one image in the outgoing request, but every one of the N elements must
    still come back labelled."""
    crop = _door_swing_crop()
    png = _shape_png(crop)
    elements = [make_shape_el(f"s{i}") for i in range(5)]
    crop_bytes = {el.id: png for el in elements}  # literally identical bytes

    resp = '[{"index": 0, "label": "door swing", "certainty": 0.92}]'
    mock_client = _mock_client(resp)
    mock_client_fn.return_value = mock_client

    result, status = label_complex_shapes(elements, crop_bytes)

    assert mock_client.chat.completions.create.call_count == 1  # one group -> one call
    assert _shape_content_image_count(mock_client) == 1
    assert status["state"] == "success"
    for el in result:
        assert el.llm_label == "door swing"
        assert el.llm_label_certainty == pytest.approx(0.92)


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_visually_different_crops_are_not_merged(mock_client_fn):
    """A door swing and a window must never be collapsed into one label --
    both get sent, both get their own answer."""
    elements = [make_shape_el("door"), make_shape_el("window")]
    crop_bytes = {
        "door": _shape_png(_door_swing_crop()),
        "window": _shape_png(_window_crop()),
    }
    resp = (
        '[{"index": 0, "label": "door swing", "certainty": 0.9}, '
        '{"index": 1, "label": "window", "certainty": 0.85}]'
    )
    mock_client = _mock_client(resp)
    mock_client_fn.return_value = mock_client

    result, status = label_complex_shapes(elements, crop_bytes)

    assert _shape_content_image_count(mock_client) == 2  # not deduped
    by_id = {e.id: e for e in result}
    assert by_id["door"].llm_label == "door swing"
    assert by_id["window"].llm_label == "window"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_near_identical_crop_with_minor_jitter_still_groups(mock_client_fn):
    """This repo's chosen matching rule is a small Hamming-distance tolerance
    (SHAPE_HASH_MAX_HAMMING = 3), not pure byte/hash equality -- justified in
    _group_similar_crops because on real line-art crops a 1-2px difference in
    crop bounds (the realistic source of variation between two independently
    detected instances of the same symbol) lands well inside that tolerance
    while a genuinely different symbol lands far outside it. This test proves
    that tolerance actually does its job: two crops of the "same" symbol
    generated with a 1px draw offset (simulating that crop-bound jitter) must
    still collapse into a single LLM call."""
    elements = [make_shape_el("a"), make_shape_el("b")]
    crop_bytes = {
        "a": _shape_png(_door_swing_crop(dx=0, dy=0)),
        "b": _shape_png(_door_swing_crop(dx=1, dy=0)),
    }
    resp = '[{"index": 0, "label": "door swing", "certainty": 0.9}]'
    mock_client = _mock_client(resp)
    mock_client_fn.return_value = mock_client

    result, status = label_complex_shapes(elements, crop_bytes)

    assert _shape_content_image_count(mock_client) == 1  # jitter still grouped
    by_id = {e.id: e for e in result}
    assert by_id["a"].llm_label == "door swing"
    assert by_id["b"].llm_label == "door swing"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_dedup_does_not_change_outcome_when_every_crop_is_unique(mock_client_fn):
    """No repeats in the input -> dedup is a no-op: same number of images sent
    as elements, and every element still gets the right label by index."""
    elements = [make_shape_el("door"), make_shape_el("window"), make_shape_el("triangle")]
    crop_bytes = {
        "door": _shape_png(_door_swing_crop()),
        "window": _shape_png(_window_crop()),
        "triangle": _shape_png(_triangle_crop()),
    }
    resp = (
        '[{"index": 0, "label": "door swing", "certainty": 0.9}, '
        '{"index": 1, "label": "window", "certainty": 0.8}, '
        '{"index": 2, "label": "triangle", "certainty": 0.7}]'
    )
    mock_client = _mock_client(resp)
    mock_client_fn.return_value = mock_client

    result, status = label_complex_shapes(elements, crop_bytes)

    assert _shape_content_image_count(mock_client) == 3
    by_id = {e.id: e for e in result}
    assert by_id["door"].llm_label == "door swing"
    assert by_id["window"].llm_label == "window"
    assert by_id["triangle"].llm_label == "triangle"
    assert status["state"] == "success"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_dedup_group_failure_still_degrades_gracefully_for_every_member(mock_client_fn):
    """A failed LLM call for a group must leave every member of that group
    unlabelled and return the original elements, exactly like the
    non-deduped path -- dedup must not change the failure contract."""
    from openai import RateLimitError
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={}, json={"error": "rate limited"})
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RateLimitError(
        "rate limited", response=response, body={"error": "rate limited"}
    )
    mock_client_fn.return_value = mock_client

    elements = [make_shape_el(f"s{i}") for i in range(3)]
    png = _shape_png(_door_swing_crop())
    crop_bytes = {el.id: png for el in elements}

    with patch("pipeline.llm_corrector.time.sleep"):
        result, status = label_complex_shapes(elements, crop_bytes)

    assert status["state"] == "failed"
    for el in result:
        assert el.llm_label is None
        assert el.llm_label_certainty is None


# --- shape-membership context injection (Phase 2: Img2UML-style association) ---
# See models.elements.TextElement.container_shape_id (set by
# pipeline/layout_reconstructor.py) and app.config.settings.shape_context_enabled
# for why this is off by default.

def test_build_prompt_omits_context_when_no_siblings_given():
    """The zero-cost silent path: no sibling_context_by_id arg at all (the
    settings-off case in apply_corrections) must produce the exact same prompt
    as before this feature existed."""
    elements = [make_text_el("t0", "entrnce", 0.48)]
    prompt = _build_prompt(elements)
    assert "nearby_confident_labels" not in prompt
    assert "same box/shape" not in prompt


def test_build_prompt_includes_sibling_labels_when_given():
    flagged = [make_text_el("t0", "entrnce", 0.48)]
    prompt = _build_prompt(flagged, sibling_context_by_id={"t0": ["Bank", "+id", "name"]})
    assert "nearby_confident_labels" in prompt
    assert "Bank" in prompt and "+id" in prompt
    # the model must be told these are context, not correction candidates
    assert "not things to correct" in prompt


def test_build_prompt_skips_empty_sibling_list():
    """A flagged word with a container but zero confident siblings inside it
    (measured: the common case for a small/isolated box) must not add a bare
    empty-list field or the disambiguation note to its entry."""
    flagged = [make_text_el("t0", "entrnce", 0.48)]
    prompt = _build_prompt(flagged, sibling_context_by_id={"t0": []})
    assert "nearby_confident_labels" not in prompt


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.shape_context_enabled", False)
@patch("pipeline.llm_corrector._get_client")
def test_context_disabled_by_default_sends_plain_prompt(mock_client_fn):
    """Setting defaults False -- a flagged word sharing container_shape_id with
    a confident sibling must still get the pre-Phase-2 prompt, unchanged."""
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [
        make_text_el("t0", "entrnce", 0.4, container_shape_id="shape_000"),
        make_text_el("t1", "Bank", 0.95, container_shape_id="shape_000"),
    ]
    apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    sent_text = [c["text"] for c in _sent_content(mock_client_fn.return_value) if c["type"] == "text"][0]
    assert "nearby_confident_labels" not in sent_text


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.shape_context_enabled", True)
@patch("pipeline.llm_corrector._get_client")
def test_context_enabled_sends_confident_sibling_in_same_shape(mock_client_fn):
    """The actual Phase 2 wiring: a flagged word finds its confident same-shape
    sibling's OCR text and it ends up in the outgoing prompt."""
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [
        make_text_el("t0", "entrnce", 0.4, container_shape_id="shape_000"),
        make_text_el("t1", "Bank", 0.95, container_shape_id="shape_000"),
        make_text_el("t2", "Unrelated", 0.95, container_shape_id="shape_099"),  # different shape
    ]
    apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    sent_text = [c["text"] for c in _sent_content(mock_client_fn.return_value) if c["type"] == "text"][0]
    assert "Bank" in sent_text
    assert "Unrelated" not in sent_text  # different shape, must not leak in as context


# --- guards found by measuring against a live model ------------------------------------
# All three cases below were produced by the configured model on samples/test1.jpeg, at the
# claimed certainties shown. None are hypothetical.

def test_alters_numbers_detects_every_direction():
    from pipeline.llm_corrector import _alters_numbers
    assert _alters_numbers("45", "43") is True                    # observed at certainty 1.00
    assert _alters_numbers("مجدي 27250 مي", "مجدي مي") is True     # observed: dropped
    assert _alters_numbers("غرفة النوم", "غرفة 3 النوم") is True   # invented
    assert _alters_numbers("و ١٢٣ د", "و ٤٥٦ د") is True           # Arabic-Indic too
    # Letters may change freely — that is what correction is for.
    assert _alters_numbers("فاحضرمجدي 27250 مي", "فأحضر مجدي 27250 مي") is False
    assert _alters_numbers("غرفه", "غرفة") is False


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_a_correction_that_rewrites_a_number_is_refused(mock_client_fn):
    """Observed: '45' -> '43' at certainty 1.00. A wrong digit makes a maths worksheet wrong,
    so certainty cannot be what gates this."""
    mock_client_fn.return_value = _mock_client(
        '[{"index": 0, "corrected": "43", "certainty": 1.0}]')
    result, _ = apply_corrections(
        [make_text_el("t0", "45", 0.53)], FAKE_PAGE, confidence_threshold=0.75)

    assert result[0].content == "45", "OCR's number must survive"
    assert result[0].llm_correction is None
    assert result[0].highlight is not None, "refused, so it ships for review"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_a_correction_that_drops_a_recovered_number_is_refused(mock_client_fn):
    """The worst observed case: the model returned the surrounding lines' text and took the
    27250 with it — undoing a number the two-recogniser splice had just recovered."""
    original = "فاحضرمجدي 27250 مي واحضرت رانية مبلغا يقل عن مبلغ"
    mock_client_fn.return_value = _mock_client(
        '[{"index": 0, "corrected": "فإذا أهدى مجدي بأقل ما يمكن من القطع", "certainty": 0.85}]')
    result, _ = apply_corrections(
        [make_text_el("t0", original, 0.71)], FAKE_PAGE, confidence_threshold=0.75)

    assert "27250" in result[0].content
    assert result[0].content == original


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_a_letters_only_correction_is_still_applied(mock_client_fn):
    """The guard must not block what correction exists to do."""
    mock_client_fn.return_value = _mock_client(
        '[{"index": 0, "corrected": "فأحضر مجدي 27250 مي", "certainty": 0.9}]')
    result, _ = apply_corrections(
        [make_text_el("t0", "فاحضرمجدي 27250 مي", 0.71)], FAKE_PAGE, confidence_threshold=0.75)

    assert result[0].content == "فأحضر مجدي 27250 مي"
    assert result[0].llm_correction is not None


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_a_placeholder_at_zero_certainty_never_reaches_the_document(mock_client_fn):
    """The prompt asks for an empty string when a crop can't be read. Measured, this model
    returns '<unknown>' at certainty 0.0 instead — non-empty, so the empty-string check let it
    through and the literal word '<unknown>' was written over real OCR text."""
    mock_client_fn.return_value = _mock_client(
        '[{"index": 0, "corrected": "<unknown>", "certainty": 0.0}]')
    result, _ = apply_corrections(
        [make_text_el("t0", "أمتل قود مجدي", 0.71)], FAKE_PAGE, confidence_threshold=0.75)

    assert result[0].content == "أمتل قود مجدي"
    assert result[0].llm_correction is None
    assert result[0].highlight is not None


# --- marking cells are marked, never sent ----------------------------------------------

@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_marking_cells_are_never_sent_to_the_model(mock_client_fn):
    """They are empty printed score boxes — unreadable by construction, so re-reading them
    cannot succeed. Measured on test1 they were 8 of 13 flagged items, consuming most of a
    budget the real prose lines needed."""
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [
        make_text_el("t0", "entrnce", 0.40),
        make_text_el("t1", "خا", 0.19, margin_column=True),
    ]
    apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    sent_text = [c["text"] for c in _sent_content(mock_client_fn.return_value) if c["type"] == "text"][0]
    assert "entrnce" in sent_text
    assert "خا" not in sent_text


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_an_unsent_marking_cell_still_ships_highlighted(mock_client_fn):
    """Excluded from what is *sent*, never from what is *marked*. A cell nobody could read
    must not pass as read."""
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [
        make_text_el("t0", "entrnce", 0.40),
        make_text_el("t1", "خا", 0.19, margin_column=True),
    ]
    result, _ = apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)
    assert {e.id: e.highlight for e in result}["t1"] is not None


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_a_page_of_only_marking_cells_sends_nothing_but_marks_everything(mock_client_fn):
    """The early-return path. Nothing worth sending, but the cells must still be flagged."""
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [make_text_el(f"t{i}", "خا", 0.2, margin_column=True) for i in range(3)]
    result, status = apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    assert status["reason"] == "no_flagged_words"
    assert mock_client_fn.return_value.chat.completions.create.call_count == 0
    assert all(e.highlight is not None for e in result)


# --- sentence context: the confident lines around a flagged one ------------------------
# The prose counterpart to the shape context above. See app.config.settings
# .sentence_context_enabled for why this is also off by default.

def test_build_prompt_omits_sentence_context_when_none_given():
    """The settings-off path must produce byte-identical output to before the feature."""
    prompt = _build_prompt([make_text_el("t0", "entrnce", 0.48)])
    assert "surrounding_lines" not in prompt
    assert "before and after" not in prompt


def test_build_prompt_includes_surrounding_lines_when_given():
    flagged = [make_text_el("t0", "أمتل قود", 0.48)]
    prompt = _build_prompt(flagged, sentence_context_by_id={
        "t0": ["التعليمة :", "أحسب المبلغ الذي أحضرته رانية"]})
    assert "surrounding_lines" in prompt
    assert "أحسب المبلغ الذي أحضرته رانية" in prompt
    # The scar this whole family of features carries: prompt text getting treated as an
    # answer. The instruction not to copy has to be present whenever the context is.
    assert "never copy them into your answer" in prompt


def test_build_prompt_skips_an_empty_sentence_context_list():
    prompt = _build_prompt([make_text_el("t0", "entrnce", 0.48)],
                           sentence_context_by_id={"t0": []})
    assert "surrounding_lines" not in prompt


def test_sentence_context_takes_confident_neighbours_in_reading_order():
    from pipeline.llm_corrector import _sentence_context
    elements = [
        make_text_el("t0", "line one", 0.95),
        make_text_el("t1", "line two", 0.95),
        make_text_el("t2", "flagged", 0.40),
        make_text_el("t3", "line four", 0.95),
        make_text_el("t4", "line five", 0.95),
        make_text_el("t5", "line six", 0.95),   # outside the two-line window
    ]
    context = _sentence_context(elements[2], elements, 0.75)
    assert context == ["line one", "line two", "line four", "line five"]


def test_sentence_context_excludes_unconfident_neighbours_and_margin_cells():
    """Two exclusions, both load-bearing.

    An unconfident neighbour is itself a guess — passing it as context launders a guess into
    evidence. A margin cell belongs to no sentence at all, and 'خا' as context is worse than
    no context; that exclusion is why this depends on the margin split landing first.
    """
    from pipeline.llm_corrector import _sentence_context
    elements = [
        make_text_el("t0", "alsoguessed", 0.30),
        make_text_el("t1", "خا", 0.95, margin_column=True),
        make_text_el("t2", "flagged", 0.40),
        make_text_el("t3", "solid line", 0.95),
    ]
    assert _sentence_context(elements[2], elements, 0.75) == ["solid line"]


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.sentence_context_enabled", False)
@patch("pipeline.llm_corrector._get_client")
def test_sentence_context_disabled_by_default(mock_client_fn):
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [
        make_text_el("t0", "confident neighbour", 0.95),
        make_text_el("t1", "flagd", 0.40),
    ]
    apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    sent_text = [c["text"] for c in _sent_content(mock_client_fn.return_value) if c["type"] == "text"][0]
    assert "surrounding_lines" not in sent_text


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.sentence_context_enabled", True)
@patch("pipeline.llm_corrector._get_client")
def test_sentence_context_enabled_reaches_the_outgoing_prompt(mock_client_fn):
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [
        make_text_el("t0", "confident neighbour", 0.95),
        make_text_el("t1", "flagd", 0.40),
        make_text_el("t2", "خا", 0.99, margin_column=True),
    ]
    apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    sent_text = [c["text"] for c in _sent_content(mock_client_fn.return_value) if c["type"] == "text"][0]
    assert "confident neighbour" in sent_text
    assert "خا" not in sent_text, "margin cells must never travel as sentence context"


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("app.config.settings.shape_context_enabled", True)
@patch("pipeline.llm_corrector._get_client")
def test_context_enabled_no_container_is_silent_no_op(mock_client_fn):
    """The common case per Phase 1 (most words, and every drawing with no
    detected shapes at all): no container_shape_id must not error or add a
    bare 'nearby_confident_labels': [] to the prompt."""
    mock_client_fn.return_value = _mock_client(MOCK_LLM_RESPONSE)
    elements = [make_text_el("t0", "entrnce", 0.4)]  # container_shape_id=None
    result, status = apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    sent_text = [c["text"] for c in _sent_content(mock_client_fn.return_value) if c["type"] == "text"][0]
    assert "nearby_confident_labels" not in sent_text
    assert status["state"] == "success"


# --- adaptive system prompt: no Arabic priming on an all-English chunk -----
# The measured bug: SYSTEM_PROMPT used to be one fixed constant with an Arabic
# glossary baked in, sent for every correction call regardless of what was on
# the page. On class-diagram.png (100% English UML diagram, zero Arabic
# anywhere) the model corrected the flagged word 'a' to 'غرفة' at certainty
# 1.00 -- copied straight out of that glossary. _build_system_prompt fixes
# this by only including the glossary(ies) the chunk being sent has evidence
# for.

def test_all_english_chunk_prompt_has_no_arabic_characters_at_all():
    """Assert on the Unicode character range, not on specific glossary words --
    a test that only checked "غرفة" not in prompt would pass while every other
    Arabic term (النوم، الصالة، المطبخ...) leaked through untested."""
    chunk = [
        make_text_el("t0", "a", 0.7),
        make_text_el("t1", "Y 1..", 0.69),
        make_text_el("t2", "0..", 0.71),
    ]
    prompt = _build_system_prompt(chunk)
    assert not _has_any_arabic_char(prompt)
    assert "english" in prompt.lower() or "bedroom" in prompt.lower()  # English glossary still present


def test_all_english_chunk_prompt_survives_a_stray_arabic_indic_digit():
    """Regression test for the exact live A/B scenario: PaddleOCR occasionally
    misreads a Latin digit as its Arabic-Indic lookalike ('er١', ground truth
    'user') even on an all-English page. A single stray digit in one flagged
    word must not be enough to re-invite the Arabic glossary for the whole
    chunk -- that's exactly the crack the old unconditional prompt (and a
    naive per-word ratio check) fell through."""
    chunk = [
        make_text_el("t0", "a", 0.72),
        make_text_el("t1", "er١", 0.71),
        make_text_el("t2", "Y 1..", 0.69),
        make_text_el("t3", "0..", 0.71),
    ]
    prompt = _build_system_prompt(chunk)
    assert not _has_any_arabic_char(prompt)
    assert _chunk_language(chunk) == "english"


def test_arabic_chunk_still_gets_arabic_glossary():
    """Today's behaviour (Arabic glossary present) is correct for a genuinely
    Arabic chunk -- this must not regress."""
    chunk = [make_text_el("t0", "غرفه", 0.5)]  # low-confidence misspelling
    prompt = _build_system_prompt(chunk)
    assert _has_any_arabic_char(prompt)
    assert "غرفة النوم" in prompt  # the Arabic glossary itself


def test_mixed_chunk_gets_both_glossaries():
    chunk = [make_text_el("t0", "entrnce", 0.4), make_text_el("t1", "غرفه", 0.3)]
    prompt = _build_system_prompt(chunk)
    assert _has_any_arabic_char(prompt)
    assert "bedroom" in prompt.lower()
    assert _chunk_language(chunk) == "mixed"


def test_prompt_always_includes_script_preservation_rule():
    chunk = [make_text_el("t0", "entrnce", 0.4)]
    prompt = _build_system_prompt(chunk)
    assert "script" in prompt.lower()


# --- response-side defense: reject a correction that hallucinates a script -
# the chunk gave zero evidence for -----------------------------------------

def test_introduces_unseen_script_rejects_arabic_on_english_chunk():
    """The exact measured shape of the bug: certainty 1.00, corrected text is
    Arabic, chunk had no Arabic evidence at all."""
    assert _introduces_unseen_script("english", "غرفة") is True


def test_introduces_unseen_script_allows_arabic_on_mixed_chunk():
    """Once the chunk has real Arabic evidence somewhere, script alone can no
    longer distinguish a real Arabic correction from a hallucinated one --
    this must not reject."""
    assert _introduces_unseen_script("mixed", "غرفة") is False


def test_introduces_unseen_script_allows_latin_on_arabic_chunk():
    """Deliberately NOT symmetric -- see _introduces_unseen_script's docstring.
    An Arabic-classified chunk correcting a misread into Latin characters (or
    a Latin unit like '3.5m' embedded in an Arabic label) must go through:
    OCR misreading real Arabic as Latin garbage and correction fixing it back
    is exactly the legitimate case this must not break."""
    assert _introduces_unseen_script("arabic", "user") is False
    assert _introduces_unseen_script("arabic", "3.5m") is False


def test_introduces_unseen_script_allows_english_correction_on_english_chunk():
    assert _introduces_unseen_script("english", "entrance") is False


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_hallucinated_arabic_correction_is_discarded_on_english_chunk(mock_client_fn):
    """End-to-end version of the measured bug: an all-English chunk, and the
    model answers with Arabic anyway (glossary bleed, general hallucination,
    doesn't matter which) -- the correction must be discarded, original OCR
    text kept, no llm_correction attached, exactly like any other rejected
    answer."""
    mock_client_fn.return_value = _mock_client(
        '[{"index": 0, "corrected": "غرفة", "certainty": 1.0}]'
    )

    result, status = apply_corrections([make_text_el("t0", "a", 0.72)], FAKE_PAGE, confidence_threshold=0.75)

    assert result[0].content == "a"
    assert result[0].llm_correction is None
    assert result[0].highlight is None
    assert status["state"] == "success"  # the LLM call itself succeeded; only the answer was rejected


@patch("app.config.settings.openrouter_api_key", "test-key")
@patch("pipeline.llm_corrector._get_client")
def test_legitimate_arabic_correction_on_mixed_chunk_is_applied(mock_client_fn):
    """Sanity check that the defensive check doesn't overreach: a mixed chunk
    (one English word, one genuinely Arabic word) correcting the Arabic word
    to a different, still-Arabic reading must go through unrejected."""
    elements = [
        make_text_el("t0", "entrnce", 0.4),
        make_text_el("t1", "غرفه", 0.3),  # low-confidence misspelling
    ]
    resp = (
        '[{"index": 0, "corrected": "entrance", "certainty": 0.9}, '
        '{"index": 1, "corrected": "غرفة", "certainty": 0.85}]'
    )
    mock_client_fn.return_value = _mock_client(resp)

    result, status = apply_corrections(elements, FAKE_PAGE, confidence_threshold=0.75)

    by_id = {e.id: e for e in result}
    assert by_id["t1"].llm_correction.corrected == "غرفة"
    assert status["state"] == "success"


# --- unverified low-confidence words must still be highlighted ---------------
# Highlighting used to happen only inside the "model returned a correction"
# branch, so any LLM failure shipped below-threshold OCR looking identical to
# text read at 0.99. Measured on class-diagram.png with the free tier
# rate-limited: 2 words under threshold, 0 highlighted.

def _llm_down(*_a, **_kw):
    from pipeline.llm_corrector import _LLMFailure
    raise _LLMFailure("rate_limited")


@patch("pipeline.llm_corrector._get_client", MagicMock())
@patch("pipeline.llm_corrector._request_with_fallback", side_effect=_llm_down)
def test_low_confidence_is_highlighted_when_the_llm_fails(_mock):
    els = [make_text_el("t0", "sure", 0.99), make_text_el("t1", '"^', 0.62)]
    out, status = apply_corrections(els, FAKE_PAGE, confidence_threshold=0.75)
    assert status["state"] == "failed"
    assert out[0].highlight is None, "confident text must not be marked"
    assert out[1].highlight is not None, "unverified low-confidence text shipped unmarked"


def test_low_confidence_is_highlighted_when_no_api_key():
    # apply_corrections imports settings inside the function, so patch the real
    # settings object's attribute rather than a module-level name that isn't there.
    from app.config import settings
    with patch.object(settings, "openrouter_api_key", ""):
        els = [make_text_el("t0", "sure", 0.99), make_text_el("t1", "er1", 0.71)]
        out, status = apply_corrections(els, FAKE_PAGE, confidence_threshold=0.75)
    assert status["reason"] == "not_configured"
    assert out[0].highlight is None
    assert out[1].highlight is not None


@patch("pipeline.llm_corrector._get_client", MagicMock())
@patch("pipeline.llm_corrector._request_with_fallback")
def test_word_the_model_confirmed_is_not_highlighted(mock_req):
    # Model answered and left it alone -> verified, no highlight. Distinct from
    # "no answer came back for this id", which must be highlighted.
    mock_req.return_value = (
        [{"index": 0, "corrected": "user", "certainty": 0.9}],
        {"state": "success", "reason": None, "model": "m"},
    )
    els = [make_text_el("t0", "user", 0.62)]
    out, _ = apply_corrections(els, FAKE_PAGE, confidence_threshold=0.75)
    assert out[0].content == "user"
    assert out[0].highlight is None


@patch("pipeline.llm_corrector._get_client", MagicMock())
@patch("pipeline.llm_corrector._request_with_fallback")
def test_word_the_model_never_answered_about_is_highlighted(mock_req):
    # Partial response: two flagged words, the model only answers about one.
    mock_req.return_value = (
        [{"index": 0, "corrected": "user", "certainty": 0.9}],
        {"state": "success", "reason": None, "model": "m"},
    )
    els = [make_text_el("t0", "usr", 0.62), make_text_el("t1", '"^', 0.60)]
    out, _ = apply_corrections(els, FAKE_PAGE, confidence_threshold=0.75)
    assert out[1].highlight is not None, "no answer covered t1; it must not pass as certain"
