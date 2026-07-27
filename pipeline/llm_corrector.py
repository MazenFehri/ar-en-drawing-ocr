import base64
import json
import logging
import random
import threading
import time
import cv2
import numpy as np
from models.elements import TextElement, ComplexShapeElement, LLMCorrection, Element, BBox

logger = logging.getLogger(__name__)

SHAPE_PROMPT = """You are labelling symbols cropped from an architectural drawing.
Typical symbols: door swing, window, staircase, north arrow, section marker, grid
reference, toilet, sink, bath, bed, sofa, dining table, car, tree, elevation marker.
Each image below is one cropped symbol, given in order.
Return ONLY a JSON array, one entry per image, no explanation:
[{"index": 0, "label": "door swing", "certainty": 0.0}]"""

_client_instance = None

SYSTEM_PROMPT = """You are an expert in Arabic and English architectural drawing OCR correction.
Common terms — Arabic: غرفة النوم (bedroom), الصالة (living room), المطبخ (kitchen),
الحمام (bathroom), المدخل (entrance), الفناء (courtyard), الرواق (corridor), الدرج (stairs).
English: bedroom, bathroom, kitchen, entrance, corridor, living room, dining room, parking, balcony.
Return ONLY a valid JSON array, no explanation."""

# A page with many flagged words / crops can produce a payload large enough for a
# free-tier model to reject outright, which looks identical to a rate limit or a
# network failure from the caller's point of view. Chunk requests to keep each one
# small. Each chunk now sends one small crop per word instead of the whole page,
# so N small crops is usually *lighter* than one full-page JPEG — but a chunk of
# MAX_WORDS_PER_CALL words each upscaled to MIN_CROP_HEIGHT_PX can still add up
# (worst case ~20 base64 JPEGs). If that ever proves too large for the free tier,
# lower MAX_WORDS_PER_CALL rather than shrinking crops — legibility matters more.
MAX_WORDS_PER_CALL = 20
MAX_CROPS_PER_CALL = 8

# A flagged word cropped tight has no context and a 15-20px tall Arabic word is
# unreadable to a vision model at native size — pad and upscale before sending.
# ponytail: fixed padding/scale ceiling, not adaptive to DPI or script. Upgrade
# path: derive from median glyph height per page if this stops being good enough.
CROP_PAD_FRAC_OF_HEIGHT = 0.4  # padding added on each side, as a fraction of bbox height
MIN_CROP_HEIGHT_PX = 64        # crops shorter than this get cv2.INTER_CUBIC upscaled
MAX_CROP_UPSCALE = 8.0         # cap the scale factor so a near-zero-height bbox can't explode

# Retry budget for transient failures (rate limit / network / 5xx) on a single
# model. 3 attempts, ~1s/2s/4s base backoff with jitter, capped at MAX_BACKOFF_SECONDS
# so a rate-limited free tier can't stall a request for minutes. See _backoff_seconds.
MAX_RETRY_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 8.0

# The openai SDK's own default timeout is 600s (10 minutes) and it does its own
# silent retries (max_retries=2) on top of ours unless told not to. Both are
# exactly the kind of "hangs for minutes" behaviour the caller must not see, so
# the client below is built with an explicit timeout and max_retries=0 — our
# retry loop is the only one in control of backoff and attempt count.
#
# This is a *client-level default*, not a guarantee: httpx.Timeout (which the
# SDK sits on) only has connect/read/write/pool phases, never a total one, and
# the read phase resets on every byte received rather than counting from the
# start of the call — OpenRouter's free-tier proxies reportedly trickle
# keep-alive bytes while a request sits queued, which can hold this open past
# 30s indefinitely. See _call_with_hard_timeout for the actual wall-clock cap.
REQUEST_TIMEOUT_SECONDS = 30.0

# Per-model bounds still multiply out: 3 models x 3 attempts x 30s + backoff is
# over 5 minutes with the provider down, and correction is only an enhancement —
# nobody should wait that long for a document we can already produce. One
# deadline, computed once per call (apply_corrections / label_complex_shapes)
# and threaded through every chunk, model, and retry inside it, caps the whole
# call regardless of the maths — a page with several chunks of flagged words
# no longer gets one full budget *per chunk*.
# ponytail: the deadline is shared across every chunk *within* one call, but
# word correction and shape labelling are separate calls with separate
# deadlines — a single /process request that exercises both and has both time
# out end to end can take up to 2x this budget. Rejected merging them into one
# shared deadline: doing so needs pipeline/__init__.py (the caller, a stage
# apart from this file) to own and pass a single deadline into both calls,
# which is more machinery across a module boundary than this bug is worth.
# Upgrade path: if that 2x ever matters in practice, have pipeline/__init__.py
# compute one deadline and pass it into both apply_corrections(..., deadline=)
# and label_complex_shapes(..., deadline=).
TOTAL_LLM_BUDGET_SECONDS = 60.0


class _LLMFailure(Exception):
    """Every configured model failed. Carries the classified reason for the last
    failure so the caller can report something more useful than "it broke"."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _MalformedCompletionError(Exception):
    """Raised when a chat completion comes back HTTP 200 with nothing usable in
    it: choices missing/empty, or the first choice's message.content is None.
    OpenRouter's gateway does this when an upstream provider errors but the
    gateway itself still returns 200 with an `error` field in the body instead
    of a real completion -- indexing straight into response.choices[0] then
    raises a TypeError that has nothing to do with the actual cause. Not added
    to _call_with_retry's retryable tuple: like a 400/401 this fails the model
    once and lets the fallback chain move on rather than burning retry budget
    on a response shape that won't change on retry.
    """


def _status(state: str, reason: str | None = None, model: str | None = None) -> dict:
    """The shape returned alongside results from apply_corrections / label_complex_shapes.

    state: "not_attempted" | "success" | "failed"
    reason (when not None):
      not_attempted -> "no_flagged_words" / "no_shapes_to_label" / "not_configured"
      failed        -> "rate_limited" / "invalid_model" / "unauthorized" /
                       "network" / "server_error" / "parse_error" /
                       "empty_response" / "timed_out" / "unknown"
    model: the model ID that produced a "success", else None.
    """
    return {"state": state, "reason": reason, "model": model}


def apply_corrections(
    elements: list[Element],
    image_bytes: bytes,
    confidence_threshold: float = 0.75,
) -> tuple[list[Element], dict]:
    """Call OpenRouter LLM to correct low-confidence OCR words in the element list.

    Returns (new_elements, status) — see _status() for the exact shape. LLM
    correction is an optional enhancement: any failure (rate limit, network,
    invalid model, malformed response) degrades gracefully to the original
    elements rather than failing the request, but unlike before, the caller can
    now tell *why* nothing changed instead of guessing.
    """
    flagged = [e for e in elements
               if isinstance(e, TextElement) and e.confidence < confidence_threshold]
    if not flagged:
        return elements, _status("not_attempted", "no_flagged_words")

    from app.config import settings
    if not settings.openrouter_api_key:
        logger.warning("LLM correction skipped: OPENROUTER_API_KEY is not configured")
        return elements, _status("not_attempted", "not_configured")

    # Decode once up front — every chunk crops out of the same page array, no
    # need to re-decode per chunk. A page that fails to decode can't be cropped
    # at all, so this degrades exactly like any other LLM failure: original
    # elements back, status says why, caller never sees an exception.
    page = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if page is None:
        logger.warning("LLM correction skipped: page image could not be decoded")
        return elements, _status("failed", "parse_error")

    client = _get_client()
    # Keyed by element id, not by OCR text: the model may alter the string it
    # echoes back, and two identical words at different confidences (e.g. the
    # same label twice on a page) must be corrected independently. Index-in-chunk
    # is how the model tells us *which* image an answer belongs to; id is how we
    # turn that back into the right element regardless of duplicate content.
    correction_by_id: dict[str, dict] = {}
    last_status = _status("failed", "unknown")
    any_success = False
    # One deadline for every chunk in this call, not one per chunk — see the
    # ponytail comment on TOTAL_LLM_BUDGET_SECONDS.
    deadline = time.monotonic() + TOTAL_LLM_BUDGET_SECONDS

    for chunk in _chunks(flagged, MAX_WORDS_PER_CALL):
        content = _build_word_content(page, chunk)
        try:
            corrections, chunk_status = _request_with_fallback(client, SYSTEM_PROMPT, content, deadline)
            for item in corrections:
                if isinstance(item, dict) and "index" in item and "corrected" in item:
                    i = int(item["index"])
                    if 0 <= i < len(chunk):
                        correction_by_id[chunk[i].id] = item
            last_status = chunk_status
            any_success = True
        except _LLMFailure as exc:
            logger.warning("LLM correction chunk skipped (%s)", exc.reason)
            last_status = _status("failed", exc.reason)

    if not any_success:
        return elements, last_status

    updated = []
    for el in elements:
        if not isinstance(el, TextElement) or el.confidence >= confidence_threshold:
            updated.append(el)
            continue
        corr_data = correction_by_id.get(el.id)
        # A word the model hands back unchanged was confirmed, not corrected —
        # highlighting it would paint most of the page yellow at a high threshold.
        # An empty "corrected" value is the model failing to answer, not a real
        # correction — applying it would silently blank out real OCR text.
        if corr_data and corr_data.get("corrected") and corr_data["corrected"] != el.content:
            certainty = corr_data.get("certainty", 0.0)
            updated.append(el.model_copy(update={
                "content": corr_data["corrected"],
                "llm_correction": LLMCorrection(
                    original=el.content,
                    corrected=corr_data["corrected"],
                    certainty=certainty,
                ),
                "highlight": "yellow" if certainty >= 0.60 else "red",
            }))
        else:
            updated.append(el)
    return updated, last_status


def label_complex_shapes(
    elements: list[Element],
    crop_png_bytes: dict[str, bytes],
) -> tuple[list[Element], dict]:
    """Ask the vision model what each complex shape crop depicts.

    Returns (new_elements, status) — see _status(). Fills llm_label /
    llm_label_certainty on ComplexShapeElements. Like word correction this is
    best-effort: a provider failure leaves the labels unset rather than failing
    the request, and the returned status says why.
    """
    targets = [
        e for e in elements
        if isinstance(e, ComplexShapeElement) and e.id in crop_png_bytes
    ]
    if not targets:
        return elements, _status("not_attempted", "no_shapes_to_label")

    from app.config import settings
    if not settings.openrouter_api_key:
        logger.warning("Shape labelling skipped: OPENROUTER_API_KEY is not configured")
        return elements, _status("not_attempted", "not_configured")

    client = _get_client()
    by_id: dict[str, dict] = {}
    last_status = _status("failed", "unknown")
    any_success = False
    # One deadline for every chunk in this call, not one per chunk — see the
    # ponytail comment on TOTAL_LLM_BUDGET_SECONDS.
    deadline = time.monotonic() + TOTAL_LLM_BUDGET_SECONDS

    for start in range(0, len(targets), MAX_CROPS_PER_CALL):
        chunk = targets[start:start + MAX_CROPS_PER_CALL]
        content = _build_shape_content(chunk, crop_png_bytes)
        try:
            raw, chunk_status = _request_with_fallback(client, SHAPE_PROMPT, content, deadline)
            for item in raw:
                if isinstance(item, dict) and "index" in item and "label" in item:
                    i = int(item["index"])
                    if 0 <= i < len(chunk):
                        by_id[chunk[i].id] = item
            last_status = chunk_status
            any_success = True
        except _LLMFailure as exc:
            logger.warning("Shape labelling chunk skipped (%s)", exc.reason)
            last_status = _status("failed", exc.reason)

    if not any_success:
        return elements, last_status

    return [
        el.model_copy(update={
            "llm_label": by_id[el.id]["label"],
            "llm_label_certainty": float(by_id[el.id].get("certainty", 0.0)),
        })
        if isinstance(el, ComplexShapeElement) and el.id in by_id
        else el
        for el in elements
    ], last_status


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _crop_word(page: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop one flagged word out of the full page, with padding and upscaling.

    The model can't locate a tiny word inside a whole page (that's the bug this
    whole change fixes), so we find it for it: crop generously around the bbox
    for context, then upscale if the crop is still too small to read.
    """
    h_img, w_img = page.shape[:2]
    x, y, w, h = bbox.x * w_img, bbox.y * h_img, bbox.w * w_img, bbox.h * h_img
    pad = max(h, 1.0) * CROP_PAD_FRAC_OF_HEIGHT
    x0 = max(0, int(x - pad))
    y0 = max(0, int(y - pad))
    x1 = min(w_img, max(x0 + 1, int(x + w + pad)))
    y1 = min(h_img, max(y0 + 1, int(y + h + pad)))
    crop = page[y0:y1, x0:x1]

    crop_h = crop.shape[0]
    if 0 < crop_h < MIN_CROP_HEIGHT_PX:
        scale = min(MIN_CROP_HEIGHT_PX / crop_h, MAX_CROP_UPSCALE)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _build_word_content(page: np.ndarray, chunk: list[TextElement]) -> list[dict]:
    content = []
    for el in chunk:
        crop = _crop_word(page, el.bbox)
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            # Practically unreachable — _crop_word always returns a non-empty
            # array — but the index-in-prompt/index-in-images pairing is load
            # bearing, so on the off chance encoding fails, send a 1x1 blank
            # placeholder rather than skip and shift every later index out of sync.
            _, buf = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))
        b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": _build_prompt(chunk)})
    return content


def _build_shape_content(chunk: list[ComplexShapeElement], crop_png_bytes: dict[str, bytes]) -> list[dict]:
    content = []
    for el in chunk:
        b64 = base64.standard_b64encode(crop_png_bytes[el.id]).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": f"{len(chunk)} symbol images, in order."})
    return content


def _request_with_fallback(client, system_prompt: str, user_content: list[dict],
                            deadline: float) -> tuple[list[dict], dict]:
    """Try each configured model in order (primary, then fallbacks), retrying
    transient failures on each with backoff. Raises _LLMFailure if every model
    in the chain fails, or "timed_out" once `deadline` (a single monotonic
    clock reading shared across every chunk of the call this came from) passes.
    """
    models = _model_chain()
    last_reason = "unknown"
    for i, model in enumerate(models):
        if time.monotonic() >= deadline:
            logger.warning("LLM budget of %.0fs exhausted, giving up", TOTAL_LLM_BUDGET_SECONDS)
            raise _LLMFailure("timed_out")
        try:
            content = _call_with_retry(client, model, system_prompt, user_content, deadline=deadline)
            parsed = _parse_json_array(content)
            return parsed, _status("success", None, model=model)
        except Exception as exc:
            last_reason = _classify_error(exc)
            is_last = i == len(models) - 1
            logger.warning(
                "LLM call to %s failed (%s: %s)%s",
                model, type(exc).__name__, exc,
                "" if is_last else ", trying next fallback model",
            )
    raise _LLMFailure(last_reason)


def _call_with_retry(client, model: str, system_prompt: str, user_content: list[dict],
                      max_attempts: int = MAX_RETRY_ATTEMPTS,
                      deadline: float | None = None) -> str:
    """Call one model with bounded retry on transient failures only. A 429 (rate
    limit) or 5xx / connection error is worth retrying; a 400 (bad request /
    invalid model) or 401 (bad key) will just fail the same way again, so those
    propagate immediately and let the caller move to the next fallback model.

    The deadline is now checked before *every* attempt, not just before
    sleeping between them — previously a model's first attempt on a fresh
    fallback could still start after the budget was already gone. Each attempt
    itself runs under _call_with_hard_timeout, which is what actually stops a
    single call from blowing the whole budget once it's in flight. A hard
    timeout is not retried — it already spent its share of the budget — it
    propagates so the fallback chain can move on or give up.
    """
    from openai import RateLimitError, APIConnectionError, InternalServerError
    retryable = (RateLimitError, APIConnectionError, InternalServerError)

    delay = BASE_BACKOFF_SECONDS
    for attempt in range(1, max_attempts + 1):
        remaining = (deadline - time.monotonic()) if deadline is not None else REQUEST_TIMEOUT_SECONDS
        if remaining <= 0:
            raise TimeoutError("LLM budget exhausted before attempt")
        try:
            return _call_with_hard_timeout(client, model, system_prompt, user_content, remaining)
        except retryable as exc:
            if attempt == max_attempts:
                raise
            sleep_s = _backoff_seconds(exc, delay)
            # Don't sleep into the budget: if waiting would overrun it, give up on
            # this model now so the chain can fail fast rather than fail late.
            if deadline is not None and time.monotonic() + sleep_s >= deadline:
                raise
            logger.info(
                "LLM call to %s hit %s (attempt %d/%d), retrying in %.1fs",
                model, type(exc).__name__, attempt, max_attempts, sleep_s,
            )
            time.sleep(sleep_s)
            delay = min(delay * 2, MAX_BACKOFF_SECONDS)


# httpx (which the openai SDK sits on) only exposes phase timeouts — connect/
# read/write/pool — never a *total* one (httpx.Timeout takes exactly those
# four and nothing else). Worse, the read phase is re-armed on every byte
# received rather than counted from the start of the call: httpcore's
# _sync/http11.py issues a fresh `timeout=read_timeout` on each individual
# socket read, so a provider that trickles a byte every <30s never trips it.
# OpenRouter's free-tier proxies are reported to do exactly that while a
# request sits queued. That means no combination of client/request timeout=
# can guarantee this call returns on time — a thread.join(timeout=) is the
# only thing that actually enforces wall-clock from the caller's side.
# ponytail: a timed-out call's background thread is not killed, only
# abandoned — it keeps holding its socket until the call eventually errors out
# on its own or the process exits. daemon=True at least means it can't block
# process/interpreter shutdown (verified: a non-daemon thread here would make
# the whole process wait out the stuck call before exiting). Accepted because
# the caller returning on time matters more than one leaked thread, and the
# cost is bounded to one thread per call that's actually stuck, not unlimited.
# Upgrade path: if leaked threads ever prove to matter, switch to the SDK's
# async client + asyncio.wait_for, which can actually cancel the in-flight
# request instead of abandoning a thread.
def _call_with_hard_timeout(client, model: str, system_prompt: str, user_content: list[dict],
                             timeout_s: float) -> str:
    """Run one create() call under a hard wall-clock ceiling of timeout_s.

    Raises TimeoutError (classified by _classify_error as "timed_out") if the
    call is still running once timeout_s elapses, regardless of what the
    provider or the socket are doing.
    """
    outcome: dict = {}

    def _run():
        try:
            outcome["response"] = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                # Best-effort defense in depth: in the common (non-adversarial)
                # case this makes the SDK's own timeout fire close to on time
                # instead of at REQUEST_TIMEOUT_SECONDS. It's not what enforces
                # the ceiling below — see the comment above this function.
                timeout=min(timeout_s, REQUEST_TIMEOUT_SECONDS),
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            outcome["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"LLM call to {model} exceeded {timeout_s:.1f}s hard ceiling")
    if "error" in outcome:
        raise outcome["error"]
    response = outcome["response"]
    choices = getattr(response, "choices", None)
    message = choices[0].message if choices else None
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        detail = _extract_error_detail(response)
        raise _MalformedCompletionError(
            f"empty completion from {model}" + (f": {detail}" if detail else "")
        )
    return content


def _extract_error_detail(response) -> str | None:
    """OpenRouter can return HTTP 200 with null/empty choices and the real
    explanation (provider name, upstream status, message) in a top-level
    `error` field the SDK's ChatCompletion type doesn't declare as a field.
    The SDK's response models are pydantic with extra="allow" though, so
    unknown top-level fields survive on `model_extra` instead of being
    dropped -- that's the only place left to read this, since by the time
    _call_with_hard_timeout sees `response` the SDK has already parsed the
    HTTP body and thrown the raw bytes away.
    """
    err = (getattr(response, "model_extra", None) or {}).get("error")
    if not err:
        return None
    if not isinstance(err, dict):
        return str(err)
    meta = err.get("metadata") or {}
    bits = [err.get("message"), meta.get("provider_name"), meta.get("raw")]
    return " | ".join(str(b) for b in bits if b)


def _backoff_seconds(exc: Exception, base_delay: float) -> float:
    """OpenRouter sometimes sends Retry-After on a 429 — honour it if present and
    sane, otherwise fall back to exponential backoff with jitter. Always capped
    so one slow retry can't blow the latency budget.
    """
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, MAX_BACKOFF_SECONDS)
    return min(base_delay + random.uniform(0, base_delay * 0.5), MAX_BACKOFF_SECONDS)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if not header:
        return None
    try:
        seconds = float(header)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _classify_error(exc: Exception) -> str:
    """Map an exception to a reason code the caller can act on."""
    from openai import (
        RateLimitError, BadRequestError, NotFoundError, AuthenticationError,
        PermissionDeniedError, APIConnectionError, APITimeoutError, InternalServerError,
    )
    # Our own hard wall-clock ceiling raises plain TimeoutError; the SDK's own
    # timeout (APITimeoutError) is a subclass of APIConnectionError, not of
    # TimeoutError, so it needs its own check too — both mean the same thing
    # to the caller and must be checked before the generic APIConnectionError
    # case below, since APITimeoutError would otherwise match that first.
    if isinstance(exc, (TimeoutError, APITimeoutError)):
        return "timed_out"
    if isinstance(exc, RateLimitError):
        return "rate_limited"
    if isinstance(exc, (BadRequestError, NotFoundError)):
        return "invalid_model"
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return "unauthorized"
    if isinstance(exc, APIConnectionError):
        return "network"
    if isinstance(exc, InternalServerError):
        return "server_error"
    if isinstance(exc, json.JSONDecodeError):
        return "parse_error"
    if isinstance(exc, _MalformedCompletionError):
        return "empty_response"
    return "unknown"


def _parse_json_array(content: str) -> list[dict]:
    """Parse a JSON array from the model response, tolerating markdown fences."""
    text = (content or "").strip()
    if text.startswith("```"):
        # strip ```json ... ``` or ``` ... ``` fences
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip("`").strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _build_prompt(flagged: list[TextElement]) -> str:
    # Index-based, matching _build_shape_content's "N images, in order" convention:
    # the model answers by image position, not by echoing the string back, so a
    # word the model rewrites or a duplicate word at a different confidence still
    # keys back to the right element (see correction_by_id in apply_corrections).
    words = [
        {"index": i, "current_guess": e.content, "confidence": round(e.confidence, 3)}
        for i, e in enumerate(flagged)
    ]
    return (
        f"{len(flagged)} cropped word images, in order, each padded with a little "
        "surrounding context and upscaled if small. Below is the OCR's current guess "
        f"and confidence for each image, by index:\n{json.dumps(words, ensure_ascii=False)}\n\n"
        "Re-read each cropped image and provide the corrected reading and your certainty "
        "(0.0-1.0). If the current guess was already correct, return it unchanged. If the "
        "crop truly can't be read, return an empty string and certainty 0.0 rather than guessing.\n"
        'Return ONLY a JSON array: [{"index": 0, "corrected": "...", "certainty": 0.0}]'
    )


def _model_chain() -> list[str]:
    """Primary model first, then configured fallbacks, de-duplicated in order."""
    from app.config import settings
    chain = [settings.openrouter_model, *settings.openrouter_fallback_model_list]
    seen = set()
    out = []
    for m in chain:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _get_client():
    global _client_instance
    if _client_instance is None:
        from openai import OpenAI
        from app.config import settings
        _client_instance = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,  # we do our own bounded retry/backoff above
        )
    return _client_instance
