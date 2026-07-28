import base64
import json
import logging
import random
import threading
import time
import cv2
import numpy as np
from models.elements import TextElement, ComplexShapeElement, LLMCorrection, Element, BBox
from utils.bidi import is_arabic

logger = logging.getLogger(__name__)

SHAPE_PROMPT = """You are labelling symbols cropped from an architectural drawing.
Typical symbols: door swing, window, staircase, north arrow, section marker, grid
reference, toilet, sink, bath, bed, sofa, dining table, car, tree, elevation marker.
Each image below is one cropped symbol, given in order.
Return ONLY a JSON array, one entry per image, no explanation:
[{"index": 0, "label": "door swing", "certainty": 0.0}]"""
# Checked this prompt for the same defect as the correction prompt below
# (unconditional Arabic priming) and it doesn't have it: there's no Arabic
# anywhere in it, and a shape label is a category name ("door swing"), not a
# script-sensitive transcription of what's in the crop -- there's nothing
# here for a script to leak from. Left unchanged.

_client_instance = None

# --- word-correction system prompt: built per chunk, not one fixed constant ---
# Used to be a single module-level SYSTEM_PROMPT sent unchanged for every
# correction call, Arabic glossary and all, no matter what was on the page.
# Measured on class-diagram.png (a 100% English UML diagram, zero Arabic
# anywhere): the model corrected the single flagged word 'a' to 'غرفة' at
# certainty 1.00 -- 'غرفة' appears verbatim in the glossary below. It wasn't
# reading the crop, it was copying our own prompt. _build_system_prompt below
# only includes the glossary(ies) the chunk being sent actually has evidence
# for.
_CORRECTION_PROMPT_INTRO = (
    "You are an expert in Arabic and English architectural drawing OCR correction."
)

_ARABIC_GLOSSARY = (
    "Common terms — Arabic: غرفة النوم (bedroom), الصالة (living room), المطبخ (kitchen), "
    "الحمام (bathroom), المدخل (entrance), الفناء (courtyard), الرواق (corridor), الدرج (stairs)."
)

_ENGLISH_GLOSSARY = (
    "Common terms — English: bedroom, bathroom, kitchen, entrance, corridor, living room, "
    "dining room, parking, balcony."
)
# ponytail: still floor-plan vocabulary (bedroom/kitchen/corridor) on a pipeline
# used for technical drawings generally -- class-diagram.png is full of
# id_transaction/band_amount/+created_at, none of which this glossary helps
# with. Not touched here: the ask was to stop *unconditional Arabic* priming,
# not to fix the English glossary's domain, and swapping it for something
# UML-flavoured would just as narrowly overfit the next non-floor-plan page.
# Upgrade path: drop the glossary to a handful of script/format examples
# instead of domain vocabulary, or drop it entirely and lean on the crop +
# _SCRIPT_RULE alone.

# Requirement: an explicit rule about the reading, not an enumerated
# forbidden-character list -- phrased as "what script does the crop actually
# show", which is the question the model can actually answer by looking.
_SCRIPT_RULE = (
    "Each cropped word is written in exactly one script. Read only the characters "
    "actually visible in that crop: a word written in Latin letters must be read back "
    "in Latin letters, and a word written in Arabic must be read back in Arabic. Do not "
    "borrow letters or digits from the other script, and do not let the glossary above "
    "influence your reading unless the crop genuinely shows one of those words."
)


def _has_arabic_letters(text: str) -> bool:
    """True if `text` has an Arabic-block character that ISN'T an Arabic-Indic
    digit (٠-٩, U+0660-0669, or the Persian/Urdu variant U+06F0-06F9).

    Narrower than utils.bidi.is_arabic/detect_language, which count those
    digits as Arabic characters -- correct for their own job (bidi reordering,
    ratio-classifying a full line) but the wrong signal for "does this word
    contain Arabic vocabulary". PaddleOCR occasionally misreads a Latin digit
    as its Arabic-Indic lookalike on an all-English page (the flagged word
    'er١' on class-diagram.png -- ground truth 'user' per a separate English-
    model OCR run); is_arabic's >30%-of-string threshold trips on that single
    stray digit in a 3-character word. Excluding the digit ranges here is what
    keeps that word from swinging a whole chunk's language classification.
    """
    return any(
        "؀" <= c <= "ۿ"
        and not ("٠" <= c <= "٩")
        and not ("۰" <= c <= "۹")
        for c in text
    )


def _chunk_language(chunk: list[TextElement]) -> str:
    """'english' / 'arabic' / 'mixed' for the whole batch of flagged words in
    one call -- decides which glossary(ies) _build_system_prompt sends, and
    gates _introduces_unseen_script below.

    Deliberately NOT TextElement.language (set by layout_reconstructor's own
    detect_language(word.text) at OCR time) and not utils.bidi.detect_language
    re-run on this text either. Both are ratio-based over a single string, and
    the words landing here are, almost by definition, the short/garbled ones
    OCR wasn't confident about -- exactly where one stray character can cross
    a 30% threshold (see _has_arabic_letters). Voting _has_arabic_letters per
    word and combining across the whole chunk sidesteps that: one noisy short
    word can't flip the whole chunk to "contains Arabic", and a genuinely
    Arabic word isn't diluted away just because it's outnumbered by English
    words in the same batch (concatenating the chunk into one string and
    running detect_language on that was tried and rejected for exactly this
    reason -- a short real Arabic word can still lose a ratio vote against
    several longer English ones in the same chunk).
    """
    has_arabic = any(_has_arabic_letters(e.content) for e in chunk)
    has_english = any(c.isalpha() and ord(c) < 128 for e in chunk for c in e.content)
    if has_arabic and has_english:
        return "mixed"
    if has_arabic:
        return "arabic"
    return "english"  # also covers a chunk with no letters at all (e.g. all-digit OCR guesses)


def _build_system_prompt(chunk: list[TextElement]) -> str:
    lang = _chunk_language(chunk)
    parts = [_CORRECTION_PROMPT_INTRO]
    if lang in ("arabic", "mixed"):
        parts.append(_ARABIC_GLOSSARY)
    if lang in ("english", "mixed"):
        parts.append(_ENGLISH_GLOSSARY)
    parts.append(_SCRIPT_RULE)
    parts.append("Return ONLY a valid JSON array, no explanation.")
    return "\n".join(parts)


def _introduces_unseen_script(chunk_lang: str, corrected: str) -> bool:
    """Response-side backstop for the exact failure measured on
    class-diagram.png: _SCRIPT_RULE is a request, not a guarantee, and the
    model followed the old prompt's Arabic glossary into an unrelated English
    crop's answer at certainty 1.00. Only rejects when the WHOLE chunk was
    classified "english" (no Arabic evidence anywhere in the batch sent -- see
    _chunk_language) and the correction contains Arabic. Two deliberate
    limits, both there to avoid discarding a real correction:

    1. Skipped for "mixed" chunks. Once the batch has genuine Arabic content
       somewhere, script alone can no longer distinguish "this correction is
       real Arabic" from "this correction is hallucinated Arabic" -- so this
       stops second-guessing and the model is trusted, same as before this
       change existed.

    2. Deliberately NOT symmetric: an "arabic"-classified chunk coming back
       with Latin characters is never rejected. Arabic architectural labels
       routinely carry embedded Latin digits/units ("3.5m" -- see
       utils/bidi.py's _LTR_RUN comment), so a legitimate correction can need
       to add or fix Latin characters inside what started as an Arabic
       reading -- and OCR misreading real Arabic as Latin garbage is exactly
       the case correction exists to fix (an all-Arabic chunk producing an
       all-Latin correction would still go through). Guarding that direction
       too, with no live evidence of it actually happening, risks silently
       eating a correct answer to prevent a bug that was never observed.
       Only the direction with live evidence ('a' -> 'غرفة' at 1.00,
       'er١' -> '١م' at 0.71, 'Y 1..' -> '١..' at 0.69, all on a page with
       zero real Arabic) is guarded.
    """
    return chunk_lang == "english" and is_arabic(corrected)

# A page with many flagged words / crops can produce a payload large enough for a
# free-tier model to reject outright, which looks identical to a rate limit or a
# network failure from the caller's point of view. Chunk requests to keep each one
# small. Each chunk now sends one small crop per word instead of the whole page,
# so N small crops is usually *lighter* than one full-page JPEG — but a chunk of
# MAX_WORDS_PER_CALL words each upscaled to MIN_CROP_HEIGHT_PX can still add up
# (worst case ~20 base64 JPEGs). If that ever proves too large for the free tier,
# lower MAX_WORDS_PER_CALL rather than shrinking crops — legibility matters more.
MAX_WORDS_PER_CALL = 20

# Highlight for a below-threshold word that no model answer ever covered. Red
# rather than yellow: yellow means "corrected, and the model was fairly sure",
# which is a stronger claim than anything we can make about an unchecked word.
_UNVERIFIED_HIGHLIGHT = "red"
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

# _call_with_retry used to hand one attempt the *entire* remaining deadline as
# its hard timeout — fine when the deadline was a fresh 60s per call, but with
# TOTAL_LLM_BUDGET_SECONDS now shared across both LLM stages (see above), one
# hanging first attempt could eat the whole shared budget and the 3-model
# fallback chain would never get a turn. Cap a single attempt well below the
# budget so several models can actually be tried before the deadline is gone.
# 60s budget / 15s cap = up to 4 attempts get to run their full cap before the
# call gives up outright — enough for the primary model plus both configured
# fallbacks with a little room left for backoff, even if every one of them
# hangs to the cap rather than failing fast. Also comfortably under
# REQUEST_TIMEOUT_SECONDS (30s), so this is the number that actually governs
# _call_with_hard_timeout's ceiling in practice, not just a paper constant.
# ponytail: a flat cap, not split per-model or weighted toward the primary.
# Upgrade path: if one particular model in the chain is known to be
# consistently slow-but-eventually-correct, give it a larger per-model cap
# instead of shrinking this for everyone.
PER_ATTEMPT_CAP_SECONDS = 15.0

# Per-model bounds still multiply out: 3 models x 3 attempts x 30s + backoff is
# over 5 minutes with the provider down, and correction is only an enhancement —
# nobody should wait that long for a document we can already produce. One
# deadline, computed once per call (apply_corrections / label_complex_shapes)
# and threaded through every chunk, model, and retry inside it, caps the whole
# call regardless of the maths — a page with several chunks of flagged words
# no longer gets one full budget *per chunk*.
#
# Both functions take an optional `deadline` param. pipeline/__init__.py
# computes ONE deadline before stage 6 and passes the same value into both
# apply_corrections(..., deadline=) and label_complex_shapes(..., deadline=),
# so a /process request that exercises both stages shares one
# TOTAL_LLM_BUDGET_SECONDS budget end to end instead of paying it twice (was:
# up to 120s worst case; now: up to 60s). `deadline=None` (the default) keeps
# each function computing its own full-budget deadline, which is what every
# test and any other single-stage caller still gets.
#
# See PER_ATTEMPT_CAP_SECONDS below for why sharing one deadline across both
# stages is safe rather than letting whichever stage runs first (word
# correction) starve the other of the whole budget.
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
    deadline: float | None = None,
) -> tuple[list[Element], dict]:
    """Call OpenRouter LLM to correct low-confidence OCR words in the element list.

    Returns (new_elements, status) — see _status() for the exact shape. LLM
    correction is an optional enhancement: any failure (rate limit, network,
    invalid model, malformed response) degrades gracefully to the original
    elements rather than failing the request, but unlike before, the caller can
    now tell *why* nothing changed instead of guessing.

    deadline: a monotonic clock reading shared with another LLM call (see
    TOTAL_LLM_BUDGET_SECONDS). None (the default) computes a fresh full-budget
    deadline for this call alone, same as before this param existed.
    """
    flagged = [e for e in elements
               if isinstance(e, TextElement) and e.confidence < confidence_threshold]
    if not flagged:
        return elements, _status("not_attempted", "no_flagged_words")

    from app.config import settings
    if not settings.openrouter_api_key:
        logger.warning("LLM correction skipped: OPENROUTER_API_KEY is not configured")
        return _mark_unverified(elements, confidence_threshold), _status(
            "not_attempted", "not_configured")

    # Decode once up front — every chunk crops out of the same page array, no
    # need to re-decode per chunk. A page that fails to decode can't be cropped
    # at all, so this degrades exactly like any other LLM failure: original
    # elements back, status says why, caller never sees an exception.
    page = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if page is None:
        logger.warning("LLM correction skipped: page image could not be decoded")
        return _mark_unverified(elements, confidence_threshold), _status("failed", "parse_error")

    client = _get_client()
    # Confident sibling text sharing a flagged word's innermost detected shape
    # (see TextElement.container_shape_id) — e.g. the other attribute labels
    # already read correctly inside the same UML class box. Off by default; see
    # settings.shape_context_enabled for why. Computed once here (not per-chunk):
    # it only depends on `elements`, which chunking doesn't change.
    sibling_context_by_id: dict[str, list[str]] = {}
    if settings.shape_context_enabled:
        sibling_context_by_id = {
            el.id: _sibling_context(el, elements, confidence_threshold) for el in flagged
        }

    # Keyed by element id, not by OCR text: the model may alter the string it
    # echoes back, and two identical words at different confidences (e.g. the
    # same label twice on a page) must be corrected independently. Index-in-chunk
    # is how the model tells us *which* image an answer belongs to; id is how we
    # turn that back into the right element regardless of duplicate content.
    correction_by_id: dict[str, dict] = {}
    # Which chunk-level script classification (see _chunk_language) produced
    # each correction — needed at apply-time below to run
    # _introduces_unseen_script against the same "english"/"arabic"/"mixed"
    # verdict the prompt itself was built from, not a value recomputed later
    # from possibly-already-corrected content.
    chunk_lang_by_id: dict[str, str] = {}
    last_status = _status("failed", "unknown")
    any_success = False
    # One deadline for every chunk in this call, not one per chunk — see the
    # comment on TOTAL_LLM_BUDGET_SECONDS. Also shared with label_complex_shapes
    # when the caller passes one in (pipeline/__init__.py does).
    if deadline is None:
        deadline = time.monotonic() + TOTAL_LLM_BUDGET_SECONDS
    # Models that already failed permanently on an earlier chunk of *this*
    # call don't get retried on the next one — see _PERMANENT_FAILURE_REASONS.
    # Local to this call, never module-level: a bad model must not stay
    # blacklisted past the request that discovered it.
    failed_models: dict[str, str] = {}

    for chunk in _chunks(flagged, MAX_WORDS_PER_CALL):
        chunk_lang = _chunk_language(chunk)
        content = _build_word_content(page, chunk, sibling_context_by_id)
        try:
            corrections, chunk_status = _request_with_fallback(
                client, _build_system_prompt(chunk), content, deadline, failed_models,
            )
            for item in corrections:
                if isinstance(item, dict) and "index" in item and "corrected" in item:
                    i = int(item["index"])
                    if 0 <= i < len(chunk):
                        correction_by_id[chunk[i].id] = item
                        chunk_lang_by_id[chunk[i].id] = chunk_lang
            last_status = chunk_status
            any_success = True
        except _LLMFailure as exc:
            logger.warning("LLM correction chunk skipped (%s)", exc.reason)
            last_status = _status("failed", exc.reason)

    if not any_success:
        return _mark_unverified(elements, confidence_threshold), last_status

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
            corrected_text = corr_data["corrected"]
            # See _introduces_unseen_script: the prompt already told the model
            # not to do this (_SCRIPT_RULE), this is the backstop for when it
            # ignores that instruction.
            if _introduces_unseen_script(chunk_lang_by_id.get(el.id), corrected_text):
                logger.info(
                    "Discarding correction for %s: %r -> %r introduces a script this "
                    "chunk's OCR gave no evidence for", el.id, el.content, corrected_text,
                )
                updated.append(el)
                continue
            certainty = corr_data.get("certainty", 0.0)
            updated.append(el.model_copy(update={
                "content": corrected_text,
                "llm_correction": LLMCorrection(
                    original=el.content,
                    corrected=corrected_text,
                    certainty=certainty,
                ),
                "highlight": "yellow" if certainty >= 0.60 else "red",
            }))
        elif corr_data:
            # The model answered about this word and left it alone — confirmed,
            # not merely unchecked. No highlight; see the comment above.
            updated.append(el)
        else:
            # No answer came back for this id at all (partial response, chunk
            # failure). OCR wasn't confident and nothing verified it, so it
            # ships marked rather than passing as certain.
            updated.append(el.model_copy(update={"highlight": _UNVERIFIED_HIGHLIGHT}))
    return updated, last_status


def _mark_unverified(elements: list[Element], confidence_threshold: float) -> list[Element]:
    """Highlight every below-threshold word, for the paths where no LLM answer arrived.

    Highlighting used to happen only when the model returned a correction, so any
    LLM failure — rate limit, missing key, undecodable page — shipped low-confidence
    OCR looking exactly like text read at 0.99. That is the wrong way round: the
    whole point of the confidence threshold is to surface uncertainty for review,
    and whether a third-party API answered has no bearing on whether the OCR was
    sure. Measured on class-diagram.png with the free tier rate-limited: 2 words
    below threshold ('"^' at 0.62, '١er' at 0.71), 0 highlighted.

    ponytail: one colour for "nobody checked this", distinct from the yellow/red
    the model's own certainty picks when it does answer. Ceiling: a reviewer can't
    tell "LLM was down" from "LLM answered with low certainty" — both are red.
    Upgrade path if that matters: the sidecar's llm_status already says which, and
    a third highlight colour is a one-line change here.
    """
    return [
        el.model_copy(update={"highlight": _UNVERIFIED_HIGHLIGHT})
        if isinstance(el, TextElement) and el.confidence < confidence_threshold
        else el
        for el in elements
    ]


def label_complex_shapes(
    elements: list[Element],
    crop_png_bytes: dict[str, bytes],
    deadline: float | None = None,
) -> tuple[list[Element], dict]:
    """Ask the vision model what each complex shape crop depicts.

    Returns (new_elements, status) — see _status(). Fills llm_label /
    llm_label_certainty on ComplexShapeElements. Like word correction this is
    best-effort: a provider failure leaves the labels unset rather than failing
    the request, and the returned status says why.

    deadline: a monotonic clock reading shared with another LLM call (see
    TOTAL_LLM_BUDGET_SECONDS). None (the default) computes a fresh full-budget
    deadline for this call alone, same as before this param existed.

    A floor plan repeats the same door swing / window / grid marker dozens of
    times, and shape labelling is the dominant cost of a request when it's
    enabled (~16-20s of a ~20s call) because every crop used to be sent, paid
    for and waited on separately. _group_similar_crops groups visually-
    identical crops so only one representative per group is actually sent to
    the model; its label is then copied onto every member of the group below.
    This changes *how many* LLM calls are made, never *which* elements end up
    labelled — every target still gets llm_label / llm_label_certainty set.
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

    groups = _group_similar_crops(targets, crop_png_bytes)
    representatives = [group[0] for group in groups]
    # Every group member (including the representative itself) maps back to
    # its representative's id, so a label found under the representative's id
    # can be copied onto the whole group once the calls come back.
    representative_of = {el.id: group[0].id for group in groups for el in group}
    logger.info(
        "Shape labelling: %d crops deduped to %d unique (%d LLM calls saved before chunking)",
        len(targets), len(representatives), len(targets) - len(representatives),
    )

    client = _get_client()
    by_id: dict[str, dict] = {}
    last_status = _status("failed", "unknown")
    any_success = False
    # One deadline for every chunk in this call, not one per chunk — see the
    # comment on TOTAL_LLM_BUDGET_SECONDS. Also shared with apply_corrections
    # when the caller passes one in (pipeline/__init__.py does).
    if deadline is None:
        deadline = time.monotonic() + TOTAL_LLM_BUDGET_SECONDS
    # Models that already failed permanently on an earlier chunk of *this*
    # call don't get retried on the next one — see _PERMANENT_FAILURE_REASONS.
    # Local to this call, never module-level: a bad model must not stay
    # blacklisted past the request that discovered it.
    failed_models: dict[str, str] = {}

    for start in range(0, len(representatives), MAX_CROPS_PER_CALL):
        chunk = representatives[start:start + MAX_CROPS_PER_CALL]
        content = _build_shape_content(chunk, crop_png_bytes)
        try:
            raw, chunk_status = _request_with_fallback(
                client, SHAPE_PROMPT, content, deadline, failed_models,
            )
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

    def _labelled(el: ComplexShapeElement) -> dict | None:
        return by_id.get(representative_of.get(el.id))

    return [
        el.model_copy(update={
            "llm_label": _labelled(el)["label"],
            "llm_label_certainty": float(_labelled(el).get("certainty", 0.0)),
        })
        if isinstance(el, ComplexShapeElement) and _labelled(el) is not None
        else el
        for el in elements
    ], last_status


# Aspect ratio is a cheap discriminator to check before trusting a hash match:
# resizing to a fixed grid for the hash throws the original aspect away, so
# two genuinely different symbols could in principle collide there. A crop
# whose width/height ratio differs from the group's by more than this
# fraction is never merged into it, no matter what the hash says.
SHAPE_HASH_ASPECT_TOLERANCE = 0.08

# Max dHash Hamming distance (out of 64 bits) to still call two crops "the
# same symbol". Measured on real crops cut from sample_drawing.png (clean
# vector line art, not photos — representative of what this pipeline
# actually labels): the *same* symbol region cropped with bounding boxes
# jittered by 1-2px (the realistic variance between two independently
# detected instances of one repeated symbol) came back at Hamming distance
# 0-2. Genuinely *different* symbols (circle vs. triangle vs. pentagon) came
# back at 14-26. That's a wide gap with plenty of margin either side, so 3
# was picked to comfortably cover the jitter case without getting anywhere
# near the different-symbol range. Pure exact-match (0) was tried first per
# the "bias toward strict matching" guidance, but real anti-aliasing/crop-
# bound jitter routinely produces Hamming 1-2 even for the same symbol, which
# would make dedup fire on almost nothing — the same "silently does nothing"
# failure mode the raw-bytes approach has, just one level down. If this ever
# proves too loose in production (a wrong label spreading across a group),
# drop it back toward 0 first before reaching for anything fancier.
SHAPE_HASH_MAX_HAMMING = 3


def _group_similar_crops(
    targets: list[ComplexShapeElement], crop_png_bytes: dict[str, bytes]
) -> list[list[ComplexShapeElement]]:
    """Group shape crops that are visually the same symbol so only one member
    per group needs labelling.

    Matching is a perceptual difference-hash (dHash) on the greyscale crop
    within SHAPE_HASH_MAX_HAMMING bits, gated by aspect ratio (see the
    constants above for the measured justification of both thresholds). A
    crop is compared against each existing group's *first* member (not a
    running average) — simple, and good enough given how tight the threshold
    is. A crop that fails to decode is never merged with anything (its own
    group of one), so a bad crop just costs one extra call instead of
    borrowing — or donating — a wrong label.
    # ponytail: this only catches near-pixel-equivalent renderings — a
    # mirrored or rotated instance of the same symbol (a door swinging the
    # other way) hashes completely differently and will not be merged, so
    # it's sent and labelled on its own like today. Upgrade path: hash the
    # crop under a small set of rotations/flips (0/90/180/270 x mirror) and
    # match against the representative's set instead of a single hash, if
    # rotated symbols turn out to be common enough to matter.
    """
    groups: list[dict] = []
    for el in targets:
        sig = _crop_signature(crop_png_bytes.get(el.id))
        match = None
        if sig is not None:
            h, aspect = sig
            for g in groups:
                if g["sig"] is None:
                    continue
                g_hash, g_aspect = g["sig"]
                if abs(g_aspect - aspect) > SHAPE_HASH_ASPECT_TOLERANCE * max(g_aspect, aspect, 1e-6):
                    continue
                if bin(g_hash ^ h).count("1") <= SHAPE_HASH_MAX_HAMMING:
                    match = g
                    break
        if match is not None:
            match["members"].append(el)
        else:
            groups.append({"sig": sig, "members": [el]})
    return [g["members"] for g in groups]


def _crop_signature(png_bytes: bytes | None) -> tuple[int, float] | None:
    """(dHash, aspect_ratio) for one shape crop, or None if it can't be decoded.

    dHash: greyscale, shrink to 9x8 with area averaging (averaging rather than
    nearest/linear resampling is what makes this tolerant of a couple of
    stray anti-aliased pixels — see _group_similar_crops), then one bit per
    pixel for "brighter than the pixel to its right" — 8 rows x 8 comparisons
    = 64 bits. Not cryptographic, doesn't need to be: two visually identical
    crops should shrink to the same light/dark pattern.
    """
    if not png_bytes:
        return None
    img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    aspect = w / h if h else 0.0
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits, aspect


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


def _sibling_context(el: TextElement, elements: list[Element], confidence_threshold: float) -> list[str]:
    """Confident OCR readings sharing el's innermost detected shape (see
    TextElement.container_shape_id) — e.g. the other attribute labels already
    read correctly inside the same UML class box as a flagged one. Empty list
    (not None) when there's no container or no confident sibling, so callers
    never need a None check.
    """
    if not el.container_shape_id:
        return []
    return [
        sib.content for sib in elements
        if isinstance(sib, TextElement) and sib.id != el.id
        and sib.container_shape_id == el.container_shape_id
        and sib.confidence >= confidence_threshold
    ]


def _build_word_content(
    page: np.ndarray, chunk: list[TextElement], sibling_context_by_id: dict[str, list[str]] | None = None,
) -> list[dict]:
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
    content.append({"type": "text", "text": _build_prompt(chunk, sibling_context_by_id)})
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


# A model that fails with one of these reasons will fail the exact same way
# on every later chunk of the same call — invalid_model (400/404, e.g. "this
# model doesn't support image input") and unauthorized (401/403, bad key /
# no permission) are properties of the (model, request) pair, not of the
# payload or the moment in time. Retrying them per chunk (up to 6 chunks for
# shape labelling) just burns PER_ATTEMPT_CAP_SECONDS for nothing.
#
# Deliberately NOT included: rate_limited, network, server_error, timed_out,
# parse_error, empty_response, unknown. All of those are plausibly transient
# or payload-dependent — a 429 can clear by the next chunk, a stalled
# connection can recover, a model that garbled the JSON for one chunk's crops
# might get the next chunk's right. Blacklisting on those would risk
# permanently benching a model for one bad chunk within an otherwise-working
# call, which is worse than the wasted retry this is meant to avoid.
_PERMANENT_FAILURE_REASONS = frozenset({"invalid_model", "unauthorized"})


def _request_with_fallback(client, system_prompt: str, user_content: list[dict],
                            deadline: float, failed_models: dict[str, str]) -> tuple[list[dict], dict]:
    """Try each configured model in order (primary, then fallbacks), retrying
    transient failures on each with backoff. Raises _LLMFailure if every model
    in the chain fails, or "timed_out" once `deadline` (a single monotonic
    clock reading shared across every chunk of the call this came from) passes.

    failed_models is mutated in place: models that fail for a reason in
    _PERMANENT_FAILURE_REASONS are recorded here (by the caller's chunk loop,
    across calls to this function) and skipped without being retried.
    """
    models = _model_chain()
    last_reason = "unknown"
    for i, model in enumerate(models):
        if time.monotonic() >= deadline:
            logger.warning("LLM budget of %.0fs exhausted, giving up", TOTAL_LLM_BUDGET_SECONDS)
            raise _LLMFailure("timed_out")
        if model in failed_models:
            last_reason = failed_models[model]
            logger.info(
                "Skipping %s: already failed permanently (%s) earlier in this call",
                model, last_reason,
            )
            continue
        try:
            content = _call_with_retry(client, model, system_prompt, user_content, deadline=deadline)
            parsed = _parse_json_array(content)
            return parsed, _status("success", None, model=model)
        except Exception as exc:
            last_reason = _classify_error(exc)
            if last_reason in _PERMANENT_FAILURE_REASONS:
                failed_models[model] = last_reason
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
        # Never hand one attempt more than PER_ATTEMPT_CAP_SECONDS, no matter
        # how much budget is left — see the comment on that constant.
        remaining = min(remaining, PER_ATTEMPT_CAP_SECONDS)
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


def _build_prompt(flagged: list[TextElement], sibling_context_by_id: dict[str, list[str]] | None = None) -> str:
    # Index-based, matching _build_shape_content's "N images, in order" convention:
    # the model answers by image position, not by echoing the string back, so a
    # word the model rewrites or a duplicate word at a different confidence still
    # keys back to the right element (see correction_by_id in apply_corrections).
    sibling_context_by_id = sibling_context_by_id or {}
    words = []
    for i, e in enumerate(flagged):
        entry = {"index": i, "current_guess": e.content, "confidence": round(e.confidence, 3)}
        siblings = sibling_context_by_id.get(e.id)
        if siblings:
            entry["nearby_confident_labels"] = siblings
        words.append(entry)

    context_note = (
        " Some entries include \"nearby_confident_labels\": other text already read with "
        "high confidence from the same box/shape on the drawing. Use those only as context "
        "to help you disambiguate the flagged image — they are not things to correct, and "
        "must never be copied into your answer unless the flagged crop genuinely shows that "
        "same text."
        if any("nearby_confident_labels" in w for w in words) else ""
    )
    return (
        f"{len(flagged)} cropped word images, in order, each padded with a little "
        "surrounding context and upscaled if small. Below is the OCR's current guess "
        f"and confidence for each image, by index:\n{json.dumps(words, ensure_ascii=False)}\n\n"
        "Re-read each cropped image and provide the corrected reading and your certainty "
        "(0.0-1.0). If the current guess was already correct, return it unchanged. If the "
        f"crop truly can't be read, return an empty string and certainty 0.0 rather than guessing.{context_note}\n"
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
