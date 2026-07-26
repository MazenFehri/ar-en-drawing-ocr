import base64
import json
import logging
from models.elements import TextElement, ComplexShapeElement, LLMCorrection, Element

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


def apply_corrections(
    elements: list[Element],
    image_bytes: bytes,
    confidence_threshold: float = 0.75,
) -> list[Element]:
    """Call OpenRouter LLM to correct low-confidence OCR words in the element list.

    Returns a new list with corrections applied and highlights set.
    """
    flagged = [e for e in elements
               if isinstance(e, TextElement) and e.confidence < confidence_threshold]
    if not flagged:
        return elements

    # LLM correction is an optional enhancement. If the provider is unavailable
    # (rate limit, network error, invalid model, malformed response), skip it and
    # return the elements with their original OCR text rather than failing the
    # whole request.
    try:
        client = _get_client()
        corrections = _call_llm(client, image_bytes, flagged)
        correction_map = {
            c["original"]: c
            for c in corrections
            if isinstance(c, dict) and "original" in c and "corrected" in c
        }
    except Exception as exc:
        logger.warning("LLM correction skipped (%s): %s", type(exc).__name__, exc)
        return elements

    updated = []
    for el in elements:
        if not isinstance(el, TextElement) or el.confidence >= confidence_threshold:
            updated.append(el)
            continue
        corr_data = correction_map.get(el.content)
        # A word the model hands back unchanged was confirmed, not corrected —
        # highlighting it would paint most of the page yellow at a high threshold.
        if corr_data and corr_data["corrected"] != el.content:
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
    return updated


def label_complex_shapes(
    elements: list[Element],
    crop_png_bytes: dict[str, bytes],
) -> list[Element]:
    """Ask the vision model what each complex shape crop depicts.

    Fills llm_label / llm_label_certainty on ComplexShapeElements. Like word
    correction this is best-effort: a provider failure leaves the labels unset
    rather than failing the request.
    """
    targets = [
        e for e in elements
        if isinstance(e, ComplexShapeElement) and e.id in crop_png_bytes
    ]
    if not targets:
        return elements

    try:
        client = _get_client()
        raw = _call_shape_llm(client, targets, crop_png_bytes)
        by_index = {
            int(item["index"]): item
            for item in raw
            if isinstance(item, dict) and "index" in item and "label" in item
        }
    except Exception as exc:
        logger.warning("Shape labelling skipped (%s): %s", type(exc).__name__, exc)
        return elements

    labels = {
        el.id: by_index[i] for i, el in enumerate(targets) if i in by_index
    }
    return [
        el.model_copy(update={
            "llm_label": labels[el.id]["label"],
            "llm_label_certainty": float(labels[el.id].get("certainty", 0.0)),
        })
        if isinstance(el, ComplexShapeElement) and el.id in labels
        else el
        for el in elements
    ]


def _call_shape_llm(client, targets, crop_png_bytes: dict[str, bytes]) -> list[dict]:
    content = []
    for el in targets:
        b64 = base64.standard_b64encode(crop_png_bytes[el.id]).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": f"{len(targets)} symbol images, in order."})

    response = client.chat.completions.create(
        model=_get_model(),
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SHAPE_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return _parse_json_array(response.choices[0].message.content)


def _call_llm(client, image_bytes: bytes, flagged: list[TextElement]) -> list[dict]:
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = _build_prompt(flagged)

    response = client.chat.completions.create(
        model=_get_model(),
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )
    return _parse_json_array(response.choices[0].message.content)


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
    words = [{"original": e.content, "confidence": round(e.confidence, 3)} for e in flagged]
    return (
        f"Flagged words from OCR (confidence below threshold):\n{json.dumps(words, ensure_ascii=False)}\n\n"
        "For each word, provide the corrected spelling and your certainty (0.0–1.0).\n"
        'Return: [{"original": "...", "corrected": "...", "certainty": 0.0}]'
    )


def _get_client():
    global _client_instance
    if _client_instance is None:
        from openai import OpenAI
        from app.config import settings
        _client_instance = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
    return _client_instance


def _get_model() -> str:
    from app.config import settings
    return settings.openrouter_model
