import base64
import json
from models.elements import TextElement, LLMCorrection, Element

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

    client = _get_client()
    corrections = _call_llm(client, image_bytes, flagged)
    correction_map = {c["original"]: c for c in corrections}

    updated = []
    for el in elements:
        if not isinstance(el, TextElement) or el.confidence >= confidence_threshold:
            updated.append(el)
            continue
        corr_data = correction_map.get(el.content)
        if corr_data:
            certainty = corr_data["certainty"]
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
    return json.loads(response.choices[0].message.content)


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
