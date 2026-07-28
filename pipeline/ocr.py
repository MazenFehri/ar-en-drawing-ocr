from dataclasses import dataclass
import numpy as np

from utils.bidi import is_arabic

# One cached PaddleOCR instance per model language — each is ~100 MB of weights,
# so they are built on first use and kept.
_ocr_instances: dict[str, object] = {}

# The caller's language_hint mapped to the PaddleOCR models to run, in order.
#
# "ar+en" runs BOTH. The "arabic" model does emit Latin letters and digits, but
# its recognition decoder is RTL-trained and mangles them: measured on the
# repo's all-English class-diagram.png, 55 of 80 position-matched words came
# back wrong — identifiers fragmented ("+allocated_amount" -> "+allocated."),
# and token order reversed ("+created_at" -> "at_+created", "+is_active" ->
# "active_is", "transaction_id" -> "id_transaction"). It reports ~0.92
# confidence on that garbage, so the confidence-gated LLM corrector never sees
# it either. The "en" model scored 0.991 on the same page. So mixed drawings
# need both models and a per-region choice, not one model that "handles both".
_LANG_MODELS = {"ar+en": ("en", "arabic"), "ar": ("arabic",), "en": ("en",)}
DEFAULT_LANGUAGE_HINT = "ar+en"

# How much two boxes must overlap to be treated as covering the same region.
# Measured as intersection over the *smaller* box, not IoU. Plain IoU at 0.5
# was tried first and does not work: the two models segment lines differently.
# On sample_drawing.png the en model returns "GROUND FLOOR PLAN" as one box
# while the arabic model returns three word boxes inside it, so every pair
# scores IoU 0.24-0.47 — under any sane threshold — and both readings survive
# as overlapping duplicate text. Intersection-over-smaller scores those pairs
# ~1.0, which is the relationship that actually holds: same region, different
# granularity.
_MERGE_OVERLAP_THRESHOLD = 0.5


@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox_px: dict        # {x, y, w, h}
    flagged: bool = False


def run_ocr(
    image: np.ndarray,
    confidence_threshold: float = 0.75,
    language_hint: str = DEFAULT_LANGUAGE_HINT,
) -> list[OcrWord]:
    """Run PaddleOCR on image, return words with bounding boxes and confidence.

    A single-script hint ("ar" / "en") runs one model. The mixed hint "ar+en"
    runs both and merges them per detected region, keeping whichever model's
    reading matches the script that is actually there.
    """
    model_langs = _LANG_MODELS.get(language_hint, _LANG_MODELS[DEFAULT_LANGUAGE_HINT])
    per_model = [
        _run_model(image, lang, confidence_threshold) for lang in model_langs
    ]
    if len(per_model) == 1:
        return per_model[0]
    en_words, arabic_words = per_model
    return _merge_by_script(arabic_words, en_words)


def _run_model(
    image: np.ndarray, model_lang: str, confidence_threshold: float
) -> list[OcrWord]:
    """One model's pass over the whole image."""
    ocr = _get_ocr(model_lang)
    raw = ocr.ocr(image, cls=True)
    words = []
    if not raw or not raw[0]:
        return words
    for line in raw[0]:
        points, (text, conf) = line
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = {
            "x": int(min(xs)),
            "y": int(min(ys)),
            "w": int(max(xs) - min(xs)),
            "h": int(max(ys) - min(ys)),
        }
        words.append(OcrWord(
            text=text,
            confidence=conf,
            bbox_px=bbox,
            flagged=conf < confidence_threshold,
        ))
    return words


def _merge_by_script(
    arabic_words: list[OcrWord],
    en_words: list[OcrWord],
    overlap_threshold: float = _MERGE_OVERLAP_THRESHOLD,
) -> list[OcrWord]:
    """Pick one model's reading per region of the page from the two outputs.

    Boxes that overlap across the two models are grouped together (transitively,
    so one model's single line box and the other's three word boxes form one
    group). Each group is then resolved whole:

      * only one model detected it        -> keep that model's boxes
      * the arabic model read Arabic there -> keep the arabic model's boxes,
        since the en model cannot read Arabic at all
      * otherwise                          -> keep the en model's boxes, because
        the arabic model's RTL decoder reverses and fragments Latin tokens

    ponytail: the group is the unit of the decision, so a group is all-Arabic or
    all-Latin and one box that genuinely mixes both scripts ("مدخل MAIN") pulls
    its whole group to the arabic model, leaving the Latin half mangled. Real
    drawing labels are single-script per detected box, so this has not come up.
    Upgrade path if it does: split such a box on the detection polygon and
    resolve each side against its own model before this grouping runs.
    """
    # Bipartite adjacency between the two models' boxes, then union them into
    # connected groups.
    parent = list(range(len(arabic_words) + len(en_words)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    offset = len(arabic_words)
    for i, arabic_word in enumerate(arabic_words):
        for j, en_word in enumerate(en_words):
            if _overlap(arabic_word.bbox_px, en_word.bbox_px) >= overlap_threshold:
                union(i, offset + j)

    groups: dict[int, tuple[list[OcrWord], list[OcrWord]]] = {}
    for i, word in enumerate(arabic_words):
        groups.setdefault(find(i), ([], []))[0].append(word)
    for j, word in enumerate(en_words):
        groups.setdefault(find(offset + j), ([], []))[1].append(word)

    merged: list[OcrWord] = []
    for arabic_side, en_side in groups.values():
        if not en_side:
            merged.extend(arabic_side)
        elif not arabic_side:
            merged.extend(en_side)
        elif any(is_arabic(w.text) for w in arabic_side):
            merged.extend(arabic_side)
        else:
            merged.extend(en_side)
    # Top-to-bottom, left-to-right so the result does not depend on which model
    # happened to detect a region. Reading order proper is redone in
    # pipeline/layout_reconstructor.py.
    merged.sort(key=lambda w: (w.bbox_px["y"], w.bbox_px["x"]))
    return merged


def _overlap(a: dict, b: dict) -> float:
    """Intersection of two {x, y, w, h} boxes over the area of the smaller one."""
    overlap_w = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    overlap_h = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    smaller = min(a["w"] * a["h"], b["w"] * b["h"])
    return (overlap_w * overlap_h) / smaller if smaller > 0 else 0.0


def _get_ocr(model_lang: str):
    """Cached PaddleOCR instance for one model language ("arabic" or "en")."""
    if model_lang not in _ocr_instances:
        from pipeline import _paddle_patch  # noqa: F401  (patches paddle before predictor build)
        from paddleocr import PaddleOCR
        _ocr_instances[model_lang] = PaddleOCR(
            lang=model_lang, use_angle_cls=True, show_log=False
        )
    return _ocr_instances[model_lang]
