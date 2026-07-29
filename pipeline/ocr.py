"""OCR: one detection pass, then per-crop recognition routed by script.

The old 2.7.3 design ran the `en` and `arabic` PaddleOCR pipelines as two complete
detect+rec passes over the whole page and reconciled their disagreeing segmentations by
bbox overlap (a union-find merge, since the two detectors returned 8 vs 21 boxes on the
same drawing). That merge existed *only* because there were two detectors.

There is now one. `PP-OCRv6_tiny_det` runs once and is language-agnostic — verified on
sample_drawing.png, every detector tested returned exactly the same 12 polygons covering
5/5 Arabic and 7/7 Latin regions. Each polygon is cropped once and recognised:

  * the Latin recogniser reads every crop;
  * the Arabic recogniser reads only the crops the Latin one was unsure about;
  * the winner per crop is decided by script (see `_pick`).

Measured on this hardware (i7-13650HX, 20 threads, no AVX-512, CPU only), best-of-3 warm:
2.63s on class-diagram.png and 0.82s on sample_drawing.png, against 11.57s / 2.55s for
the two-pass v4 design. A naive version bump that kept two full passes measured 38.11s /
12.53s — the win here is the single segmentation, not the model version.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from utils.bidi import is_arabic

# One cached predictor per model name — each is tens of MB of weights, so they are built
# on first use and kept for the process lifetime.
_predictors: dict[str, object] = {}

# One detector for every language. It finds where text is, not what it says.
DET_MODEL = "PP-OCRv6_tiny_det"
# Latin/digits, and the default reading for every crop.
LATIN_REC_MODEL = "PP-OCRv6_small_rec"
# Arabic. Only ever consulted for crops the Latin recogniser was unsure about.
ARABIC_REC_MODEL = "arabic_PP-OCRv5_mobile_rec"

# The caller's language_hint mapped to the recognisers to consider, in order.
#
# "ar+en" considers BOTH, but does not run both over everything — see
# ARABIC_ROUTING_CONFIDENCE. A single-script hint skips the routing entirely and uses one
# recogniser for every crop, which is both faster and strictly more correct for that
# input: the Arabic model's RTL decoder mangles Latin while reporting ~0.92 confidence on
# the result (measured on class-diagram.png under v4: identifiers fragmented,
# "+created_at" came back as "at_+created"), so "en" must never be routed through it.
_LANG_RECOGNISERS = {
    "ar+en": (LATIN_REC_MODEL, ARABIC_REC_MODEL),
    "ar": (ARABIC_REC_MODEL,),
    "en": (LATIN_REC_MODEL,),
}
DEFAULT_LANGUAGE_HINT = "ar+en"

# Latin-recogniser confidence below which a crop is also shown to the Arabic recogniser.
# This is what keeps "run both models" from costing two full recognition passes: measured
# at 0.90, it fires on 8 of 89 crops on class-diagram.png and 5 of 12 on
# sample_drawing.png, so the second recogniser sees ~10% of the page, not all of it.
#
# ponytail: 0.90 is tuned on exactly two images. It sits in a wide empty gap on both of
# them — Latin crops read Latin score 0.97+, Arabic crops read by the Latin model score
# well under 0.7 — but "wide gap on two drawings" is not "separates the populations in
# general". The failure mode if it is too low is silent: an Arabic label the Latin model
# happens to hallucinate confidently is never shown to the Arabic model at all. The
# ceiling is that this is a confidence threshold standing in for a script detector.
# Upgrade path if a real corpus disagrees: classify the crop's script directly (a tiny
# CNN, or the far cheaper trick of asking both recognisers on a sampled subset and
# fitting the threshold), rather than moving this number by hand. Note the worst case is
# bounded and still fast — every crop through both recognisers measured 7.44s / 1.28s,
# under the 11.57s / 2.55s the two-pass v4 design cost.
ARABIC_ROUTING_CONFIDENCE = 0.90

# A crop this much taller than it is wide is vertical text; rotate it upright before
# recognition. This is PaddleOCR's own convention from get_rotate_crop_image, kept because
# the 3-model design has no text-line orientation classifier to do it properly.
_VERTICAL_CROP_ASPECT = 1.5

# Pixels of margin sampled outside the detected polygon on every side. The detector's
# box hugs the glyphs, and a recogniser handed a crop that clips ascenders/descenders by
# a pixel reads it noticeably worse; the benchmark that chose this architecture used the
# same 2px and reported 0.9876 mean confidence with it.
_CROP_PAD_PX = 2


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
    """Run OCR on image, return words with bounding boxes and confidence.

    One detection pass locates the text. A single-script hint ("ar" / "en") then reads
    every crop with that script's recogniser. The mixed hint "ar+en" reads every crop
    with the Latin recogniser and re-reads only the low-confidence ones with the Arabic
    one, keeping whichever reading matches the script that is actually there.
    """
    recognisers = _LANG_RECOGNISERS.get(
        language_hint, _LANG_RECOGNISERS[DEFAULT_LANGUAGE_HINT]
    )
    polys = _detect(image)
    if not polys:
        return []
    crops = [_crop_polygon(image, poly) for poly in polys]
    keep = [i for i, crop in enumerate(crops) if crop is not None]
    if not keep:
        return []

    primary = _recognise(recognisers[0], [crops[i] for i in keep])

    if len(recognisers) == 1:
        readings = primary
    else:
        # Only the crops the Latin recogniser doubted go to the Arabic one. This is the
        # whole cost argument for the design: it is one recognition pass plus a tail.
        # An empty reading is routed too regardless of the score attached to it: "the
        # Latin model found nothing here" is exactly the signal that the crop may not be
        # Latin, and it is the one case where a high confidence means nothing at all.
        unsure = [
            n for n, (text, conf) in enumerate(primary)
            if conf < ARABIC_ROUTING_CONFIDENCE or not text
        ]
        readings = list(primary)
        if unsure:
            secondary = _recognise(
                recognisers[1], [crops[keep[n]] for n in unsure]
            )
            for n, arabic_reading in zip(unsure, secondary):
                readings[n] = _pick(primary[n], arabic_reading)

    words = []
    for n, i in enumerate(keep):
        text, conf = readings[n]
        if not text:
            continue
        words.append(OcrWord(
            text=text,
            confidence=conf,
            bbox_px=_poly_bbox(polys[i]),
            flagged=conf < confidence_threshold,
        ))
    # Top-to-bottom, left-to-right, so the output order is a property of the page rather
    # than of detector iteration order. Reading order proper is redone in
    # pipeline/layout_reconstructor.py.
    words.sort(key=lambda w: (w.bbox_px["y"], w.bbox_px["x"]))
    return words


def _pick(
    latin: tuple[str, float], arabic: tuple[str, float]
) -> tuple[str, float]:
    """Choose between the two recognisers' readings of one crop.

    Script decides, not confidence. The Arabic reading wins if and only if it actually
    came back as Arabic — the Latin model cannot emit Arabic script at all, so an Arabic
    reading here means the crop really is Arabic and the Latin model's low-confidence
    guess was noise. In every other case the Latin reading is kept, *including* when the
    Arabic model reports the higher confidence: on Latin input its RTL decoder reverses
    and fragments tokens while still scoring ~0.92, so its confidence is not comparable
    and must not be allowed to win a numeric contest against a genuine Latin reading.
    """
    arabic_text, _ = arabic
    if arabic_text and is_arabic(arabic_text):
        return arabic
    return latin


def _detect(image: np.ndarray) -> list[np.ndarray]:
    """Text polygons for the whole page, from the one shared detector."""
    result = _predict_one(DET_MODEL, image)
    if result is None:
        return []
    polys = _field(result, "dt_polys")
    return [] if polys is None else [np.asarray(p, dtype=np.float32) for p in polys]


def _recognise(model_name: str, crops: list[np.ndarray]) -> list[tuple[str, float]]:
    """Read a batch of crops with one recogniser. Returns (text, confidence) per crop."""
    if not crops:
        return []
    predictor = _get_predictor(model_name)
    out: list[tuple[str, float]] = []
    for result in predictor.predict(crops):
        text = _field(result, "rec_text") or ""
        score = _field(result, "rec_score")
        out.append((str(text).strip(), float(score) if score is not None else 0.0))
    # predict() is documented to yield one result per input, but a short yield would
    # silently misalign every reading after it against the wrong crop, which is far worse
    # than a missing word. Pad rather than zip-truncate.
    while len(out) < len(crops):
        out.append(("", 0.0))
    return out[:len(crops)]


def _predict_one(model_name: str, image: np.ndarray):
    for result in _get_predictor(model_name).predict(image):
        return result
    return None


def _field(result, key: str):
    """Read one field from a paddlex prediction result.

    3.x results are dict-like objects; older/newer point releases have differed on
    whether attribute access also works, so try both rather than pinning to one.
    """
    try:
        return result[key]
    except (TypeError, KeyError, IndexError):
        return getattr(result, key, None)


def _poly_bbox(poly: np.ndarray) -> dict:
    """Axis-aligned {x, y, w, h} around a detection polygon."""
    xs, ys = poly[:, 0], poly[:, 1]
    x, y = int(np.floor(xs.min())), int(np.floor(ys.min()))
    return {
        "x": max(x, 0),
        "y": max(y, 0),
        "w": max(int(np.ceil(xs.max())) - x, 1),
        "h": max(int(np.ceil(ys.max())) - y, 1),
    }


def _crop_polygon(image: np.ndarray, poly: np.ndarray):
    """Cut one detected text region out of the page, un-rotating it.

    The detector returns *polygons*, not boxes, and they are only axis-aligned when the
    text is. Taking the axis-aligned bbox of a tilted line would hand the recogniser a
    crop containing the line plus two triangular wedges of whatever is diagonally
    adjacent to it, with the glyphs themselves running at an angle across it — the
    recogniser expects a horizontal strip and reads that badly. A perspective transform
    onto the quad's own dimensions gives it the strip it expects. This is PaddleOCR's own
    get_rotate_crop_image, reimplemented here because the 3.x package only applies it
    inside the full pipeline object this module deliberately does not use.

    Returns None for a degenerate polygon rather than a zero-sized array, which
    cv2.warpPerspective would refuse and paddle would happily read as an empty string.
    """
    points = np.asarray(poly, dtype=np.float32)
    if points.ndim != 2 or len(points) < 4:
        return None
    if len(points) > 4:
        # Polygon-mode detection (curved text) gives more than four points. minAreaRect
        # is the honest reduction: the tightest rotated box that still contains the whole
        # run, so nothing is clipped and the un-rotation still happens.
        points = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float32)
        points = _order_quad(points)

    width = int(round(max(
        np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]),
    )))
    height = int(round(max(
        np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]),
    )))
    if width < 2 or height < 2:
        return None

    # The quad maps to an inset rect, so the extra _CROP_PAD_PX ring around it samples
    # the page just outside the detection rather than stretching its edge pixels.
    pad = _CROP_PAD_PX
    dst = np.array(
        [[pad, pad], [pad + width, pad],
         [pad + width, pad + height], [pad, pad + height]], dtype=np.float32,
    )
    crop = cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(points, dst),
        (width + 2 * pad, height + 2 * pad),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    if crop.shape[0] >= crop.shape[1] * _VERTICAL_CROP_ASPECT:
        # Vertical text. The recogniser only reads horizontal strips, so stand it up.
        # ponytail: this guesses the rotation direction (always counter-clockwise) rather
        # than determining it — the 3-model design has no text-line orientation
        # classifier, and adding one would be a fourth model's weights and latency for
        # something neither test drawing contains. Upgrade path if upside-down or
        # clockwise-rotated labels turn up: add PP-LCNet_x1_0_textline_ori and use it here.
        crop = np.ascontiguousarray(np.rot90(crop))
    return crop


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order four corners as top-left, top-right, bottom-right, bottom-left.

    cv2.boxPoints' starting corner depends on the rect's angle, and getPerspectiveTransform
    maps corner i to corner i — so an unordered quad produces a mirrored or rotated crop.
    """
    ordered = points[np.argsort(points[:, 1])]
    top, bottom = ordered[:2], ordered[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(-bottom[:, 0])]
    return np.vstack([top, bottom]).astype(np.float32)


def _get_predictor(model_name: str):
    """Cached paddle predictor for one model.

    enable_mkldnn=False is load-bearing, not tuning. Stock paddle 3.3.1 on a CPU without
    AVX-512 (Alder/Raptor Lake, e.g. this i7-13650HX) dies during inference with
      NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
      [pir::ArrayAttribute<pir::DoubleAttribute>]   (onednn_instruction.cc:116)
    Verified: neither `ir_optim=False` nor the `FLAGS_enable_pir_api=0` env var avoids it,
    and the `FLAGS_use_mkldnn=0` env var that 2.x honoured is ignored outright by 3.x.
    Only this constructor kwarg works, so every construction needs it — a predictor built
    without it is an immediate hard crash, not a slow path.

    This replaces pipeline/_paddle_patch.py, which monkeypatched paddle.inference to
    disable IR optimization for the AVX-512 SIGILL of Paddle#76111. That crash does not
    reproduce on 3.3.1 (re-tested repeatedly, clean exit), and its fix would not have
    helped here anyway.
    """
    if model_name not in _predictors:
        from paddleocr import TextDetection, TextRecognition
        cls = TextDetection if model_name == DET_MODEL else TextRecognition
        _predictors[model_name] = cls(model_name=model_name, enable_mkldnn=False)
    return _predictors[model_name]
