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
import difflib
import re
from dataclasses import dataclass, field

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
# Consulted ONLY for the digits it can see and ARABIC_REC_MODEL cannot. See _splice_digits.
ARABIC_DIGIT_DONOR_MODEL = "arabic_PP-OCRv3_mobile_rec"

# How similar the two Arabic readings of one crop must be before digits are carried across.
#
# Two recognisers reading the same strip normally agree closely on the Arabic even when
# they disagree on a letter or two. When they don't agree at all, one of them has misread
# the crop wholesale and its digits are not evidence of anything — splicing from it would
# invent a number that is not on the page, which is far worse than the missing number this
# whole mechanism exists to fix.
# ponytail: one ratio over the whole string, so a donor that nails the half of the line
# containing the number but garbles the other half can still fall under the bar and be
# ignored. Upgrade path if that shows up in practice: score the similarity of the aligned
# neighbourhood around each digit run rather than of the entire reading.
DIGIT_DONOR_MIN_SIMILARITY = 0.5

# A digit run: Western digits plus the separators that appear inside one number.
#
# [0-9] and not \d, which is load-bearing. Python's \d matches every Unicode decimal digit
# including Arabic-Indic ٠-٩, and ARABIC_REC_MODEL reads *those* perfectly well — it only
# drops Western ones. Matching them made the donor's Arabic-Indic misreads look like
# recoverable losses, and measured on these pages it spliced a spurious ٢ into
# "ما هوثمن العصير؟" and a ١ into "و د", costing test2 accuracy (0.8678 -> 0.8623).
_DIGIT_RUN = re.compile(r"[0-9][0-9.,٫٬]*[0-9]|[0-9]")

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

# Longest side an image is scaled up to before detection, if it arrives smaller.
#
# Low-DPI scans are not merely "a bit worse" for Arabic, they fail in a specific way: the
# marks that separate one letter from another are single dots, and at ~45 DPI a dot is one
# or two pixels. Measured on a 375x533 scan of a school register (14-16px text lines), the
# native-resolution read returned 157 Arabic characters at 0.651 mean confidence; the same
# page cubic-upscaled 4x to 1500px returned 346 characters at 0.756, for +0.42s. The gain
# is mostly recall — whole table rows that came back as 'ال' and 'ل  ا' at native size read
# as real phrases once the recogniser is handed a crop it can resolve.
#
# Interpolation does not add information, so this cannot fix a dot that was never sampled:
# the same page reads 'اللفب' for 'اللقب' at 1x, 2x, 3x and 4x alike. What it fixes is the
# recogniser's own downstream resize — a 14px line scaled up to the 48px input height is
# reconstructing from almost nothing, and doing that step once, well, with INTER_CUBIC on
# the whole page beats letting it happen per-crop on a strip.
#
# This deliberately does NOT live in pipeline/preprocessor.py. That module's output is the
# coordinate space for shape detection, and pipeline/shape_detector.py is tuned throughout
# in absolute pixels (MIN_AREA_PX, RECT_MIN_SIDE_PX, ELLIPSE_MIN_AXIS_PX, ...) against
# images at native scale — scaling the page under it would redefine every one of those
# thresholds by the same factor, silently. Boxes are divided back down before they leave
# run_ocr, so nothing outside this module sees the upscaled space.
#
# ponytail: one number, chosen from a 1x/2x/3x/4x sweep on one low-DPI page — 4x scored
# best there and 1500 is what 4x came to. Images already at or above it are untouched, so
# the normal path is unaffected. Upgrade path if this matters more: scale by measured text
# height (run detection once at native size, take the median box height, target ~48px)
# rather than by page size, which would also stop a large page of tiny text falling through.
OCR_MIN_DIM_PX = 1500

# Detector geometry. Both of these are measured wins over the shipped defaults, scored as
# mean per-line best match against a hand-transcribed ground truth for two ~45 DPI Arabic
# worksheets: combined 0.8134 -> 0.9145, with no measurable latency change.
#
# unclip_ratio is the dominant one (+0.083 of the +0.101 on its own). DB detectors emit a
# tight shrunken region and dilate it back out by this factor; the 1.4 default over-grows
# an already line-tight box on dense prose and pulls the neighbouring rows' ascenders and
# descenders into the crop, so the recogniser reads a strip with contamination above and
# below it. Same box, same geometry, tightened to 1.0:
#     1.4: 'السنلاحتفال بعيد ملاد أمهم قر الإخوة الساهمة بذخاتهم'      conf 0.72
#     1.0: 'السند للاحتقال بعيد ميلاد أمهم قرر الإخوة المساهمة بمذخراتهم'  conf 0.88
# Recognised characters rose 469->567 and 407->629 on the two pages. Sweeping it spans
# 0.708 (at 2.0) to 0.915 (at 1.0), monotonically.
#
# limit_side_len is the smaller half (+0.018). PP-OCRv6_tiny_det is in paddlex's
# _TEXT_DET_MAX_LIMIT_MODELS, so it defaults to (960, "max") and *downscales* anything
# larger before inference — the OCR_MIN_DIM_PX upscale to 1500 was being cut straight back
# to 960, netting 1.81x rather than 2.82x. Note the fix is NOT "stop downscaling": accuracy
# peaks around 1000-1200 detector pixels and falls off above it (1200: 0.9145, 1400:
# 0.8947, 1500: 0.8914), so the detector genuinely does not want the full upscale. Cubic up
# to 1500 then linear down to 1200 measured better than feeding it 1200 directly.
#
# Measured inert and deliberately not set: box_thresh (0.30/0.40/0.55 give bit-identical
# accuracy — it only adds or drops low-score polygons carrying no text) and thresh (<0.01
# either way).
#
# ponytail: tuned on two pages of one document type at one resolution. 1200 vs 1000 is a
# 0.002 difference and is not meaningfully tuned — treat the pair as "tighter unclip, mild
# size bump", not as precise constants. Upgrade path if a real corpus disagrees: these are
# per-call constructor arguments, so they could become request parameters or be chosen from
# the measured median text height rather than fixed here.
DET_LIMIT_SIDE_LEN = 1200
DET_UNCLIP_RATIO = 1.0

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
    # Numeric-integrity signal from the digit donor. `digits_recovered` lists the runs put
    # back into this line; `digit_disagreement` says the donor saw a number here that could
    # NOT be placed, so a value is probably missing from the text and no one can say which.
    # See _recover_dropped_digits. Both travel to the sidecar so the caller can force review
    # on exactly these lines instead of trusting the page as a whole.
    digits_recovered: list[str] = field(default_factory=list)
    digit_disagreement: bool = False


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
    # Everything from here to _poly_bbox works in upscaled pixels; the caller never sees
    # them. See OCR_MIN_DIM_PX.
    image, scale = _upscale_for_ocr(image)
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

    readings, digit_notes = _recover_dropped_digits(readings, [crops[i] for i in keep])

    words = []
    for n, i in enumerate(keep):
        text, conf = readings[n]
        if not text:
            continue
        recovered, unplaced = digit_notes.get(n, ([], []))
        words.append(OcrWord(
            text=text,
            confidence=conf,
            bbox_px=_poly_bbox(polys[i], scale),
            flagged=conf < confidence_threshold,
            digits_recovered=recovered,
            digit_disagreement=bool(unplaced),
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


def _digit_notes(primary: str, donor: str, spliced: str) -> tuple[list[str], list[str]]:
    """(recovered, unplaced) digit runs, derived from what the splice actually did.

    Computed from the three strings rather than threaded out of _splice_digits' opcode loop,
    which keeps that function's contract (and its ten regression assertions) untouched.

    `unplaced` is the interesting half and the reason this exists. The donor can see a number
    that the splice then refuses to place — because the two readings disagree about the whole
    strip, or because the alignment offers no gap to put it in. Silently discarding that is
    the one outcome nobody can audit: the page ends up missing a value and reads as though it
    never had one. Reported instead, so the caller can demand a human look at that line.

    Membership is a substring test, matching the `missing` check in _splice_digits — so a
    donor run of "2" counts as already-present when primary holds "27250". Deliberate: it is
    the same rule both halves are judged by, and a looser one would flag every line whose
    numbers merely differ in grouping.
    """
    seen = _DIGIT_RUN.findall(donor)
    recovered = [run for run in seen if run not in primary and run in spliced]
    unplaced = [run for run in seen if run not in primary and run not in spliced]
    return recovered, unplaced


def _recover_dropped_digits(
    readings: list[tuple[str, float]], crops: list[np.ndarray],
) -> tuple[list[tuple[str, float]], dict[int, tuple[list[str], list[str]]]]:
    """Put back the numbers ARABIC_REC_MODEL silently deletes from Arabic sentences.

    arabic_PP-OCRv5_mobile_rec drops Western digits embedded mid-line in RTL text, usually
    leaving a double space where the number was and reading the surrounding Arabic
    correctly. Measured on two Arabic maths worksheets: `27250`, `8500` and `3240` are all
    lost this way, while `43500` survives only because it sits on its own line and is
    therefore its own crop. The digits are inside the detected polygon, so this is a
    decoder defect, not a detection or image one — it survived all 45 preprocessing
    variants and all 23 detector configurations tested against it.

    The Latin recogniser cannot help: on these crops it returns an empty string at 0.00,
    so there is nothing of its to merge. arabic_PP-OCRv3_mobile_rec, the previous
    generation, *does* emit the digits (3 of 4 numbers against v5's 1 of 4) but is worse at
    Arabic overall (0.8924 vs 0.9145), so it is not a replacement — it is used here purely
    as a digit donor, and none of its Arabic text is ever kept.

    Only crops whose chosen reading is Arabic are re-read, so a page with no Arabic pays
    nothing, and the second pass sees at most the Arabic subset rather than the page.

    Returns (readings, notes), where notes maps a reading's index to its
    (recovered, unplaced) digit runs and omits indices where neither happened.
    """
    targets = [
        n for n, (text, _) in enumerate(readings)
        if text and is_arabic(text) and n < len(crops)
    ]
    if not targets:
        return readings, {}

    donor = _recognise(ARABIC_DIGIT_DONOR_MODEL, [crops[n] for n in targets])
    out = list(readings)
    notes: dict[int, tuple[list[str], list[str]]] = {}
    for n, (donor_text, _) in zip(targets, donor):
        text, conf = out[n]
        spliced = _splice_digits(text, donor_text)
        out[n] = (spliced, conf)
        recovered, unplaced = _digit_notes(text, donor_text, spliced)
        if recovered or unplaced:
            notes[n] = (recovered, unplaced)
    return out, notes


def _splice_digits(primary: str, donor: str) -> str:
    """Return `primary` with digit runs the donor saw and it missed put back in place.

    Only digits cross over, and only at the position the alignment puts them — the donor's
    own Arabic is always discarded, because it is the weaker reader of everything except
    these numbers. Returns `primary` unchanged when there is nothing to add, so the common
    case (no digits on the line, or both readers agree) is a no-op.
    """
    missing = [run for run in _DIGIT_RUN.findall(donor) if run not in primary]
    if not missing:
        return primary
    if difflib.SequenceMatcher(None, primary, donor).ratio() < DIGIT_DONOR_MIN_SIMILARITY:
        # The two disagree about the whole strip, so the donor's digits are not evidence.
        return primary

    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, primary, donor).get_opcodes():
        segment = primary[i1:i2]
        out.append(segment)
        if tag in ("insert", "replace"):
            # Where the donor has digits and `primary` has none, the number was dropped.
            # A segment where primary *also* has digits is an ordinary misread, not a
            # deletion, and primary wins those as it wins every other character contest.
            runs = [r for r in _DIGIT_RUN.findall(donor[j1:j2]) if r in missing]
            if runs and not _DIGIT_RUN.search(segment):
                # Padded on both sides, then collapsed below: an inserted run otherwise
                # fuses onto the neighbouring word ("3240مي") when the gap primary left
                # behind was a single space rather than the usual double.
                out.append(f" {' '.join(runs)} ")
    return " ".join("".join(out).split())


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


def _upscale_for_ocr(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale a small page up to OCR_MIN_DIM_PX. Returns the image and the factor applied.

    Only ever enlarges — pipeline/preprocessor.py already caps the longest side, so an
    image at or above the target passes through untouched at scale 1.0 and the normal path
    pays nothing.
    """
    longest = max(image.shape[:2])
    if longest >= OCR_MIN_DIM_PX:
        return image, 1.0
    scale = OCR_MIN_DIM_PX / longest
    upscaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return upscaled, scale


def _poly_bbox(poly: np.ndarray, scale: float = 1.0) -> dict:
    """Axis-aligned {x, y, w, h} around a detection polygon, in pre-upscale pixels."""
    xs, ys = poly[:, 0] / scale, poly[:, 1] / scale
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
        if model_name == DET_MODEL:
            # Geometry kwargs go on the detector only — TextRecognition has no notion of
            # either, and paddleocr's mixins raise on unexpected keywords rather than
            # ignoring them, so the single shared `cls(...)` call this replaced could not
            # carry them.
            _predictors[model_name] = TextDetection(
                model_name=model_name, enable_mkldnn=False,
                limit_side_len=DET_LIMIT_SIDE_LEN, unclip_ratio=DET_UNCLIP_RATIO,
            )
        else:
            _predictors[model_name] = TextRecognition(
                model_name=model_name, enable_mkldnn=False,
            )
    return _predictors[model_name]
