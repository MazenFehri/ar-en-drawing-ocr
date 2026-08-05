"""Measure OCR accuracy against hand-transcribed ground truth.

Every quality number this project quotes elsewhere is a *confidence score* — the model's
opinion of itself. This measures correctness instead, so a tuning change can be judged
rather than argued about, and so a regression is visible.

    python tools/eval.py                  # every page that has ground truth
    python tools/eval.py test1 test3      # named pages only
    python tools/eval.py --lang ar        # override the language hint
    python tools/eval.py --self-test      # arithmetic check, needs no models or images

Ground truth lives in tests/fixtures/groundtruth/<name>.txt, one text line per line, in
reading order. The image is found by stem in samples/ or docs/images/.

Deliberately stops after layout reconstruction: no LLM. The LLM stage is networked,
rate-limited and nondeterministic, and it only ever touches words OCR already doubted.
Measuring without it gives the honest floor and a repeatable number.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRUTH_DIR = ROOT / "tests" / "fixtures" / "groundtruth"
IMAGE_DIRS = (ROOT / "samples", ROOT / "docs" / "images", ROOT)

# Numeric tokens, defined here rather than imported from pipeline.ocr on purpose: an
# evaluator that shares a regex with the code under test inherits its blind spots. Covers
# Western and Arabic-Indic digits plus the vulgar fractions (½ ¼ ¾), which are content on a
# maths worksheet and which OCR engines routinely flatten to "12".
_DIGITS = r"0-9٠-٩۰-۹"
_NUM = re.compile(rf"[{_DIGITS}¼-¾]+(?:[.,٫٬][{_DIGITS}]+)*")


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance over any two sequences. Two rows, so memory is O(len(b))."""
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (item_a != item_b),  # substitution
            ))
        previous = current
    return previous[-1]


# Sub/superscript digits folded to ASCII. PP-OCRv6 reads the "4" of a typeset "D₄" as
# U+2084 while reading D1-D3 as plain digits, so without this the evaluator scores a
# correct reading as both a character error and a lost number. Done as a targeted table
# rather than by NFKC, which would also fold ½ into "1/2" and destroy the fraction cases
# this is built to catch.
_SCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def normalise(text: str) -> str:
    """Collapse whitespace and fold presentation forms, so typography isn't scored as error.

    NFC rather than NFKC: it composes Arabic letter+mark pairs into single codepoints,
    which is what we want, without the compatibility folding that would destroy fractions.
    """
    return " ".join(unicodedata.normalize("NFC", text).translate(_SCRIPT_DIGITS).split())


def cer(truth: str, hypothesis: str) -> float:
    """Character error rate. 0.0 is perfect; can exceed 1.0 when output is longer than truth."""
    t, h = normalise(truth), normalise(hypothesis)
    if not t:
        return 0.0 if not h else 1.0
    return edit_distance(list(t), list(h)) / len(t)


def wer(truth: str, hypothesis: str) -> float:
    """Word error rate, on whitespace-separated tokens."""
    t, h = normalise(truth).split(), normalise(hypothesis).split()
    if not t:
        return 0.0 if not h else 1.0
    return edit_distance(t, h) / len(t)


def numeric_exactness(truth: str, hypothesis: str) -> tuple[int, int, list[str]]:
    """(recovered, expected, missing) for the numbers in `truth`.

    Order-preserving subsequence match, not a set intersection: on a maths worksheet the
    same value appears more than once, and a page that reads 8500 twice while dropping 3240
    must not score as complete.

    This is the metric that matters most here and the one a naive check gets wrong. The
    Arabic recogniser's failure mode is *deletion* — it leaves a double space where the
    number was — so comparing values between two engines finds nothing to compare. What
    must be measured is presence.
    """
    expected = _NUM.findall(normalise(truth))
    remaining = _NUM.findall(normalise(hypothesis))
    missing = []
    cursor = 0
    for number in expected:
        try:
            cursor += remaining[cursor:].index(number) + 1
        except ValueError:
            missing.append(number)
    return len(expected) - len(missing), len(expected), missing


def read_page(
    image_path: Path, language_hint: str, include_margin: bool = False, use_llm: bool = False,
) -> tuple[str, dict]:
    """Run preprocess -> OCR -> reading order, and return the page text, one line per row.

    Cells of an edge marking column are excluded by default, because the fixtures exclude
    them too — they are unreadable at these scan resolutions and transcribing a guess would
    be worse than omitting them. Scoring text the ground truth deliberately does not contain
    measures the fixture, not the pipeline.

    Pass include_margin=True to score everything. That is the honest number to quote when
    comparing against a run from before the column was identified at all, because moving
    those cells changes their edit-distance cost even though the text is unchanged.

    use_llm runs the correction stage too. Off by default because it is networked,
    rate-limited and nondeterministic — but it is the only way to measure a change to the
    prompt. Returns (text, info); `info` carries the LLM status and correction count, without
    which a rate-limited run is indistinguishable from a change that did nothing.
    """
    import cv2
    from pipeline.preprocessor import preprocess
    from pipeline.ocr import run_ocr
    from pipeline.layout_reconstructor import reconstruct_layout
    from models.elements import TextElement

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"could not decode {image_path}")
    processed = preprocess(image)
    height, width = processed.shape[:2]
    words = run_ocr(processed, language_hint=language_hint)
    elements = reconstruct_layout(words, [], width, height)

    info: dict = {}
    if use_llm:
        from pipeline.llm_corrector import apply_corrections
        from app.config import settings
        threshold = settings.confidence_threshold
        # Below threshold, and of those the ones actually sent — marking cells are excluded
        # from the send (see llm_corrector.apply_corrections). Reporting only the first number
        # overstates the work by 2.6x on test1 and hides that the skip is doing anything.
        below = [e for e in elements if isinstance(e, TextElement) and e.confidence < threshold]
        info["flagged"] = len(below)
        info["sent"] = sum(1 for e in below if not e.margin_column)
        _, buf = cv2.imencode(".jpg", processed)
        elements, status = apply_corrections(
            elements, buf.tobytes(), confidence_threshold=threshold,
        )
        info["llm"] = f"{status['state']}/{status['reason'] or 'ok'}"
        info["corrections"] = sum(
            1 for e in elements if isinstance(e, TextElement) and e.llm_correction is not None
        )

    text = "\n".join(
        e.content for e in elements
        if isinstance(e, TextElement) and (include_margin or not e.margin_column)
    )
    return text, info


def find_image(stem: str) -> Path:
    for directory in IMAGE_DIRS:
        for suffix in (".png", ".jpeg", ".jpg", ".tif", ".bmp"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    raise SystemExit(f"no image found for {stem!r} in {[str(d) for d in IMAGE_DIRS]}")


def self_test() -> None:
    """Check the arithmetic without needing models, images or ground truth."""
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], list("abc")) == 3
    assert edit_distance(list("abc"), []) == 3
    assert cer("abcd", "abcd") == 0.0
    assert cer("abcd", "abXd") == 0.25
    assert wer("one two three", "one two three") == 0.0
    assert wer("one two three", "one two") == 1 / 3

    # Whitespace differences are layout, not error.
    assert cer("a  b", "a b") == 0.0

    # The defect this exists for: v5 drops the number and leaves a gap behind.
    got, want, missing = numeric_exactness("bought for 27250 mi", "bought for  mi")
    assert (got, want, missing) == (0, 1, ["27250"]), (got, want, missing)

    # Order matters, and repeats are counted once each.
    assert numeric_exactness("1 2 1", "1 2 1")[:2] == (3, 3)
    assert numeric_exactness("1 2 1", "1 1")[0] == 2
    assert numeric_exactness("8500 3240", "3240 8500")[0] == 1

    # Arabic-Indic digits and fractions are numbers too.
    assert numeric_exactness("١٢٣ pieces", "١٢٣ pieces")[0] == 1
    assert numeric_exactness("½ dinar", "12 dinar") == (0, 1, ["½"])

    # A typeset subscript is the same number, not an error. Measured on test3: OCR reads
    # "D4" as "D₄" while reading D1-D3 as plain digits.
    assert numeric_exactness("D4 Capacity", "D₄ Capacity") == (1, 1, [])
    assert cer("D4", "D₄") == 0.0
    # ...but the fraction must survive that folding, or the ½ cases stop being detectable.
    assert numeric_exactness("½ dinar", "½ dinar")[0] == 1

    # A page with no numbers is complete, not divided by zero.
    assert numeric_exactness("no digits", "no digits") == (0, 0, [])
    print("self-test: all assertions passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pages", nargs="*", help="ground-truth stems; default is all of them")
    parser.add_argument("--lang", default="ar+en", help="language_hint passed to run_ocr")
    parser.add_argument("--self-test", action="store_true", help="check the metrics, then exit")
    parser.add_argument("--show", action="store_true", help="print the OCR output for each page")
    parser.add_argument("--include-margin", action="store_true",
                        help="also score edge marking-column cells, which the fixtures omit")
    parser.add_argument("--llm", action="store_true",
                        help="run the correction stage too (networked, nondeterministic)")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if not TRUTH_DIR.exists():
        raise SystemExit(f"no ground truth directory at {TRUTH_DIR}")
    wanted = set(args.pages)
    truths = sorted(p for p in TRUTH_DIR.glob("*.txt") if not wanted or p.stem in wanted)
    if not truths:
        raise SystemExit(f"no ground truth found in {TRUTH_DIR} for {args.pages or 'any page'}")

    rows = []
    for truth_path in truths:
        truth = truth_path.read_text(encoding="utf-8")
        output, info = read_page(
            find_image(truth_path.stem), args.lang, args.include_margin, args.llm)
        if args.show:
            print(f"\n--- {truth_path.stem} ---\n{output}\n")
        recovered, expected, missing = numeric_exactness(truth, output)
        rows.append({
            "page": truth_path.stem,
            "cer": cer(truth, output),
            "wer": wer(truth, output),
            "num": (recovered, expected),
            "missing": missing,
            "info": info,
        })

    width = max(len(r["page"]) for r in rows)
    print(f"\n{'page'.ljust(width)}   {'CER':>7} {'WER':>7} {'numbers':>9}")
    print("-" * (width + 28))
    for r in rows:
        got, want = r["num"]
        numbers = f"{got}/{want}" if want else "--"
        print(f"{r['page'].ljust(width)}   {r['cer']:>7.3f} {r['wer']:>7.3f} {numbers:>9}")
        if r["missing"]:
            print(f"{' ' * width}   lost: {', '.join(r['missing'])}")
        if r["info"]:
            # Always shown with --llm. A rate-limited run produces zero corrections and an
            # unchanged CER, which looks exactly like a prompt change that achieved nothing.
            print(f"{' ' * width}   llm: {r['info'].get('llm')} · "
                  f"{r['info'].get('corrections')} corrections · "
                  f"{r['info'].get('sent')} sent of {r['info'].get('flagged')} flagged")

    # Macro average — each page counts once, so a long page cannot mask several short bad
    # ones. With five pages that is the more honest summary; revisit if the set grows.
    mean_cer = sum(r["cer"] for r in rows) / len(rows)
    mean_wer = sum(r["wer"] for r in rows) / len(rows)
    got = sum(r["num"][0] for r in rows)
    want = sum(r["num"][1] for r in rows)
    print("-" * (width + 28))
    print(f"{'mean'.ljust(width)}   {mean_cer:>7.3f} {mean_wer:>7.3f} {f'{got}/{want}':>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
