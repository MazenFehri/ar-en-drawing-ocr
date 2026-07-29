import arabic_reshaper
from bidi.algorithm import get_display

# There used to be a `to_logical_order()` here, and pipeline/layout_reconstructor.py
# called it on every recognised word. It reversed Arabic strings, because PP-OCRv4's
# arabic model emitted text in *visual* order — the order the glyphs sit on the page,
# left to right — which is the reverse of Unicode storage order, and both the sidecar and
# Word (which runs its own bidi algorithm over <w:bidi/> paragraphs) expect logical order.
#
# arabic_PP-OCRv5_mobile_rec does not do that. It returns whole phrases already in
# logical order. Verified on sample_drawing.png by codepoint, not by rendering (rendered
# Arabic looks plausible either way, which is exactly how this bug hides): the model
# returns U+0645 062E 0637 0637 0020 0627 0644 0637 0627 0628 0642 ... for
# "مخطط الطابق الأرضي", which is the correct storage order, not its reverse.
#
# So the compensation is gone. Applying it now would be a second correction on top of a
# correct string and would silently ship reversed Arabic in every document — the failure
# mode being that it still *renders* as Arabic and still round-trips through
# is_arabic()/detect_language() unchanged, so nothing downstream would complain.
# If a future recogniser goes back to visual order, restore the reversal here rather than
# in the caller, and pin it with a codepoint-level test like the one above.


def is_arabic(text: str) -> bool:
    """True if more than 30% of the characters are Arabic Unicode block."""
    if not text:
        return False
    arabic_chars = sum(1 for c in text if '؀' <= c <= 'ۿ')
    return arabic_chars / len(text) > 0.3


def detect_language(text: str) -> str:
    """Classify text as 'arabic', 'english', or 'mixed' based on character ratios."""
    if not text.strip():
        return "english"
    arabic_chars = sum(1 for c in text if '؀' <= c <= 'ۿ')
    latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    total = arabic_chars + latin_chars
    if total == 0:
        return "english"
    arabic_ratio = arabic_chars / total
    if arabic_ratio > 0.7:
        return "arabic"
    if arabic_ratio < 0.3:
        return "english"
    return "mixed"


def reshape_for_display(text: str) -> str:
    """Reshape and apply bidi algorithm to Arabic text for correct visual rendering.

    English text is returned unchanged.
    """
    if not is_arabic(text):
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)
