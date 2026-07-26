import re

import arabic_reshaper
from bidi.algorithm import get_display

# Runs of Latin letters/digits (and the punctuation that binds them, e.g. "3.5m")
_LTR_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,:/\-]*")


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


def to_logical_order(text: str) -> str:
    """Convert visually-ordered Arabic into logical (storage) order.

    PaddleOCR reports Arabic in the order the glyphs sit on the page, left to right,
    which is the reverse of how the string is stored in Unicode. Everything
    downstream — the sidecar handed to the caller, and Word, which applies its own
    bidi algorithm to <w:bidi/> paragraphs — expects logical order, so a raw OCR
    string would render backwards in both.

    Reversing the whole string recovers Arabic order; embedded Latin/digit runs were
    already left-to-right on the page, so they get flipped back.

    ponytail: single-line heuristic, correct for the short labels a drawing contains.
    Swap in python-bidi's reverse transform if multi-line paragraphs ever show up.
    """
    if not is_arabic(text):
        return text
    reversed_text = text[::-1]
    return _LTR_RUN.sub(lambda m: m.group()[::-1], reversed_text)


def reshape_for_display(text: str) -> str:
    """Reshape and apply bidi algorithm to Arabic text for correct visual rendering.

    English text is returned unchanged.
    """
    if not is_arabic(text):
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)
