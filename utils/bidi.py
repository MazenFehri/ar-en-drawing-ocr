import arabic_reshaper
from bidi.algorithm import get_display


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
    if arabic_ratio > 0.85:
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
