from utils.bidi import is_arabic, detect_language, reshape_for_display

# The four to_logical_order tests that used to open this file are gone with the function.
# They pinned a v4 compensation: PP-OCRv4's arabic model returned visual order, so the
# pipeline reversed every Arabic string on the way out. arabic_PP-OCRv5_mobile_rec
# returns logical order already, so the reversal became a corruption rather than a fix.
# The replacement guard is a real one, at the layer where it can actually catch a
# regression: test_layout_reconstructor.py asserts a recognised Arabic phrase reaches the
# element unchanged, codepoint for codepoint.


def test_is_arabic_with_arabic_text():
    assert is_arabic("غرفة النوم") is True


def test_is_arabic_with_english_text():
    assert is_arabic("bedroom") is False


def test_is_arabic_with_mixed_text():
    # Majority Arabic characters → True
    assert is_arabic("غرفة النوم 3.5m") is True


def test_is_arabic_empty_string():
    assert is_arabic("") is False


def test_detect_language_arabic():
    assert detect_language("غرفة النوم") == "arabic"


def test_detect_language_english():
    assert detect_language("entrance hall") == "english"


def test_detect_language_mixed():
    # "ab" (2 Latin) + "غر" (2 Arabic) = 50% Arabic → "mixed"
    assert detect_language("ab غر") == "mixed"


def test_detect_language_numbers_only():
    # Pure numbers have no alphabetic chars → defaults to "english"
    result = detect_language("3.5")
    assert result in ("english", "mixed")  # acceptable either way


def test_reshape_arabic_returns_string():
    result = reshape_for_display("غرفة النوم")
    assert isinstance(result, str)
    assert len(result) > 0


def test_reshape_english_passthrough():
    result = reshape_for_display("entrance")
    assert result == "entrance"
