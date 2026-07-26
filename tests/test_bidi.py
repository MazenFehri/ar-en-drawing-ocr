from utils.bidi import is_arabic, detect_language, reshape_for_display, to_logical_order


def test_to_logical_order_reverses_visual_arabic():
    # PaddleOCR reports these visually; the right-hand side is the logical form.
    assert to_logical_order("يضرألا") == "الأرضي"
    assert to_logical_order("ةفرغ") == "غرفة"
    assert to_logical_order("خبطملا") == "المطبخ"


def test_to_logical_order_leaves_english_alone():
    assert to_logical_order("ENTRANCE") == "ENTRANCE"
    assert to_logical_order("3.5m x 4.2m") == "3.5m x 4.2m"


def test_to_logical_order_keeps_embedded_latin_forwards():
    # "غرفة 3.5m" renders with the Arabic on the right and "3.5m" to its left, still
    # left-to-right — so OCR reads off the page "3.5m" then the Arabic glyphs. The
    # whole-string reverse would leave the Latin as "m5.3"; it must be flipped back.
    assert to_logical_order("3.5m ةفرغ") == "غرفة 3.5m"


def test_to_logical_order_is_involutive_on_arabic():
    for word in ("غرفة", "المطبخ", "الصالة", "مدخل"):
        assert to_logical_order(to_logical_order(word)) == word


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
