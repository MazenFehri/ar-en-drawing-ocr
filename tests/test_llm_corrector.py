import pytest
from unittest.mock import patch, MagicMock
from models.elements import TextElement, BBox, LLMCorrection
from pipeline.llm_corrector import apply_corrections, _build_prompt


def test_unchanged_word_is_not_highlighted():
    """The model echoing a word back unchanged is a confirmation, not a correction."""
    from unittest.mock import patch as _patch, MagicMock as _MagicMock
    from pipeline.llm_corrector import apply_corrections as _apply
    from models.elements import TextElement as _TE, BBox as _BBox

    el = _TE(id="t0", bbox=_BBox(x=0.0, y=0.0, w=0.1, h=0.02),
             content="GROUND", language="english", confidence=0.92)
    resp = _MagicMock()
    resp.choices = [_MagicMock(message=_MagicMock(
        content='[{"original": "GROUND", "corrected": "GROUND", "certainty": 1.0}]'))]
    client = _MagicMock()
    client.chat.completions.create.return_value = resp

    with _patch("pipeline.llm_corrector._get_client", return_value=client):
        out = _apply([el], b"img", confidence_threshold=0.99)

    assert out[0].highlight is None
    assert out[0].llm_correction is None
    assert out[0].content == "GROUND"


def make_text_el(id_, text, conf):
    return TextElement(
        id=id_,
        bbox=BBox(x=0.0, y=0.0, w=0.1, h=0.02),
        content=text,
        language="english" if text.isascii() else "arabic",
        confidence=conf,
    )


MOCK_LLM_RESPONSE = '[{"original": "entrnce", "corrected": "entrance", "certainty": 0.97}]'
LOW_CERTAINTY_RESPONSE = '[{"original": "entrnce", "corrected": "entrance", "certainty": 0.4}]'


@patch("pipeline.llm_corrector._get_client")
def test_corrects_flagged_word(mock_client_fn):
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_LLM_RESPONSE
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    mock_client_fn.return_value = mock_client

    elements = [
        make_text_el("t0", "غرفة", 0.91),
        make_text_el("t1", "entrnce", 0.48),
    ]
    result = apply_corrections(elements, b"fake_image", confidence_threshold=0.75)
    corrected = [e for e in result if e.llm_correction is not None]
    assert len(corrected) == 1
    assert corrected[0].llm_correction.corrected == "entrance"
    assert corrected[0].llm_correction.certainty == pytest.approx(0.97)


@patch("pipeline.llm_corrector._get_client")
def test_high_certainty_gets_yellow(mock_client_fn):
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_LLM_RESPONSE
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    mock_client_fn.return_value = mock_client

    result = apply_corrections([make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75)
    assert result[0].highlight == "yellow"


@patch("pipeline.llm_corrector._get_client")
def test_low_certainty_gets_red(mock_client_fn):
    mock_choice = MagicMock()
    mock_choice.message.content = LOW_CERTAINTY_RESPONSE
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    mock_client_fn.return_value = mock_client

    result = apply_corrections([make_text_el("t1", "entrnce", 0.48)], b"fake", confidence_threshold=0.75)
    assert result[0].highlight == "red"


@patch("pipeline.llm_corrector._get_client")
def test_high_confidence_not_sent(mock_client_fn):
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    apply_corrections([make_text_el("t0", "entrance", 0.95)], b"fake", confidence_threshold=0.75)
    mock_client.chat.completions.create.assert_not_called()


def test_build_prompt_contains_words():
    elements = [make_text_el("t0", "entrnce", 0.48)]
    prompt = _build_prompt(elements)
    assert "entrnce" in prompt
    assert "0.48" in prompt
