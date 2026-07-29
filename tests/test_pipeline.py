import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline import process_image, PipelineResult

NOT_ATTEMPTED = {"state": "not_attempted", "reason": "no_flagged_words", "model": None}

FAKE_OCR_WORD = MagicMock(
    text="entrance", confidence=0.91,
    bbox_px={"x": 10, "y": 10, "w": 100, "h": 20},
    flagged=False
)


@patch("pipeline.label_complex_shapes")
@patch("pipeline.apply_corrections", return_value=([], NOT_ATTEMPTED))
@patch("pipeline.detect_shapes")
@patch("pipeline.run_ocr", return_value=[FAKE_OCR_WORD])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_word_correction_cannot_spend_the_whole_shared_llm_budget(
    mock_pre, mock_ocr, mock_shapes, mock_llm, mock_labels
):
    """Both LLM stages share one budget, so word correction (which runs first)
    could otherwise consume all of it and leave shape labelling nothing — that
    was measured happening on 2 of 3 runs of a real 167-element page. Word
    correction must be handed an earlier deadline than shape labelling."""
    from pipeline import CORRECTION_BUDGET_SHARE
    from pipeline.llm_corrector import TOTAL_LLM_BUDGET_SECONDS

    complex_shape = MagicMock(
        shape_type="complex", bbox_px={"x": 5, "y": 5, "w": 40, "h": 40},
        confidence=0.7, crop=np.ones((40, 40, 3), dtype=np.uint8) * 128,
    )
    mock_shapes.return_value = [complex_shape]
    mock_labels.return_value = ([], NOT_ATTEMPTED)

    process_image(np.ones((400, 600, 3), dtype=np.uint8) * 255, label_shapes=True)

    correction_deadline = mock_llm.call_args.kwargs["deadline"]
    labelling_deadline = mock_labels.call_args.kwargs["deadline"]
    assert correction_deadline < labelling_deadline, (
        "word correction got the full shared budget; shape labelling can be starved"
    )
    reserved = labelling_deadline - correction_deadline
    expected = TOTAL_LLM_BUDGET_SECONDS * (1 - CORRECTION_BUDGET_SHARE)
    assert reserved == pytest.approx(expected, abs=1.0)


def fake_line(y, width):
    return MagicMock(text="سطر", confidence=0.7,
                     bbox_px={"x": 10, "y": y, "w": width, "h": 15}, flagged=False)


@patch("pipeline.assemble_document", return_value=b"docx")
@patch("pipeline.apply_corrections", side_effect=lambda els, *a, **k: (els, NOT_ATTEMPTED))
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[fake_line(50 + i * 30, 500) for i in range(7)])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_prose_page_is_assembled_as_flowing_text(
    mock_pre, mock_ocr, mock_shapes, mock_llm, mock_assemble
):
    """Seven lines spanning 500 of 600px — the worksheet shape. Pinned textboxes
    scatter that page, so the classifier must flip word assembly into flow mode."""
    result = process_image(np.ones((400, 600, 3), dtype=np.uint8) * 255)
    assert mock_assemble.call_args.kwargs["flow_text"] is True
    assert result.sidecar["stats"]["document_type"] == "text"
    assert result.sidecar["stats"]["median_text_height_px"] == 15.0
    assert result.sidecar["stats"]["low_resolution"] is True


@patch("pipeline.assemble_document", return_value=b"docx")
@patch("pipeline.apply_corrections", side_effect=lambda els, *a, **k: (els, NOT_ATTEMPTED))
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[FAKE_OCR_WORD])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_drawing_page_keeps_pinned_textboxes(
    mock_pre, mock_ocr, mock_shapes, mock_llm, mock_assemble
):
    result = process_image(np.ones((400, 600, 3), dtype=np.uint8) * 255)
    assert mock_assemble.call_args.kwargs["flow_text"] is False
    assert result.sidecar["stats"]["document_type"] == "drawing"


@patch("pipeline.apply_corrections", return_value=([], NOT_ATTEMPTED))
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[FAKE_OCR_WORD])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_process_image_returns_pipeline_result(
    mock_pre, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert isinstance(result, PipelineResult)
    assert isinstance(result.docx_bytes, bytes)
    assert isinstance(result.sidecar, dict)


@patch("pipeline.apply_corrections", return_value=([], NOT_ATTEMPTED))
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[FAKE_OCR_WORD])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_process_image_sidecar_page_dimensions(
    mock_pre, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert result.sidecar["page_dimensions"]["width_px"] == 600
    assert result.sidecar["page_dimensions"]["height_px"] == 400


@patch("pipeline.apply_corrections", return_value=([], NOT_ATTEMPTED))
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_process_image_empty_elements(
    mock_pre, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert result.sidecar["stats"]["total_elements"] == 0
    assert isinstance(result.docx_bytes, bytes)
