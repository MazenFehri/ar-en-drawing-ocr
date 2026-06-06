import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline import process_image, PipelineResult


FAKE_OCR_WORD = MagicMock(
    text="entrance", confidence=0.91,
    bbox_px={"x": 10, "y": 10, "w": 100, "h": 20},
    flagged=False
)


@patch("pipeline.apply_corrections", return_value=[])
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[FAKE_OCR_WORD])
@patch("pipeline.segment_layout", return_value=[])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_process_image_returns_pipeline_result(
    mock_pre, mock_layout, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert isinstance(result, PipelineResult)
    assert isinstance(result.docx_bytes, bytes)
    assert isinstance(result.sidecar, dict)


@patch("pipeline.apply_corrections", return_value=[])
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[FAKE_OCR_WORD])
@patch("pipeline.segment_layout", return_value=[])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_process_image_sidecar_page_dimensions(
    mock_pre, mock_layout, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert result.sidecar["page_dimensions"]["width_px"] == 600
    assert result.sidecar["page_dimensions"]["height_px"] == 400


@patch("pipeline.apply_corrections", return_value=[])
@patch("pipeline.detect_shapes", return_value=[])
@patch("pipeline.run_ocr", return_value=[])
@patch("pipeline.segment_layout", return_value=[])
@patch("pipeline.preprocess", side_effect=lambda x: x)
def test_process_image_empty_elements(
    mock_pre, mock_layout, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert result.sidecar["stats"]["total_elements"] == 0
    assert isinstance(result.docx_bytes, bytes)
