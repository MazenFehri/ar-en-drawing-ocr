import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline.layout import segment_layout, LayoutRegion

MOCK_RESULT = [
    {"type": "text",   "bbox": [10, 20, 300, 60]},
    {"type": "figure", "bbox": [50, 100, 250, 250]},
    {"type": "text",   "bbox": [10, 270, 300, 310]},
]

@patch("pipeline.layout._get_structure")
def test_returns_layout_regions(mock_get):
    mock_engine = MagicMock(return_value=MOCK_RESULT)
    mock_get.return_value = mock_engine
    regions = segment_layout(np.ones((400, 400, 3), dtype=np.uint8) * 255)
    assert len(regions) == 3
    assert all(isinstance(r, LayoutRegion) for r in regions)

@patch("pipeline.layout._get_structure")
def test_text_regions_identified(mock_get):
    mock_engine = MagicMock(return_value=MOCK_RESULT)
    mock_get.return_value = mock_engine
    regions = segment_layout(np.ones((400, 400, 3), dtype=np.uint8) * 255)
    assert sum(1 for r in regions if r.region_type == "text") == 2

@patch("pipeline.layout._get_structure")
def test_figure_regions_identified(mock_get):
    mock_engine = MagicMock(return_value=MOCK_RESULT)
    mock_get.return_value = mock_engine
    regions = segment_layout(np.ones((400, 400, 3), dtype=np.uint8) * 255)
    assert sum(1 for r in regions if r.region_type == "figure") == 1

@patch("pipeline.layout._get_structure")
def test_bbox_px_fields(mock_get):
    mock_engine = MagicMock(return_value=MOCK_RESULT)
    mock_get.return_value = mock_engine
    regions = segment_layout(np.ones((400, 400, 3), dtype=np.uint8) * 255)
    r = regions[0]
    assert r.bbox_px["x"] == 10
    assert r.bbox_px["y"] == 20
    assert r.bbox_px["w"] == 290   # x2 - x1
    assert r.bbox_px["h"] == 40
