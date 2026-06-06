import numpy as np
import cv2
import pytest
from pipeline.preprocessor import preprocess, detect_quality


def test_preprocess_returns_ndarray(blank_image):
    result = preprocess(blank_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == blank_image.shape


def test_preprocess_handles_skewed_image():
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.line(img, (100, 200), (500, 200), (0, 0, 0), 3)
    result = preprocess(img)
    assert result.shape[0] > 0
    assert result.shape[1] > 0


def test_detect_quality_returns_float(blank_image):
    score = detect_quality(blank_image)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_detect_quality_blurry_is_low():
    blurry = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(blurry, (21, 21), 0)
    score = detect_quality(blurry)
    assert score < 0.5


def test_preprocess_preserves_color_channels(blank_image):
    result = preprocess(blank_image)
    assert len(result.shape) == 3
    assert result.shape[2] == 3
