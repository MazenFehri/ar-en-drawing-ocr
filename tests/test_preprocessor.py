import numpy as np
import cv2
import pytest
from pipeline.preprocessor import preprocess, detect_quality, downscale, _deskew


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


def test_downscale_shrinks_oversized_image_keeping_aspect():
    big = np.ones((4000, 3000, 3), dtype=np.uint8) * 255
    out = downscale(big, max_dim=2000)
    assert max(out.shape[:2]) == 2000
    assert out.shape[0] / out.shape[1] == pytest.approx(4000 / 3000, abs=0.01)


def test_downscale_leaves_small_image_untouched():
    small = np.ones((300, 400, 3), dtype=np.uint8) * 255
    assert downscale(small, max_dim=2000) is small


def test_deskew_ignores_implausible_angle():
    """A page whose ink is dominated by a diagonal yields a ~45 deg min-area rect.
    Rotating by that would destroy the drawing, so it must be left alone."""
    img = np.ones((800, 800, 3), dtype=np.uint8) * 255
    cv2.line(img, (100, 700), (700, 100), (0, 0, 0), 6)   # roof slope / section cut
    cv2.putText(img, "ROOF", (300, 750), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    assert _deskew(img) is img


def test_deskew_still_corrects_real_skew():
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    for y in range(80, 320, 40):
        cv2.line(img, (60, y), (540, y), (0, 0, 0), 4)
    M = cv2.getRotationMatrix2D((300, 200), 6.0, 1.0)
    skewed = cv2.warpAffine(img, M, (600, 400), borderMode=cv2.BORDER_REPLICATE)
    assert _deskew(skewed) is not skewed   # a plausible skew is acted on
