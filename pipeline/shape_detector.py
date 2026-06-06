import math
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

MIN_AREA_PX = 400  # Ignore contours smaller than 20×20 px


@dataclass
class ShapeResult:
    shape_type: str   # "circle", "ellipse", "triangle", "rect", "square", "line", "complex"
    bbox_px: dict     # {x, y, w, h} in pixel coordinates
    confidence: float
    crop: Optional[np.ndarray] = None  # Only set for complex shapes


def detect_shapes(image: np.ndarray, text_bboxes_px: list[dict]) -> list[ShapeResult]:
    """Detect non-text shapes in an image.

    text_bboxes_px: list of {x, y, w, h} dicts in pixel coords — these regions are masked.
    Returns list of ShapeResult, one per detected contour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Mask text regions so OCR text regions don't appear as shapes
    for tb in text_bboxes_px:
        x, y, w, h = int(tb["x"]), int(tb["y"]), int(tb["w"]), int(tb["h"])
        cv2.rectangle(binary, (x, y), (x + w, y + h), 0, -1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        shape_type, confidence = _classify(cnt)
        crop = image[y:y + h, x:x + w].copy() if shape_type == "complex" else None
        results.append(ShapeResult(
            shape_type=shape_type,
            bbox_px={"x": x, "y": y, "w": w, "h": h},
            confidence=confidence,
            crop=crop,
        ))
    return results


def _classify(contour) -> tuple[str, float]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "triangle", 0.95

    if vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        # Extent: ratio of contour area to its bounding box area.
        # Rectangles fill ~100% of their bbox; ellipses fill ~π/4 ≈ 78%.
        extent = cv2.contourArea(contour) / bbox_area if bbox_area > 0 else 1.0
        if extent < 0.85:
            # Low extent means the shape is curved (ellipse/circle), not a flat rect
            circ = _circularity(contour)
            if circ > 0.85:
                return "circle", float(circ)
            return "ellipse", float(circ)
        ar = w / h if h > 0 else 1.0
        if 0.9 <= ar <= 1.1:
            return "square", 0.93
        return "rect", 0.93

    # For 5+ vertices: use circularity + solidity
    circ = _circularity(contour)
    solidity = _solidity(contour)

    if circ > 0.85:
        return "circle", float(circ)
    if circ > 0.80 and solidity > 0.92:
        # High circularity + high solidity → smooth convex curve = ellipse
        return "ellipse", float(circ)
    return "complex", 0.70


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return (4 * math.pi * area) / (peri ** 2)


def _solidity(contour) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return cv2.contourArea(contour) / hull_area
