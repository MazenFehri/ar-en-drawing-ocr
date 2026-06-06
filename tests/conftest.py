import numpy as np
import cv2
import pytest


@pytest.fixture
def blank_image():
    return np.ones((800, 600, 3), dtype=np.uint8) * 255


@pytest.fixture
def circle_image():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img, (100, 100), 50, (0, 0, 0), -1)
    return img


@pytest.fixture
def triangle_image():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    pts = np.array([[100, 20], [20, 180], [180, 180]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    return img


@pytest.fixture
def rect_image():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 60), (170, 140), (0, 0, 0), -1)
    return img
