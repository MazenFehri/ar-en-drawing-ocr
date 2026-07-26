import cv2
import numpy as np

# Above this, denoising and OCR get slow and the base64 image sent to the LLM
# gets large, with no accuracy gain — architectural text is legible well below it.
MAX_DIM_PX = 2000

# Page-level skew beyond this is almost certainly a misdetection (a diagonal
# section line or a sparse page dominating the min-area rect), not a tilted scan.
MAX_SKEW_DEG = 15.0


def preprocess(image: np.ndarray) -> np.ndarray:
    """Run all preprocessing stages on a BGR image."""
    img = downscale(image)
    img = _deskew(img)
    img = _denoise(img)
    img = _enhance_contrast(img)
    return img


def downscale(image: np.ndarray, max_dim: int = MAX_DIM_PX) -> np.ndarray:
    """Shrink an image so its longest side is at most max_dim. Smaller images pass through."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def detect_quality(image: np.ndarray) -> float:
    """Return a sharpness score in [0, 1]. Below 0.1 is likely too blurry for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(laplacian_var / 1000.0, 1.0))


def _deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    if abs(angle) > MAX_SKEW_DEG:
        # Detection failed — rotating here would wreck the page. A drawing whose
        # ink is dominated by a diagonal (roof slope, section cut) or a nearly
        # blank page both yield ~45 deg here.
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
