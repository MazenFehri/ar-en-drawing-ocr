import cv2
import numpy as np

# Above this, denoising and OCR get slow and the base64 image sent to the LLM
# gets large, with no accuracy gain — architectural text is legible well below it.
MAX_DIM_PX = 2000

# Page-level skew beyond this is almost certainly a misdetection (a diagonal
# section line or a sparse page dominating the min-area rect), not a tilted scan.
MAX_SKEW_DEG = 15.0


def preprocess(image: np.ndarray) -> np.ndarray:
    """Run all preprocessing stages on a BGR image.

    The CLAHE contrast stage that used to end this function is now `enhance_contrast`,
    applied by the caller to the shape-detection input only — see there for why. OCR reads
    the un-enhanced page.

    Also measured and NOT changed, because weakening it made things worse monotonically:
    the denoise strength. h=10 (kept) scored 0.8122, h=5 0.8206 but losing test2
    characters, h=3 0.7732, and no denoising at all 0.7986. The theory that
    fastNlMeansDenoisingColored was erasing the 1-2px Arabic letter dots is not supported —
    denoising helps at this resolution.
    """
    img = downscale(image)
    img = _deskew(img)
    img = _denoise(img)
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



def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE over the LAB lightness channel. For shape detection, NOT for OCR.

    This used to run at the end of preprocess(), so OCR and shape detection saw the same
    enhanced page. They want opposite things, and doing it once was quietly costing both.

    OCR is better without it. Scored as mean per-line best match against a hand-transcribed
    ground truth for two ~45 DPI Arabic worksheets, two identical reps:

        with CLAHE     combined 0.8955    17 of 32 lines fully recovered
        without        combined 0.9066    21 of 32

    That was the only variant of 45 tested that beat baseline both before and after the
    detector geometry fix, so it is a real effect, not something compensating for bad
    crops. Expected mechanism: local equalisation over an 8x8 grid amplifies per-tile noise
    on a low-DPI scan, and a tile straddling a text line and the white margin stretches the
    two apart in a way that hardens antialiased glyph edges.

    Shape detection is much better WITH it, which is why it survives rather than being
    deleted. Measured on the same pages, same run:

        test2.jpeg        with: 29 shapes (circle 5, ellipse 1, line 17, polyline 4, rect 2)
                       without: 19 shapes (line 17, rect 2)
        class-diagram     with: 91        without: 87
        sample_drawing    with:  7        without:  7

    Every circle on test2 disappears without it — and on that worksheet the circles are the
    coin values, i.e. content, not decoration. Faint printed strokes on a low-contrast scan
    need the local stretch to survive Otsu; glyphs do not.

    ponytail: the caller applies this on top of preprocess()'s output rather than
    preprocess() returning both variants, so a page is walked once more (~0.01s here).
    Upgrade path if a much larger page makes that matter: return a pair.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
