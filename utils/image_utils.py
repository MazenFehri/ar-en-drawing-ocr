import io
import numpy as np
from PIL import Image


def ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert a BGR numpy array to PNG bytes."""
    # Convert BGR to RGB for PIL
    rgb = arr[:, :, ::-1] if arr.ndim == 3 and arr.shape[2] == 3 else arr
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
