from dataclasses import dataclass
import numpy as np

_structure_instance = None


@dataclass
class LayoutRegion:
    region_type: str   # "text", "figure", "table", "title"
    bbox_px: dict      # {x, y, w, h}


def segment_layout(image: np.ndarray) -> list[LayoutRegion]:
    """Segment image into regions using PaddleOCR ppstructure."""
    engine = _get_structure()
    raw = engine(image)
    regions = []
    for item in raw:
        x1, y1, x2, y2 = item["bbox"]
        rtype = item["type"] if item["type"] in ("text", "figure", "table", "title") else "figure"
        regions.append(LayoutRegion(
            region_type=rtype,
            bbox_px={"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
        ))
    return regions


def _get_structure():
    global _structure_instance
    if _structure_instance is None:
        from pipeline import _paddle_patch  # noqa: F401  (patches paddle before predictor build)
        from paddleocr import PPStructure
        pp = PPStructure(show_log=False)
        _structure_instance = lambda img: [
            {"type": r["type"], "bbox": r["bbox"]} for r in pp(img)
        ]
    return _structure_instance
