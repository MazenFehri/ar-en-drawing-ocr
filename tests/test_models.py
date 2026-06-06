from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement, LLMCorrection

def test_bbox_relative_coords():
    bbox = BBox(x=0.1, y=0.2, w=0.3, h=0.05)
    assert bbox.x == 0.1
    assert bbox.w == 0.3

def test_text_element_defaults():
    el = TextElement(
        id="text_001",
        bbox=BBox(x=0.0, y=0.0, w=0.5, h=0.03),
        content="غرفة النوم",
        language="arabic",
        confidence=0.91,
    )
    assert el.type == "text"
    assert el.llm_correction is None
    assert el.highlight is None

def test_llm_correction_attached():
    correction = LLMCorrection(original="entrnce", corrected="entrance", certainty=0.97)
    el = TextElement(
        id="text_002",
        bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.02),
        content="entrance",
        language="english",
        confidence=0.48,
        llm_correction=correction,
        highlight="yellow",
    )
    assert el.llm_correction.corrected == "entrance"
    assert el.highlight == "yellow"

def test_simple_shape_element():
    el = SimpleShapeElement(
        id="shape_001",
        bbox=BBox(x=0.3, y=0.2, w=0.1, h=0.1),
        shape="circle",
        confidence=0.97,
    )
    assert el.type == "simple_shape"

def test_complex_shape_element():
    el = ComplexShapeElement(
        id="shape_002",
        bbox=BBox(x=0.5, y=0.3, w=0.15, h=0.1),
        llm_label="door swing",
        llm_label_certainty=0.94,
    )
    assert el.type == "complex_shape"
    assert el.embedded_as == "image"
