"""Send a page to the running service and save the Word document it returns.

    python try_it.py                        # uses sample_drawing.png
    python try_it.py my_scan.jpg            # uses your own image
    python try_it.py my_scan.jpg --ar       # Arabic only (skips the Latin reader, faster)
    python try_it.py my_scan.jpg --en       # English only
    python try_it.py my_scan.jpg --label    # also ask the LLM to name unrecognised shapes

Writes <image-name>.docx and <image-name>.sidecar.json next to the image, so testing a
second image does not overwrite the first. Open the .docx in Word.
"""
import base64
import json
import mimetypes
import pathlib
import sys
import time

import httpx

# Windows consoles default to cp1252, which cannot print Arabic.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "http://localhost:8000"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}

img = pathlib.Path(args[0] if args else "sample_drawing.png")
if not img.exists():
    sys.exit(f"No such image: {img}")

hint = "ar" if "--ar" in flags else "en" if "--en" in flags else "ar+en"
# Off by default: shape labelling costs a second LLM round trip, and on a free
# OpenRouter tier that is usually a ~30s wait to be told "rate_limited".
label = "true" if "--label" in flags else "false"

print(f"health : {httpx.get(f'{API}/health', timeout=10).json()}")
print(f"sending: {img.name}  (language_hint={hint}, label_shapes={label})")

start = time.time()
response = httpx.post(
    f"{API}/process",
    files={"image": (img.name, img.read_bytes(), mimetypes.guess_type(img.name)[0] or "image/png")},
    # try confidence_threshold=0.9 to push more words through the AI reviewer
    data={"confidence_threshold": "0.75", "language_hint": hint, "label_shapes": label},
    timeout=900,
)
print(f"status : {response.status_code} in {time.time() - start:.1f}s")
response.raise_for_status()
result = response.json()

docx = img.with_suffix(".docx")
docx.write_bytes(base64.b64decode(result["docx_base64"]))
sidecar = result["sidecar"]
img.with_suffix(".sidecar.json").write_text(
    json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
)

stats = sidecar["stats"]
print(f"\nwrote  : {docx.name} and {img.with_suffix('.sidecar.json').name}")
print(f"found  : {stats['text_elements']} text, "
      f"{stats['simple_shapes']} simple shapes, {stats['polyline_shapes']} freeform, "
      f"{stats['complex_shapes']} embedded images")
print(f"llm    : {stats['llm_status']['state']}"
      + (f" ({stats['llm_status']['reason']})" if stats["llm_status"].get("reason") else ""))

# The one thing worth interrupting for: confidences look merely mediocre on a scan
# that was never readable, so say it in words rather than leaving it in the JSON.
if stats.get("low_resolution"):
    print(f"\n  !! LOW RESOLUTION: text lines are only {stats['median_text_height_px']:.0f}px tall.")
    print("     Arabic letters differ by dots 1-2px across at this size, so they are not")
    print("     sampled at all and the reading is partly guesswork. Rescan at 200-300 DPI")
    print("     if you can — no OCR model recovers what the scan never captured.")

print("\nwhat it found:")
for el in sidecar["elements"]:
    if el["type"] == "text":
        mark = f"  [{el['highlight']}]" if el["highlight"] else ""
        print(f"  text  {el['confidence']:.0%}  {el['content']}{mark}")
    else:
        print(f"  shape       {el.get('shape')}  {el.get('llm_label') or ''}")
