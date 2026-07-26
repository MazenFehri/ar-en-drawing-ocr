"""Send a drawing to the running service and save the Word document it returns.

    python try_it.py                      # uses sample_drawing.png
    python try_it.py my_drawing.jpg       # uses your own image

Writes output.docx and sidecar.json next to the image. Open output.docx in Word.
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

img = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "sample_drawing.png")
if not img.exists():
    sys.exit(f"No such image: {img}")

print(f"health : {httpx.get(f'{API}/health', timeout=10).json()}")
print(f"sending: {img.name}")

start = time.time()
response = httpx.post(
    f"{API}/process",
    files={"image": (img.name, img.read_bytes(), mimetypes.guess_type(img.name)[0] or "image/png")},
    # try confidence_threshold=0.9 to push more words through the AI reviewer
    data={"confidence_threshold": "0.75", "language_hint": "ar+en", "label_shapes": "true"},
    timeout=900,
)
print(f"status : {response.status_code} in {time.time() - start:.1f}s")
response.raise_for_status()
result = response.json()

docx = img.with_name("output.docx")
docx.write_bytes(base64.b64decode(result["docx_base64"]))
sidecar = result["sidecar"]
img.with_name("sidecar.json").write_text(
    json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(f"\nwrote  : {docx.name} and sidecar.json")
print(f"stats  : {json.dumps(sidecar['stats'], indent=9)}")
print("\nwhat it found:")
for el in sidecar["elements"]:
    if el["type"] == "text":
        mark = f"  [{el['highlight']}]" if el["highlight"] else ""
        print(f"  text  {el['confidence']:.0%}  {el['content']}{mark}")
    else:
        print(f"  shape       {el.get('shape')}  {el.get('llm_label') or ''}")
