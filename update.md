# Arabic Architectural OCR — Progress Update

**Date:** 26 July 2026
**Status:** Working end to end, running in Docker, verified against a test drawing.

---

## What it does

You give it a photo or scan of an architectural drawing. It returns a Word document
with the text, shapes and symbols placed roughly where they were on the original,
plus a JSON file describing everything found and how confident it was.

The hard part is that these drawings mix Arabic and English — two languages that read
in opposite directions sharing one page.

---

## What we used, and why

| Part | Choice | Why | Trade-off |
|---|---|---|---|
| Text recognition | **PaddleOCR** (PP-OCRv4, Arabic model) | Best free Arabic OCR. Runs locally, so no per-page cost and drawings never leave the machine. | Slower than a cloud OCR API, and its Arabic model is a version behind the newest. |
| Shape detection | **OpenCV** contour analysis | Geometry is maths, not AI — it is instant, free and predictable. | Only recognises simple shapes. Anything irregular is cropped as a picture instead. |
| Layout analysis | **PaddleOCR PP-Structure** | Already in the stack, separates text areas from drawing areas. | Trained on documents, not floor plans, so it is the weakest link. We compensate by also masking the detected words. |
| Text correction | **Vision LLM via OpenRouter** (currently Gemma 4, free tier) | Sees the whole drawing, so it can fix OCR errors using visual context. One provider, swappable models. | Free tier rate-limits unpredictably. Treated as optional — if it fails we return the raw OCR text. |
| Word output | **python-docx** with positioned text boxes | Puts every element at an absolute position, which is what "keep the layout" requires. | Output is a canvas of boxes, not flowing editable paragraphs. |
| Storage | **PostgreSQL** | Stores user corrections for the feedback loop. | — |
| Packaging | **Docker Compose** | One command starts the service and its database anywhere. | — |

**The overall trade-off:** OCR runs locally and free, and the AI is used only for the
few words OCR is unsure about. That keeps cost near zero and keeps the service working
even when the AI provider is down.

---

## What we fixed this session

This was the first run on a real image rather than unit tests. It surfaced several
genuine problems, all now fixed.

- **Arabic came out backwards.** The OCR reports Arabic in the order letters *appear*
  on the page, the reverse of how Arabic is stored. Word would then reverse it again,
  so documents showed nonsense. Most serious issue found; invisible until we looked at
  real output.
- **Words were in the wrong order.** "GROUND FLOOR PLAN" came out as "FLOOR PLAN
  GROUND". Lines are now read in their own direction, right-to-left for Arabic.
- **Page-straightening could destroy a drawing.** A long diagonal — a roof slope or
  section line, common in these drawings — was misread as page tilt and rotated the
  whole image 45 degrees.
- **Shapes and text were double-counted**, so text appeared twice: once as words, once
  as a pasted picture.
- **Large scans were processed at full size**, wasting seconds and sending huge images
  to the AI on every request.
- **Missing API options.** Callers can now set the confidence threshold, language, and
  symbol labelling, with invalid values rejected.

**Test result:** a floor plan with 3 rooms, a circle, a triangle, an irregular symbol
and 12 mixed Arabic/English labels. All 21 text labels found at 96% average
confidence, all shapes classified correctly, irregular symbol correctly identified as
a grid reference marker.

**Proof the AI correction works:** OCR misread "LIVING" as "LMVING" — but at 93%
confidence, so it was never flagged. Forcing every word through the AI reviewer caught
it, corrected it, and highlighted it yellow. That threshold is a tuning decision worth
your input: lower is faster but misses confident-but-wrong readings like this one.

---

## Speed

| | |
|---|---|
| Core pipeline (OCR, shapes, Word generation) | **3.2 seconds** |
| Plus symbol naming | 7–27 seconds |

The core is fast and predictable. All variability comes from the external AI service
on its free tier. Everything runs on CPU — no special hardware needed.

---

## Open items

- **No real drawing tested yet.** Everything above used a drawing we generated
  ourselves — clean, printed, perfectly straight. Real scans are skewed, noisy,
  handwritten. **Getting 5–10 real drawings is the most valuable next step.**
- **Free AI tier is not production-ready.** It rate-limited us mid-testing and forced
  a model switch. A paid key or a locally hosted model is needed before real use.
- **Feedback loop half-built.** User corrections are saved to the database as designed,
  but not yet fed back to improve future results.
- **Deployment target undecided.** Whether the host has a graphics card changes several
  technical choices, so we have not optimised for hardware we may not have.

---

## Next steps

1. Get real drawings and test against them.
2. Tune the confidence threshold using those results.
3. Decide the deployment target.
4. Connect the correction feedback loop.
5. Move off the free AI tier.

78 automated tests, all passing.
