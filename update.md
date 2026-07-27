# Arabic Architectural OCR — Progress Update

**Date:** 27 July 2026
**Status:** Working end to end, verified in Docker against the real pipeline. Two rounds of fixes since the last update, both triggered by testing on a genuine document rather than our own test image.

---

## What it does

You give it a photo or scan of an architectural drawing. It returns a Word document
with the text, shapes and symbols placed where they were on the original, plus a JSON
file describing everything found and how confident it was.

The hard part is that these drawings mix Arabic and English — two languages that read
in opposite directions sharing one page.

---

## Verified working

Measured in Docker, on the real pipeline (not shortcuts or mocks):

| Check | Result |
|---|---|
| Speed, core pipeline (warm) | 2.8–3.3 seconds |
| Speed, with symbol naming | ~20 seconds |
| Speed, with every word double-checked | 25–27 seconds |
| Word document validity | Valid; every element placed at an exact position |
| Page shape | Matches the source image correctly (see below) |
| Elements placed off the page | Zero |
| Arabic reading order | Correct, on the real OCR path |
| Automated tests | 117 passing |

---

## What we used, and why

| Part | Choice | Why | Trade-off |
|---|---|---|---|
| Text recognition | **PaddleOCR** (PP-OCRv4, Arabic model) | Best free Arabic OCR. Runs locally, so no per-page cost and drawings never leave the machine. | Slower than a cloud OCR API, and its Arabic model is a version behind the newest (see Open items). |
| Shape detection | **OpenCV** contour analysis | Geometry is maths, not AI — it is instant, free and predictable. | Only recognises simple shapes. Anything irregular is cropped as a picture instead. |
| Layout analysis | **PaddleOCR PP-Structure** | Already in the stack, separates text areas from drawing areas. | Trained on documents, not floor plans, so it is the weakest link. We compensate by also masking the detected words. |
| Text correction | **Vision LLM via OpenRouter** (currently a free-tier Gemma model) | Double-checks words the OCR is unsure about, by looking at the image. One provider, swappable models. | Free tier rate-limits unpredictably. Treated as optional — if it fails we return the raw OCR text. |
| Word output | **python-docx** with positioned text boxes | Puts every element at an exact position, which is what "keep the layout" requires. | Output is a canvas of boxes, not flowing editable paragraphs. |
| Storage | **PostgreSQL** | Stores user corrections for the feedback loop. | — |
| Packaging | **Docker Compose** | One command starts the service and its database anywhere. | — |

**The overall trade-off:** OCR runs locally and free, and the AI is used only to
double-check the few words OCR is unsure about. That keeps cost near zero and keeps
the service working even when the AI provider is down.

---

## Round 2 — what the first real document exposed

The first genuine document we ran (a printed Arabic university registration table)
exposed three faults our own test image was never shaped in a way to reveal.

1. **Positions were wrong on every document.** Elements were placed as if every page
   were the same fixed shape, regardless of the actual shape of the original. The test
   document was distorted 44%, worse toward the bottom of the page. The page now
   matches the proportions of whatever is fed in; measured error afterwards: **0.001%**.
   Separately, the document had also been the wrong paper size all along (US Letter
   instead of A4), while the positioning maths assumed A4 — that is now consistent.
2. **Most of the drawing was being thrown away.** Only the outermost outline of each
   shape was kept, so anything drawn *inside* a room or a table — interior walls,
   fixtures, furniture — never reached the document. On the test drawing this went
   from 11 elements captured to 30. Anything still unrecognised is now cut out of the
   original image and pasted in at the right position, so nothing on the page is
   silently lost.
3. **There was no way to tell whether the AI reviewer had run.** "0 corrections"
   looked identical whether the model had checked every word and approved it, or had
   never been reached at all. The result now says which, and why.

---

## Round 3 — the AI reviewer, fixed properly

Our working theory had been that the AI reviewer was being rate-limited. Live testing
ruled that out: the key was valid, the model responded, and the connection worked.
The real problem was how we were asking the question. The reviewer was shown the
*entire page* and asked what a handful of small, low-confidence words said, with no
indication of where on the page they were. It could not find them, so it simply
repeated the OCR's original guess back — at maximum confidence, even when that guess
was wrong.

It now receives a close-up, enlarged photo of each individual word instead of the
whole page.

To test this properly, we designed a check the model could not pass by guessing: each
word was deliberately mislabelled with a plausible but *unrelated* wrong answer (the
word "BEDROOM" was flagged as if it might say "PARKING"), so the only way to answer
correctly was to actually read the image.

| | Old (whole page) | New (word close-ups) |
|---|---|---|
| Correctly re-read | 0 out of 5 | 5 out of 5 |

Along the way we also found the reviewer occasionally returns a blank answer instead
of a word. The old code would have written that blank into the document, silently
erasing a correct piece of text. Blank answers are now rejected rather than applied.

**Confirmed on the real service.** Running the full pipeline with every word sent for
review, the reviewer found and fixed a genuine error — the OCR had read the word
"LIVING" as "LMVING" — and correctly left the other eleven words it checked untouched,
including all the Arabic. Nothing correct was changed into something wrong, which is
the failure we were most concerned about.

---

## Security

A routine dependency check found that the library used to decode uploaded images is
being held at an old version — a requirement of the OCR engine — and that old version
has a publicly known flaw, already exploited elsewhere, affecting one image format
(WebP).

Checking the file's declared type was not enough, because that label can be faked by
whoever uploads the file. So the service now inspects the actual file contents and
refuses any image that is really a WebP, regardless of what it claims to be. Uploads
in other formats (JPEG, PNG, TIFF, BMP) are unaffected.

We also removed two unused packages found during the same check.

---

## Open items

- **Only one real document has been tested**, and it was a printed table rather than
  an architectural drawing. Getting 5–10 real drawings remains the most valuable next
  step.
- **The free AI tier still rate-limits unpredictably**, and is sometimes simply slow.
  A paid key is needed before production use. In the meantime the system retries,
  falls back to alternative models, and enforces a 60-second limit on each of the two
  AI stages — so a request that uses both and hits the limit on each can take about two
  minutes at worst before returning the document from the raw OCR text. Testing caught
  an earlier version of this limit not being enforced at all; it now is.
- **The feedback loop is half-built.** User corrections are saved to the database but
  not yet used to improve future results.
- **Deployment target still undecided.**
- **A newer version of the OCR engine (PP-OCRv5) would improve Arabic accuracy**, but
  it is a substantial migration rather than a simple upgrade — it changes several
  internal interfaces and risks reintroducing a crash previously seen on certain Intel
  processors. Worth scoping as its own piece of work.

---

## Next steps

1. Get real drawings and test against them.
2. Move off the free AI tier.
3. Decide the deployment target.
4. Connect the correction feedback loop.
5. Scope the OCR engine upgrade separately.
