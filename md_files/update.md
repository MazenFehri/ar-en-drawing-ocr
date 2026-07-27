# UPDATE


## How it works

A drawing goes through eight steps, in this order:

1. **Clean up the image.** Large scans are shrunk,
2. **Split the page into regions**, using PaddleOCR PP-Structure : separates text
   areas from drawing areas so each is handled correctly further down the line.
3. **Read the words**, using PaddleOCR PP-OCRv4's Arabic model very word is read,
   each with a confidence score.
4. **Find the shapes**, using OpenCV. This runs *after* the words, on purpose, so
   recognised text can be blanked out first and never gets captured twice, once as text
   and once as a picture. It works by tracing each dark region's outline and measuring
   its geometry — how round, how many corners, how solid — arithmetic, not AI, so it is
   instant and predictable. Whatever it cannot identify is still cut out of the original
   image and carried through as a picture, so nothing on the page is silently dropped.
5. **Rebuild the layout.** Positions become proportional rather than fixed pixels,
   words group into lines, each line reads in its own direction (right to left for
   Arabic), and Arabic converts from the order it is drawn in to the order it is
   actually stored and read.
6. **Let an AI double-check the hard parts**, using a vision model via OpenRouter.
   Two separate, optional passes: unsure words are each shown as an enlarged close-up
   of just that one word, and unclassified shapes are shown to the AI to name.
7. **Build the Word document**, using python-docx — every element placed at its exact
   position, on a page shaped to match the original drawing's proportions.
8. **Write the results file** — a JSON file listing everything found, with its
   position, confidence, and whether the AI reviewed it.

## What we used, and why

| Part | Choice | Why | Trade-off |
|---|---|---|---|
| Text recognition | **PaddleOCR** (PP-OCRv4, Arabic model) | Best free Arabic OCR. Runs locally, so no per-page cost and drawings never leave the machine. | Slower than a cloud OCR API, and its Arabic model is a version behind the newest (see Open items). |
| Shape detection | **OpenCV** contour analysis | Geometry is maths, not AI — it is instant, free and predictable. | Only recognises simple shapes. Anything irregular is cropped as a picture instead. |
| Layout analysis | **PaddleOCR PP-Structure** | Already in the stack, separates text areas from drawing areas. | Trained on documents, not floor plans, so it is the weakest link. We compensate by also masking the detected words. |
| Text correction | **Vision LLM via OpenRouter** (currently a free-tier Gemma model) | Double-checks words the OCR is unsure about, now by looking at an enlarged close-up of each individual word rather than the whole page. One provider, swappable models. | Free tier rate-limits unpredictably. Treated as optional — if it fails we return the raw OCR text. |
| Word output | **python-docx** with positioned text boxes | Puts every element at an exact position, which is what "keep the layout" requires. | Output is a canvas of boxes, not flowing editable paragraphs. |
| Storage | **PostgreSQL** | Stores user corrections for the feedback loop. | — |
| Packaging | **Docker Compose** | One command starts the service and its database anywhere. | — |

**The overall trade-off:** OCR runs locally and free, and the AI is used only to
double-check the few words OCR is unsure about. That keeps cost near zero and keeps
the service working even when the AI provider is down.

