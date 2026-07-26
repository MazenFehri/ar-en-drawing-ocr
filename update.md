# Arabic Architectural OCR — Progress Update

**Date:** 26 July 2026
**Status:** Working end to end. Running in Docker, verified against a test drawing.

---

## What this service does

You give it a photo or scan of an architectural drawing. It gives you back a Word
document with the text, shapes and symbols placed roughly where they were on the
original, plus a JSON file describing everything it found and how confident it was.

It handles drawings that mix Arabic and English, which is the hard part — the two
languages read in opposite directions and have to share a page.

---

## Where we are

The whole pipeline runs. A drawing goes in, a Word document comes out, and the
positions are right. All eight processing stages are built, tested and packaged in
Docker, so it starts with a single command on any machine.

This session was the first time the service was run end to end on a real image
rather than in unit tests. That surfaced several genuine problems, all now fixed.

**Test result.** We fed it a floor plan containing three rooms, a circle, a
triangle, an irregular symbol, and twelve labels in mixed Arabic and English.

| What we expected | What it produced |
|---|---|
| 3 room rectangles | 3 rectangles, correct positions |
| Circle and triangle | Recognised as circle and triangle |
| One irregular symbol | Cropped and embedded as a picture |
| 21 text labels | All 21 found, average confidence 96% |

---

## What we fixed this session

**Arabic was coming out backwards.** This was the most serious problem and it was
invisible until we looked at real output. The OCR engine reports Arabic in the order
the letters *appear on the page*, which is the reverse of how Arabic is actually
stored and typed. Every Arabic label was reversed — and worse, Word would have
reversed it a second time when rendering, so the final document showed nonsense.
Arabic now comes through correctly.

**Text was in the wrong order.** The service sorted labels top to bottom but did
nothing about words sharing a line, so "GROUND FLOOR PLAN" came out as
"FLOOR PLAN GROUND". Lines are now read in their own direction — left to right for
English, right to left for Arabic.

**Skewed-page correction could destroy a drawing.** The service tries to straighten
tilted scans. But on a drawing dominated by a long diagonal — a roof slope or a
section line, which are everywhere in architectural drawings — it misread the
diagonal as the page being tilted and rotated the whole thing by 45 degrees. It now
refuses to apply corrections larger than a plausible scan tilt.

**Shapes and text were being double-counted.** Text the layout stage missed was
being detected a second time as a "shape" and embedded as a picture next to its own
text. Fixed by reordering the pipeline so text detection informs shape detection.

**Large scans were being processed at full size.** A 300 dpi A4 scan wasted several
seconds on image cleanup and sent a multi-megabyte image to the AI service on every
request. Images are now scaled down first, with no loss of readable detail.

**The API was missing options the design called for.** Callers can now set the
confidence threshold, choose the language, and turn symbol labelling on or off.
Invalid values are rejected rather than silently ignored.

**Symbol naming now works.** Unrecognised symbols used to be labelled "unknown".
The service now asks the vision model what they are — in our test it correctly
identified the irregular shape as a grid reference marker.

**Over-eager highlighting.** Words the AI reviewed and confirmed as already correct
were being highlighted as if they had been changed. Only genuine corrections are
highlighted now.

---

## Does the AI correction actually work?

Yes, and we proved it. The OCR misread "LIVING" as "LMVING" — but at 93% confidence,
so it was never flagged as suspicious. When we forced every word through the AI
reviewer, it caught the error, corrected it, and highlighted it yellow in the Word
document.

This is worth flagging because it shows the confidence threshold is doing real work
and is a tuning decision, not a technical one. Set it low and the service is fast but
misses confident-but-wrong readings like this one. Set it high and it catches more
but costs more time. **We should tune this against real drawings.**

The AI service is treated as optional throughout. If it is rate-limited or
unavailable, the service logs a warning and returns the document using the raw OCR
text rather than failing the request. We saw this happen live during testing and the
fallback behaved correctly.

---

## Speed

| Stage | Time |
|---|---|
| Core pipeline (OCR, shapes, Word generation) | **3.2 seconds** |
| Plus symbol naming | 7–27 seconds |
| Plus word correction, when words are flagged | varies |

The core pipeline is fast and predictable. All the variability comes from the
external AI service, which is currently on a free tier with unpredictable latency and
rate limits. Everything runs on CPU — no special hardware needed.

---

## Open items

**No real drawing has been tested yet.** Everything above was verified against a
drawing we generated ourselves. It exercises every code path, but it is clean,
printed, and perfectly straight. Real scans are skewed, noisy, handwritten and
photocopied. **This is the single most valuable next step and we need sample
drawings to do it.**

**The free AI tier is not production-ready.** It rate-limits unpredictably; we had to
switch models once during testing. Fine for development, but a paid key or a locally
hosted model will be needed before this goes live.

**User corrections are stored but not yet used.** When a user corrects the output, it
is saved to the database as designed. Feeding those corrections back to improve future
results is built but not yet connected.

**Mixed-language lines.** A single line containing both Arabic and English is ordered
by whichever language dominates. Lines in one language are handled correctly, which
covers normal drawing labels.

---

## Suggested next steps

1. **Get 5–10 real drawings.** Nothing else gives us as much. Everything below is
   guesswork until we see real input.
2. **Tune the confidence threshold** against those drawings.
3. **Decide where this will be deployed.** Whether the host has a graphics card
   changes several technical choices. We deliberately have not optimised for hardware
   we may not have.
4. **Connect the correction feedback loop.**
5. **Move off the free AI tier** before any real use.

---

## Deployment

Runs anywhere Docker runs, with one command. The service and its database start
together. First run downloads the Arabic OCR models (~200 MB) and caches them, so
later starts are quick.

Test coverage: 78 automated tests, all passing.
