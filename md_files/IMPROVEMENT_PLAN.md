# Improvement Plan

Derived from a close reading of `etude-ocr-arabe-reconstruction-examens.html` — a feasibility
study for a mobile app that reconstructs Arabic exam papers. It targets a different product,
but the same documents: its hero demo is `samples/test1.jpeg`, watermark and all.

Two parts. **Part 1** is what the study offers that we don't already have. **Part 2** is the
execution plan that follows from it.

---

## Part 1 — Findings

### Already shipped here. Do not redo.

The study proposes these as future work. They exist in this repo today.

| Study proposes | Where it lives |
|---|---|
| §10 — double-read numbers with a second engine | `pipeline/ocr.py:269` `_recover_dropped_digits` — v3 as digit donor, difflib splice |
| §04 — upscale before recognition | `pipeline/ocr.py` `OCR_MIN_DIM_PX = 1500` |
| §04 — recognise per region, not whole page | One detection pass, per-crop recognition |
| §04 — store logical order, never visual | No bidi reordering; pinned by a codepoint-level test |
| §02 strategy B — crop unreadable regions from the image | Residual-ink raster pass in `shape_detector.py` |
| §04 stage 1 — preprocessing | `preprocessor.py`, and **better than the study's annexe A** |

That last row is worth stating plainly. The study's annexe A `nettoyer()` ends in
`adaptiveThreshold`. Binarising destroys the grey information that figure and token detection
needs — the same detection the study's own §04 stage 6 depends on. Measured here: CLAHE costs
OCR accuracy (0.9066 → 0.8955) but finds 29 shapes instead of 19, including the five circles
that hold the coin values on `test2.jpeg`. Preprocessing must **fork per consumer, not chain**.
`pipeline/__init__.py:78` does exactly that.

### Not applicable

§06 mobile architecture · §07 Chromium/PDF rendering (we emit `.docx`) · §08 cost model (local,
free) · §05 levels 2–3 inpainting · §12 legal · §13 pivots.

### Genuinely open

**1. There is no accuracy measurement at all.** Every quality number in this project is a
*confidence score* — the model's opinion of itself. No CER, no WER, no regression gate. The
`enhance_contrast` docstring cites scoring against "a hand-transcribed ground truth for two
~45 DPI Arabic worksheets", so ground truth existed at some point; it is not in the repo, and
neither is the scorer. Every tuning constant — the 0.90 routing threshold, the 20px
low-resolution gate, CLAHE on/off, the 0.5 donor similarity — was decided once, by hand, on two
images. The study's §11 phase-0 rule is correct and we skipped it.

**2. The numeric divergence is computed, then discarded.** `_recover_dropped_digits` knows, per
Arabic line, whether the v3 donor saw a number v5 lost (`ocr.py:298-303`). Nothing records it.
The .NET caller cannot force human review on exactly the lines that need it.

Note the failure mode, because it shapes the check: v5 **deletes** the number, it does not
invert it. A validator that compares numeric values between two engines sees nothing — there is
no number to compare. The signal is *divergence*, not disagreement about a value.

**3. The marginal grading column will corrupt exam pages.** `_reading_order`
(`layout_reconstructor.py:83`) groups words into lines by vertical centre with tolerance
`median_h * 0.6`, then sorts each line by x. The narrow notation cells down the page edge share
their y with body lines, so they interleave *into the middle of sentences*. Scope was widened to
dense school pages; the reconstructor was built for drawings. The study called this in §04.

**4. No watermark handling exists.** Verified — zero occurrences in code or docs. `test1.jpeg`
and `test2.jpeg` both carry one. The study's §05 trimodal-histogram observation (ink 0–70,
watermark 170–230, paper 240–255) is cheap to test. Payoff unproven: the watermark may not be
costing anything.

**5. Numbers in the `.docx` have no directional isolation.** Paragraphs carry `<w:bidi/>`
(`word_assembler.py:225`) but Latin digit runs inside Arabic paragraphs rely entirely on Word's
implicit BiDi. Cheap insurance on digits we fought to recover.

**6. No table structure model.** The grid is an emergent result of accurate rectangle placement.
The study's §04/§09 note that *bordered* tables are the easy case (TEDS 80–92%) via line
detection — and EdgeDrawing already finds those lines and assembles rectangles here. Reachable,
but the largest item on this list.

### One correction to the study, from our data

Its §01 encadré names the wrong failure mode for numbers. It predicts BiDi inversion
(`27250` → `05272`). What actually happens on that exact page is deletion. And its §03/§08 are
mutually incompatible: §08's thesis is confidence-driven review, but the VLM engines §03
recommends emit no per-word confidence. Verified — Unlimited-OCR returned 576 chars on that page
with zero confidences and zero bboxes. That incompatibility is the reason this project's
architecture exists in the shape it does.

---

## Part 2 — Execution plan

Ordered by value × cheapness. The ordering is negotiable everywhere except Phase 0.

### Phase 0 — The measurement bench — **DONE** *(2026-08-05)*

- `tools/eval.py` — CER, WER, NER. Edit distance inline; no new dependency.
- `tests/fixtures/groundtruth/{test1,test3}.txt` + a README on per-page trust.
- Baseline committed: **`md_files/EVAL_BASELINE.md`**.

Results: `test3` (English, high-res) **0.000 / 0.000 / 27-27** — solved, and that sets the
reference: given legible input the pipeline reads the page exactly. `test1` (Arabic, 45 DPI)
0.166 / 0.432 / 7-11, CER an upper bound.

Four things the first run surfaced, three of which change the plan below:

1. **The digit-donor fix is confirmed.** `27250`, `8500`, `3240` all recovered. It shipped
   without a measurement; it has one now, and a regression would show as a number here.
2. **A second, distinct digit defect exists** — the question list markers `1/ 2/ 4/` are
   dropped where `3/` survives. The donor does not recover these. New item, 1c below.
3. **Margin-cell pollution is observed, not predicted** — `خا · د · محا · ح1 · حا` appear as
   their own lines interleaved into the body. Confirms 1b.
4. **The watermark is read as text** — `aasaaaaaaaaeeaeaa · med · e`. Promotes the Phase 4
   experiment: it has a measured cost now, not a hypothetical one.

*Two pages, not five. Enough to catch a regression in what it covers; not enough to
characterise the system. `test2.jpeg` is the next page and needs someone who reads Arabic.*

### Phase 1 — **DONE** *(2026-08-05)* — 246 tests green

Bench after: `test1` CER 0.145 / WER 0.396, `test3` 0.000 / 0.000, numbers 34/38. See
`EVAL_BASELINE.md` — and read the note there before comparing against the Phase 0 figures, as
the metric definition changed alongside 1b.

**1a — done. Surface numeric divergence.** Thread `digits_recovered: list[str]` from `OcrWord` through
`TextElement` to the sidecar, populated by the splice that already runs. When the donor's digits
are rejected by the `DIGIT_DONOR_MIN_SIMILARITY < 0.5` bail, record `digit_disagreement: true`
and force `highlight: "red"` regardless of confidence. A field threaded through — no new logic.

Shipped as `digits_recovered` / `digit_disagreement` on `OcrWord` → `TextElement` → sidecar,
with a red highlight set at layout time so it does not depend on the optional LLM stage.
Measured: 6 recovered runs on `test1`, 7 on `test2`, 0 disagreements on either.

**1b — done. Isolate the margin column.** Before `_reading_order`, detect words whose boxes form a
narrow vertical band within ~8% of either page edge, at a width well under the median line
width. Emit them as their own ordered run appended after the body. Guard: fire only when 3+ such
boxes stack vertically, so a single edge label on a drawing isn't misfiled.

Shipped as `margin_column` on `TextElement`; cells are lifted out of the prose and appended
after it, top to bottom, never dropped. Four conditions must hold at once — narrow, inside an
edge band, tightly x-aligned, ≥3 deep, **and outside the body text block**. That last one was
added after running it on real pages: without it the rule swallowed `test3`'s `S1/S2/S3` row
labels and 19 of class-diagram's UML attributes. Every synthetic test passed with the broken
rule; only the real pages caught it.

**1c — diagnosed, closed as not fixable.** A per-crop trace showed the v3 donor is blind to the
list markers too (it emits the `/` but no digit). The donor mechanism can only recover what one
of the two readers sees, so it cannot help here. Inferring the value from position would be
inventing a number. Recorded in `EVAL_BASELINE.md`; not worth further effort at this input
resolution.

*Actual: ~1 day for 1a/1b/1c.*

### Phase 2 — **MEASURED, REJECTED** *(2026-08-06)* — stays off, now on evidence

A working key arrived, so the A/B finally ran: 4 reps per arm on `test1`, the only page with
flagged words.

| arm | CER across reps | |
|---|---|---|
| context **off** | 0.145 · 0.145 · 0.145 · 0.145 | never degrades |
| context **on** | 0.145 · 0.145 · 0.145 · **0.176** | still degrades |

Baseline OCR-only is 0.145. Context never once helped and sometimes hurt, so
`SENTENCE_CONTEXT_ENABLED` **stays false** — now for a measured reason rather than an
inherited one. The Kanerva et al. prior holds for this model tier.

The failure is exactly what the prompt's warning was written to prevent. Unguarded, with
context on, the model merged the surrounding lines into the answer:

```
'فاحضرمجدي 27250 مي واحضرت رانية مبلغا يقل عن مبلغ'          (0.85)
  -> 'فإذا أهدى مجدي بأقل ما يمكن من القطع التقديرية والأوراق المالية'
```

and in a later rep inserted the English word **"prostaglandin"** into an Arabic maths
worksheet at certainty 0.92. The instruction not to copy context is a request, not a
guarantee — same lesson as the glossary scar.

**Bigger finding: the correction stage delivers no measurable benefit at this model tier.**
Across 14 reps, guarded and unguarded, it never improved CER on `test1` even once. Best case
is "unchanged from baseline". That is worth knowing before paying for a larger model — the
value case for correction rests on a better model, not on prompt engineering.

#### The guards this measurement forced *(269 tests green)*

Two real defects, both found only by running against a live model:

- **`certainty <= 0.0` is now refused.** The prompt asks for an empty string when a crop can't
  be read; this model returns `"<unknown>"` instead — non-empty, so the existing empty-string
  check passed it and the literal word `<unknown>` was written over real OCR text.
- **Corrections that alter numbers are now refused outright** (`_alters_numbers`). Measured:
  `'45'` → `'43'` at **certainty 1.00**, and separately a dropped `27250` that the v3 digit
  donor had just recovered. Certainty cannot gate this, because the model is fully confident
  when it does it. Rejects the whole correction rather than restoring the digits: a reading
  that got the number wrong was not reading the crop carefully, and its letters are worth no
  more than the OCR's.

After the guards, the shipping config is flat at baseline — `test1` CER 0.147 with the LLM on
against 0.145 without, numbers 7/11 either way. Correction can no longer make the page worse
in the one way that matters.

**Known gap, recorded not fixed:** `_introduces_unseen_script` is deliberately asymmetric and
so permits Latin inside an Arabic chunk — which is how "prostaglandin" got through. That
asymmetry protects legitimate embedded units like `3.5m`, and the one observation came from
the arm now disabled. Tighten it only if it recurs in the shipping config.

### Phase 2 — original build note *(2026-08-05)* — 253 tests green

`_sentence_context()` gives a flagged line the confident lines either side of it in reading
order, behind `SENTENCE_CONTEXT_ENABLED` (**default false**). Reuses the existing single LLM
call — no second pass. Margin cells are excluded from the context, which is why 1b had to land
first. `tools/eval.py --llm` was added to measure it.

**The measurement is blocked, and the gate therefore stands: leave it off.** Every model in the
configured chain returns `failed/rate_limited` — 0 corrections on 13 flagged lines, on repeated
attempts. The status reporting is what makes that visible; without it a rate-limited run and a
prompt change that achieved nothing produce an identical table.

So this is shipped dark. The code is tested against a mocked client and the built prompt was
inspected on the real page (context is coherent and correctly disambiguating), but **no claim
is made that it improves anything.** Enabling it requires a working key and a measured CER
delta on `test1` — the only fixture with flagged words at all; `test3` has none.

Two things the attempt exposed, both worth acting on independently:

- **8 of `test1`'s 13 flagged lines are margin cells** — `خا`, `د`, `ح1`. The correction budget
  is mostly spent re-reading unreadable marking boxes instead of the 5 real prose lines. Now
  that they are tagged, skipping or deprioritising them is a small change with a real payoff.
- **A confident-but-garbage watermark token (`med`) enters the context**, since the filter is
  confidence-based and the watermark is read confidently. Another argument for Phase 4.

*Actual: ~0.5 days to build. Measurement deferred until a key works.*

### Phase 3 — **DONE** *(2026-08-05)* — 264 tests green

**Review queue** — `stats.review_queue` in the sidecar: element ids needing a human look, most
urgent first. Numeric disagreements lead (they can carry a *high* confidence, so sorting by
confidence alone buries them), marking cells sink to the back (unreadable by construction),
everything else ascending by confidence. Verified on the real pages: `test1` 13 items ordered
prose-then-cells, `test2` sinks its 0.54 cell behind 0.65 prose, `test3` empty.

**Marking cells are no longer sent to the model** — excluded from what is *sent*, never from
what is *marked*. Measured: `test1` drops from 13 items to **5**, a 62% cut in correction work;
`test2` 8 → 7. Asking a vision model to re-read an empty printed score box cannot succeed, and
those cells were consuming the budget the real prose lines needed.

**Directional isolates — considered and deliberately not implemented.** The digits in
`مجدي ب 8500 مي` already resolve correctly under UAX#9 (rule W2 makes a European number
following an Arabic letter an Arabic-number, laid out properly inside the RTL run). There is no
observed defect, and none is observable from here without rendering in real Word. Injecting
invisible control characters into text that is already correct is the same mistake as the
visual-order reversal this pipeline had to delete. Pinned by a test asserting no control
characters reach the `.docx`.

**What was found instead:** Arabic runs carried `w:szCs` (complex-script font size) but no
`w:rtl`, so Word never treated them as complex-script and that size was written and ignored on
every Arabic line the service has produced. Fixed, and the `rPr` children reordered into CT_RPr
schema sequence — Word tolerates any order, LibreOffice and the .NET caller's parser are
entitled not to. Nothing asserted any of this before; three tests do now.

*Actual: ~0.5 days.*

### Phase 4 — **RUN, REJECTED** *(2026-08-06)* — no code added

**Watermark luminance suppression: measured across 12 threshold variants, not implemented.**
The experiment ran entirely outside the repo, which is the point — an unproven idea should not
leave code behind.

The study's §05 premise is **false for these scans**. It claims the histogram is trimodal and
"very separated" — ink 0–70, watermark 170–230, paper 240–255. Measured on `test1`:

| band | test1 (45 DPI) | test3 (high-res) |
|---|---|---|
| ink 0–70 | 3.08% | 6.38% |
| 71–120 | 3.41% | 1.04% |
| 121–170 | 4.53% | 2.37% |
| 171–230 | 6.60% | 2.77% |
| 231–244 | 6.91% | 1.04% |
| paper 245–255 | 75.47% | 86.39% |

There is no valley between ink and watermark at 45 DPI — antialiasing smears every glyph edge
straight across the band the study says is watermark-only. `test3` shows the separation the
study assumes, which is consistent: **the premise holds at high resolution and fails at low**,
and low is where the watermarked pages live.

Aggressive variants are catastrophic — the study's own `120/165` gives CER 0.396 (vs 0.145
baseline) and **3/11 numbers**; `ink<=100` gives 0.710 and **0/11**. It destroys the digits
Phase 1 recovered.

One conservative variant (`211–250`) does beat baseline: CER 0.145 → **0.119**, WER and numbers
unchanged, and it removes the watermark tokens `med`, `aasaaaaaaaaeeaeaa`, `a` outright. It
also improves real words (`السن`→`السند`, `قر`→`قرر`, `الماهمة`→`المساهمة`). **Rejected
anyway**, for three reasons:

1. **It trades dot accuracy for watermark removal** — new errors appear in exactly the failure
   class this project has established as its binding constraint (`احضرته`→`اجضرته`,
   `توجه`→`توخه`, `المبلغ`→`الميلغ`).
2. **The threshold response is non-monotonic** — 211→0.119, 221→0.138, 231→0.212, 241→0.187.
   A real effect varies smoothly; this looks like fitting one page's noise.
3. **One page of provisional ground truth.** The CLAHE decision that set the house standard was
   made across multiple pages and two reps, and survived a detector change. A 0.026 gain on a
   single page does not meet that bar.

**Revisit when `test2` has ground truth** — it is the other watermarked page, and it would turn
one data point into two. Until then this is closed, and re-running it is wasted effort.

- **Bordered-table assembly.** Still deferred. The emergent grid is not visibly broken, and
  nothing measured since has changed that.

### Carried over from `PROJECT_REPORT.md`

A paid LLM key (the free tier's daily quota dominates correction reliability, and no engineering
compensates for a quota). The missing mixed Arabic+English test image, the only case that can
settle script-aware routing. Wiring `get_few_shot_examples()`, whose upstream half already
stores.

### If a GPU becomes available

A 2B-class specialised Arabic recogniser slots in *behind* PaddleOCR detection without moving
anything: boxes, confidences, layout reconstructor, correction gate and highlights all stay.
It is the only known path past the mobile-model ceiling — PaddleOCR 3.7 has exactly two Arabic
recognisers, both mobile class, and no server-class model exists.

Not to be confused with replacing the pipeline with a VLM, which is closed for a reason that has
not changed: the gap is in *localisation*, and this pipeline is built entirely on spatial
reconstruction from boxes. Attempt only **after** Phase 0, or the gain is not demonstrable.
