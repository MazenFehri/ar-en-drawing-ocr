# OCR accuracy baseline

First measured accuracy this project has had. Everything quoted elsewhere is a *confidence
score* — the model's opinion of itself. These are error rates against transcribed pages.

Reproduce:

```bash
docker run --rm -v "$PWD:/src" -w /src -e PYTHONPATH=/src ocrtask-ocr-api python tools/eval.py
python tools/eval.py --self-test    # metric arithmetic, needs no models or images
```

Measured 2026-08-05, `ocrtask-ocr-api:latest`, CPU only, `language_hint=ar+en`, no LLM stage.

**Current — after Phase 1**, margin-column cells excluded (the default; fixtures omit them too):

| page | CER | WER | numbers | |
|---|---|---|---|---|
| `test3` — English, high resolution | **0.000** | **0.000** | 27/27 | clean |
| `test1` — Arabic worksheet, ~45 DPI | 0.145 | 0.396 | 7/11 | lost list markers `2`, `4` |
| **mean** | **0.073** | **0.198** | **34/38** | |

**Phase 0 baseline**, for comparison. Scored with everything included, which is what
`--include-margin` reproduces:

| page | CER | WER | numbers |
|---|---|---|---|
| `test1` | 0.166 | 0.432 | 7/11 |
| `test3` | 0.000 | 0.000 | 27/27 |

> **Do not read 0.166 → 0.145 as a quality gain.** Phase 1b changed no recognised character —
> it reorders and labels. Scored the *old* way it reads 0.190, i.e. worse, purely because
> clumping the marking cells at the end turns them into pure insertions where scattered they
> partly aligned by chance. The metric definition changed alongside the code, and both numbers
> are reported so neither hides the other. See the fixtures README.

`test1`'s CER is an **upper bound**, not a measurement — the fixture omits the coin-token row
and the marginal grading cells because they cannot be read reliably at 45 DPI, so text the OCR
does emit there scores as insertions. The bias is constant, so deltas across runs are valid.
See `tests/fixtures/groundtruth/README.md`.

---

## What the first run found

### 1. The English high-resolution case is solved

Zero character errors, zero word errors, every one of 27 numbers recovered including the whole
5×5 transportation grid. That is now a fact with a command behind it rather than a claim.

It also sets the reference: **the pipeline is not the limiting factor.** Given legible input it
reads the page exactly. Everything below is about input quality and layout, not recognition.

### 2. The digit-donor fix is confirmed working — measured for the first time

`27250`, `8500` and `3240` are all present in the `test1` output. These are the three numbers
`arabic_PP-OCRv5_mobile_rec` silently deletes, and the v3-donor splice
(`pipeline/ocr.py:269`) recovers all three. That fix shipped without a measurement behind it.
It now has one, and a regression in it would show up as a number in this table.

### 3. But the *list markers* are still being dropped — and the donor cannot fix it

**Diagnosed in Phase 1c.** A per-crop trace of both recognisers on `test1` shows the v3 donor
is blind to these too:

```
crop 29   v5 'أمتل قود مجدي بأقل...'      v3 '/أمتل نقود مجدي بأقل...'   ← slash, no digit
crop 25   v5 'أحسب المبلغ الذي احضرته'   v3 'أحسب المبلغ الذيأحضرته'    ← neither has "2"
crop 22   v5 '3/ أحسب نقود أحمد'          v3 '3أحسب نقود حمد'            ← both saw "3"
```

Not a detection-boundary issue: crops 22 and 25 share a right edge at x=314, so the marker is
inside both boxes. Both models simply fail to emit it, and the donor mechanism can only recover
what *one* of them sees. **Not economically fixable at this resolution** — inferring "1" from
position would be inventing a number, which this pipeline does not do.

This also bounds the Phase 1a signal honestly: `digit_disagreement` catches "the donor saw a
number we could not place", not "both models missed one". On `test1` it fires zero times, which
is correct — there is no disagreement, there is shared blindness.

Ground truth has questions numbered `1/ 2/ 3/ 4/ 5 /`. The output keeps only `3/` and `5`:

```
أمتل قود مجدي بأقل مايمكن م القطع التقدية والأوراق المالية     ← "1/" gone
أحسب المبلغ الذي احضرته رانية                                  ← "2/" gone
3/ أحسب نقود أحمد                                              ← survived
أحسب المبلغ الجملي المجمع لدى الأطفال                          ← "4/" gone
```

Same *symptom* as the money-amount defect, but the donor does not recover these. Worth a
separate look: these sit at the leading edge of an RTL line rather than mid-sentence, which is
plausibly why the splice's alignment does not place them.

Low severity — a missing question number does not make an exercise wrong the way a missing
amount does — but it is a real, reproducible loss and it was invisible before this bench.

### 4. The marginal grading cells pollute the text stream — **fixed in Phase 1b**

The baseline carried these as their own lines, interleaved into the body:

```
خا · د · محا · ح1 · 1e · حا · a
```

Narrow notation boxes down the page edge, read, failed, and emitted *between* the body
sentences. Now separated and appended after the prose, tagged `margin_column: true`.

Measured across all four sample pages: 9 cells found on `test1`, 4 on `test2`, **0 on `test3`
and 0 on `class-diagram`**. Those last two matter — an earlier version of the rule swallowed
`test3`'s `S1/S2/S3` row labels and 19 of class-diagram's UML attributes, all of which are
genuine content. Narrow, edge-adjacent and pixel-aligned describes both a marking column and a
table's row-label column. What separates them is that a marking column sits *outside the text
block*; a row label shares its horizontal span with the prose. Only running it on the real
pages exposed that — the synthetic tests all passed with the broken rule.

### 5. The watermark is OCR'd as garbage — but suppressing it costs more than it saves

```
aasaaaaaaaaeeaeaa · med · e
```

The diagonal `medrassatouna.com` overlay, recognised as text. Real cost: pure insertions
against ground truth, inflating CER and WER on every watermarked page, and they sit at the
*front* of the review queue (0.22 and 0.26 confidence) where a reviewer looks first.

**Phase 4 tested luminance suppression across 12 threshold variants and rejected it.** The
study's trimodal-histogram premise is false at 45 DPI — there is no valley between ink and
watermark, because antialiasing smears glyph edges across the whole mid range. The same
measurement on high-resolution `test3` *does* show the separation, so the premise holds at
300 DPI and fails at 45.

Aggressive settings destroy the numbers Phase 1 recovered (the study's own thresholds:
CER 0.396, 3/11 numbers). One conservative band beats baseline on CER but introduces new dot
errors and responds non-monotonically to its own threshold. Full reasoning and the band table
are in `IMPROVEMENT_PLAN.md` Phase 4. Do not re-run this without more ground truth.

---

## The LLM correction stage, measured *(2026-08-06)*

`tools/eval.py --llm` runs the correction stage. With a working key:

```
test1   llm: success/ok · 1 corrections · 5 sent of 13 flagged      CER 0.147
test3   llm: not_attempted/no_flagged_words · 0 sent of 0 flagged   CER 0.000
```

Against 0.145 OCR-only. **Correction has never once improved CER on this page** — across 14
reps, with and without sentence context, guarded and unguarded, the best outcome was
"unchanged from baseline". Before it was also frequently *worse*; see below.

Two structural notes: `test3` can never exercise correction at all (every line reads above
threshold), so this measurement rests on one page. And only **5 of 13** flagged items are sent
— the marking cells are skipped, which is the Phase 3 change doing its job.

### What running against a live model exposed

Three failures, all at high claimed certainty, none hypothetical:

```
'45'  ->  '43'                                                   certainty 1.00
'أمتل قود مجدي'  ->  '<unknown>'                                  certainty 0.00
'فاحضرمجدي 27250 مي واحضرت رانية مبلغا يقل عن مبلغ'                certainty 0.85
  -> 'فإذا أهدى مجدي بأقل ما يمكن من القطع التقديرية والأوراق المالية'
```

The first rewrote a number. The second wrote the literal string `<unknown>` over real OCR text
— the prompt asks for an empty string when a crop is unreadable, and the empty-string check
therefore let a non-empty placeholder through. The third copied the surrounding-context lines
into the answer *and* destroyed a `27250` the two-recogniser splice had just recovered.

Guards added for the first two (`_alters_numbers`, and refusing `certainty <= 0.0`). The third
is why sentence context stays disabled. After the guards, correction is flat at baseline: it
can no longer make a page wrong in the way that matters.

## Caveats on the bench itself

- **No LLM stage by default.** Deliberate — it is networked, rate-limited, nondeterministic,
  and only ever touches words OCR already doubted. The default measures the honest floor,
  repeatably.
- **Two pages.** Enough to catch a regression in what it covers, not enough to characterise the
  system. `test2.jpeg` is the obvious next page and needs someone who can read the source.
- **One evaluator bug already found and fixed.** OCR read the typeset `D₄` as U+2084 while
  reading `D1`–`D3` as ASCII; the evaluator scored that correct reading as both a character
  error and a lost number. Sub/superscript digits are now folded before comparison, and the
  fold is pinned by a self-test assertion that also checks `½` survives it. Worth stating
  because it is the failure mode of every eval harness: an evaluator bug looks exactly like a
  regression in the thing being evaluated.
