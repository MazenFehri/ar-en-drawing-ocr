# Ground truth

One `.txt` per page, named after the image stem. `tools/eval.py` finds the image by that
stem in `samples/`, `docs/images/`, then the repo root.

> **`samples/` is gitignored**, so these transcriptions are version-controlled but the pages
> they describe are not — the bench will not run from a fresh clone until the images are put
> back. That looks deliberate: `test1`/`test2` are third-party school worksheets carrying
> `medrassatouna.com` and `matheleve.net` marks. If the bench should be reproducible in CI,
> the fix is to add redistributable pages, not to commit these.

Line breaks carry no meaning — `eval.normalise()` collapses all whitespace before scoring.
They are there so a human can diff a page against its source.

## Confidence, per page

Read this before quoting an absolute CER from any of them.

| Page | Trust | Note |
|---|---|---|
| `test3` | **high** | Clean high-resolution English render. Every glyph unambiguous. Absolute CER from this page is meaningful. |
| `test1` | **provisional** | ~45 DPI Arabic scan. Body prose and all numbers transcribed with confidence; the coin-token row and the marginal grading cells are **omitted** because they cannot be read reliably at this resolution. |

## What "provisional" costs you

Two distinct biases, and they behave differently:

**Omitted regions inflate CER.** Text the OCR reads but the fixture doesn't contain scores as
insertions. So `test1`'s CER is an **upper bound**, not a measurement.

**That bias is constant only while the omitted text stays put.** It cancels in a comparison —
which is the point — but *only* for changes that don't move the omitted regions. Phase 1b broke
that assumption and proved the caveat matters: separating the marking column changed no
recognised character, yet CER moved 0.166 → 0.190, because clumping those cells at the end made
them pure insertions where scattered they had partly aligned by luck.

Hence `tools/eval.py` now **excludes margin-column cells by default**, so the fixture and the
measurement agree on what is being scored. `--include-margin` restores the old behaviour, which
is the number to quote when comparing against a run from before the column was identified.
Neither number is wrong; they answer different questions.

**Numbers are exempt from both.** Digits are unambiguous even at 45 DPI. The `numbers` column
is a real measurement on every page, and it is the metric that matters most here — the Arabic
recogniser's known failure mode is silently *deleting* embedded Latin digits.

## Adding a page

Transcribe the whole page if you can read the whole page; otherwise omit what you cannot and
add a row above saying so. A confident guess is worse than an omission — it moves the
measurement without anyone knowing it moved.

`test2.jpeg` is deliberately absent. It is the other ~45 DPI Arabic worksheet and it is dense
with coin tokens, which would make it mostly guesswork. It is the best candidate for the next
page, by someone who can read the source.
