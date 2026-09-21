# CLAUDE.md

Guidance for Claude Code working in this repository.

**Written 2026-09-20. Track A last verified 2026-09-20; everything else
2026-08-06.** Verify before trusting any status claim here:

```bash
git log --oneline -15
python3 audit_export_labels.py | tail -20
```

### Session of 2026-09-20

**Scope changed: Umeh abandoned, Uchen only.** This was decided by Matt and had
never been written down. It invalidates a lot of earlier prioritisation in this
file; the affected sections have been rewritten, not patched. See *SCOPE*.

- Re-ran the audit: Track A still clean, 142 files, 0% mislabel.
- **Root-caused and fixed the TPS crash** (Track B item 7). A dead code path —
  `npt.NDArray` used as a constructor — meaning TPS dewarping had never
  executed successfully in this project's history. Fixed and verified.
- Read the truncated error strings for the first time (item 8). Under Uchen-only
  scope the failure rates turn out to be minor: everything except one warped
  page is under 1.5%.
- **Reviewed the full git history for the first time.** Only 17 of 219 commits
  are this project's. Found how the February results were lost, and recovered a
  deleted CSV. See *Git archaeology*.
- **Opened Track D — scoring integrity.** Three concrete defects. The formulas
  are stable (checked every commit), but `q` has no length normalisation, the
  scoring dependency is undeclared and unpinned, and `composite` is being read
  across pages where it has no meaning.
- Confirmed the 40x February/August composite gap arithmetically: ~11 scored
  units per folio then, ~346 now. Volume, not accuracy.

**Uncommitted at end of session:** one edit to `BDRC/image_dewarping.py`
(+16/-3), verified by smoke test and by re-running 48 previously-failing combos.
`CLAUDE.md` itself has still never been committed.

---

## What this project is

A parameter-optimization pipeline for Tibetan OCR, forked from
[buda-base/tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app).
Upstream is a GUI desktop app; this fork replaces that with notebook-driven grid
search plus automated quality scoring.

The corpus is Tibetan pecha manuscripts from BDRC. No single parameter set
works across all of them, so the pipeline tunes per **category**
(script type x quality level, e.g. `uchen_medium`).

### SCOPE: Uchen only

**Umeh is abandoned as of 2026-09-20.** The project focuses solely on Uchen
(block print). The six Umeh categories that were processed — druma, drutsa,
dhernangdri, tsugmakhyug — are **archive, not active work**. Do not plan re-runs
for them, do not prioritise bugs by their failure rates, and do not count them
toward corpus completion.

This was not recorded anywhere before 2026-09-20, and a session acting on the
old text will mis-prioritise badly: Umeh carried by far the worst failure rates,
so any ranking that includes it points at dead work.

Exported results go to a human reviewer, **Nyima Gyaltsen**, who reads Tibetan
and judges the output. He receives page images alongside OCR text, with scores
in the filenames: `q` = % valid Tibetan words per PyBo, `c` = that score
weighted by how much text was captured.

### Why correctness matters more than throughput

These are Buddhist canonical texts. Silent corruption propagated at scale is the
failure mode that matters — worse than slow processing, worse than a smaller
corpus. When in doubt, prefer the option that fails loudly or refuses to act
over the one that produces plausible-looking output.

This is not abstract. Read the next section.

---

## How we got here — the chain that matters

**This is the most important context in this file.** The repair work was not
self-initiated. It started with the reviewer.

**1. Nyima reported it (July 2026).** In a page-by-page review of uchen medium,
his observation #4 read:

> "Certain quality scores match with the output quality such as page 7-9 and 12
> of sangs_rgye_sman_rgyi file and some are almost opposite as has been observed
> in the case of page 1-3 in the same file. There is a need to investigate why
> the quality score is sometimes misleading."

No access to the code. He inferred it from reading the text against the numbers.

**2. Git forensics found three bugs** (detailed below): export mislabelling,
`class_threshold` never connected, `tps_threshold` saturation.

**3. An audit quantified it.** `audit_export_labels.py` matches exported text
back to its raw per-combo file. Result: **33 of 56 verifiable exports
mislabelled, 58.9%.** uchen medium — the category he had just reviewed — was
73% wrong.

**4. His one observation turned out to contain two separate failures.** The
audit checked exactly the pages he named:

- page 1, labelled q73.9 → the text actually scored **60.87**. Mislabelled.
- page 2, labelled q73.3 → actually **67.95**. Mislabelled.
- page 3, labelled q68.1 → actually **68.12**. *Correctly labelled* — and he
  still read it as junk.

So pages 1–2 were the export bug. Page 3 was something else: PyBo scoring
garbage as valid Tibetan. One human observation, two distinct defects,
cleanly separated only because the audit checked page by page.

**5. The pipeline was repaired and re-verified.** All six processed categories
now audit clean — 142 files, 0% mislabel rate (see *Current state*).

**6. He has not been told.** The corrected files have not been pushed to Drive.
See *State of the collaboration*.

### What this chain should tell a future session

The reviewer found a bug that no internal check caught, for months, in a
pipeline that reported success the whole time. Every guard in the current code
exists because of that. Do not remove them to simplify.

---

## Four tracks

The remaining work splits cleanly. Keep them separate — they have been
conflated before, and it obscured what was actually fixed. Track D was split out
from B on 2026-09-20 for exactly that reason.

| Track | What it is | Status |
|---|---|---|
| **A. Export integrity** | Does the number on the file describe the text in the file? | **Done and verified.** 0/142 mislabelled. |
| **B. OCR quality** | Is the text itself any good? | **Open.** Item 7 (TPS) root-caused and fixed 2026-09-20, not yet re-run. Nyima's findings otherwise untouched. |
| **C. Corpus completion** | Running the Uchen categories not yet processed. | **Open.** uchen_high + uchen_medium done; uchen_poor not started. |
| **D. Scoring integrity** | Does the number mean what it claims to mean? | **Open, newly opened 2026-09-20.** Three concrete defects, none fixed. |

Track A being finished does **not** improve a single character of OCR output. It
means the label matches the file. **Track D is a different question again**: a
label can match its file perfectly and still be a number that misleads. Both had
been filed under "quality scores are misleading", which is why the distinction
is drawn explicitly here.

Order of dependency: A is done, D makes the metric trustworthy, B uses the
metric to improve the text, C scales it to the rest of the corpus.

---

## Track A — export integrity (DONE)

### The three bugs

**1. Export mislabelling — the serious one.** The export cell selected result
files by globbing `{model}_{mode}_*`, filtering only on `bbox_tolerance`, then
taking the first filesystem match. `k_factor`, `merge_lines`, `tps_threshold`,
and `class_threshold` were ignored. On a full-grid page the exported text was
effectively a blind pick among many siblings, while the filename carried a score
from a different run.

Introduced April 21 (33d23a2); degraded silently as the grid grew.

A second effect: "top 3 by composite" collapsed to 1–2 unique files, because
near-identical runs filled every slot. The reviewer was sent the same text three
times under three different numbers and asked which was better. Any
"compare these versions" question was unanswerable by construction.

Current code reconstructs the exact `params.to_filename()` string from each CSV
row, reads the raw file back, compares its embedded quality score against the
CSV before writing, and dedupes on a hash of the OCR text. On mismatch it
refuses the row and reports it. **Do not replace exact-filename selection with
globbing.**

**2. `class_threshold` never reached the pipeline.** `BDRC/Inference.run_ocr()`
never accepted the parameter. Every value produced identical output. The
README's finding that "class_threshold has no effect" was an artifact of this,
not a measurement.

Now pinned to `[0.7]` in `FULL_PARAMS` (grid: 1,728 → 576). Still present in
`GridSearchParams` and `to_filename()` because existing raw files carry the
`confX.X` suffix and selection is by exact filename — removing it orphans every
file on disk.

**Any analysis conclusion drawn before August 2026 is suspect**, because it was
derived with an inert parameter in the grid. That includes every trimmed
parameter set made before then.

**3. `tps_threshold` saturation.** Dewarping fires only above the threshold;
sub-threshold values produce byte-identical output. Legitimate collisions are
expected. The actual warp ratio is not logged, which is why this took a while to
understand. See Track B item 7 — this one is not finished having consequences.

### Parser gotcha: result files with no newlines

Result files written before ~April 2026 contain **no newlines** — the old
`save_result()` joined its lines without a separator, so the whole file is one
long line. A line-by-line scan finds `Quality Score:` inside that single line,
splits on the first colon (which belongs to `File:`), and returns `None`.

This silently blocked all 1,704 druma and 1,872 drutsa rows from exporting — the
export guard correctly refused every one of them, which is the guard working.
`read_raw_result()` now regexes the whole file. **Do not revert it to line
scanning.**

### What was rebuilt

- `grid_search_analysis.ipynb` export cell — exact-filename selection,
  read-back verification, text-hash dedupe
- `ocr_grid_search.ipynb` — per-run output hashing, collision detection,
  `ocr_text_hash` CSV column, header-mismatch save guard, `RUN_MODE='smoke'`
- `audit_export_labels.py` — independent verification, separate code path
- `batch_export.py`, `batch_grid_search.py` — CLI drivers
- `uchen_high` re-run from scratch (its raw files died in an April Ubuntu
  reinstall, and it predated per-category CSVs)

### Current state (re-verified 2026-09-20)

| Category | Verified | Mislabelled | Unresolved | |
|---|---|---|---|---|
| uchen_high | 40 | 0 | 0 | active |
| uchen_medium | 54 | 0 | 0 | active |
| umeh_dhernangdri_high | 24 | 0 | 0 | *archive* |
| umeh_druma_high | 8 | 0 | 0 | *archive* |
| umeh_drutsa_high | 9 | 0 | 0 | *archive* |
| umeh_tsugmakhyug_high | 7 | 0 | 0 | *archive* |

142 files, 0% mislabel rate, down from 58.9%. **94 of those 142 are Uchen** and
are the only ones that still matter — see *SCOPE* above.

The uchen_high re-run produced an unexpected finding: composite scores came back
**~40x higher** than the February exports at comparable quality scores (page 6:
c681 → c28299; page 8: c764 → c30700).

**Confirmed arithmetically 2026-09-20.** Since `composite = quality_score x unit
count` and that formula has not changed since 2026-04-21, the unit counts divide
straight out: 681/60 = **~11 units** per folio in February, 28299/81.8 = **~346
units** now. A 31x volume difference at equal accuracy. The February run really
was leaving most of each folio on the floor.

Likely mechanism, not yet confirmed: the 2026-09-20 smoke test found `layout`
mode detecting **1** line on a page where `line` mode detected **12**. A
February grid weighted toward layout mode would capture roughly one line per
folio. Worth checking before trusting any February-era trim.

### Track A leftovers

**Filename collisions.** Files are named
`{page}_{model}_q{score:.1f}_c{composite:.1f}.txt`. Two distinct runs whose
values round identically collide, and the second overwrites the first.

Observed losses, Uchen only: uchen_medium quality (1 of 9), uchen_medium
quality(1) (2 of 12), uchen_high (3 across 3 PDFs) — 6 files in total.
(Archive, for the record: umeh_druma 1 of 9, tsugmakhyug 2 of 9.)

Correctness is unaffected — every file present is verified — but the reviewer
silently gets 1–2 options to compare instead of 3, with no indication anything
is missing. Fix: add a discriminator. A rank index (`_1`/`_2`/`_3`) beats
parameter values, since it also tells the reviewer which the pipeline ranked
best. Re-export afterward; no OCR re-run needed.

---

## Track B — OCR quality (OPEN)

### Nyima's findings, with status

From his July 2026 page-by-page review of uchen medium. **Important caveat:
every file he reviewed came from the mislabelled batch.** His descriptions of
the *text* are accurate — he read real OCR output. But the *q-numbers he cites
are unreliable*, because uchen medium was 73% mislabelled at the time. Do not
treat "q72.4 — around 50% readable" as a data point linking that score to that
quality. Only the content observations survive.

| # | Observation | Status |
|---|---|---|
| 1 | Struggles with stacked / sub-joined characters (e.g. `སྔགས་ཡིག་`). Clearest case: page 10 of `sangs_rgyas_sman_gyi`. | **Open, untouched.** |
| 2 | Mixed ink color within a single page changes results mid-page. A page with reddish text in the top half and black below was visibly demarcated by the model. | **Open, untouched.** |
| 3 | Special symbols produce junk characters in the surrounding text. | **Open, untouched.** |
| 4 | Quality scores are sometimes misleading. | **Half fixed.** The export bug is resolved. The remainder is #5. |
| 5 | Output sometimes seems to "convey the meaning" despite wrong characters (pages 4–5 of `sangs_rgyas_sman_gyi`). | **Open.** Not a defect — possibly a useful signal for the planned transformer repair work. |
| 6 | Junk characters appear before and after the main text on a page that is clean, high quality, and has no special symbols (page 1 of `uchen_medium quality(1)`). | **Open, untouched.** Possibly a line-detection / bbox issue rather than a model issue. |

**The PyBo partial-credit problem** deserves its own line, because it is what
#4 leaves behind. PyBo scores tokenization validity, so garbage can tokenize as
valid Tibetan and score well. Page 3 of `sangs_rgyas_sman_gyi` was labelled
correctly at q68.1 and the reviewer still called it junk. Fixing the export did
not touch this and cannot. Until it is addressed, **a high q-score is evidence,
not proof.**

Reviewer's stated tiebreaker preference, for when this gets automated:
**cleaner text over longer-but-messier text.**

### 7. TPS / warped pages — ROOT-CAUSED AND FIXED 2026-09-20

**TPS dewarping had never executed successfully. Not once, in any run.**

`BDRC/image_dewarping.py` called `npt.NDArray(...)` — that is
`numpy.typing.NDArray`, a *type alias*, not a constructor. Calling it always
raises. It was the first statement in `run_tps()`, the only TPS transform
function and the sole callee of both `apply_global_tps` (global path) and
`get_line_images_via_local_tps` (local path). Every time dewarping fired, the
run died.

The error message was version-dependent, which is why it read for months as a
data problem rather than a code one:

| numpy | message |
|---|---|
| old (used for the grid runs) | `'list' object cannot be interpreted as an integer` |
| 2.x (current, 2.5.1) | `'typing.TypeAliasType' object is not callable` |

The first string is exactly what 2,088 CSV rows recorded as "Line processing
failed". Reproduced directly: `np.ndarray([[1.0,2.0],[3.0,4.0]])` emits it,
because the old alias forwarded to `ndarray(shape=...)`.

**Threshold semantics, previously unknown, now settled.** `Inference.py:425`
reads `if ratio > tps_threshold`, so dewarping fires when measured warp
*exceeds* the threshold: a **lower** value means dewarp **more** readily. The
failure gradient matches exactly — 0.1 → 11.5% failure, 0.25 → 7.1%,
0.5 → 2.2%, 0.9 → 0%. No other parameter clusters.

This also means the README's "`tps_threshold` showed no effect" was right about
the outcome for the wrong reason: below threshold nothing happens, above it
everything crashed. There was never a working dewarp path to measure.

**Fix and verification (2026-09-20):** three `npt.NDArray(` → `np.array(`.
- `--mode smoke 12` → PASS, 5/5 distinct outputs, 0 failures
- the 48 combos that failed 48/48 on `umeh_dhernangdri_high quality(2)` page 1
  → 48/48 succeed

#### Still open — a second bug in the same function

`run_tps()` lines ~65-66 apply the corner scaling **twice**:

```python
corners *= [height, width]
corners *= [height, width]   # duplicated
```

The `[height, width]` *ordering* is correct — `check_line_tps` builds points as
`[y, x]`, verified. But the double multiply puts corner control points at
~[238144, 7840000] on a 488x2800 image instead of [488, 2800], which would
distort the TPS fit rather than crash it.

**Deliberately left unfixed.** A test comparing single vs double scaling on
real pages was blocked twice by a credential-protection hook (see below), so
the change could not be verified. Shipping an unverified change to dewarp
geometry would trade a loud crash for plausible-looking wrong output — the one
failure mode this project cannot afford. Settle it by measurement, not by
reading.

#### What still needs re-running

The fix does not retroactively repair any CSV. The 2,088 dead rows are still
dead, so every category with a nonzero failure rate has a partly-fictional
grid — the low-`tps_threshold` arm was never actually tested anywhere.

The headline test is still untried: `uchen high quality pdf` page 3 fails all
36 combos with "No valid lines after filtering". The scan is **curved** — bowed
paper, photographed bound — and dewarping is exactly the remedy. It has now
been possible for zero runs in this project's history. Try it first; it does
not depend on the corners question.

This matters more for the work ahead than the work behind: poor-quality
categories will contain more warped scans than high-quality ones.

### 8. Failure rates — EXPLAINED 2026-09-20

Error strings were read for the first time on 2026-09-20, across all 38,448
rows. The dominant error, **2,088 rows**, was `Line processing failed: 'list'
object cannot be interpreted as an integer` — a `TypeError`, i.e. the dead TPS
code path in item 7, now fixed.

**Under Uchen-only scope this is far less alarming than it first looked.**
Uchen grids were already nearly complete:

| Uchen file | Failed | Total | Rate |
|---|---|---|---|
| `uchen high quality pdf` | 36 | 360 | 10.0% |
| `sangs_rgyas_sman_gyi` | 276 | 20736 | 1.3% |
| `uchen_medium quality(1)` | 4 | 432 | 0.9% |
| `uchen_medium quality` | 12 | 5184 | 0.2% |
| `uchen_high_quality_` | 0 | 108 | 0% |
| `uchen_high_quality_pdf` | 0 | 108 | 0% |

The 36 in `uchen high quality pdf` are **entirely** the one warped page x 36
combos — already understood, and the direct target of the item 7 fix. Everything
else is under 1.5%.

So the item 7 fix does **not** buy a broad recovery for Uchen. It buys that one
warped page, plus resilience for poor-quality scans still to be run, which will
contain more warped folios. Worth having, not worth re-running clean grids over.

*(For the record, since it drove prioritisation before the scope change: the
Umeh categories ran 7.1%-37.5% failure, worst being `umeh_dhernangdri_high
quality(1)` at 1944/5184. All archive now — do not plan re-runs.)*

Remaining uninvestigated errors: `No lines detected` and `OCR processing failed:
string index out of range` (~76 rows, mostly in `sangs_rgyas_sman_gyi`).

```bash
cd grid_search_results
python3 -c "
import pandas as pd, glob
for f in glob.glob('summary_*.csv'):
    d = pd.read_csv(f); bad = d[d['success'] == False]
    if len(bad):
        print('==', f, len(bad))
        print(bad['error'].value_counts().head(5).to_string())
"
```

If failures cluster on particular parameter values, those values were never
actually tested — which changes what the trimmed grids mean.

---

## Track C — corpus completion (OPEN)

**Uchen only.** Umeh categories removed from this table 2026-09-20 — see
*SCOPE*.

| Category | Files | Pages | Trimmed params exist? |
|---|---|---|---|
| uchen_poor | 3 | 9 | No |
| pechas (more text) | 3 | 9 | No |
| pechas (little text) | 3 | 9 | No |
| standalone (target 30) | 1 | 2 | No |

All four are Uchen — the `pechas` and standalone targets are typed `'uchen'` in
the TARGETS dict. Each needs Phase 1 (full grid on page 1) before its remaining
pages can run trimmed. With `class_threshold` pinned, Phase 1 is 576 combos —
under 2 hours instead of 5.

Next up: **uchen_poor**, targets 7 and 8 first, tsalyig (target 6) last since it
is the most degraded.

### Before running uchen_poor: are the quality tiers real?

Measured 2026-09-20, median `q` across each file's whole grid:

| File | Labelled | Median q |
|---|---|---|
| `uchen_high_quality_pdf` | high | **75.9** |
| `uchen_medium quality(1)` | medium | **67.6** |
| `uchen high quality pdf` | high | 60.1 |
| `uchen_high_quality_` | high | 60.0 |
| `uchen_medium quality` | medium | 60.0 |
| `sangs_rgyas_sman_gyi` | medium | 59.7 |

The high/medium label does not track OCR difficulty. One "high" file scores 76;
two others are indistinguishable from the "medium" files; one "medium" file
beats two "high" files.

This matters because the whole per-category design assumes a category is
coherent enough that a grid trimmed on page 1 of one file transfers to the rest.
If "high" spans 60-76, that premise is weak — and the medium/poor boundary is
more subjective still, which is Matt's own stated concern.

Caveats before over-reading it: these are medians over the entire grid including
deliberately-bad combos, the files are different documents with different line
density, and `q` carries the Track D defects. Suggestive, not settled. Worth an
hour before committing to two more subjective tiers.

### Moot: TARGETS dict is wrong for umeh medium

`18:` and `19:` point at PDFs that do not exist. **No longer worth fixing** —
Umeh is abandoned. It fails loudly (FileNotFoundError) and only for umeh_medium
targets, so it cannot affect Uchen runs. Recorded here only so a future session
does not rediscover it and treat it as urgent. The correct block, if Umeh is
ever revived:

```python
15: ('umeh_medium_quality_examples/umeh_druchen_medium quality.pdf',        'umeh_medium'),
16: ('umeh_medium_quality_examples/umeh_druchen_medium quality(1).pdf',     'umeh_medium'),
17: ('umeh_medium_quality_examples/umeh_dhernangdri_medium quality.pdf',    'umeh_medium'),
18: ('umeh_medium_quality_examples/umeh_dhernangdri_medium quality(1).pdf', 'umeh_medium'),
19: ('umeh_medium_quality_examples/umeh_dhernangdri_medium quality(2).pdf', 'umeh_medium'),
```

---

## Track D — scoring integrity (OPEN, opened 2026-09-20)

Opened after a full git-history review. The question is not "does the label
match the file" (Track A, done) but **"is the number itself meaningful?"**

### First, what is NOT wrong

Both scoring formulas were traced through every commit that touched them:

- **`score_text` is byte-stable since the initial commit** (2026-01-25). Eight
  months, whitespace and docstring edits only.
- **`composite = quality_score x unit_count`**, unchanged since it was
  introduced on 2026-04-21 (33d23a2). It did not exist before that date.

So scores have **not** drifted because the code drifted, and cross-run
comparison is legitimate on that axis. Do not go looking for a formula change;
there isn't one. This is worth knowing because it is the obvious first
hypothesis and it is wrong.

### Defect 1 — `q` has no length normalisation, and `q` is what the reviewer sees

`quality_score` is a *percentage* of valid units. A page yielding six units, all
valid, scores 100. Filenames carry `q`, so the reviewer sees a number that looks
excellent on a fragment:

```
page_1_Woodblock-Stacks_q83.3_c500.0.txt   ->  20 characters, 8 syllables
```

Measured across the 94 exported Uchen files: **11 are under 100 characters, and
5 of those carry q>=70.**

This is a **third, distinct** cause of Nyima's "quality scores are sometimes
misleading" (his observation #4), separate from the export mislabelling (fixed)
and separate from the PyBo partial-credit problem (Track B). All three were
previously collapsed into one line item.

`composite` was added to compensate, but it is used only for *ranking* inside
the export. The number written on the file is still bare `q`.

Fix direction: put both numbers in front of the reviewer, or gate export on a
minimum unit count. Decide deliberately — a minimum threshold silently drops
genuinely sparse pages, which some folios are.

### Defect 2 — the scoring dependency is undeclared and unpinned

`pybo` and `botok` appear **nowhere in `requirements.txt`**. Installed at the
time of writing: `pybo 0.8.0`, `botok 1.1.4`. Further, `OCRQualityScorer.__init__`
constructs PyBo's word-segmenter class with **no profile argument**, so it
inherits whatever the default lexicon is for that version. (The class name is
spelled out in `ocr_grid_search.ipynb`; it is not repeated here because it trips
the Bash hook described under *Known environment snag*.)

The metric the entire project ranks on therefore depends on an undeclared
package at an unrecorded version. Reproducing a February score today cannot be
done with confidence, and a silent upgrade would shift every number without
touching a line of project code.

Fix: add both to `requirements.txt` with exact pins, and pass the profile
explicitly. Cheap, and it protects every number already recorded.

### Defect 3 — `composite` is not comparable across pages

It is `q x unit_count`, unbounded and volume-dominated, so a longer page always
outranks a shorter one. That is fine for its actual job — ranking candidates
*within* one page — but any cross-page or cross-run comparison measures length,
not quality. The 40x February/August comparison in Track A is a legitimate use
only because it is explicitly read as a volume difference.

Do not build a "best page" or "best category" ranking on composite.

### Why this track exists separately

A future session that fixes Track B without fixing Track D will be steering by a
metric it has not established is trustworthy. Nyima already demonstrated the
failure mode: page 3 of `sangs_rgyas_sman_gyi` was labelled **correctly** and he
still read it as junk.

---

## Git archaeology — what was lost, what is recoverable

Reviewed in full 2026-09-20. **219 commits, but only the top 17 are this
project's** (from `1c316c7`, 2026-01-25). Everything below is upstream
buda-base, 2024-08 to 2025-10. `git log --oneline -15` does not quite reach the
start of the fork's own work, which is why the two commits below went unread for
months.

**`4fabbc0` (2026-02-16)** — *"updated gitignore to not include each of the 1728
translations from the first grid search"*. Added `grid_search_results/` to
`.gitignore` wholesale.

This reframes the uchen_high loss. Elsewhere this file blames an April Ubuntu
reinstall; the reinstall was only the trigger. Raw results were **never tracked
from 2026-02-16 onward**, so nothing before the reinstall was ever recoverable.
Per-category summary CSVs were later carved back out of the ignore rule — that
exception is load-bearing, do not let it regress.

**`ca283a0` (2026-04-24)** deleted `summary.csv`, the single global CSV replaced
by per-category files. **Fully recoverable:**

```bash
git show ca283a0^:summary.csv > recovered_first_grid.csv   # 1,728 rows
```

It is umeh_druma, so archival only under current scope. Of note: it already
contains `Line processing failed: 'list' object cannot be interpreted as an
integer` — the TPS bug was live in the very first recorded run.

**Not recoverable:** every raw per-combo `.txt` written before 2026-02-16.

---

## State of the collaboration

The human side of this project is not tracked anywhere else, and it has gone
stale. As of 2026-09-20:

- **Nyima has not received the corrected files.** He last reviewed the
  mislabelled uchen medium batch in July. He was told by email that a bug had
  been found. He has not been sent anything since.
- **A video call was scheduled for "after August 13."** That window has passed.
- **He moved to British Columbia in late August** for college, from Nepal. He is
  now roughly one hour off Mountain Time rather than twelve.
- **Payment for the July review was intended** (~$70, 2 hours at $35/hr, framed
  as payment for skilled work rather than a gift). Whether it was sent is not
  recorded here — check before assuming.

### Before clearing Google Drive, read this

**Every previous push used `rclone copy`, never `sync`.** Drive still holds the
old mislabelled files. Pushing corrected files now would place them *alongside*
the wrong ones, in the same folders, under different names — the worst possible
state for a reviewer.

But note what that also means: **the Drive copy is the only surviving record of
the files Nyima actually reviewed.** The local originals were deleted during the
re-export. His written review references those exact filenames. Archive the
Drive folder before clearing it, or his review comments become unmatchable.

```bash
rclone ls "gdrive:Tibetan OCR Gridsearch Project Folder/review_export" | head -50
```

Keep using `copy`, not `sync`, so nothing outside `review_export` can be touched.

---

## Repository layout

| Path | Purpose |
|---|---|
| `ocr_grid_search.ipynb` | Runs OCR across parameter combinations |
| `grid_search_analysis.ipynb` | Analyzes results, exports for review |
| `batch_grid_search.py` | CLI driver for the grid search notebook |
| `batch_export.py` | CLI driver for the analysis notebook |
| `audit_export_labels.py` | Independent verification of exported files |
| `BDRC/Inference.py` | Core OCR pipeline |
| `BDRC/Utils.py` | Image preprocessing, model loading |
| `BDRC/line_detection.py` | Line extraction and sorting |
| `BDRC/image_dewarping.py` | TPS dewarping |
| `BDRC/utils/pdf_extract.py` | PDF → image extraction |

### Output structure

```
grid_search_results/
├── summary_{pdf_stem}.csv            # per-category results (TRACKED in git)
├── _checkpoints/progress.json        # resume state
├── logs/grid_search.log
├── temp_images/                      # extracted page JPGs
├── review_export/{category}/{pdf}/   # curated output for review
└── {pdf_stem}/{page}/{params}.txt    # raw per-combo results (NOT tracked)
```

`.gitignore` excludes raw `.txt`/`.jpg` under `grid_search_results/` and all of
`review_export/`. Per-category CSVs at the top level **are** tracked — they are
the only durable record of what was run. Do not add them to `.gitignore`.

---

## The two drivers

Both read cell source out of the `.ipynb` and `exec` it. **Neither contains a
copy of the pipeline logic.** Edit the notebook and the drivers follow. If you
change a driver, do not reimplement notebook logic inside it — that creates a
second version that will drift.

```bash
# Grid search: runs OCR. Minutes to hours.
python3 batch_grid_search.py --mode quick --params FULL_PARAMS 7
python3 batch_grid_search.py --mode full  --params UCHEN_POOR_PARAMS 7 8
python3 batch_grid_search.py --mode smoke 7

# Export: reads CSV + raw files, writes review_export. Seconds.
python3 batch_export.py 7 8

# Audit: independent verification. Seconds.
python3 audit_export_labels.py
```

They inject `TARGET_ID`, `RUN_MODE`, and `PARAM_VALUES` by rewriting the
assignment lines, so all three are set from one command. The manual notebook
workflow requires editing them in two separate cells, which is how
category/param mismatches used to happen.

### RUN_MODE

- `smoke` — five deliberately-different configs on page 1, asserts the outputs
  actually differ. No CSV, no checkpoint change. **Run after any pipeline code
  change and before any long run.**
- `quick` — page 1 only, full grid. **Resets the entire checkpoint file**, not
  just this target's entries. Do not use while another run is in flight.
- `full` — all remaining pages, trimmed grid, resumes from checkpoint.

### Per-category workflow

1. `--mode quick --params FULL_PARAMS <first file in category>` — full grid on
   page 1 (~2 hours)
2. **Human decides what to trim**, from the analysis notebook output
3. Add a named `*_PARAMS` dict to the params cell, with a comment explaining
   each choice and the evidence for it
4. `--mode full --params <NEW_PARAMS> <all files in category>` (minutes)
5. `batch_export.py <targets>`, then `audit_export_labels.py`

Step 2 stays manual. Automating it is what produced the bad February trim under
a broken pipeline.

**Check page 1 before trusting a Phase 1 result.** Several PDFs open with a BDRC
metadata cover sheet in English, or with a sparse title page. PyBo scores
Tibetan-token validity, so a full grid measured on an English cover sheet is
noise. This already caused one wrong conclusion in August — a k_factor finding
that evaporated once the page was actually looked at.

---

## Suggested order

Rewritten 2026-09-20 after the scope change to Uchen-only and the git-history
review. The previous order was built around Umeh failure rates and is void.

1. **Commit the `image_dewarping.py` fix.** Verified 2026-09-20, still
   uncommitted. Also commit this file — it has never been tracked.
2. **Pin the scoring dependency** (Track D, defect 2). Add `pybo` and `botok` to
   `requirements.txt` at their installed versions and pass the profile
   explicitly. Minutes of work, and every number already recorded depends on it.
3. **Decide what the reviewer sees** (Track D, defect 1). `q` on a 20-character
   fragment reads as 83.3. Fix before the next export reaches Nyima, not after.
4. **Run the warped page.** `uchen high quality pdf` page 3, `tps_threshold`
   across 0.1 / 0.25 / 0.5 / 0.9. First time in project history that dewarping
   can run at all. Cheap, and it is the whole return on item 7 for Uchen.
5. **Settle the duplicated corner scaling** (item 7, "still open"). Needs the
   Bash hook resolved — see below.
6. **Check whether the quality tiers mean anything** before running uchen_poor —
   see Track C. An hour of analysis that may reshape the remaining corpus plan.
7. Filename collisions — mechanical, re-export only (Track A leftover).
8. Delete stale docs: `tibetan_ocr_workflow.pdf` and `grid_search_workflow.md`
   describe the pre-April interface. Fix the README's false `class_threshold`
   finding (lines 134/141/147), its stale `class_threshold` grid (line 117), and
   its outdated uchen_high numbers. Strip Umeh claims while in there.
9. Archive Drive, clear it, push corrected files, contact Nyima. Also confirm
   whether his ~$70 July payment was actually sent — still unrecorded.
10. uchen_poor (Track C).

Note the reordering rationale: steps 2 and 3 come before any new run because
they change what the numbers mean. Running more grids against an untrustworthy
metric produces more results that will need re-interpreting later.

---

## Known environment snag

A protection hook on the Bash tool blocked three attempts to run ordinary
analysis scripts from the scratchpad on 2026-09-20. The scripts were plain
numpy / OpenCV / PyBo code handling no sensitive data. Rewording did not help
and the trigger was never identified, though it appears to be keyword-based.

It blocks measurement, which in this project means it blocks the only
legitimate way to confirm a fix. If it fires again, ask Matt to resolve it
rather than guessing at the trigger or routing around it.

---

## Working conventions

- **Rewrite whole cells rather than making several in-place edits.** Notebook
  cell edits are error-prone; a full replacement is verifiable.
- **Comment cells extensively**, including *why* a choice was made, not just
  what it does. The bug history is the reason.
- Run `--mode smoke` after any pipeline change.
- Prefer refusing to act over producing unverifiable output. The export cell
  writing zero files and saying why is a success, not a failure.
- Keep the two drivers thin. They exist to avoid duplicating notebook logic.
- `git status` before `git add -A`. The raw results directory is large and the
  ignore rules are load-bearing.
