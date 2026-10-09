# Tibetan OCR Grid Search & Parameter Optimization

A systematic parameter optimization pipeline for Tibetan OCR, built on top of [BDRC's tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app). Instead of manually tuning OCR settings one at a time, this fork tests hundreds of parameter combinations per page automatically and scores the results using Tibetan NLP tokenization.

## Scope

**Uchen (block print) only.** Work on the cursive Umeh scripts was stopped in September 2026. Six Umeh files were processed before then; their results are kept as an archive and are not maintained.

## What This Fork Adds

The original BDRC app is a GUI desktop application for running Tibetan OCR on images and PDFs. This fork replaces the GUI workflow with a notebook-based grid search pipeline that:

- Tests **576 parameter combinations** per page: 3 OCR models and 5 tunable parameters
- **Scores each result automatically** using [PyBo](https://github.com/OpenPecha/pybo): the percentage of words in the output that are valid Tibetan words
- Ranks the versions of a page by **number of valid words**, so a short fragment cannot outrank a full page
- **Resumes from a checkpoint** if interrupted; results are saved page by page
- **Checks itself**: every output is hashed, a smoke test confirms that different settings really produce different text, and an independent audit confirms that every exported file carries the score of the text inside it
- **Exports the top three versions of each page** for human review, next to the page image

## Background

[BDRC](https://www.bdrc.io) holds a collection of 30-40 million folios of Tibetan pecha manuscripts spanning centuries of Buddhist scholarship, at widely varying levels of preservation. No single set of OCR parameters works well across all of them.

This pipeline looks for the best parameters for each quality level of Uchen text (high, medium, poor), working toward large-scale digitization of the BDRC archive.

## Quick Start

### Prerequisites

- Python 3.12+
- Git with Git LFS

### Setup

```bash
git clone https://github.com/bcomeyes/tibetan-ocr-app.git
cd tibetan-ocr-app
git lfs pull
pip install -r requirements.txt
pip install jupyter pandas matplotlib seaborn scikit-learn ipykernel ipywidgets
```

`requirements.txt` pins the scoring libraries (`pybo==0.8.0`, `botok==1.1.4`). The word list the score is measured against is kept in `scoring_data/` and checked by fingerprint at every start. Changing any of these changes every score.

> **Note:** Some OpenPecha dependencies (pyewts, tibetan-sort) require installation from source:
> ```bash
> pip install wheel
> pip install --no-build-isolation pyewts tibetan-sort
> ```

## How to Use

Three command-line scripts drive the two notebooks. They run the notebooks' own code, so there is only one copy of the logic. The number at the end is the target ID of a PDF, listed in the `TARGETS` dict in `ocr_grid_search.ipynb`.

```bash
# 1. Smoke test: five different settings on page 1, checks the outputs differ.
python batch_grid_search.py --mode smoke 7

# 2. Grid search: runs the OCR. Hours.
python batch_grid_search.py --mode quick --params FULL_PARAMS 7     # page 1 only
python batch_grid_search.py --mode full  --params FULL_PARAMS 7 8   # all remaining pages

# 3. Export the top three versions of each page for review. Seconds.
python batch_export.py 7 8

# 4. Audit: confirm every exported file is labelled with its own score. Seconds.
python audit_export_labels.py
```

| Mode | What it does |
|------|--------------|
| `smoke` | Page 1, five settings, no results saved. Run it after any change to the pipeline code and before any long run. |
| `quick` | Page 1 only, the whole grid. **Resets the checkpoint for every file**, so do not use it while another run is in progress. |
| `full` | All pages not yet done. Resumes from the checkpoint. |

The notebooks can also be run by hand: set `TARGET_ID`, `RUN_MODE` and `PARAM_VALUES` in `ocr_grid_search.ipynb` and run all cells, then set the same `TARGET_ID` in `grid_search_analysis.ipynb`.

A full grid on one page takes roughly 1.5 to 3.5 hours on a laptop, depending on image size. The run uses about 3 GB of memory.

### Sync to Google Drive

```bash
rclone copy grid_search_results/review_export/ "gdrive:Tibetan OCR Gridsearch Project Folder/review_export" --progress
```

Use `copy`, not `sync`, so nothing already on Drive is deleted.

---

## Per-Category Workflow

A **category** is a quality level of Uchen text, for example `uchen_medium`.

1. **Look at page 1 first.** Several PDFs open with a cover sheet or a title page. A grid measured on one of those tells you nothing.
2. **Full grid on the first real page** (`--mode quick --params FULL_PARAMS`).
3. **A person decides what to trim**, from the output of `grid_search_analysis.ipynb`, and adds a named `*_PARAMS` set to the params cell with a comment giving the evidence. This step is deliberately not automated.
4. **Run the remaining pages** (`--mode full`) with the trimmed set, or with `FULL_PARAMS` if the category is small or its files differ a lot. `uchen_poor` was run untrimmed for that reason.
5. **Export, then audit.**

---

## Parameters Tested

| Parameter | Values | What It Does |
|-----------|--------|-------------|
| `ocr_model_name` | Woodblock, Woodblock-Stacks, Modern | Which OCR model to use |
| `line_mode` | line, layout | Line-only vs full layout detection |
| `k_factor` | 2.0, 2.5, 3.0 | Line extraction expansion factor |
| `bbox_tolerance` | 2.5, 3.5, 4.0, 5.0 | Bounding box merge tolerance |
| `merge_lines` | True, False | Whether to merge line segments |
| `tps_threshold` | 0.1, 0.25, 0.5, 0.9 | When to straighten (dewarp) the page. Lower means straighten more readily. |
| `class_threshold` | fixed at 0.7 | Not tested; see below |

3 x 2 x 3 x 4 x 2 x 4 = **576 combinations per page**.

**`class_threshold` has never been tested.** Until August 2026 the value was written into filenames and results but never passed to the OCR code, so every value gave identical output. Earlier versions of this README reported that it "showed no effect"; that was this bug, not a measurement. It is now fixed at one value. Parameter choices made before August 2026 were worked out with that dead parameter in the grid and should not be relied on.

**`tps_threshold` only matters on pages the pipeline judges to be bent or tilted.** The pipeline measures the share of lines that are not straight and straightens the page only if that share is above the threshold. On a flat page all four values give identical output. Straightening crashed on every attempt until a fix in September 2026, so results from before then never include a straightened page.

## Quality Scoring

Each OCR output gets two numbers, both shown in the exported filename:

- **`q`, quality score (0-100):** the percentage of words in the output that PyBo recognises as valid Tibetan words.
- **`c`, composite:** `q` x total words, which equals 100 x the number of valid words. `c29900` means 299 valid words. It is used to rank the versions of **one page**. Do not compare it between pages: a longer page always scores higher.

What `q` can and cannot tell you:

- **A low `q` is a reliable warning.** Random letters score about 30.
- **A high `q` is not proof of a good reading.** PyBo checks that words exist, not that they make sense. A well-read page scored 91; the same syllables shuffled into random order scored 88.
- **`q` is a percentage, so a fragment can score very high.** Six words, all valid, score 100.

This is why the results go to a human reader.

## Results So Far

All three Uchen categories have been run, exported and audited: **120 exported files, none mislabelled**.

| Category | Files | Pages | q of the best version of each page |
|----------|-------|-------|------------------------------------|
| Uchen high | 3 | 16 | median 81 (74 to 89) |
| Uchen medium | 3 | 19 | median 68 (42 to 78) |
| Uchen poor | 2 | 7 | 68 to 80 on the six pages of body text |

High and medium leave out covers, title pages and fragments (pages where the best version has under 50 words).

- The high and medium labels do track how hard a page is to read by machine: every high file scores above every medium file.
- "Poor" is not uniformly poor. Its best versions overlap medium and reach the bottom of high.
- On the tilted scans in the poor set, the best version of every body page is a straightened one.

These scores have not yet been checked against a human reading. That calibration review is the next stage of the project.

## Test Corpus

PDFs are in `input_files/tibetan_texts/`. The Uchen part of the corpus:

| Category | Files | Pages | Status |
|----------|-------|-------|--------|
| Uchen high | 3 | 16 | Done |
| Uchen medium | 3 | 19 | Done |
| Uchen poor | 2 | 7 | Done |
| Pechas (more text) | 3 | 9 | Not yet run |
| Pechas (little text) | 3 | 9 | Not yet run |
| Standalone | 1 | 2 | Not yet run |

A third poor-quality file, `uchen_tsalyig_poor quality.pdf`, is left out: both of its pages are title pages. The Umeh PDFs are still in the folder but are out of scope.

## Output Structure

```
grid_search_results/
├── summary_{pdf_stem}.csv         # One row per run (tracked in git)
├── _checkpoints/
│   └── progress.json              # Resume state
├── logs/
│   └── grid_search.log
├── temp_images/                   # Extracted page images
├── review_export/                 # Curated outputs for review (synced to Drive)
│   └── {category}/
│       └── {pdf_name}/
│           ├── page_1_original.jpg
│           ├── page_1_rank1_Woodblock_q68.2_c8800.4.txt
│           ├── page_1_rank2_Woodblock_q68.0_c8700.2.txt
│           └── page_1_rank3_Woodblock_q61.9_c8599.9.txt
└── {pdf_stem}/                    # Raw results (not tracked in git)
    └── {page_name}/
        └── {model}_{mode}_{params}.txt
```

`rank1` is the version with the most valid words. The raw result files exist only on the machine that ran the grid; the export is rebuilt from them, so do not delete them.

## Project Files

| File | Purpose |
|------|---------|
| `ocr_grid_search.ipynb` | Main grid search notebook |
| `grid_search_analysis.ipynb` | Results analysis and export notebook |
| `batch_grid_search.py` | Command-line driver for the grid search notebook |
| `batch_export.py` | Command-line driver for the export |
| `audit_export_labels.py` | Independent check that exported files are correctly labelled |
| `scoring_data/` | The word list the quality score is measured against |
| `CLAUDE.md` | Detailed project history, open problems and working notes |
| `BDRC/Inference.py` | Core OCR pipeline |
| `BDRC/Utils.py` | Image preprocessing, model loading |
| `BDRC/Data.py` | Data classes and enums |
| `BDRC/line_detection.py` | Line extraction and sorting |
| `BDRC/image_dewarping.py` | TPS dewarping (page straightening) |
| `BDRC/utils/pdf_extract.py` | PDF to image extraction |

## Planned

- Calibration review: a Tibetan reader checks what the scores mean in practice
- The remaining Uchen files (pechas, standalone)
- Binarization parameter exploration (`block_size` and `c` values in `Utils.binarize()`), currently hardcoded but potentially important for degraded documents

## Acknowledgements

- [Buddhist Digital Resource Center](https://www.bdrc.io) and Gene Smith's vision for Tibetan text preservation
- Eric Werner for the original OCR application and pipeline
- [OpenPecha](https://github.com/OpenPecha) for PyBo and Tibetan NLP tools
- Nyima Gyaltsen, Tenzin, and Chozin for Tibetan language expertise and sample collection

## Upstream

Forked from [buda-base/tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app).
