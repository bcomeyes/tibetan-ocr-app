# Tibetan OCR Grid Search & Parameter Optimization

A systematic parameter optimization pipeline for Tibetan OCR, built on top of [BDRC's tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app). Instead of manually tuning OCR settings one at a time, this fork tests thousands of parameter combinations automatically and scores the results using Tibetan NLP tokenization.

## What This Fork Adds

The original BDRC app is a GUI desktop application for running Tibetan OCR on images and PDFs. This fork strips out the GUI workflow and adds a Jupyter notebook-based grid search pipeline that:

- Tests **1,728 parameter combinations** per image across 6 tunable parameters and 3-5 OCR models
- **Automatically scores OCR quality** using [PyBo](https://github.com/OpenPecha/pybo) tokenization — the percentage of output that consists of valid Tibetan words
- Uses **composite scoring** (quality × token count) to avoid misleading high scores on sparse pages
- **Resumes from checkpoint** if interrupted, so multi-hour runs don't lose progress
- Supports **per-category parameter trimming** — run the full grid once per script type / degradation level, then use a reduced grid for remaining images in that category
- Outputs a **per-category summary CSV** and per-result text files for analysis
- **Exports curated top results** for human review, organized by page with original JPGs

## Background

[BDRC](https://www.bdrc.io) holds a collection of 30-40 million folios of Tibetan pecha manuscripts spanning centuries of Buddhist scholarship. These texts exist in multiple script types — block print (Uchen) and various cursive styles (Umeh: Tsugring, Tsugtung, Drutsa, Khyug, Petsug) — at widely varying levels of preservation. No single set of OCR parameters works well across all of them.

This pipeline finds the best parameters for each script type and degradation level, working toward the goal of large-scale digitization of the BDRC archive.

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
pip install pybo jupyter pandas tqdm matplotlib seaborn scikit-learn ipykernel ipywidgets
```

> **Note:** Some OpenPecha dependencies (pyewts, tibetan-sort) require installation from source:
> ```bash
> pip install wheel
> pip install --no-build-isolation pyewts tibetan-sort
> ```

## How to Use

### Step 1 — Configure `ocr_grid_search.ipynb`

In the **CONFIG cell**, set three values:

```python
TARGET_ID    = 10           # which PDF to process (see TARGETS dict)
RUN_MODE     = 'quick'      # 'quick' = page 1 only; 'full' = all remaining pages
PARAM_VALUES = FULL_PARAMS  # which parameter grid to use
```

That's it. Models are auto-selected based on script type (Uchen vs Umeh).

### Step 2 — Run the notebook

Run all cells top to bottom. The RUN cell handles everything based on `RUN_MODE`.

### Step 3 — Analyze results in `grid_search_analysis.ipynb`

Set `TARGET_ID` to match what you just ran. Run all cells. The notebook:
- Loads the correct per-category CSV automatically
- Shows quality and composite scores by parameter
- Identifies which parameters to trim
- Exports top results for review

### Step 4 — Sync to Google Drive

```bash
rclone copy ~/Documents/tibetan-ocr-app/grid_search_results/review_export/ "gdrive:Tibetan OCR Gridsearch Project Folder/review_export" --progress
```

---

## Per-Category Workflow

Each script type / quality combination is a **category** (e.g. `umeh_druma_high`, `uchen_poor`).

### Phase 1: Full Grid on First Page (~5 hours)

1. Set `TARGET_ID` to first file in new category
2. Set `RUN_MODE = 'quick'`
3. Set `PARAM_VALUES = FULL_PARAMS`
4. Run all cells — processes page 1 with 1,728 combinations

### Phase 2: Trimmed Grid on Remaining Pages (~minutes)

1. Analyze page 1 results in `grid_search_analysis.ipynb`
2. Add trimmed `PARAM_VALUES` to the params cell (e.g. `UMEH_DRUMA_HIGH_PARAMS`)
3. Set `RUN_MODE = 'full'`
4. Set `PARAM_VALUES` to the new trimmed set
5. Run all cells — checkpoint skips page 1, runs remaining pages

### Moving to Next Category

1. Change `TARGET_ID` to first file of new category
2. Set `RUN_MODE = 'quick'`, `PARAM_VALUES = FULL_PARAMS`
3. Repeat from Phase 1

---

## Parameters Tested

| Parameter | Values | What It Does |
|-----------|--------|-------------|
| `ocr_model_name` | Woodblock, Woodblock-Stacks, Modern (Uchen) / Ume_Druma, Ume_Petsuk, Modern (Umeh) | Which OCR model to use |
| `line_mode` | line, layout | Line-only vs full layout detection |
| `k_factor` | 2.0, 2.5, 3.0 | Line extraction expansion factor |
| `bbox_tolerance` | 2.5, 3.5, 4.0, 5.0 | Bounding box merge tolerance |
| `merge_lines` | True, False | Whether to merge line segments |
| `tps_threshold` | 0.1, 0.25, 0.5, 0.9 | Thin Plate Spline dewarping sensitivity |
| `class_threshold` | 0.7, 0.8, 0.9 | Detection confidence threshold |

## Quality Scoring

Each OCR output is scored two ways:

- **Quality score (0-100):** percentage of tokens with valid Tibetan POS tags (via PyBo tokenization)
- **Composite score:** quality × total_tokens — penalizes high scores on sparse pages (e.g. title pages with 5 words)

Good OCR produces mostly recognized Tibetan words; garbage OCR produces mostly unrecognized tokens.

## Results (So Far)

### Uchen High Quality
- **Best model:** Woodblock-Stacks
- `line` mode beats `layout` by ~5pts
- `merge_lines=True` beats False by ~10pts
- `tps_threshold` and `class_threshold` showed no effect — trimmed to single values
- **Trimmed grid:** 8 combos per image (from 1,728)

### Umeh Druma High Quality
- **Best model:** Ume_Druma (composite), Ume_Petsuk competitive on full pages
- `layout` and `line` tied
- `bbox_tolerance=2.5` wins on composite
- `tps_threshold` and `class_threshold` no effect
- **Trimmed grid:** 16 combos per image

### Umeh Drutsa High Quality
- **Best model:** Ume_Druma (composite on full pages), Modern wins raw quality but captures fewer lines
- `bbox_tolerance` meaningful spread — keep all 4 values
- `tps_threshold` and `class_threshold` no effect
- **Trimmed grid:** 32 combos per image

## Test Corpus

33 PDFs organized by script type and degradation in `input_files/tibetan_texts/`:

| Category | Files | Pages |
|----------|-------|-------|
| Uchen High | 3 | 16 |
| Uchen Medium | 2 | 7 |
| Uchen Poor | 3 | 9 |
| Umeh High | 6 | 18 |
| Umeh Medium | 5 | 25 |
| Umeh Poor | 6 | 18 |
| Pechas (more text) | 3 | 9 |
| Pechas (little text) | 3 | 9 |
| Standalone | 2 | 14 |
| **Total** | **33** | **125** |

## Output Structure

```
grid_search_results/
├── summary_{pdf_stem}.csv         # Per-category results (tracked in git)
├── _checkpoints/
│   └── progress.json              # Resume state
├── logs/
│   └── grid_search.log
├── temp_images/                   # Extracted page JPGs
├── review_export/                 # Curated outputs for Nyima (synced to Drive)
│   └── {category}/
│       └── {pdf_name}/
│           ├── page_1_original.jpg
│           └── page_1_Ume_Druma_q73.1_c1900.1.txt
└── {pdf_stem}/                    # Raw results (not tracked in git)
    └── {page_name}/
        └── {model}_{mode}_{params}.txt
```

## Project Files

| File | Purpose |
|------|---------|
| `ocr_grid_search.ipynb` | Main grid search notebook |
| `grid_search_analysis.ipynb` | Results analysis and export notebook |
| `BDRC/Inference.py` | Core OCR pipeline |
| `BDRC/Utils.py` | Image preprocessing, model loading |
| `BDRC/Data.py` | Data classes and enums |
| `BDRC/line_detection.py` | Line extraction and sorting |
| `BDRC/image_dewarping.py` | TPS dewarping |
| `BDRC/utils/pdf_extract.py` | PDF to image extraction |

## Phase 2 (Planned)

- Binarization parameter exploration (`block_size` and `c` values in `Utils.binarize()`) — currently hardcoded but potentially critical for degraded historical documents
- Expanded test corpus from BDRC archive
- Synthetic training data generation for underrepresented script types

## Acknowledgements

- [Buddhist Digital Resource Center](https://www.bdrc.io) and Gene Smith's vision for Tibetan text preservation
- Eric Werner for the original OCR application and pipeline
- [OpenPecha](https://github.com/OpenPecha) for PyBo and Tibetan NLP tools
- Nyima Gyaltsen, Tenzin, and Chozin for Tibetan language expertise and sample collection

## Upstream

Forked from [buda-base/tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app).
