# Tibetan OCR Grid Search & Parameter Optimization

A systematic parameter optimization pipeline for Tibetan OCR, built on top of [BDRC's tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app). Instead of manually tuning OCR settings one at a time, this fork tests thousands of parameter combinations automatically and scores the results using Tibetan NLP tokenization.

## What This Fork Adds

The original BDRC app is a GUI desktop application for running Tibetan OCR on images and PDFs. This fork strips out the GUI workflow and adds a Jupyter notebook-based grid search pipeline that:

- Tests **1,728 parameter combinations** per image across 6 tunable parameters and 3-5 OCR models
- **Automatically scores OCR quality** using [PyBo](https://github.com/OpenPecha/pybo) tokenization — the percentage of output that consists of valid Tibetan words
- **Resumes from checkpoint** if interrupted, so multi-hour runs don't lose progress
- Supports **per-category parameter trimming** — run the full grid once per script type / degradation level, then use a reduced grid for remaining images in that category
- Outputs a summary CSV and per-result text files for analysis

## Background

[BDRC](https://www.bdrc.io) holds a collection of 30-40 million folios of Tibetan pecha manuscripts spanning centuries of Buddhist scholarship. These texts exist in multiple script types — block print (Uchen) and various cursive styles (Umeh: Tsugring, Tsugtung, Drutsa, Khyug, Petsug) — at widely varying levels of preservation. No single set of OCR parameters works well across all of them.

This pipeline finds the best parameters for each script type and degradation level, working toward the goal of large-scale digitization of the BDRC archive.

## Quick Start

### Prerequisites

1. [Git LFS](https://git-lfs.com) installed
2. Python 3.10+ with conda recommended
3. Jupyter notebook

### Setup

```bash
git clone https://github.com/bcomeyes/tibetan-ocr-app.git
cd tibetan-ocr-app
git lfs pull

conda create -n tibetan-ocr python=3.11
conda activate tibetan-ocr

pip install -r requirements.txt
pip install pybo jupyter pandas tqdm
```

### Running the Grid Search

1. Open `ocr_grid_search.ipynb` in Jupyter
2. Run cells 1-6 in order (setup, config, logging, scorer, params, engine)
3. In Cell 2, uncomment the TARGET PDF you want to process
4. In Cell 5, select `PARAM_VALUES = FULL_PARAMS` for first run of a category
5. Run Cell 7 (quick test — 1 image) to verify everything works
6. Run Cell 8 (full grid search) for real results
7. Run Cell 9 or open `grid_search_analysis.ipynb` to analyze results

### Per-Category Workflow

Each script type / degradation combination gets its own optimization:

1. **Phase 1**: Run full 1,728-combo grid on one representative image (~5 hours)
2. **Analyze**: Check which parameters matter vs which are irrelevant for this category
3. **Trim**: Create a reduced parameter set (e.g. 12 combos instead of 1,728)
4. **Phase 2**: Run trimmed grid on remaining images in that category (~minutes)
5. **Repeat** for next category

See `grid_search_workflow.md` for the detailed checklist.

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

Manually reviewing 1,728 OCR outputs per image is impossible. We needed an automated way to score OCR quality without reading Tibetan.

We explored two Tibetan NLP tokenizers (see `botok_exploration.ipynb` and `pybo_explorer.ipynb`):

- **[Botok](https://github.com/OpenPecha/Botok)** — OpenPecha's more advanced tokenizer with multiple tokenization modes, lemmatization, and richer linguistic attributes. Dictionary-assisted with comprehensive lexicons.
- **[PyBo](https://github.com/OpenPecha/pybo)** — A rule-based word tokenizer using the THL (Tibetan Himalayan Library) lexicon for word matching, with POS tagging.

**PyBo was chosen** because it's simpler, fast enough for our use case (scoring 1,728 outputs in reasonable time), and its rule-based approach provides a clear valid/invalid signal that works well for classical Tibetan texts like the pecha manuscripts in our corpus. Botok's additional features (lemmatization, multiple tokenization modes) weren't needed for binary quality scoring.

The scoring method: each OCR output is tokenized, and the percentage of tokens with valid POS tags (excluding punctuation and NON_WORD markers) becomes the quality score (0-100). Good OCR produces mostly recognized Tibetan words; garbage OCR produces mostly unrecognized tokens. This approach was a breakthrough insight that made automated parameter optimization feasible at scale.

## Test Corpus

The `input_files/tibetan_texts/` directory contains 33 PDFs organized by script type and degradation:

- **Uchen** (block print): 8 files across high, medium, and poor quality
- **Umeh** (cursive): 16 files across high, medium, and poor quality, including Drutsa, Druma, Petsug, Khyug, and Dhernangdri variants
- **Pechas**: 6 files of varying text density
- **Standalone**: 2 additional test files

## Results (So Far)

### Uchen High Quality — First Run

Tested on `uchen high quality pdf.pdf` page 1 (1,728 combinations):

- **Best model**: Woodblock-Stacks (64.8% avg quality)
- **line mode** beats layout mode by 5 points
- **merge_lines=True** beats False by 10 points
- **bbox_tolerance** shows meaningful spread (keep all values)
- **k_factor**, **tps_threshold**, **class_threshold** showed no difference on clean pages — trimmed to single values for this category
- **Trimmed grid**: 12 combos per image (from 1,728)

## Output Structure

```
grid_search_results/
├── summary.csv                    # Master results file
├── _checkpoints/
│   └── progress.json              # Resume state
├── logs/
│   └── grid_search.log            # Execution log
└── {pdf_stem}/
    └── {page_name}/
        └── {model}_{mode}_{params}.txt  # Individual OCR results
```

## Project Files

| File | Purpose |
|------|---------|
| `ocr_grid_search.ipynb` | Main grid search notebook (11 cells) |
| `grid_search_analysis.ipynb` | Results analysis notebook |
| `grid_search_workflow.md` | Per-category workflow checklist |
| `pybo_explorer.ipynb` | PyBo tokenizer exploration and benchmarking |
| `botok_exploration.ipynb` | Botok tokenizer exploration and comparison |
| `BDRC/Inference.py` | Core OCR pipeline |
| `BDRC/Utils.py` | Image preprocessing, model loading |
| `BDRC/Data.py` | Data classes and enums |
| `BDRC/line_detection.py` | Line extraction and sorting |
| `BDRC/image_dewarping.py` | TPS dewarping |
| `BDRC/utils/pdf_extract.py` | PDF to image extraction |

## Upstream

Forked from [buda-base/tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app). The original GUI application and its OCR models were developed by Eric Werner for the Buddhist Digital Resource Center. See the upstream repo for the desktop application, model training code, and evaluation tools.

## Phase 2 (Planned)

- Binarization parameter exploration (`block_size` and `c` values in `Utils.binarize()`) — currently hardcoded but potentially critical for degraded historical documents
- Expanded test corpus from BDRC archive
- Synthetic training data generation for underrepresented script types

## Acknowledgements

- [Buddhist Digital Resource Center](https://www.bdrc.io) and Gene Smith's vision for Tibetan text preservation
- Eric Werner for the original OCR application and pipeline
- [OpenPecha](https://github.com/OpenPecha) for PyBo and Tibetan NLP tools
- Nyima, Tenzin, and Chozin for Tibetan language expertise and sample collection
