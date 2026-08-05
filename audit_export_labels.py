#!/usr/bin/env python3
"""
Audit review_export/ for the label-mismatch bug.

The export cell globbed '{model}_{mode}_*' and took the first filesystem match,
ignoring k_factor, merge_lines, tps_threshold and class_threshold. So the q-score
baked into an exported filename may describe a DIFFERENT run than the text inside.

This script settles it retroactively:
  1. Read each exported .txt (clean OCR text only, no header).
  2. Find the raw per-combo files for that same page.
  3. Locate the raw file whose OCR text is byte-identical to the exported text.
  4. Compare that raw file's embedded 'Quality Score' to the q-value in the
     exported filename.

  Score agrees  -> the export was correctly labelled.
  Score differs -> MISLABELLED: the filename describes a different run.

Read-only. Changes nothing.

USAGE:
  python3 audit_export_labels.py
"""

import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path.home() / "Documents" / "tibetan-ocr-app"
OUTPUT_DIR = BASE_DIR / "grid_search_results"
REVIEW_EXPORT = OUTPUT_DIR / "review_export"

# page_1_Ume_Druma_q73.1_c1900.1.txt
EXPORT_RE = re.compile(
    r"^(?P<page>page[_ ]?\d+)_(?P<model>.+?)_q(?P<q>[\d.]+)_c(?P<c>[\d.]+)\.txt$"
)

SCORE_RE = re.compile(r"Quality Score:\s*([\d.]+)")
TEXT_MARKER = "OCR TEXT"


def split_raw_file(path: Path):
    """Return (embedded_quality_score, clean_ocr_text) from a raw result file."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None

    m = SCORE_RE.search(content)
    score = float(m.group(1)) if m else None

    if TEXT_MARKER in content:
        idx = content.index(TEXT_MARKER) + len(TEXT_MARKER)
        while idx < len(content) and content[idx] in "\n=":
            idx += 1
        text = content[idx:].strip()
    else:
        text = content.strip()

    return score, text


def main() -> None:
    if not REVIEW_EXPORT.exists():
        print(f"ERROR: {REVIEW_EXPORT} not found")
        return

    print(f"Auditing: {REVIEW_EXPORT}\n")

    ok, mislabelled, unresolved = [], [], []
    raw_cache = {}

    for category_dir in sorted(REVIEW_EXPORT.iterdir()):
        if not category_dir.is_dir():
            continue

        for pdf_dir in sorted(category_dir.iterdir()):
            if not pdf_dir.is_dir():
                continue

            file_name = pdf_dir.name

            for exported in sorted(pdf_dir.glob("*.txt")):
                m = EXPORT_RE.match(exported.name)
                if not m:
                    unresolved.append(
                        (category_dir.name, file_name, exported.name,
                         "filename doesn't match expected export pattern")
                    )
                    continue

                page_label = m.group("page").replace("_", " ")
                labelled_model = m.group("model")
                labelled_q = float(m.group("q"))

                page_name = f"{file_name} - {page_label}"
                raw_dir = OUTPUT_DIR / file_name / page_name

                if not raw_dir.exists():
                    unresolved.append(
                        (category_dir.name, file_name, exported.name,
                         f"no raw results dir: {file_name}/{page_name}")
                    )
                    continue

                # Cache the parsed raw directory (some have 1,728 files)
                if raw_dir not in raw_cache:
                    parsed = []
                    for raw in raw_dir.glob("*.txt"):
                        score, text = split_raw_file(raw)
                        if text is not None:
                            parsed.append((raw, score, text))
                    raw_cache[raw_dir] = parsed

                exported_text = exported.read_text(encoding="utf-8").strip()
                matches = [
                    (raw, score) for raw, score, text in raw_cache[raw_dir]
                    if text == exported_text
                ]

                if not matches:
                    unresolved.append(
                        (category_dir.name, file_name, exported.name,
                         "exported text matches no raw result for this page")
                    )
                    continue

                # If ANY identical-text raw file carries the labelled score,
                # the label is defensible.
                scores = {s for _, s in matches if s is not None}
                # Exported filenames carry one decimal, so a CSV value of
                # 41.96 legitimately becomes q42.0. Under 0.05 is rounding.
                if any(abs(s - labelled_q) < 0.05 for s in scores):
                    ok.append((category_dir.name, file_name, exported.name))
                else:
                    actual = ", ".join(f"{s:.2f}" for s in sorted(scores))
                    sample = matches[0][0].name
                    mislabelled.append(
                        (category_dir.name, file_name, exported.name,
                         f"label says q{labelled_q:.2f}, actual text scores {actual}",
                         f"actual run: {sample}",
                         labelled_model)
                    )

    print("=" * 70)
    print(f"CORRECTLY LABELLED ({len(ok)})")
    print("=" * 70)
    for cat, pdf, name in ok:
        print(f"  {cat}/{pdf}/{name}")

    if mislabelled:
        print()
        print("=" * 70)
        print(f"MISLABELLED ({len(mislabelled)})  <- score does not describe this text")
        print("=" * 70)
        for cat, pdf, name, why, actual, labelled_model in mislabelled:
            print(f"  {cat}/{pdf}/{name}")
            print(f"    -> {why}")
            print(f"    -> {actual}")
            if labelled_model not in actual:
                print(f"    -> NOTE: labelled model '{labelled_model}' "
                      f"differs from the matched run")

    if unresolved:
        print()
        print("=" * 70)
        print(f"UNRESOLVED ({len(unresolved)})  <- could not verify")
        print("=" * 70)
        for cat, pdf, name, why in unresolved:
            print(f"  {cat}/{pdf}/{name}")
            print(f"    -> {why}")

    print()
    print("=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    per_cat = Counter()
    for cat, _, _ in ok:
        per_cat[(cat, "ok")] += 1
    for cat, _, _, _, _, _ in mislabelled:
        per_cat[(cat, "mislabelled")] += 1
    for cat, _, _, _ in unresolved:
        per_cat[(cat, "unresolved")] += 1

    cats = sorted({c for c, _ in per_cat})
    for cat in cats:
        o = per_cat[(cat, "ok")]
        m = per_cat[(cat, "mislabelled")]
        u = per_cat[(cat, "unresolved")]
        print(f"  {cat:30s}  ok={o:3d}  mislabelled={m:3d}  unresolved={u:3d}")

    print()
    total = len(ok) + len(mislabelled)
    if total:
        rate = len(mislabelled) / total * 100
        print(f"Mislabel rate (of verifiable files): {rate:.1f}%")
    print()
    if mislabelled:
        print("VERDICT: Confirmed mislabelling. These q-scores describe different")
        print("         runs than the text Nyima read. Re-export after fixing")
        print("         selection to use the exact params.to_filename() string.")
    elif ok:
        print("VERDICT: No mislabelling detected in verifiable files.")
    else:
        print("VERDICT: Nothing could be verified — check paths above.")


if __name__ == "__main__":
    main()
