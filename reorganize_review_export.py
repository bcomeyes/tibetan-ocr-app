#!/usr/bin/env python3
"""
Reorganize review_export from quality-only folders (umeh_high/, umeh_medium/, ...)
into script-style + quality folders (umeh_druma_high/, umeh_dhernangdri_high/, ...).

The PDF filename already encodes the script style (e.g. umeh_druma_high quality.pdf),
so this is a deterministic reorganization — no judgment calls needed.

Mirrors the spreadsheet category labels exactly:
  - umeh_druma_high
  - umeh_drutsa_high
  - umeh_dhernangdri_high
  - umeh_tsugmakhyug_high
  - umeh_druchen_medium
  - umeh_dhernangdri_medium
  - umeh_dhernangdri_poor
  - umeh_drutsa_poor
  - umeh_petsug_poor
  - umeh_khyugyig_poor

Uchen and Pechas folders stay as-is (no script-style sub-categorization needed).

USAGE:
  1. Dry-run first (no changes made):
       python3 reorganize_review_export.py

  2. Apply changes:
       python3 reorganize_review_export.py --apply

  3. Then sync to Drive:
       rclone copy ~/Documents/tibetan-ocr-app/grid_search_results/review_export/ \\
         "gdrive:Tibetan OCR Gridsearch Project Folder/review_export_v2" --progress

     (Using review_export_v2 keeps the old one on Drive untouched until verified.)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path.home() / "Documents" / "tibetan-ocr-app"
REVIEW_EXPORT = BASE_DIR / "grid_search_results" / "review_export"

# Output goes to a sibling folder so the original is preserved.
OUTPUT_ROOT = BASE_DIR / "grid_search_results" / "review_export_by_script"


# Map (script_style, quality) → new bucket name.
# These names mirror the spreadsheet row labels.
SCRIPT_KEYWORDS = {
    "druma":         "druma",
    "drutsa":        "drutsa",
    "dhernangdri":   "dhernangdri",
    "tsugma khyug":  "tsugmakhyug",
    "tsugmakhyug":   "tsugmakhyug",
    "druchen":       "druchen",
    "petsug":        "petsug",
    "khyugyig":      "khyugyig",
}


def detect_script_style(pdf_stem: str) -> str | None:
    """
    Look at the PDF stem (e.g. 'umeh_druma_high quality') and return the
    script-style key (e.g. 'druma'). Returns None if no Umeh style is detected.
    """
    lowered = pdf_stem.lower()
    for keyword, normalized in SCRIPT_KEYWORDS.items():
        if keyword in lowered:
            return normalized
    return None


def detect_quality(quality_folder: str) -> str | None:
    """
    Map a top-level review_export folder name to a quality tier.
    Returns 'high', 'medium', 'poor', or None.
    """
    name = quality_folder.lower()
    if "high" in name:    return "high"
    if "medium" in name:  return "medium"
    if "med" in name:     return "medium"
    if "poor" in name:    return "poor"
    return None


def plan_moves(review_export: Path) -> list[tuple[Path, Path, str]]:
    """
    Walk review_export/<quality_bucket>/<pdf_stem>/ and decide where each
    pdf_stem directory should go.

    Returns a list of (source_dir, target_dir, reason) tuples.
    """
    moves = []

    if not review_export.exists():
        print(f"ERROR: {review_export} does not exist", file=sys.stderr)
        return moves

    for quality_bucket in sorted(review_export.iterdir()):
        if not quality_bucket.is_dir():
            continue

        bucket_name = quality_bucket.name
        is_umeh = bucket_name.lower().startswith("umeh")
        quality = detect_quality(bucket_name)

        for pdf_dir in sorted(quality_bucket.iterdir()):
            if not pdf_dir.is_dir():
                continue

            if is_umeh:
                script_style = detect_script_style(pdf_dir.name)
                if script_style and quality:
                    new_bucket = f"umeh_{script_style}_{quality}"
                    target = OUTPUT_ROOT / new_bucket / pdf_dir.name
                    moves.append((pdf_dir, target, "umeh: split by script style"))
                else:
                    # Couldn't classify — leave under original bucket name
                    target = OUTPUT_ROOT / bucket_name / pdf_dir.name
                    reason = f"umeh: could not detect script style in '{pdf_dir.name}' — kept under '{bucket_name}'"
                    moves.append((pdf_dir, target, reason))
            else:
                # Uchen, pechas, standalone — keep as-is
                target = OUTPUT_ROOT / bucket_name / pdf_dir.name
                moves.append((pdf_dir, target, f"non-umeh: kept under '{bucket_name}'"))

    return moves


def print_plan(moves: list[tuple[Path, Path, str]]) -> None:
    """Print the move plan in a readable format."""
    if not moves:
        print("Nothing to move.")
        return

    # Group by target bucket
    by_bucket: dict[str, list[tuple[Path, Path, str]]] = {}
    for src, tgt, reason in moves:
        bucket = tgt.parent.name
        by_bucket.setdefault(bucket, []).append((src, tgt, reason))

    print(f"\nReorganizing {len(moves)} PDF folders into {len(by_bucket)} buckets:\n")
    for bucket in sorted(by_bucket):
        items = by_bucket[bucket]
        print(f"  {bucket}/  ({len(items)} folder{'s' if len(items) != 1 else ''})")
        for src, _, _ in items:
            old_bucket = src.parent.name
            print(f"    ← {old_bucket}/{src.name}")
        print()


def apply_moves(moves: list[tuple[Path, Path, str]]) -> None:
    """Copy each source dir to the target location."""
    for src, tgt, _ in moves:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if tgt.exists():
            print(f"  SKIP (exists): {tgt}")
            continue
        shutil.copytree(src, tgt)
        print(f"  COPIED: {src.parent.name}/{src.name} → {tgt.parent.name}/{tgt.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Actually copy the files. Without this flag, prints the plan only.")
    args = parser.parse_args()

    print(f"Source:      {REVIEW_EXPORT}")
    print(f"Destination: {OUTPUT_ROOT}")

    moves = plan_moves(REVIEW_EXPORT)
    print_plan(moves)

    if not args.apply:
        print("DRY RUN — no changes made. Re-run with --apply to copy.")
        return

    if OUTPUT_ROOT.exists():
        print(f"\nWARNING: {OUTPUT_ROOT} already exists. Existing folders will NOT be overwritten.")
        response = input("Continue? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    print("\nCopying...")
    apply_moves(moves)
    print(f"\nDone. New structure is at: {OUTPUT_ROOT}")
    print("\nNext steps:")
    print("  1. Spot-check a few folders to make sure files landed in the right buckets")
    print("  2. Sync to Drive (using a NEW name so the old one is preserved):")
    print(f"       rclone copy {OUTPUT_ROOT}/ \\")
    print(f'         "gdrive:Tibetan OCR Gridsearch Project Folder/review_export_by_script" --progress')


if __name__ == "__main__":
    main()
