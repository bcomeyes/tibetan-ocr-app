#!/usr/bin/env python3
"""
Verify that each post-OCR result folder is grouped consistently with its filename.

For each subfolder in review_export/ (e.g. umeh_dhernangdri_high/), check that
every PDF-stem subfolder inside it has a filename consistent with the parent's
claimed script style.

This does NOT verify the actual PDF contents (only Nyima can do that).
It only confirms that the OCR routing didn't put e.g. a 'druma' filename
under a 'dhernangdri' folder.

USAGE:
  python3 audit_review_export.py
"""

from pathlib import Path

BASE_DIR = Path.home() / "Documents" / "tibetan-ocr-app"
REVIEW_EXPORT = BASE_DIR / "grid_search_results" / "review_export"

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


def detect_script_style(name: str) -> str | None:
    lowered = name.lower()
    for keyword, normalized in SCRIPT_KEYWORDS.items():
        if keyword in lowered:
            return normalized
    return None


def detect_quality(name: str) -> str | None:
    n = name.lower()
    if "high" in n:    return "high"
    if "medium" in n:  return "medium"
    if "med" in n:     return "medium"
    if "poor" in n:    return "poor"
    return None


def folder_claims(folder_name: str) -> tuple[str | None, str | None]:
    """Return (script_style, quality) the folder name claims."""
    return detect_script_style(folder_name), detect_quality(folder_name)


def main() -> None:
    print(f"Auditing: {REVIEW_EXPORT}\n")

    if not REVIEW_EXPORT.exists():
        print(f"ERROR: {REVIEW_EXPORT} not found")
        return

    ok = []
    mismatch = []
    unclear = []

    for bucket in sorted(REVIEW_EXPORT.iterdir()):
        if not bucket.is_dir():
            continue
        bucket_script, bucket_quality = folder_claims(bucket.name)

        for pdf_dir in sorted(bucket.iterdir()):
            if not pdf_dir.is_dir():
                continue

            pdf_script = detect_script_style(pdf_dir.name)
            pdf_quality = detect_quality(pdf_dir.name)

            # Non-umeh buckets (uchen, pechas) don't have a script-style sub-claim
            if bucket_script is None:
                # Just check quality consistency
                if bucket_quality and pdf_quality and bucket_quality != pdf_quality:
                    mismatch.append((bucket.name, pdf_dir.name,
                                     f"quality mismatch: folder={bucket_quality} pdf={pdf_quality}"))
                else:
                    ok.append((bucket.name, pdf_dir.name))
                continue

            # Umeh buckets: check both script and quality
            if pdf_script is None:
                unclear.append((bucket.name, pdf_dir.name,
                                "pdf stem has no recognizable script keyword"))
            elif pdf_script != bucket_script:
                mismatch.append((bucket.name, pdf_dir.name,
                                 f"script mismatch: folder={bucket_script} pdf={pdf_script}"))
            elif bucket_quality and pdf_quality and bucket_quality != pdf_quality:
                mismatch.append((bucket.name, pdf_dir.name,
                                 f"quality mismatch: folder={bucket_quality} pdf={pdf_quality}"))
            else:
                ok.append((bucket.name, pdf_dir.name))

    print("=" * 70)
    print(f"OK ({len(ok)})")
    print("=" * 70)
    for b, p in ok:
        print(f"  {b}/{p}")

    if mismatch:
        print()
        print("=" * 70)
        print(f"MISMATCH ({len(mismatch)})  ← these need attention")
        print("=" * 70)
        for b, p, why in mismatch:
            print(f"  {b}/{p}")
            print(f"    → {why}")

    if unclear:
        print()
        print("=" * 70)
        print(f"UNCLEAR ({len(unclear)})  ← couldn't auto-classify")
        print("=" * 70)
        for b, p, why in unclear:
            print(f"  {b}/{p}")
            print(f"    → {why}")

    print()
    print("=" * 70)
    if not mismatch and not unclear:
        print("VERDICT: All post-OCR results are grouped consistently with their filenames.")
        print("         (This does NOT verify the actual PDF contents — only Nyima can do that.)")
    elif mismatch:
        print("VERDICT: Some folders contain PDFs whose filename disagrees with the folder name.")
        print("         These need to be moved or relabeled.")
    else:
        print("VERDICT: Routing looks consistent, but some entries couldn't be auto-classified.")
        print("         Review the UNCLEAR list manually.")


if __name__ == "__main__":
    main()
