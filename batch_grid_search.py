#!/usr/bin/env python3
"""
batch_grid_search.py — run ocr_grid_search.ipynb from the command line.

Same idea as batch_export.py, pointed at the other notebook. It reads the code
out of the .ipynb and executes it; it does NOT contain its own copy of the
pipeline. Edit the notebook, this picks the change up.

WHAT IT INJECTS
    Three values that otherwise have to be set by hand in two different cells:
        TARGET_ID      (config cell)
        RUN_MODE       (config cell)
        PARAM_VALUES   (params cell)
    Setting all three from one command removes the main way these runs go
    wrong — editing TARGET_ID but forgetting PARAM_VALUES still points at the
    previous category's trimmed grid.

USAGE
    python3 batch_grid_search.py --mode quick --params UCHEN_HIGH_V2_PARAMS 1
    python3 batch_grid_search.py --mode full  --params UCHEN_HIGH_V2_PARAMS 1 2 3
    python3 batch_grid_search.py --mode smoke 1

MODES (defined in the notebook's run cell)
    smoke  five deliberately-different configs on page 1, asserts outputs
           differ. No CSV written, checkpoint untouched.
    quick  page 1 only. RESETS THE CHECKPOINT FIRST — see warning below.
    full   all remaining pages, resumes from checkpoint.

WARNING ABOUT 'quick'
    The notebook's quick branch calls checkpoint.reset(), which deletes the
    whole progress file — not just this target's entries. Any partially
    finished run of another PDF loses its resume state and would restart from
    page 1. That is usually harmless (finished categories are already exported
    and their CSVs are on disk), but do not use 'quick' while something else
    is mid-run.

    Because 'quick' also passes resume=False and max_images=1, it re-processes
    page 1 even if that page was done before. That is what makes it the right
    mode for a one-page comparison run.

NOTE ON TIME
    This actually runs OCR. Unlike batch_export.py, which is seconds, this can
    be minutes to hours. Output streams live so you can watch progress.
"""

import re
import sys
import json
import argparse
import traceback
from pathlib import Path

NOTEBOOK = Path.home() / "Documents" / "tibetan-ocr-app" / "ocr_grid_search.ipynb"

# Executed in this order. 'analyze' is skipped — it only prints, and the run
# cell already reports what matters.
CELLS_TO_RUN = ["imports", "config", "logging_setup", "scorer", "params", "engine", "run"]


def load_cells(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = {}
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and cell.get("id") in CELLS_TO_RUN:
            cells[cell["id"]] = "".join(cell["source"])
    missing = [c for c in CELLS_TO_RUN if c not in cells]
    if missing:
        raise SystemExit(f"ERROR: notebook missing cell(s): {missing}")
    return cells


def rewrite(src: str, name: str, value: str, quote: bool = False) -> str:
    """Rewrite a top-level `NAME = ...` assignment. Fails loudly if absent."""
    repl = f"{name} = {value!r}" if quote else f"{name} = {value}"
    new, n = re.subn(rf"^{name}\s*=\s*\S+", repl, src, count=1, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"Could not find a `{name} = ...` line to rewrite.")
    return new


def run_one(cells: dict, target_id: int, mode: str, params: str | None) -> bool:
    """Execute the pipeline for one target. Output is NOT captured — OCR runs
    are long and you want to see the progress bar."""
    ns: dict = {"__name__": "__main__"}
    try:
        for cid in CELLS_TO_RUN:
            src = cells[cid]
            if cid == "config":
                src = rewrite(src, "TARGET_ID", str(target_id))
                src = rewrite(src, "RUN_MODE", mode, quote=True)
            elif cid == "params" and params:
                src = rewrite(src, "PARAM_VALUES", params)
            exec(compile(src, f"<{cid}>", "exec"), ns)
        return True
    except Exception:
        traceback.print_exc()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ocr_grid_search.ipynb across targets.")
    ap.add_argument("targets", nargs="+", type=int, help="TARGET_ID values")
    ap.add_argument("--mode", required=True, choices=["smoke", "quick", "full"])
    ap.add_argument("--params", default=None,
                    help="Name of a param dict in the params cell, "
                         "e.g. FULL_PARAMS or UCHEN_HIGH_V2_PARAMS. "
                         "Omit to use whatever the notebook already has set.")
    args = ap.parse_args()

    if not NOTEBOOK.exists():
        raise SystemExit(f"ERROR: notebook not found at {NOTEBOOK}")

    cells = load_cells(NOTEBOOK)

    print("=" * 70)
    print(f"BATCH GRID SEARCH — targets {args.targets}")
    print(f"Mode:   {args.mode}")
    print(f"Params: {args.params or '(as set in notebook)'}")
    print("=" * 70)

    failures = []
    for tid in args.targets:
        print("\n" + "#" * 70)
        print(f"# TARGET {tid}")
        print("#" * 70)
        if not run_one(cells, tid, args.mode, args.params):
            failures.append(tid)
            print(f"\n*** TARGET {tid} FAILED — continuing ***")

    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Attempted: {len(args.targets)}   Failed: {len(failures)}"
          + (f"  -> {failures}" if failures else ""))


if __name__ == "__main__":
    main()
