#!/usr/bin/env python3
"""
batch_export.py — run the analysis notebook's export for several targets in a row.

WHY THIS EXISTS
    Re-exporting seven PDFs by hand means editing TARGET_ID and re-running the
    notebook seven times. Each pass is a chance to mis-set the config or forget
    to run a cell. This does the same work in one command.

WHAT IT DOES *NOT* DO
    It does not contain a copy of the export logic. It reads the code straight
    out of grid_search_analysis.ipynb and runs it. If you change the notebook,
    this picks the change up automatically — there is no second version of the
    logic to drift out of sync.

    Specifically it runs three of the notebook's cells, in order:
        config  -> resolves TARGET_ID to a PDF and a category, finds the CSV
        load    -> reads the CSV, computes composite scores
        export  -> writes the verified review_export files
    It skips by_param / top20 / inspect, which only print analysis to screen.

HOW IT WORKS
    For each target, it takes the config cell's source text and rewrites the
    single line `TARGET_ID = <something>` to the target we want, then executes
    the three cells in a fresh namespace. A fresh namespace per target means no
    leftover variables from the previous run can leak across.

USAGE
    cd ~/Documents/tibetan-ocr-app
    python3 batch_export.py 9 10 11 12 13 14

    Or with no arguments, it uses the DEFAULT_TARGETS list below.

SAFETY
    Read-only with respect to the notebook — it never writes to the .ipynb.
    If any target raises, it reports the failure and CONTINUES to the next one,
    then lists every failure again at the end so nothing gets lost in scrollback.
"""

import io
import re
import sys
import json
import traceback
import contextlib
from pathlib import Path

# The seven umeh re-exports, in category order.
DEFAULT_TARGETS = [9, 10, 11, 12, 13, 14]

NOTEBOOK = Path.home() / "Documents" / "tibetan-ocr-app" / "grid_search_analysis.ipynb"

# Only these cells are executed, in this order. Analysis/printing cells are skipped.
CELLS_TO_RUN = ["config", "load", "export"]


def load_cells(nb_path: Path) -> dict:
    """Return {cell_id: source_text} for the cells we care about."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = {}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cid = cell.get("id")
        if cid in CELLS_TO_RUN:
            cells[cid] = "".join(cell["source"])

    missing = [c for c in CELLS_TO_RUN if c not in cells]
    if missing:
        raise SystemExit(
            f"ERROR: notebook is missing expected cell(s): {missing}\n"
            f"       Found: {[c.get('id') for c in nb['cells']]}"
        )
    return cells


def set_target_id(config_src: str, target_id: int) -> str:
    """Rewrite the `TARGET_ID = N` line in the config cell source."""
    new_src, n = re.subn(
        r"^TARGET_ID\s*=\s*\d+",
        f"TARGET_ID = {target_id}",
        config_src,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(
            "Could not find a `TARGET_ID = <number>` line to rewrite in the "
            "config cell. Has the cell been edited?"
        )
    return new_src


def run_one(cells: dict, target_id: int) -> tuple[bool, str]:
    """
    Execute config/load/export for one target in a fresh namespace.

    Returns (ok, captured_output). Output is captured so that a failure
    mid-way still shows us how far it got.
    """
    ns: dict = {"__name__": "__main__"}
    buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            for cid in CELLS_TO_RUN:
                src = cells[cid]
                if cid == "config":
                    src = set_target_id(src, target_id)
                exec(compile(src, f"<{cid}>", "exec"), ns)
        return True, buf.getvalue()
    except Exception:
        buf.write("\n--- TRACEBACK ---\n")
        buf.write(traceback.format_exc())
        return False, buf.getvalue()


def main() -> None:
    args = sys.argv[1:]
    if args:
        try:
            targets = [int(a) for a in args]
        except ValueError:
            raise SystemExit(f"Targets must be integers. Got: {args}")
    else:
        targets = DEFAULT_TARGETS

    if not NOTEBOOK.exists():
        raise SystemExit(f"ERROR: notebook not found at {NOTEBOOK}")

    cells = load_cells(NOTEBOOK)

    print("=" * 70)
    print(f"BATCH EXPORT — {len(targets)} target(s): {targets}")
    print(f"Notebook: {NOTEBOOK}")
    print("=" * 70)

    failures = []

    for target_id in targets:
        print()
        print("#" * 70)
        print(f"# TARGET {target_id}")
        print("#" * 70)

        ok, output = run_one(cells, target_id)
        print(output.rstrip())

        if not ok:
            failures.append(target_id)
            print(f"\n*** TARGET {target_id} FAILED — continuing to next target ***")

    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Attempted: {len(targets)}")
    print(f"Succeeded: {len(targets) - len(failures)}")
    print(f"Failed:    {len(failures)}" + (f"  -> {failures}" if failures else ""))
    if failures:
        print("\nScroll up to the FAILED targets above for their tracebacks.")
    else:
        print("\nAll targets exported. Next: re-run audit_export_labels.py")


if __name__ == "__main__":
    main()
