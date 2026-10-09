"""Check the blind repairs against the image readings.

For each page that has both `page_N_blind_repair.txt` (made from OCR text
only) and `page_N_image_reading.txt` (made from the page image), line up the
syllables and count:

  ocr_acc     share of the page's syllables the raw OCR text got right
  repair_acc  share of the page's syllables the blind repair got right
  conf_wrong  share of the repair's UNBRACKETED syllables that are not on
              the page: confident-looking wrong fixes

"The page's syllables" means the image reading, leaving out syllables the
image reader bracketed as unsure or illegible. The image reading is itself
unverified, so these are measurements against Claude's reading of the image,
not against ground truth.

Usage: .venv/bin/python grid_search_results/claude_review/score_blind_vs_image.py
Writes blind_vs_image.csv next to this file.
"""
import csv, re, unicodedata
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
EXPORT = HERE.parent / "review_export"
SPLIT = re.compile(r"[་༌།༎༏༐༑༔\s]+")
DROP = re.compile(r"[༄༅༆༈༼༽()]")
MARKERS = {"junk", "?", "illegible"}


def syllables(text):
    """-> list of (syllable, unsure). Bracketed spans are flagged unsure;
    [junk] / [?] / [illegible] markers yield no syllables."""
    text = unicodedata.normalize("NFC", DROP.sub("", text))
    out = []
    for i, part in enumerate(re.split(r"\[([^\]]*)\]", text)):
        unsure = i % 2 == 1
        if unsure and part.strip() in MARKERS:
            continue
        out += [(s, unsure) for s in SPLIT.split(part) if s]
    return out


def body(path):
    return "\n".join(l for l in path.read_text(encoding="utf-8").splitlines()
                     if not l.startswith("#"))


def header(path):
    return {m.group(1): m.group(2).strip()
            for l in path.read_text(encoding="utf-8").splitlines()
            if (m := re.match(r"# (\w+): (.*)", l))}


def matched_flags(truth, cand):
    """Which truth / candidate syllables sit in an aligned matching block."""
    a, b = [s for s, _ in truth], [s for s, _ in cand]
    t_hit, c_hit = [False] * len(a), [False] * len(b)
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            t_hit[blk.a + k] = c_hit[blk.b + k] = True
    return t_hit, c_hit


rows = []
for rep in sorted(HERE.rglob("page_*_blind_repair.txt")):
    img = rep.with_name(rep.name.replace("blind_repair", "image_reading"))
    if not img.exists():
        continue
    tier, pdf = rep.parts[-3], rep.parts[-2]
    hr, hi = header(rep), header(img)
    truth = syllables(body(img))
    sure = [i for i, (_, u) in enumerate(truth) if not u]
    ocr = syllables((EXPORT / tier / pdf / hr["source"]).read_text(encoding="utf-8"))
    repair = syllables(body(rep))
    t_ocr, _ = matched_flags(truth, ocr)
    t_rep, c_rep = matched_flags(truth, repair)
    conf = [i for i, (_, u) in enumerate(repair) if not u]
    pct = lambda hits, idx: round(100 * sum(hits[i] for i in idx) / len(idx), 1) if idx else None
    q = re.search(r"_q([\d.]+)_c", hr["source"])
    wrong = pct(c_rep, conf)
    rows.append(dict(
        tier=tier, pdf=pdf, page=int(re.search(r"page_(\d+)", rep.name).group(1)),
        page_type=hi.get("page_type", ""), q=float(q.group(1)),
        as_is=int(hr["as_is"]), recoverable_guess=int(hr["recoverable_guess"]),
        page_syllables=len(truth), unsure_in_reading=len(truth) - len(sure),
        ocr_acc=pct(t_ocr, sure), repair_acc=pct(t_rep, sure),
        conf_wrong=None if wrong is None else round(100 - wrong, 1),
        reading_confidence=hi.get("reading_confidence", ""),
        leaned_on_memory=hi.get("leaned_on_memory", "").split()[0] if hi.get("leaned_on_memory") else "",
    ))

rows.sort(key=lambda r: (r["tier"], r["pdf"], r["page"]))
if rows:
    with open(HERE / "blind_vs_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
cols = ["pdf", "page", "page_type", "q", "as_is", "recoverable_guess", "page_syllables",
        "unsure_in_reading", "ocr_acc", "repair_acc", "conf_wrong"]
print("  ".join(cols))
for r in rows:
    print("  ".join(str(r[c]) for c in cols))
print(len(rows), "pages scored")
