# Reading Tibetan page images — brief

You are doing one stage of an experiment on Tibetan (Uchen block print) OCR.
An earlier stage produced "blind repairs" of OCR text without seeing the
pages. **Your job is to produce an independent reading of each page from its
image**, which those repairs will be checked against. Your reading is the
yardstick, so honesty about what you cannot read matters more than
completeness.

## Hard rules

- Do NOT open any file named `*blind_repair*`, `blind_pass.csv`, or
  `BLIND_BRIEF.md`. Your reading must not be influenced by them.
- You MAY read the OCR texts for the page (`page_N_rank1/2/3_*.txt`) as an
  aid, but the image is the authority: where image and OCR disagree, go
  with what you see; where you cannot see, do not copy the OCR — bracket.
- No web searches. Do not modify anything outside `claude_review/` and the
  scratch folder below.

## Method, per page

1. Open `page_N_original.jpg` with the Read tool. The pages are wide strips
   and the default view is usually too small to read letter by letter. Crop
   and enlarge: write a small script (run with
   `/home/matt/Documents/tibetan-ocr-app/.venv/bin/python`, cv2 is
   installed) that cuts the page into overlapping pieces (e.g. each text
   line split into 2-4 horizontal segments, upscaled 2-3x) saved as PNG
   under
   `/tmp/claude-1000/-home-matt-Documents-tibetan-ocr-app/bc24aa58-641d-4cf8-95e6-f5f72efd5422/scratchpad/crops/<your tier>/`
   and Read those. Spend the effort: actually look at every line.
2. Write the reading: **one output line per printed line of the page**, top
   to bottom, body text only (leave out margin folio labels/captions, but
   mention them in notes). Use `།` and `་` as printed.
   - Square brackets around syllables you are unsure of: `[གཟུངས་རིག་]`.
   - `[illegible]` where you cannot read the print at all. Do not fill
     such gaps from memory of a known text. If you recognise the text and
     memory supplies a word the image cannot confirm, it goes in brackets.
   - Keep the page's own spellings, abbreviations and archaic orthography;
     do not normalise.
3. If the page is a cover, metadata sheet, title page or non-Tibetan, say
   so in the header and transcribe only what Tibetan there is.

## Output

One file per page:
`/home/matt/Documents/tibetan-ocr-app/grid_search_results/claude_review/{tier}/{pdf folder name}/page_N_image_reading.txt`

```
# source_image: page_N_original.jpg
# page_type: body | title | cover | metadata | other
# printed_lines: <number of printed text lines you see>
# reading_confidence: <0-100: share of syllables you believe you read correctly>
# leaned_on_memory: <none | some | heavy> — how much recognising the text helped beyond what the image shows
# notes: <one line: legibility, damage, margins, anything odd>
<line 1>
<line 2>
...
```

## Final report back

Compact table, one row per page: folder, page, page_type, printed_lines,
reading_confidence, leaned_on_memory, bracketed/illegible span count
(rough). Then 2-4 sentences: what limited your reading, and which pages'
readings should NOT be trusted as a yardstick. No Tibetan text dumps.
