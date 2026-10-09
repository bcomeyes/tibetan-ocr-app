# Blind repair of Tibetan OCR output — brief

You are doing one stage of an experiment on Tibetan (Uchen block print) OCR
output. The question: **how much of the true page text can a language model
recover from the garbled OCR text alone, without seeing the page?** A later,
separate stage will check your repairs against the page images. Your work is
only useful if it is genuinely blind.

## Hard rules (blindness)

- Read ONLY files whose names match `page_N_rank1_*.txt` in the folders you
  are assigned. Use the Read tool on them.
- Do NOT open any `.jpg`, any `rank2`/`rank3` file, any PDF, anything under
  `temp_images/`, any raw result folder, or any CSV. Do not search the web.
  Do not read other agents' output folders.
- Do not use the q/c numbers in the filename to shape your judgement.

## For each rank1 file

1. Read the OCR text.
2. Write your **repair**: your best reconstruction of what the page actually
   says, inferred from the OCR text plus your knowledge of Tibetan and of
   Buddhist literature. Typical OCR damage here: spurious tsheg splitting
   words into single letters (ཕྱ་ག for ཕྱག), wrong/missing vowels and
   stacked letters, confused look-alike letters, junk before/after lines.
   - Keep **one repaired line per OCR line, in the same order** (so line 3
     of your repair corresponds to line 3 of the OCR file). If an OCR line
     is pure junk (not page text), write `[junk]` on that line. Keep blank
     lines as blank lines.
   - Put square brackets around any syllable(s) you are unsure of:
     `[གཟུངས་རིག་]`. Where you cannot recover anything, write `[?]`.
   - Repair only what the OCR line gives evidence for. If you recognise a
     known text, you may use that to fix words, but do **not** add text the
     OCR line shows no trace of, and do not silently "normalise" toward a
     famous version when the OCR letters clearly say something else. A
     confident wrong fix is the failure this experiment is trying to detect,
     so bracket honestly.
3. Give two grades, each 0-100:
   - `as_is`: how intelligible is the OCR text **as it stands** to a
     Tibetan reader? 90-100 reads like the page, a few wrong letters;
     70-89 can follow every line, noticeable errors; 50-69 get the gist,
     some lines lost; 30-49 scattered recognisable phrases, cannot follow;
     0-29 mostly unintelligible.
   - `recoverable_guess`: your own estimate of what share of the true
     page's words your repair gets right. (This will be checked.)

## Output

Write one file per page to
`/home/matt/Documents/tibetan-ocr-app/grid_search_results/claude_review/{tier}/{pdf folder name}/page_N_blind_repair.txt`
(create folders as needed; `{tier}` and `{pdf folder name}` are the same as
in the input path). Format exactly:

```
# source: <rank1 filename>
# as_is: <0-100>
# recoverable_guess: <0-100>
# identified_as: <what text/genre you think it is, or "unknown">
# notes: <one line: main kinds of damage; anything odd, e.g. looks like a title page / non-Tibetan / fragment>
<repaired line 1>
<repaired line 2>
...
```

Do not modify or create anything outside `claude_review/`.

## Final report back

A compact table, one row per page: folder, page, as_is, recoverable_guess,
identified_as (short), bracketed-syllable count (rough). Then 2-4 sentences
on patterns you saw. No Tibetan text dumps in the report.
