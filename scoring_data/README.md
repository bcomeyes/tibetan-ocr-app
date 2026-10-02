# scoring_data/

The word list PyBo checks OCR output against. The quality score (`q`) is the
percentage of words found here, so **changing these files changes every score**.

- Source: botok "dialect pack" `general`, release **v0.0.1** of
  https://github.com/Esukhia/botok-data (published 2020-07-27; still the only
  release as of 2026-10-02).
- Copied 2026-10-02 from `~/Documents/pybo/dialect_packs/general`, botok's
  default download location, where it had been downloaded on 2026-04-17.
  Byte-identical to that download (`diff -r`).
- Fingerprint (checked by `OCRQualityScorer` in `ocr_grid_search.ipynb`,
  which refuses to score on mismatch):
  `eb304a395ac210977ffb32a6bb861186fd8a45eb532f3ecf8f7cb823d3e6bf78`

Why it is in the repo: botok downloads "whatever is newest" on first use, to a
folder outside the project. A deleted folder or a new machine would have
silently fetched a possibly different list.

`*.pickled` is botok's cache of the list. It is rebuilt from these files at
every scorer start and is not tracked.
