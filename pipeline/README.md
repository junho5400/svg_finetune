# Data pipeline

One-time scripts that build the training dataset from a fresh Google Fonts clone.
**Already executed for the current dataset** — these are here for reproducibility, not
day-to-day use.

Run from the project root, in this order:

```bash
# Pre-req: shallow-clone Google Fonts to data/google-fonts/
git clone --depth 1 https://github.com/google/fonts.git data/google-fonts

# 1. Parse METADATA.pb files + tags/families.csv into a font index CSV
python pipeline/build_font_index.py
# → data/font_index.csv

# 2. Extract glyphs as SVGs across all fonts
python pipeline/extract_corpus.py
# → data/glyphs.parquet (uses pipeline/extract_glyph.py)

# 3. Add metadata captions
python pipeline/build_captions_metadata.py
# → data/glyphs.parquet (adds caption_desc, caption_name, caption_combined columns)

# 4. Augment with LLM-generated style descriptions (vision via claude CLI)
python pipeline/augment_captions.py
# → data/font_descriptions.json (3 vibe descriptions per font)

# 5. Build train/val/test splits
python pipeline/build_splits.py
# → data/{train,val,test}.parquet

# 6. (optional) Sanity-check token-length distribution + render samples
python pipeline/sanity_check.py
# → data/sanity_renders/
```

## Other files

- `extract_glyph.py` — utility (no `__main__`); imported by `extract_corpus.py`
- `filter_quality.py` — one-off analysis of /Quality/ tag distribution; informed
  the decision to NOT filter by quality. See `notes/decisions.md`.

## Notes

- These scripts use `ROOT = Path(__file__).resolve().parent.parent` to find the
  project root, since they live one level deep.
- Output paths are all under `data/` (project root), gitignored.
- See `notes/decisions.md` for the reasoning behind each step's design choices.
