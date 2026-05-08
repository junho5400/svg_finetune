"""
Build train/val/test parquet files from data/glyphs.parquet
(metadata-captioned) + data/font_descriptions.json (LLM-captioned).

For each glyph row, we emit one training pair per available caption:
  - 3 metadata captions (caption_desc, caption_name, caption_combined)
  - 3 LLM captions if the font has them (form, character, use)
The same SVG is paired with multiple captions — that variety is what
keeps the model from memorizing one shape per font.

Splits are by FONT, not by row. Train/val/test fonts are disjoint, so
val/test glyphs come from fonts the model has never seen during training.
Val loss is a memorization detector + training-stability signal, NOT a
'is this output correct' metric. The 'test' set is reserved for final
qualitative demo inspection.

Filters applied:
  - Drop fonts with <30 glyph coverage (degenerate edge cases like
    Indic-script fonts that only contain a few Latin characters).
  - Drop individual glyphs over 20,000 chars (top ~0.5% — pathological
    layered/decorative outliers that would blow training context).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GLYPHS = ROOT / "data" / "glyphs.parquet"
LLM_CAPS_PATH = ROOT / "data" / "font_descriptions.json"
OUT_DIR = ROOT / "data"

MIN_COVERAGE = 30        # drop fonts with fewer than this many glyphs
MAX_SVG_CHARS = 20_000   # drop individual glyphs longer than this

VAL_FRAC = 0.05
TEST_FRAC = 0.05
SEED = 42

LLM_CAPTION_TEMPLATE = "the letter '{char}' in {description}"


def main() -> None:
    rng = random.Random(SEED)

    print(f"Loading {GLYPHS}...")
    df = pd.read_parquet(GLYPHS)
    print(f"  {len(df):,} glyph rows, {df.font_family.nunique()} fonts")

    # --- Filter 1: drop low-coverage fonts ---
    coverage = df.groupby("font_family").size()
    low_cov = set(coverage[coverage < MIN_COVERAGE].index)
    if low_cov:
        print(f"\nDropping {len(low_cov)} font(s) with <{MIN_COVERAGE} glyph coverage:")
        for fam in sorted(low_cov):
            print(f"  {fam}: {coverage[fam]} glyphs")
        df = df[~df.font_family.isin(low_cov)]
        print(f"  {len(df):,} rows remain ({df.font_family.nunique()} fonts)")

    # --- Filter 2: drop pathological-length glyphs ---
    too_long_mask = df.svg.str.len() > MAX_SVG_CHARS
    n_too_long = int(too_long_mask.sum())
    if n_too_long:
        print(f"\nDropping {n_too_long} glyph(s) over {MAX_SVG_CHARS:,} chars (top "
              f"{100 * n_too_long / len(df):.2f}%)")
        df = df[~too_long_mask]
        print(f"  {len(df):,} rows remain")

    # --- Load LLM captions (may be partial if augmentation still running) ---
    llm_caps: dict[str, list[str]] = {}
    if LLM_CAPS_PATH.exists():
        llm_caps = json.loads(LLM_CAPS_PATH.read_text())
    fonts_remaining = set(df.font_family.unique())
    llm_covered = sum(1 for fam in fonts_remaining if fam in llm_caps)
    print(f"\nLLM captions available for {llm_covered}/{len(fonts_remaining)} remaining fonts")

    # --- Expand each glyph into multiple training pairs ---
    pairs: list[dict] = []
    for row in df.itertuples(index=False):
        fam = row.font_family
        # 3 metadata variants — already char-aware
        pairs.append({"caption": row.caption_desc, "svg": row.svg, "font_family": fam})
        pairs.append({"caption": row.caption_name, "svg": row.svg, "font_family": fam})
        pairs.append({"caption": row.caption_combined, "svg": row.svg, "font_family": fam})
        # LLM variants (if available for this font) — wrap with letter context
        for desc in llm_caps.get(fam, []):
            pairs.append({
                "caption": LLM_CAPTION_TEMPLATE.format(char=row.char, description=desc),
                "svg": row.svg,
                "font_family": fam,
            })

    pairs_df = pd.DataFrame(pairs)
    print(f"\nExpanded into {len(pairs_df):,} training pairs")
    print(f"  avg captions/glyph: {len(pairs_df) / len(df):.2f}")

    # --- Split by font (disjoint sets) ---
    fonts = sorted(fonts_remaining)
    rng.shuffle(fonts)
    n_test = int(len(fonts) * TEST_FRAC)
    n_val = int(len(fonts) * VAL_FRAC)
    test_fonts = set(fonts[:n_test])
    val_fonts = set(fonts[n_test:n_test + n_val])
    train_fonts = set(fonts[n_test + n_val:])

    print(f"\nSplit by font:")
    print(f"  train: {len(train_fonts)} fonts")
    print(f"  val  : {len(val_fonts)} fonts")
    print(f"  test : {len(test_fonts)} fonts")

    train = pairs_df[pairs_df.font_family.isin(train_fonts)].reset_index(drop=True)
    val = pairs_df[pairs_df.font_family.isin(val_fonts)].reset_index(drop=True)
    test = pairs_df[pairs_df.font_family.isin(test_fonts)].reset_index(drop=True)

    print(f"\nFinal splits (rows):")
    print(f"  train: {len(train):,}")
    print(f"  val  : {len(val):,}")
    print(f"  test : {len(test):,}")

    train.to_parquet(OUT_DIR / "train.parquet", index=False, compression="zstd")
    val.to_parquet(OUT_DIR / "val.parquet", index=False, compression="zstd")
    test.to_parquet(OUT_DIR / "test.parquet", index=False, compression="zstd")

    for split_name, path in [("train", OUT_DIR / "train.parquet"),
                              ("val", OUT_DIR / "val.parquet"),
                              ("test", OUT_DIR / "test.parquet")]:
        size_mb = path.stat().st_size / 1e6
        print(f"  wrote {path.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
