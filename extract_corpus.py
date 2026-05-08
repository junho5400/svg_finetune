"""
Run extract_glyph.extract_glyph_svg across every font in
data/font_index.csv and every character in CHARS.

Output: data/glyphs.parquet with columns (font_family, char, svg).

Skips silently:
  - fonts that fail to open (corrupt / unsupported variant table)
  - characters the font doesn't contain (e.g. lowercase missing in a
    caps-only display font)
  - characters whose outline is empty (e.g. punctuation that maps to
    a degenerate glyph)

Per-font: opens TTF once, extracts all CHARS from that single open.
Re-opening per character was ~50x slower in dry runs.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd
from fontTools.ttLib import TTFont
from tqdm import tqdm

from extract_glyph import extract_glyph_svg

ROOT = Path(__file__).parent
INDEX = ROOT / "data" / "font_index.csv"
OUT = ROOT / "data" / "glyphs.parquet"

CHARS = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
    + list(".,!?'\"-=+&@#/:;")
)


def main() -> None:
    with INDEX.open() as f:
        index = list(csv.DictReader(f))

    rows: list[dict] = []
    skipped: Counter = Counter()
    skipped_examples: list[tuple[str, str]] = []
    glyphs_per_font: list[int] = []

    for entry in tqdm(index, desc="extracting"):
        ttf_path = ROOT / entry["ttf_path"]
        try:
            font = TTFont(str(ttf_path))
        except Exception as e:
            skipped[f"open_error:{type(e).__name__}"] += 1
            if len(skipped_examples) < 5:
                skipped_examples.append((entry["family"], type(e).__name__))
            continue

        n = 0
        for ch in CHARS:
            try:
                svg = extract_glyph_svg(font, ch)
            except Exception:
                continue
            if svg is None:
                continue
            rows.append({
                "font_family": entry["family"],
                "char": ch,
                "svg": svg,
            })
            n += 1
        glyphs_per_font.append(n)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False, compression="zstd")

    print("\n=== Extraction complete ===")
    print(f"  rows            : {len(df):,}")
    print(f"  fonts processed : {len(index) - sum(skipped.values()):,}")
    print(f"  fonts skipped   : {sum(skipped.values())} {dict(skipped)}")
    if glyphs_per_font:
        gpf = sorted(glyphs_per_font)
        n = len(gpf)
        print(f"  glyphs / font   : min={gpf[0]} median={gpf[n // 2]} max={gpf[-1]}  (target: {len(CHARS)})")
    print(f"  output          : {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    if skipped_examples:
        print(f"  skip examples   : {skipped_examples}")


if __name__ == "__main__":
    main()
