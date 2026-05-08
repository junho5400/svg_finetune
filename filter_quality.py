"""
Filter data/glyphs.parquet to high-quality fonts only, using Google
Fonts' /Quality/ tag scores (Drawing, Spacing, Wordspace, Concept).

Reads families.csv directly (the index lost sub-50 scores during
top-N filtering, so we re-parse to see the full picture). Computes a
per-font quality score, reports the distribution at several thresholds
so the threshold choice is informed, then filters.

Default: keep fonts where the *minimum* of the four /Quality/ scores
is >= QUALITY_THRESHOLD. Min (rather than mean) means a font with
great drawing but bad spacing still gets dropped — one weak dimension
is enough to disqualify, which matches how a designer would judge.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
FAMILIES_CSV = ROOT / "data" / "google-fonts" / "tags" / "all" / "families.csv"
GLYPHS = ROOT / "data" / "glyphs.parquet"

QUALITY_TAGS = ("/Quality/Drawing", "/Quality/Spacing",
                "/Quality/Wordspace", "/Quality/Concept")


def load_quality_scores() -> dict[str, dict[str, int]]:
    """family -> {/Quality/Drawing: 75, /Quality/Spacing: 80, ...}"""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with FAMILIES_CSV.open() as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            family, _, tag, score = row[0], row[1], row[2], row[3]
            if tag in QUALITY_TAGS:
                try:
                    out[family.strip()][tag] = int(score)
                except ValueError:
                    continue
    return out


def per_font_min(scores: dict[str, dict[str, int]],
                 fonts: list[str]) -> dict[str, int]:
    """Min score across the 4 quality dims. 0 if any dim is missing."""
    out = {}
    for fam in fonts:
        fam_scores = scores.get(fam, {})
        if len(fam_scores) < len(QUALITY_TAGS):
            out[fam] = 0
        else:
            out[fam] = min(fam_scores.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=None,
                    help="min quality score to keep (0–100). If omitted, only reports.")
    ap.add_argument("--write", action="store_true",
                    help="overwrite glyphs.parquet with the filtered set")
    args = ap.parse_args()

    print("Loading families.csv quality scores...")
    quality = load_quality_scores()
    print(f"  fonts with at least one quality tag: {len(quality)}")

    print(f"Loading {GLYPHS}...")
    df = pd.read_parquet(GLYPHS)
    fonts_in_data = sorted(df["font_family"].unique())
    print(f"  {len(fonts_in_data)} fonts × {len(df):,} rows")

    min_by_font = per_font_min(quality, fonts_in_data)

    vals = sorted(min_by_font.values())
    n = len(vals)

    def pct(p: float) -> int:
        return vals[max(0, min(n - 1, int(n * p)))]

    print("\n=== Per-font min(/Quality/) distribution ===")
    print(f"  fonts with all 4 quality tags : {sum(1 for v in vals if v > 0)} / {n}")
    print(f"  fonts missing some            : {sum(1 for v in vals if v == 0)} / {n}")
    print(f"  among scored: min={min(v for v in vals if v > 0)}  median={pct(0.5)}  max={max(vals)}")

    print("\n=== Pass rate by threshold ===")
    for t in (50, 60, 70, 75, 80, 85):
        passing = sum(1 for v in vals if v >= t)
        rows_kept = sum(1 for fam, s in min_by_font.items() if s >= t
                        for _ in df[df.font_family == fam].itertuples())  # heavy; for display
        # cheaper:
        keep_fams = {fam for fam, s in min_by_font.items() if s >= t}
        rows_kept = int(df["font_family"].isin(keep_fams).sum())
        print(f"  threshold {t:>3}: {passing:4d} fonts ({100*passing/n:5.1f}%) → {rows_kept:,} glyph rows")

    if args.threshold is None:
        print("\n(no --threshold passed; nothing written. Re-run with --threshold N --write to filter.)")
        return

    keep_fams = {fam for fam, s in min_by_font.items() if s >= args.threshold}
    df_filt = df[df["font_family"].isin(keep_fams)].reset_index(drop=True)
    print(f"\nApplying threshold {args.threshold}:")
    print(f"  fonts kept : {len(keep_fams)} / {n}")
    print(f"  rows kept  : {len(df_filt):,} / {len(df):,}")

    if args.write:
        df_filt.to_parquet(GLYPHS, index=False, compression="zstd")
        print(f"  wrote      : {GLYPHS} ({GLYPHS.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  (dry run; pass --write to overwrite glyphs.parquet)")


if __name__ == "__main__":
    main()
