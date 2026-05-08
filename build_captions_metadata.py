"""
Add three templated caption variants to data/glyphs.parquet , one row
per glyph still, but with new columns:

  caption_desc      "the letter 'A' in a sans-serif typeface; humanist, calm, easy to read"
  caption_name      "the letter 'A' in Roboto style"
  caption_combined  "the letter 'A' in Roboto, a sans-serif typeface; humanist, calm"

Decision context (see chat for full reasoning): we keep BOTH descriptive
and name-anchored variants so the trained model can serve users who
prompt by style and users who prompt by font name.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
INDEX = ROOT / "data" / "font_index.csv"
GLYPHS = ROOT / "data" / "glyphs.parquet"

CATEGORY_NAME = {
    "SANS_SERIF": "sans-serif",
    "SERIF": "serif",
    "DISPLAY": "display",
    "HANDWRITING": "handwriting",
    "MONOSPACE": "monospace",
}

# Quality-score tags — internal metrics about spacing / abstract aesthetic
# rating, not style words. "the letter 'A' in a sans-serif typeface; spacing,
# wordspace" doesn't help anyone. Holiday/seasonal tags ARE kept: they're
# valid vibe vocabulary (a user typing "halloween" or "valentine's" is
# making a real style request).
SKIP_TAG_LEAVES = {"Concept", "Drawing", "Spacing", "Wordspace"}

# Most leaf tag names work as-is when lowercased. A handful read awkwardly
# in caption position; rewrite those for readability.
TAG_PHRASE_REWRITE = {
    "Easy Reading": "easy to read",
    "Neo Grotesque": "neo-grotesque",
    "Old Style Garalde": "old-style garalde",
    "Fat Face": "fat-face",
    "Upright Script": "upright script",
    "Pixel": "pixel-style",
}

TOP_K_TAGS = 4  # cap how many tags appear per caption — more is noise


def parse_top_tags(s: str) -> list[tuple[str, int]]:
    """Parse "Humanist:75|Calm:81|..." into [(tag, score), ...]."""
    if not s:
        return []
    out = []
    for piece in s.split("|"):
        if ":" not in piece:
            continue
        tag, _, score = piece.rpartition(":")
        try:
            out.append((tag.strip(), int(score)))
        except ValueError:
            continue
    return out


def selected_phrases(tag_scores: list[tuple[str, int]], category: str) -> list[str]:
    """Filter unhelpful tags, drop tags that just repeat the category,
    take top K, convert to natural phrasing."""
    cat_phrase = CATEGORY_NAME.get(category, "")
    kept = [(t, s) for t, s in tag_scores
            if t not in SKIP_TAG_LEAVES
            and t.lower() != cat_phrase.lower()]
    kept.sort(key=lambda ts: ts[1], reverse=True)
    return [TAG_PHRASE_REWRITE.get(t, t.lower()) for t, _ in kept[:TOP_K_TAGS]]


def build_captions(char: str, font_name: str, category: str,
                   tag_scores: list[tuple[str, int]]) -> tuple[str, str, str]:
    cat = CATEGORY_NAME.get(category, "typeface")
    phrases = selected_phrases(tag_scores, category)
    tail = f"; {', '.join(phrases)}" if phrases else ""

    desc = f"the letter '{char}' in a {cat} typeface{tail}"
    name = f"the letter '{char}' in {font_name} style"
    combined = f"the letter '{char}' in {font_name}, a {cat} typeface{tail}"
    return desc, name, combined


def main() -> None:
    print(f"Loading font index from {INDEX}...")
    with INDEX.open() as f:
        index = {row["family"]: row for row in csv.DictReader(f)}
    print(f"  {len(index)} fonts in index")

    print(f"Loading glyphs from {GLYPHS}...")
    df = pd.read_parquet(GLYPHS)
    print(f"  {len(df):,} rows")

    descs: list[str] = []
    names: list[str] = []
    combos: list[str] = []
    missing_in_index = 0

    # Parse tag strings once per font (small dict), not per glyph (143k).
    tag_cache: dict[str, list[tuple[str, int]]] = {}

    for fam, ch in zip(df["font_family"], df["char"]):
        meta = index.get(fam)
        if meta is None:
            # font in glyphs but not in index — shouldn't happen since we
            # generated glyphs FROM the index, but defend anyway
            missing_in_index += 1
            descs.append(f"the letter '{ch}' in a typeface")
            names.append(f"the letter '{ch}' in {fam} style")
            combos.append(f"the letter '{ch}' in {fam}, a typeface")
            continue

        if fam not in tag_cache:
            tag_cache[fam] = parse_top_tags(meta["top_tags"])

        d, n, c = build_captions(ch, fam, meta["category"], tag_cache[fam])
        descs.append(d)
        names.append(n)
        combos.append(c)

    df["caption_desc"] = descs
    df["caption_name"] = names
    df["caption_combined"] = combos

    df.to_parquet(GLYPHS, index=False, compression="zstd")
    print(f"\n=== Captions added ===")
    print(f"  rows           : {len(df):,}")
    print(f"  cols           : {list(df.columns)}")
    print(f"  missing in idx : {missing_in_index}")
    print(f"  output         : {GLYPHS} ({GLYPHS.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
