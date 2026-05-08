"""
Task #7: validate the extracted glyph dataset.

  1. Parse every SVG as XML to catch any from-extraction syntax bugs.
  2. Tokenize every SVG with the Qwen2.5-Coder tokenizer to get the
     real *token* length distribution (chars-per-token isn't constant).
  3. Report what fraction of glyphs fits at common `max_length`
     choices — drives the training-config decision.
  4. Render N random glyphs back to PNG so we can eyeball whether the
     extracted SVGs actually look right.
  5. Show the top-N longest glyphs (by token count) — these are the
     candidates we may skip during training.
"""
from __future__ import annotations

import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
GLYPHS = ROOT / "data" / "glyphs.parquet"
RENDERS_DIR = ROOT / "data" / "sanity_renders"
SAMPLE_N = 30
TOKENIZER = "Qwen/Qwen2.5-Coder-0.5B"


def safe_filename(name: str, char: str) -> str:
    fam = "".join(c if c.isalnum() else "_" for c in name)[:32].strip("_")
    if char.isalnum():
        ch = char if char.isascii() else f"u{ord(char):04x}"
    else:
        ch = f"p{ord(char):03d}"  # punctuation: encode codepoint
    return f"{fam}__{ch}"


def main() -> None:
    print(f"Loading {GLYPHS}...")
    df = pd.read_parquet(GLYPHS)
    print(f"  {len(df):,} rows × {df.font_family.nunique()} fonts\n")

    # --- 1. SVG validity ---
    print("=== 1. SVG XML validity ===")
    invalid: list[tuple[str, str, str]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="  parsing"):
        try:
            ET.fromstring(row.svg)
        except ET.ParseError as e:
            invalid.append((row.font_family, row.char, str(e)[:80]))
    print(f"  parse failures: {len(invalid):,}/{len(df):,} ({100 * len(invalid) / len(df):.3f}%)")
    for fam, ch, msg in invalid[:5]:
        print(f"    {fam!r} {ch!r}: {msg}")

    # --- 2. Tokenize ---
    print(f"\n=== 2. Token-length distribution (tokenizer: {TOKENIZER}) ===")
    print("  loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    print("  tokenizing all SVGs...")
    lens: list[int] = []
    BATCH = 500
    svgs = df.svg.tolist()
    for i in tqdm(range(0, len(svgs), BATCH), desc="  tokenizing"):
        batch_ids = tok(svgs[i:i + BATCH], add_special_tokens=False).input_ids
        lens.extend(len(ids) for ids in batch_ids)

    sorted_lens = sorted(lens)
    n = len(sorted_lens)

    def pct(p: float) -> int:
        return sorted_lens[min(n - 1, int(n * p))]

    print(f"\n  n      : {n:,}")
    print(f"  min    : {sorted_lens[0]}")
    print(f"  p25    : {pct(0.25)}")
    print(f"  median : {pct(0.50)}")
    print(f"  p75    : {pct(0.75)}")
    print(f"  p90    : {pct(0.90)}")
    print(f"  p95    : {pct(0.95)}")
    print(f"  p99    : {pct(0.99)}")
    print(f"  max    : {sorted_lens[-1]:,}")
    print(f"  mean   : {statistics.mean(sorted_lens):.0f}")

    # --- 3. max_length implications ---
    print("\n=== 3. max_length implications ===")
    print("  fraction of glyphs that fit at each max_length:")
    for max_len in (256, 512, 1024, 2048, 4096, 8192):
        fits = sum(1 for v in sorted_lens if v <= max_len)
        print(f"    max_length={max_len:>5}: {fits:>7,} / {n:,} ({100 * fits / n:5.2f}%)")

    # --- 4. Render samples ---
    print(f"\n=== 4. Rendering {SAMPLE_N} random samples to {RENDERS_DIR.relative_to(ROOT)}/ ===")
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    sample = df.sample(SAMPLE_N, random_state=42)
    failed = 0
    for row in sample.itertuples(index=False):
        out = RENDERS_DIR / f"{safe_filename(row.font_family, row.char)}.png"
        try:
            cairosvg.svg2png(
                bytestring=row.svg.encode(),
                write_to=str(out),
                output_width=128,
                output_height=128,
            )
        except Exception as e:
            failed += 1
            print(f"  render failed: {row.font_family!r} {row.char!r}: {type(e).__name__}: {str(e)[:60]}")
    print(f"  rendered: {SAMPLE_N - failed}/{SAMPLE_N}")

    # --- 5. Long-tail ---
    print("\n=== 5. Top 10 longest glyphs (by tokens) ===")
    df_lens = df.copy()
    df_lens["_tokens"] = lens
    df_lens["_chars"] = df_lens.svg.str.len()
    longest = df_lens.nlargest(10, "_tokens")[["font_family", "char", "_tokens", "_chars"]]
    print(longest.to_string(index=False))


if __name__ == "__main__":
    main()
