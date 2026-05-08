"""
Render a sample image per font and ask Claude (via local CLI, Max plan)
to write 3 vibe descriptions per font based on what it actually sees.

Per call:  5 fonts → 5 sample PNGs → 1 claude --print invocation that
loads each image with the Read tool and returns JSON with 3 short
style descriptions per font.

Output: data/font_descriptions.json mapping family -> [desc, desc, desc].
Cache is incremental and the script is restartable — already-captioned
fonts are skipped on subsequent runs.

Sample images persist in data/font_samples/ so re-runs don't re-render.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "font_index.csv"
SAMPLES_DIR = ROOT / "data" / "font_samples"
CACHE = ROOT / "data" / "font_descriptions.json"

SAMPLE_TEXT = "AaBbGg 123 ?!"
SAMPLE_SIZE = (1024, 256)
FONT_PIXEL_SIZE = 120
BATCH_SIZE = 20
MODEL = "haiku"
TIMEOUT_SEC = 600

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "font_family": {"type": "string"},
                    "descriptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                "required": ["font_family", "descriptions"],
            },
        }
    },
    "required": ["results"],
}


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def render_sample(ttf_path: Path, out_path: Path) -> bool:
    """Render SAMPLE_TEXT using the font; return True on success.

    Some TTFs have hinting / table layouts that PIL's text engine can't
    handle (raises OSError "execution context too long" or similar).
    We catch *any* exception during rendering — those fonts get skipped
    rather than killing the whole pipeline.
    """
    try:
        font = ImageFont.truetype(str(ttf_path), FONT_PIXEL_SIZE)
        img = Image.new("RGB", SAMPLE_SIZE, "white")
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), SAMPLE_TEXT, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = max(10, (SAMPLE_SIZE[0] - text_w) // 2 - bbox[0])
        y = (SAMPLE_SIZE[1] - text_h) // 2 - bbox[1]
        draw.text((x, y), SAMPLE_TEXT, font=font, fill="black")
        img.save(out_path)
        return True
    except Exception as e:
        print(f"  render failed for {ttf_path.name}: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def build_prompt(batch: list[dict], image_paths: dict[str, Path]) -> str:
    cat_phrase = {
        "SANS_SERIF": "sans-serif", "SERIF": "serif", "DISPLAY": "display",
        "HANDWRITING": "handwriting", "MONOSPACE": "monospace",
    }
    image_list = "\n".join(
        f"- {image_paths[r['family']]} — font_family={r['family']!r}, "
        f"already known to be a {cat_phrase.get(r['category'], 'typeface')} typeface"
        for r in batch
    )
    return f"""You'll see sample renderings of typefaces. Write factual descriptions of each.

The category of each font (sans-serif / serif / display / handwriting / monospace) is given to
you as ground truth — do NOT contradict it. Use the image to describe specific visual
qualities WITHIN that category (stroke weight, contrast, x-height, terminals, axis tilt,
joints, character width).

For each image, write exactly 3 short descriptions (8-14 words each). The three MUST cover
distinct angles so they're not interchangeable. Be concise, descriptive, and direct — describe
what the typeface IS, not how amazing it is or where it would be "perfect" to use.

  1. **Form** — concrete typographical observations of what you see in the image: stroke
     contrast, terminals (sharp/round/slab), x-height, axis tilt, weight, joint construction,
     character width.
     Example: "humanist sans, moderate x-height, low contrast, soft joints, open apertures"

  2. **Character** — the typeface's mood or feel, using direct plain language. Emotional
     adjectives are good and expected ("warm", "cold", "friendly", "harsh", "calm",
     "playful", "serious", "casual"). What to AVOID is marketing-style evocative wrapping
     around those words.
     Examples (good): "warm and friendly; informal without being playful"
                      "cold and technical; restrained"
                      "playful but composed; approachable"
     Counter-examples (avoid): "feels remarkably warm with a quietly confident voice"
                               "carries a beautifully understated warmth"

  3. **Use** — describe where this typeface is typically found, factually.
     Example: "common in editorial body text and educational materials"
     Counter-example (avoid): "perfect for editorial designs and pairs beautifully with serifs"

Hard rules:
- NEVER include the font name in any description.
- NO marketing phrases: "perfect for", "ideal for", "great for", "well-suited to",
  "pairs beautifully/wonderfully/well with", "carries weight", "reads as remarkable", etc.
- NO hyperbolic adjectives: "stunning", "remarkable", "wonderful", "beautifully".
- Each description is a single line of prose.
- The "descriptions" array MUST have exactly 3 entries in the order above (form, character, use).

Use the Read tool to load every image first, then return JSON.

Images:
{image_list}
"""


def call_claude(prompt: str) -> dict:
    result = subprocess.run(
        [
            "claude", "--print", "--no-session-persistence",
            "--model", MODEL,
            "--output-format", "json",
            "--json-schema", json.dumps(JSON_SCHEMA),
            prompt,
        ],
        capture_output=True, text=True, timeout=TIMEOUT_SEC,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CLI exit {result.returncode}: {result.stderr[:300]}")
    parsed = json.loads(result.stdout)
    if parsed.get("is_error"):
        raise RuntimeError(f"claude error: {parsed.get('result', '')[:300]}")
    # With --json-schema, the validated structured output lives in
    # `structured_output`. The `result` field just has a free-form
    # confirmation string from the model.
    out = parsed.get("structured_output")
    if not out:
        raise RuntimeError(f"no structured_output in response: result={parsed.get('result', '')[:200]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N pending fonts (for testing)")
    args = ap.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[str]] = (
        json.loads(CACHE.read_text()) if CACHE.exists() else {}
    )

    with INDEX.open() as f:
        index = list(csv.DictReader(f))
    todo = [r for r in index if r["family"] not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Total fonts: {len(index)} | cached: {len(cache)} | to do: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(todo), BATCH_SIZE), total=n_batches, desc="batches"):
        batch = todo[i:i + BATCH_SIZE]

        image_paths: dict[str, Path] = {}
        valid_batch: list[dict] = []
        for row in batch:
            ttf = ROOT / row["ttf_path"]
            png = SAMPLES_DIR / f"{safe_filename(row['family'])}.png"
            if not png.exists():
                if not render_sample(ttf, png):
                    continue
            image_paths[row["family"]] = png
            valid_batch.append(row)

        if not valid_batch:
            continue

        try:
            data = call_claude(build_prompt(valid_batch, image_paths))
        except Exception as e:
            tqdm.write(f"  batch failed at offset {i}: {e}")
            continue

        for item in data.get("results", []):
            fam = item.get("font_family")
            descs = item.get("descriptions") or []
            if fam and len(descs) == 3:
                cache[fam] = descs

        CACHE.write_text(json.dumps(cache, indent=2))

    print(f"\nDone. {len(cache)}/{len(index)} fonts captioned. Cache: {CACHE}")


if __name__ == "__main__":
    main()
