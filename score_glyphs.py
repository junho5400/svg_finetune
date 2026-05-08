"""
OCR-based letter-recognition scoring for an eval folder.

For each `sample_NN.svg`, re-renders at high resolution, crops tightly
to the glyph, centers it on a white square, then runs OCR. Compares the
predicted character to the `target` field in `_summary.json`.

Why high-res + crop: the original 128x128 PNGs failed OCR (returns empty)
because (a) too few pixels, (b) glyph sits in upper-left corner so the
text-detection step skips it. Re-rendering at 1024x1024 and centering
fixes both.

Usage:
    pip install easyocr cairosvg pillow
    python score_glyphs.py outputs/eval/mideval
"""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import cairosvg
import easyocr  # type: ignore
import numpy as np
from PIL import Image, ImageOps

ALLOWLIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "?!.,&-+@#:;'\""
)


def svg_to_centered_png(svg_text: str, render_size: int = 1024,
                        out_size: int = 256, pad_px: int = 30) -> Image.Image:
    """Render SVG, crop to ink, center on white square, resize for OCR."""
    png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode(),
        output_width=render_size,
        output_height=render_size,
    )
    # cairosvg outputs RGBA with transparent canvas; PIL's default RGB
    # conversion composites on BLACK, so the black-ink path on transparent
    # background → all black. Explicitly composite on white instead.
    raw = Image.open(BytesIO(png_bytes)).convert("RGBA")
    img = Image.new("RGB", raw.size, "white")
    img.paste(raw, mask=raw.split()[3])
    # Crop to non-white content (invert so white→0; getbbox finds non-zero)
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    if bbox is None:
        return img.resize((out_size, out_size), Image.LANCZOS)
    cropped = img.crop(bbox)
    w, h = cropped.size
    side = max(w, h) + 2 * pad_px
    centered = Image.new("RGB", (side, side), "white")
    centered.paste(cropped, ((side - w) // 2, (side - h) // 2))
    return centered.resize((out_size, out_size), Image.LANCZOS)


def main(eval_dir: Path) -> None:
    summary_path = eval_dir / "_summary.json"
    summary = json.loads(summary_path.read_text())

    print("Loading EasyOCR (downloads ~250 MB on first run)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    by_type: dict[str, list[int]] = {"in-dist": [0, 0], "natural": [0, 0], "name": [0, 0]}
    rows: list[tuple[int, str, str, str, str]] = []
    correct_total = 0
    rendered_total = 0

    # Save preprocessed images for debugging / visual review
    preproc_dir = eval_dir / "preprocessed"
    preproc_dir.mkdir(exist_ok=True)

    for entry in summary:
        i = entry["i"]
        target = entry["target"]
        ptype = entry["type"]
        svg_path = eval_dir / f"sample_{i:02d}.svg"

        if not svg_path.exists():
            rows.append((i, target, ptype, "—", "no_svg"))
            continue

        try:
            preprocessed = svg_to_centered_png(svg_path.read_text())
            preprocessed.save(preproc_dir / f"sample_{i:02d}.png")
        except Exception as e:
            rows.append((i, target, ptype, "—", f"render_err:{type(e).__name__}"))
            continue

        out = reader.readtext(np.array(preprocessed), detail=0, allowlist=ALLOWLIST)
        pred = (out[0] if out else "").strip()

        rendered_total += 1
        by_type[ptype][1] += 1

        if pred == target:
            status = "correct"
            correct_total += 1
            by_type[ptype][0] += 1
        elif pred and pred.lower() == target.lower():
            status = f"case-mismatch ({pred!r})"
        elif pred:
            status = f"wrong→{pred!r}"
        else:
            status = "empty"

        rows.append((i, target, ptype, pred, status))

    n_total = len(summary)
    print(f"\n=== OCR-based letter accuracy ===")
    print(f"  exact match (over rendered) : {correct_total}/{rendered_total} ({100 * correct_total / max(rendered_total, 1):.0f}%)")
    print(f"  exact match (over all 30)   : {correct_total}/{n_total} ({100 * correct_total / n_total:.0f}%)")
    print()
    print(f"  by type (correct/rendered):")
    for ptype in ["in-dist", "natural", "name"]:
        c, t = by_type[ptype]
        if t > 0:
            print(f"    {ptype:>8} : {c}/{t} ({100 * c / t:.0f}%)")

    print(f"\n=== Per-sample (i, target, type, predicted, status) ===")
    for i, target, ptype, pred, status in rows:
        print(f"  [{i:02d}] {target!r:>4}  {ptype:>8}  predicted={pred!r:>5}  {status}")

    print(f"\nPreprocessed PNGs saved to: {preproc_dir}/")
    print("(open them to visually verify the OCR was looking at proper images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    args = ap.parse_args()
    main(args.eval_dir)
