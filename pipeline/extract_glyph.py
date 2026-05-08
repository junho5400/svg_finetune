"""
Glyph-to-SVG extractor.

Given an open TTFont and a single character, returns a self-contained
<svg> element with the glyph as one <path>, normalized to a 0 0 1000
1000 viewBox.

Two coordinate-system gotchas this handles:

  1. TTF coordinates are y-up (math convention); SVG is y-down (screen
     convention). Without a flip, every glyph renders upside-down.
  2. Different fonts use different "units per em" and different
     ascender/descender heights. We normalize to a fixed 1000x1000
     viewBox so the model sees a consistent coordinate frame regardless
     of source font, and we use full typographic height
     (ascender + |descender|) as the scale denominator so descenders
     never get clipped.

The transform is baked into the path coordinates (via TransformPen)
rather than left as a transform="..." attribute. That keeps the SVG
short and means the model never sees per-font transform numbers as
boilerplate before the real path data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

VIEWBOX = 1000


def extract_glyph_svg(font: TTFont, char: str) -> str | None:
    """Return the glyph SVG for `char` from an open TTFont, or None if missing."""
    if len(char) != 1:
        raise ValueError(f"expected single character, got {char!r}")

    glyph_name = font.getBestCmap().get(ord(char))
    if glyph_name is None:
        return None  # font doesn't have this character

    ascender = font["hhea"].ascender
    descender = font["hhea"].descender  # negative
    total_height = ascender - descender
    scale = VIEWBOX / total_height
    baseline_y = ascender * scale

    # Affine matrix in fontTools 6-tuple form (xx, xy, yx, yy, dx, dy):
    # x_new = scale * x_font
    # y_new = baseline_y - scale * y_font  (the negative yy flips, dy is the offset)
    matrix = (scale, 0, 0, -scale, 0, baseline_y)

    glyph_set = font.getGlyphSet()
    # Round to integers — sub-pixel precision is invisible at 1000-unit
    # scale, and integer coords mean shorter SVG strings (= fewer tokens).
    pen = SVGPathPen(glyph_set, ntos=lambda v: str(round(v)))
    transform_pen = TransformPen(pen, matrix)
    glyph_set[glyph_name].draw(transform_pen)
    d = pen.getCommands()
    if not d:
        return None  # glyph has no outline (e.g. space)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {VIEWBOX} {VIEWBOX}">'
        f'<path d="{d}"/>'
        f'</svg>'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttf", required=True)
    ap.add_argument("--char", required=True)
    ap.add_argument("--out-svg", required=True)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    font = TTFont(args.ttf)
    svg = extract_glyph_svg(font, args.char)
    if svg is None:
        print(f"glyph for {args.char!r} not found in {args.ttf}")
        sys.exit(1)

    Path(args.out_svg).write_text(svg)
    print(f"wrote SVG ({len(svg)} chars): {args.out_svg}")
    print(f"  preview: {svg[:200]}{'...' if len(svg) > 200 else ''}")

    if args.out_png:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg.encode(),
            write_to=args.out_png,
            output_width=256,
            output_height=256,
        )
        print(f"wrote PNG: {args.out_png}")


if __name__ == "__main__":
    main()
