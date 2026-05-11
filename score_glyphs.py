"""
CLIP-based letter-recognition scoring for an eval folder.

For each `sample_NN.svg`, re-renders at high resolution, crops tightly
to the glyph, centers it on a white square, then runs CLIP zero-shot
classification across all candidate characters. The argmax candidate
is the prediction; compare to the `target` field in `_summary.json`.

Why CLIP instead of EasyOCR: EasyOCR is trained on document text and
systematically undercounts stylized glyphs (e.g. fails on bold serif
'L', misreads Helvetica 'O' as '0', returns empty for 'k' that's clearly
the letter k). CLIP was trained on web images including logos, posters,
and decorative typography — it handles stylization much better. Trade-off
is occasional confusion on visually similar pairs (O/0, l/1) but those
are unsolvable without document context anyway.

Why high-res + crop preprocessing: see notes/gotchas.md "Glyph eval
preprocessing" — same reasoning as before, applies to CLIP too.

Usage:
    python score_glyphs.py outputs/eval/mideval_50000
"""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import cairosvg
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

# Candidate letter set — uppercase only. CLIP's text encoder + our centered-glyph
# preprocessing can't reliably distinguish 'h' vs 'H' (no size cue, weak case
# signal in the text encoder), so including both lowercase + uppercase just
# wastes prompts and lets uppercase always win the tiebreaker. We score
# case-insensitively below.
CANDIDATES = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "?!&"
)

# Multiple prompt templates — averaging their text embeddings improves CLIP
# zero-shot accuracy by ~2-5% (standard zero-shot trick from the CLIP paper).
PROMPT_TEMPLATES = [
    "a single letter {c}",
    "an image of the character {c}",
    "the letter {c} in a typeface",
]

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def svg_to_centered_png(svg_text: str, render_size: int = 1024,
                        out_size: int = 224, pad_px: int = 30) -> Image.Image:
    """Render SVG, crop to ink, center on white square, resize for CLIP.

    out_size=224 matches CLIP ViT-B/32's input resolution — saves an extra
    resize inside the processor.
    """
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


def build_text_embeddings(model: CLIPModel, processor: CLIPProcessor,
                          device: torch.device) -> torch.Tensor:
    """Pre-compute one text embedding per candidate, averaged across templates."""
    prompts = [t.format(c=c) for t in PROMPT_TEMPLATES for c in CANDIDATES]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        emb = model.text_projection(text_outputs.pooler_output)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    # Reshape to (n_templates, n_candidates, dim) and average across templates
    emb = emb.view(len(PROMPT_TEMPLATES), len(CANDIDATES), -1).mean(dim=0)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # (n_candidates, dim)


def main(eval_dir: Path) -> None:
    summary_path = eval_dir / "_summary.json"
    summary = json.loads(summary_path.read_text())

    print(f"Loading CLIP ({CLIP_MODEL_ID}, downloads ~150 MB on first run)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    text_embeds = build_text_embeddings(model, processor, device)

    by_type: dict[str, list[int]] = {"in-dist": [0, 0], "natural": [0, 0], "name": [0, 0]}
    rows: list[tuple[int, str, str, str, str, float]] = []
    correct_total = 0
    rendered_total = 0

    # Save preprocessed images for visual debugging
    preproc_dir = eval_dir / "preprocessed"
    preproc_dir.mkdir(exist_ok=True)

    for entry in summary:
        i = entry["i"]
        target = entry["target"]
        ptype = entry["type"]
        svg_path = eval_dir / f"sample_{i:02d}.svg"

        if not svg_path.exists():
            rows.append((i, target, ptype, "—", "no_svg", 0.0))
            continue

        try:
            preprocessed = svg_to_centered_png(svg_path.read_text())
            preprocessed.save(preproc_dir / f"sample_{i:02d}.png")
        except Exception as e:
            rows.append((i, target, ptype, "—", f"render_err:{type(e).__name__}", 0.0))
            continue

        img_inputs = processor(images=preprocessed, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=img_inputs.pixel_values)
            img_emb = model.visual_projection(vision_outputs.pooler_output)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        sims = (img_emb @ text_embeds.T).squeeze(0)
        top_idx = int(sims.argmax().item())
        pred = CANDIDATES[top_idx]
        confidence = float(sims[top_idx].item())

        rendered_total += 1
        by_type[ptype][1] += 1

        # Case-insensitive comparison (CLIP can't distinguish case here)
        if pred.upper() == target.upper():
            status = "correct"
            correct_total += 1
            by_type[ptype][0] += 1
        else:
            status = f"wrong→{pred!r}"

        rows.append((i, target, ptype, pred, status, confidence))

    n_total = len(summary)
    print(f"\n=== CLIP-based letter accuracy ===")
    print(f"  exact match (over rendered) : {correct_total}/{rendered_total} ({100 * correct_total / max(rendered_total, 1):.0f}%)")
    print(f"  exact match (over all {n_total})   : {correct_total}/{n_total} ({100 * correct_total / n_total:.0f}%)")
    print()
    print(f"  by type (correct/rendered):")
    for ptype in ["in-dist", "natural", "name"]:
        c, t = by_type[ptype]
        if t > 0:
            print(f"    {ptype:>8} : {c}/{t} ({100 * c / t:.0f}%)")

    print(f"\n=== Per-sample (i, target, type, predicted, status, conf) ===")
    for i, target, ptype, pred, status, conf in rows:
        conf_str = f"  conf={conf:.2f}" if conf > 0 else ""
        print(f"  [{i:02d}] {target!r:>4}  {ptype:>8}  predicted={pred!r:>4}  {status}{conf_str}")

    print(f"\nPreprocessed PNGs saved to: {preproc_dir}/")
    print("(open them to visually verify CLIP was looking at proper images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    args = ap.parse_args()
    main(args.eval_dir)
