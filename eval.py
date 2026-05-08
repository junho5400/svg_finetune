"""
Evaluate model on a fixed prompt set. Use:
  python eval.py --baseline                         # untuned Qwen2.5-Coder-0.5B
  python eval.py --adapter outputs/checkpoint/final  # after training
  python eval.py --adapter junho5400/svg-finetune-qwen-0.5b-lora   # from Hub

For each prompt:
  - Generate continuation from the model.
  - Try to extract an <svg>...</svg> block.
  - Check if it's valid XML; render to PNG if possible.
  - Save raw generation + extracted SVG + PNG.

Aggregates: SVG extraction rate, XML validity rate, render success rate.

Designed so we can run BEFORE training (baseline floor) and AFTER
training (improvement) and compare side by side.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config

ROOT = Path(__file__).parent


# Fixed prompt set used for baseline AND post-training so comparisons are
# apples-to-apples. Each prompt tagged with its type for per-type stat breakdown:
#   "in-dist"  — matches caption_desc training format
#   "natural"  — natural-language style description (closer to real user prompts)
#   "name"     — font name as the style anchor (matches caption_name training format)
EVAL_PROMPTS: list[tuple[str, str, str]] = [
    # ----- In-distribution (15): matches caption_desc training format -----
    # 3 per category × 5 categories
    ("the letter 'A' in a sans-serif typeface; humanist, calm, easy to read", "A", "in-dist"),
    ("the letter 'g' in a sans-serif typeface; geometric, neo-grotesque", "g", "in-dist"),
    ("the letter 'h' in a sans-serif typeface; rounded, friendly", "h", "in-dist"),
    ("the letter 'B' in a serif typeface; transitional, formal", "B", "in-dist"),
    ("the letter 'q' in a serif typeface; old-style, vintage", "q", "in-dist"),
    ("the letter 'R' in a serif typeface; modern, high-contrast", "R", "in-dist"),
    ("the letter 'M' in a display typeface; playful, rounded, blobby", "M", "in-dist"),
    ("the letter '7' in a display typeface; vintage, fat-face", "7", "in-dist"),
    ("the letter 'W' in a display typeface; futuristic, geometric", "W", "in-dist"),
    ("the letter 'p' in a handwriting typeface; informal, cursive", "p", "in-dist"),
    ("the letter 'L' in a handwriting typeface; calligraphic, formal", "L", "in-dist"),
    ("the letter 'j' in a handwriting typeface; casual, script", "j", "in-dist"),
    ("the letter 'X' in a monospace typeface; technical, clean", "X", "in-dist"),
    ("the letter '2' in a monospace typeface; geometric, neo-grotesque", "2", "in-dist"),
    ("the letter 'k' in a monospace typeface; technical, programmer-style", "k", "in-dist"),

    # ----- Natural (10): style words only, no "in a typeface" scaffolding -----
    # Tests whether the model generalizes to user-style prompts.
    ("warm humanist 'C', calm and easy to read", "C", "natural"),
    ("geometric 'd' in modern style", "d", "natural"),
    ("formal transitional 'E', classical and elegant", "E", "natural"),
    ("old-style vintage 's', traditional character", "s", "natural"),
    ("playful 'N', blobby and bouncy", "N", "natural"),
    ("vintage decorative '5', ornate and bold", "5", "natural"),
    ("flowing 'y' in calligraphic style", "y", "natural"),
    ("hand-drawn casual 'T', informal handwriting", "T", "natural"),
    ("technical clean '?', monospaced precision", "?", "natural"),
    ("sharp futuristic '&', geometric", "&", "natural"),

    # ----- Name references (5) -----
    # Mix of in-corpus (Roboto, Pacifico) and famous-but-not-in-corpus
    # (Comic Sans, Helvetica, Times) — tests generalization to known names.
    ("the letter 'h' in Roboto style", "h", "name"),
    ("the letter 'S' in Pacifico style", "S", "name"),
    ("the letter 'A' in Comic Sans style", "A", "name"),
    ("the letter 'O' in Helvetica style", "O", "name"),
    ("the letter 'r' in Times New Roman style", "r", "name"),
]


def extract_svg(text: str) -> str | None:
    """Pull the first complete <svg>...</svg> block from a generation."""
    start = text.find("<svg")
    if start < 0:
        return None
    end = text.find("</svg>", start)
    if end < 0:
        return None
    return text[start:end + len("</svg>")]


def is_valid_xml(s: str) -> bool:
    try:
        ET.fromstring(s)
        return True
    except ET.ParseError:
        return False


def main(adapter: str | None, output_root: Path, max_new_tokens: int) -> None:
    cfg = Config()

    print(f"Loading tokenizer + base model ({cfg.base_model})...")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter}...")
        model = PeftModel.from_pretrained(model, adapter)
        run_name = adapter.replace("/", "__").replace("\\", "__")
    else:
        print("BASELINE run: no LoRA adapter loaded.")
        run_name = "baseline"

    model.eval()
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from data import format_prompt
    results: list[dict] = []
    for i, (prompt, target_char, ptype) in enumerate(EVAL_PROMPTS):
        text_in = format_prompt(prompt, cfg)
        inputs = tok(text_in, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for deterministic eval
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out_ids[0][inputs.input_ids.shape[1]:]
        gen_text = tok.decode(new_tokens, skip_special_tokens=True)

        svg = extract_svg(gen_text)
        valid_xml = bool(svg) and is_valid_xml(svg)
        rendered = False

        # Always save the raw generation so we can read what the model did
        (out_dir / f"sample_{i:02d}_raw.txt").write_text(gen_text)
        if svg:
            (out_dir / f"sample_{i:02d}.svg").write_text(svg)
            try:
                cairosvg.svg2png(
                    bytestring=svg.encode(),
                    write_to=str(out_dir / f"sample_{i:02d}.png"),
                    output_width=128, output_height=128,
                )
                rendered = True
            except Exception:
                pass

        status_marks = []
        if svg: status_marks.append("svg")
        if valid_xml: status_marks.append("xml")
        if rendered: status_marks.append("png")
        status = "+".join(status_marks) or "—"

        print(f"  [{i:02d}] {target_char!r:>4}  {ptype:>8}  ({status:>11})  "
              f"gen_tokens={int(new_tokens.shape[0]):>4}  "
              f"prompt={prompt[:55]}...")

        results.append({
            "i": i,
            "prompt": prompt,
            "target_char": target_char,
            "type": ptype,
            "n_gen_tokens": int(new_tokens.shape[0]),
            "svg_extracted": svg is not None,
            "valid_xml": valid_xml,
            "rendered": rendered,
        })

    (out_dir / "_summary.json").write_text(json.dumps(results, indent=2))

    n = len(results)
    print(f"\n=== Summary ({run_name}) ===")
    print(f"  total prompts            : {n}")
    print(f"  total extracted <svg>    : {sum(r['svg_extracted'] for r in results)}/{n}")
    print(f"  total valid XML          : {sum(r['valid_xml'] for r in results)}/{n}")
    print(f"  total rendered to PNG    : {sum(r['rendered'] for r in results)}/{n}")
    print()
    print("  by type (rendered/total):")
    for ptype in ["in-dist", "natural", "name"]:
        rs = [r for r in results if r["type"] == ptype]
        if rs:
            n_rend = sum(r["rendered"] for r in rs)
            print(f"    {ptype:>8} : {n_rend}/{len(rs)}")
    print(f"\n  outputs in: {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="Path or HF Hub repo id of LoRA adapter; omit for baseline")
    ap.add_argument("--baseline", action="store_true",
                    help="Force baseline mode (ignore --adapter)")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    adapter = None if args.baseline else args.adapter
    main(adapter, Path(args.output_dir), args.max_new_tokens)
