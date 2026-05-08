# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo state

Small text-to-SVG LoRA fine-tuning side project. This file is the authoritative spec; reasoning behind specific choices and resolved gotchas live in `notes/decisions.md` and `notes/gotchas.md`.

## Layout

```
.
├── pipeline/             # Data pipeline (one-time, already run). See pipeline/README.md
├── colab/                # Colab notebooks (eval, dryrun, train_resume)
├── notes/                # decisions.md + gotchas.md
├── config.py             # Training hyperparameters (Config + DryRunConfig)
├── data.py               # Dataset loading, format helpers
├── train.py              # SFTTrainer entrypoint, auto-resume, auto-terminate
├── eval.py               # Generation eval against fixed prompt set
├── score_glyphs.py       # OCR-based letter-accuracy scorer (post-eval)
├── prep_data_hub.py      # Push parquets to HF Hub for Colab fetch
├── requirements.txt      # Pinned deps (validated via colab/dryrun_validate.ipynb)
├── data/                 # gitignored — parquets, model cache, training caches
└── outputs/              # gitignored — checkpoints, eval renders
```

## Scope

**Text → single SVG glyph, conditioned on free-form font style description.** Demo input: a vibe prompt + a target character (e.g. *"warm humanist serif, journal-like"* + `'A'`). Demo output: an SVG glyph in that style, progressively rendered as tokens stream.

Dataset is custom-curated from Google Fonts (~1859 fonts × ~80 chars). One-shot alphabet completion from a single example was scoped out and should not be proposed. Editing existing SVGs by instruction is a stretch goal — see `notes/decisions.md` — but Stage 1 (this scope) ships first.

## Hard constraints

- **LoRA only** on `Qwen/Qwen2.5-Coder-7B`. No full fine-tune. (Earlier 0.5B failed baseline check — couldn't draw letter shapes from scratch under LoRA capacity. See `notes/decisions.md`.)
- **Configs in `config.py`**, not in notebook cells.
- **Auto-terminate every RunPod pod** before it bills, via `runpodctl stop pod $RUNPOD_POD_ID` at the end of `train.py`. No exceptions.
- **Budget**: $30 personal cap + Colab Pro + $10 RunPod credit. Default allocation: Colab Pro for dev/smoke/eval; RunPod for the real run(s); $20 reserve for Stage 2 / reruns.
- **Non-goals**: SOTA quality, multi-GPU, custom tokenizers, image-to-SVG, animation, full-stack frontend, full-alphabet style transfer.

## Pipeline phases

- **Curate (Mac)**: extract glyphs from Google Fonts via `fonttools` (`SVGPathPen`), normalized to `viewBox="0 0 1000 1000"` with integer coords, single `<path>`. Captions = 3 metadata templates per glyph + 3 LLM-augmented "vibe" descriptions per font (form / character / use), generated via `claude` CLI vision (Max plan, Haiku).
- **Local dev (Mac, CPU)**: `train.py`, `data.py`, `config.py`, `eval.py` with a `DRY_RUN=1` mode that runs end-to-end on tiny shapes. Preserve this path — it's the wiring contract before any GPU spend.
- **Remote train (RunPod 4090, VS Code Remote-SSH)**: real shapes, WandB logging, push LoRA adapter to HF Hub, then auto-terminate.
- **Demo (HF Space, Gradio, free CPU tier)**: loads base + adapter, streams tokens, applies the progressive sanitizer.

## Python environment

Conda env **`svg_finetune`** (Python 3.12). `conda activate svg_finetune` or `conda run -n svg_finetune python ...`. Don't install into system/Homebrew Python; don't use the user's general-purpose `myenv` env for this project's deps.

## Training-time decisions baked into the data

When writing `train.py`, these must be honored:

- **Loss masking on captions** — compute loss only on SVG tokens (`DataCollatorForCompletionOnlyLM` or equivalent).
- **Consistent caption↔SVG separator** at training and inference. Bake into the tokenizer template in `data.py`.
- **Skip examples over `max_length`, don't truncate** — truncation teaches malformed output.
- **Held-out test fonts are for qualitative inspection**, not loss-based eval. Val loss = memorization detector + early-stopping signal, not a "did the output match" metric.
- **SFT first, DPO only if SFT outputs reveal specific behavioral failures.** Reasoning + reward-signal options in `notes/decisions.md`.

## RunPod workflow (mandatory)

1. Network Volume (10–20GB), region-pinned to GPU region.
2. Spin up RTX 4090 pod, mount volume, SSH via VS Code Remote.
3. `git clone`, `pip install -r requirements.txt`, `wandb login`, `huggingface-cli login`.
4. **Smoke before real**: `python train.py --dry-run` → tiny GPU run (100 rows, 5 steps) → real run.
5. Real run includes auto-terminate at the end of `train.py`:
   ```python
   import os, subprocess
   subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]])
   ```
6. Push the LoRA adapter to HF Hub *before* the auto-terminate fires.
7. Pod terminates. Keep the volume.

## Demo sanitizer edge cases

Streaming output breaks SVG mid-token. The Gradio sanitizer must handle, every N tokens or on closing-tag boundaries:

- **Mid-attribute cut** (`<path d="M10 20`) — trim back to last whole number/space.
- **Mid-tag cut** (`<pat`) — drop the partial open-tag.
- **Malformed numbers** in `d` (trailing `.`, `e`, `-`) — trim partial numerics.
- **Unclosed strings** — close the open quote before appending closing markup.
- **Unclosed tags** — append matching `/>` or `</tag>`.

## Success criteria (Stage 1 ship)

- HF Space live, works end-to-end for 5–10 sample prompts (mix of style descriptions + font-name references).
- Progressive rendering visibly works.
- README documents the project end-to-end (raw material in `notes/`).
- Total spend ≤ $30.

## Commands

No build/test/lint tooling yet. When adding: simplest thing that works (`requirements.txt` + `python train.py --dry-run`). No Poetry/Hatch/tox/pre-commit unless asked.
