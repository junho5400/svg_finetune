"""
Retroactive validation loss across historical training checkpoints.

HF Hub keeps every commit. Each "Training in progress, step XXXXX" commit on
the resume repo has the adapter weights at that step. This script:

1. Lists those commits via HfApi
2. Subsamples to N evenly-spaced steps (cheap signal, dense enough for a curve)
3. For each: loads the adapter at that revision, computes val cross-entropy
   on a subset of val examples (loss masked to SVG tokens, matching training)
4. Saves (step, val_loss) → JSON, prints a curve, optionally plots PNG

Run after training is complete. ~2-5 min per checkpoint on A100 with
default --val-samples 2000.

Usage:
    python eval_val_loss.py \\
        --repo-id junho5400/svg-finetune-qwen-7b-lora-resume \\
        --n-points 6 \\
        --val-samples 2000 \\
        --output outputs/val_loss_curve.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from config import Config
from data import filter_by_length, load_split

ROOT = Path(__file__).parent

# Match commits made by Trainer's hub push OR our FullCheckpointPushCallback
COMMIT_PATTERN = re.compile(r"(?:Training in progress|Full checkpoint),?\s*step\s*(\d+)")


def list_step_commits(api: HfApi, repo_id: str) -> list[tuple[int, str]]:
    """Return [(step, commit_id), ...] sorted by step ascending."""
    commits = api.list_repo_commits(repo_id)
    out = []
    seen_steps = set()
    for c in commits:
        m = COMMIT_PATTERN.search(c.title)
        if m:
            step = int(m.group(1))
            if step not in seen_steps:  # in case multiple commits at same step
                out.append((step, c.commit_id))
                seen_steps.add(step)
    out.sort(key=lambda x: x[0])
    return out


def subsample_evenly(items: list, n: int) -> list:
    """Pick n evenly-spaced items including first and last."""
    if len(items) <= n:
        return items
    indices = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[i] for i in indices]


@torch.no_grad()
def compute_val_loss(model, val_ds, collator, batch_size: int) -> float:
    """Mean cross-entropy on val examples (loss masked to SVG tokens)."""
    loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collator)
    total_loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        total_loss_sum += out.loss.item()
        n_batches += 1
    return total_loss_sum / max(n_batches, 1)


def main(repo_id: str, n_points: int, val_samples: int, batch_size: int,
         output_path: Path, plot: bool) -> None:
    cfg = Config()
    api = HfApi()

    # 1. Find historical step commits
    print(f"Listing commits on {repo_id}...")
    step_commits = list_step_commits(api, repo_id)
    print(f"Found {len(step_commits)} step commits "
          f"(steps {step_commits[0][0]} → {step_commits[-1][0]})")
    chosen = subsample_evenly(step_commits, n_points)
    print(f"Will evaluate {len(chosen)} checkpoints: {[s for s, _ in chosen]}")

    # 2. Load base model + tokenizer once
    print(f"\nLoading base model {cfg.base_model}...")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 3. Build val subset (filter + tokenize once)
    print(f"\nLoading val parquet, filtering by max_length={cfg.max_length}...")
    val_ds = load_split(cfg.val_path, cfg)
    val_ds = filter_by_length(val_ds, tok, cfg.max_length)
    val_ds = val_ds.shuffle(seed=42).select(range(min(val_samples, len(val_ds))))
    print(f"  using {len(val_ds)} val examples")

    def tokenize(ex):
        return tok(ex["text"], truncation=True, max_length=cfg.max_length)
    val_ds = val_ds.map(tokenize, remove_columns=["text"], desc="tokenizing")

    # 4. Loss collator — same masking as training (loss only on SVG tokens)
    response_template_ids = tok(
        cfg.response_template.lstrip("\n"),
        add_special_tokens=False,
    ).input_ids
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tok,
    )

    # 5. For each checkpoint: load adapter at revision, eval, unload
    results = []
    for step, commit in chosen:
        print(f"\n=== step {step} (commit {commit[:8]}) ===")
        try:
            model = PeftModel.from_pretrained(base, repo_id, revision=commit)
            model.eval()
            loss = compute_val_loss(model, val_ds, collator, batch_size)
            print(f"  val_loss = {loss:.4f}")
            results.append({"step": step, "val_loss": loss, "commit": commit})
            # unload merges adapter back into base or detaches it; either way
            # the next from_pretrained is clean
            base = model.unload()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({"step": step, "val_loss": None, "commit": commit,
                            "error": f"{type(e).__name__}: {e}"})

    # 6. Save + summarize
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n=== Curve saved to {output_path} ===")
    print(f"\n{'step':>8}  {'val_loss':>10}")
    for r in results:
        loss_str = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "FAILED"
        print(f"{r['step']:>8}  {loss_str:>10}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            xs = [r["step"] for r in results if r["val_loss"] is not None]
            ys = [r["val_loss"] for r in results if r["val_loss"] is not None]
            plt.figure(figsize=(8, 5))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("training step")
            plt.ylabel("val loss (cross-entropy on SVG tokens)")
            plt.title("Validation loss across training")
            plt.grid(alpha=0.3)
            plot_path = output_path.with_suffix(".png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to {plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="junho5400/svg-finetune-qwen-7b-lora-resume",
                    help="HF Hub repo with training checkpoints")
    ap.add_argument("--n-points", type=int, default=6,
                    help="Number of checkpoints to evaluate (evenly spaced)")
    ap.add_argument("--val-samples", type=int, default=2000,
                    help="Subset of val examples to use (full=38k is ~10x slower)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Per-device eval batch size")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs" / "val_loss_curve.json",
                    help="Output JSON path; PNG plot saved alongside")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip matplotlib plot generation")
    args = ap.parse_args()
    main(args.repo_id, args.n_points, args.val_samples, args.batch_size,
         args.output, plot=not args.no_plot)
