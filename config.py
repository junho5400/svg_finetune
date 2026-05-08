"""
Hyperparameters and paths for training.

Edit this file (not notebook cells, not CLI flags) to tune training.
Reproducibility means configs in source. The DryRunConfig inherits from
Config and overrides shapes for a CPU smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class Config:
    # --- Model ---
    # Qwen2.5-Coder-7B chosen after 0.5B baseline showed no letter-shape priors
    # (model knew SVG syntax but not what letters look like). 7B has letter
    # shape knowledge baked in from pretraining, so LoRA only needs to teach
    # style — what fine-tuning is actually good for. See notes/decisions.md.
    base_model: str = "Qwen/Qwen2.5-Coder-7B"

    # --- LoRA ---
    # Rank bumped 16 → 32 after few-shot test showed 7B can't generalize letter
    # shapes from in-context examples — training has to teach 76 letter prototypes
    # × style variations parametrically, which needs more adapter capacity.
    # See notes/gotchas.md "7B few-shot prompting".
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ])

    # --- Training ---
    learning_rate: float = 2e-4
    # batch=2 + grad_accum=8 (was 4+4) keeps effective batch 16 but halves
    # per-step activation memory — needed to fit 7B + LoRA at max_length=2048
    # on a 24GB 4090. See notes/gotchas.md "OOM on first real training step".
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True   # ~30% slower, ~50% less activation mem
    # 2 epochs (was 1) — same reasoning as the LoRA rank bump: model has more to
    # learn from scratch than originally assumed. Save_steps gives us early-stop
    # option if val loss plateaus.
    num_train_epochs: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    bf16: bool = True               # 4090 supports; auto-disabled on CPU
    # max_length=1024 (down from 2048). Why: cross-entropy on Qwen's ~150k vocab
    # × seq_len=2048 produces logits tensors that OOM the 4090. 1024 halves
    # logits memory and still fits ~88% of glyphs (task #7). Lose the longest
    # 12% (mostly decorative pixel-art outliers); keep the bulk of training
    # signal. See notes/gotchas.md "OOM at cross_entropy_loss with long seq_len".
    max_length: int = 1024

    # --- Data paths ---
    train_path: Path = ROOT / "data" / "train.parquet"
    val_path: Path = ROOT / "data" / "val.parquet"
    test_path: Path = ROOT / "data" / "test.parquet"

    # --- Format ---
    # Training pair format: "Generate an SVG glyph for: {caption}\nSVG:\n{svg}"
    # The "SVG:\n" separator is what the model is taught to start its
    # response after. Critical for an untrained-model baseline that knows
    # SVG syntax but doesn't volunteer it from a natural-language prompt.
    instruction_prefix: str = "Generate an SVG glyph for: "
    response_template: str = "\nSVG:\n"

    # --- Output ---
    output_dir: Path = ROOT / "outputs" / "checkpoint"
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    save_total_limit: int = 2

    # --- Hub ---
    hub_repo_id: str = "junho5400/svg-finetune-qwen-7b-lora"
    push_to_hub: bool = True

    # --- WandB ---
    wandb_project: str = "svg-finetune"
    wandb_run_name: str | None = None  # None → trainer auto-names

    # --- Misc ---
    seed: int = 42


@dataclass
class DryRunConfig(Config):
    """Tiny shapes for smoke test on either Mac CPU or RunPod GPU.

    bf16 is True so 7B fits on a 4090 (float32 would OOM at 28GB > 24GB VRAM).
    On Mac CPU this auto-disables in train.py since it's gated on
    torch.cuda.is_available()."""
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    max_length: int = 256
    bf16: bool = True
    push_to_hub: bool = False
    output_dir: Path = ROOT / "outputs" / "dryrun"
    logging_steps: int = 1
    eval_steps: int = 5
    save_steps: int = 50
    wandb_project: str = "svg-finetune-dryrun"
