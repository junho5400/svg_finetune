"""
Train Qwen2.5-Coder-0.5B with LoRA on (caption → SVG glyph) pairs.

Modes:
  python train.py --dry-run    Tiny shapes on CPU. Catches wiring bugs cheaply
                                before any GPU spend. Mandatory before any
                                real run.
  python train.py              Real training run (bf16 + WandB + push to HF Hub).

If RUNPOD_POD_ID is in the environment at end of a real run, the script
calls `runpodctl stop pod` to terminate the pod (CLAUDE.md mandates this).

Format details:
  - Each training pair is `{caption}\\n{svg}`.
  - DataCollatorForCompletionOnlyLM is given response_template=`\\n<svg`,
    which uniquely starts the SVG portion. Loss is computed only on
    tokens after that template — caption tokens contribute no gradient.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import torch

# torch 2.6+ changed torch.load to default weights_only=True for security.
# Our saved checkpoints include numpy objects (rng_state.pth) which strict
# weights_only mode rejects with `Unsupported global: numpy.core.multiarray
# ._reconstruct`. Since we only ever load our own trusted checkpoints, force
# weights_only=False on every torch.load call.
# See notes/gotchas.md "torch 2.6+ weights_only default breaks RNG resume".
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from config import Config, DryRunConfig
from data import load_datasets


def main(dry_run: bool) -> None:
    cfg: Config = DryRunConfig() if dry_run else Config()
    set_seed(cfg.seed)

    print(f"Loading tokenizer + model ({cfg.base_model})...")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_bf16 = cfg.bf16 and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"Wrapping with LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # PEFT + gradient_checkpointing requires this — without it, the inputs to
    # checkpointed sub-modules don't have requires_grad=True (because the base
    # model is frozen), and backward fails with "element 0 ... does not require
    # grad". See notes/gotchas.md "Gradient checkpointing + PEFT".
    if getattr(cfg, "gradient_checkpointing", False):
        model.enable_input_require_grads()

    print("Loading datasets...")
    train_ds, val_ds = load_datasets(cfg, tok)

    if dry_run:
        # tiny subset for fast smoke test
        train_ds = train_ds.select(range(min(20, len(train_ds))))
        val_ds = val_ds.select(range(min(5, len(val_ds))))

    if not dry_run:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=getattr(cfg, "gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        bf16=use_bf16,
        max_seq_length=cfg.max_length,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_only_model=getattr(cfg, "save_only_model", False),
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_repo_id if cfg.push_to_hub else None,
        report_to=["wandb"] if not dry_run else "none",
        run_name=cfg.wandb_run_name,
        dataset_text_field="text",
        seed=cfg.seed,
        # max_steps override via SVG_MAX_STEPS_OVERRIDE env var lets a single
        # Colab session cap training at e.g. 30k steps without editing config.
        max_steps=(
            5 if dry_run
            else int(os.environ["SVG_MAX_STEPS_OVERRIDE"]) if os.environ.get("SVG_MAX_STEPS_OVERRIDE")
            else -1
        ),
        # eval disabled — val set is too large to eval at training cadence
        # without dominating runtime (~1-2h per eval × 425 evals at eval_steps=200
        # = days of eval alone). Real evaluation is qualitative on the test set
        # post-training (see notes/decisions.md "Validation framework").
        eval_strategy="no",
    )

    # Pass response template as token IDs (not as a string) and drop the
    # leading "\n" — see notes/gotchas.md "Response template tokenization
    # mismatch": some captions end with characters that merge with the
    # following "\n" into a single token under BPE, so the standalone
    # tokenization of "\nSVG:\n" doesn't match the in-context tokens. Using
    # IDs starting from "SVG:" (no leading newline) avoids this.
    response_template_ids = tok(
        cfg.response_template.lstrip("\n"),
        add_special_tokens=False,
    ).input_ids
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tok,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    print(f"Starting {'DRY-RUN' if dry_run else 'training'}...")
    # resume_from_checkpoint=True tells Trainer to auto-detect the latest
    # checkpoint in output_dir if one exists, otherwise start fresh. Safe
    # to always pass — does the right thing in both cases.
    trainer.train(resume_from_checkpoint=True if cfg.output_dir.exists()
                  and any(cfg.output_dir.glob("checkpoint-*"))
                  else None)

    print("Saving final adapter...")
    final_dir = cfg.output_dir / "final"
    trainer.save_model(str(final_dir))

    if cfg.push_to_hub:
        print(f"Pushing to {cfg.hub_repo_id}...")
        trainer.push_to_hub()

    print("Done.")

    # Auto-terminate on RunPod (CLAUDE.md mandates this for any real run).
    if not dry_run and os.environ.get("RUNPOD_POD_ID"):
        pod_id = os.environ["RUNPOD_POD_ID"]
        print(f"Auto-terminating RunPod pod {pod_id}...")
        subprocess.run(["runpodctl", "stop", "pod", pod_id], check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU smoke test with tiny shapes (no WandB, no Hub push)")
    args = ap.parse_args()

    dry_run = args.dry_run or os.environ.get("DRY_RUN") == "1"
    main(dry_run)
