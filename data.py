"""
Load train/val/test parquet → HF Dataset.

Per training pair, the formatted text is:
    {instruction_prefix}{caption}{response_template}{svg}

i.e. "Generate an SVG glyph for: {caption}\\nSVG:\\n{svg}".

SFTTrainer's response_template (`\\nSVG:\\n`) handles caption-loss
masking automatically inside the data collator. This format is what
makes a baseline LLM actually emit SVG: without the explicit
"Generate ... SVG:" framing, the model continues the prompt as prose.

Examples whose tokenized length exceeds max_length are *skipped*, not
truncated — truncation would teach the model to produce SVGs that cut
off mid-path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset


def format_text(caption: str, svg: str, cfg) -> str:
    return f"{cfg.instruction_prefix}{caption}{cfg.response_template}{svg}"


def format_prompt(caption: str, cfg) -> str:
    """Inference-time prompt (no SVG yet — model generates it)."""
    return f"{cfg.instruction_prefix}{caption}{cfg.response_template}"


def load_split(parquet_path: Path, cfg) -> Dataset:
    df = pd.read_parquet(parquet_path)
    # Vectorized string concat — ~100x faster than df.apply on 770k rows.
    df["text"] = cfg.instruction_prefix + df["caption"] + cfg.response_template + df["svg"]
    return Dataset.from_pandas(df[["text"]], preserve_index=False)


def filter_by_length(ds: Dataset, tokenizer, max_length: int,
                     num_proc: int = 4) -> Dataset:
    """Drop examples whose tokenized length exceeds max_length."""
    def is_short_enough(ex):
        ids = tokenizer(ex["text"], add_special_tokens=False).input_ids
        return len(ids) <= max_length
    return ds.filter(is_short_enough, num_proc=num_proc)


def load_datasets(cfg, tokenizer) -> tuple[Dataset, Dataset]:
    """Load train + val splits, filter by max_length."""
    train = load_split(cfg.train_path, cfg)
    val = load_split(cfg.val_path, cfg)
    print(f"  loaded     : train={len(train):,}  val={len(val):,}")

    print(f"  filtering by max_length={cfg.max_length}...")
    train = filter_by_length(train, tokenizer, cfg.max_length)
    val = filter_by_length(val, tokenizer, cfg.max_length)
    print(f"  after filter: train={len(train):,}  val={len(val):,}")

    return train, val
