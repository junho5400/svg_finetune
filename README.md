# svg_finetune

Text-to-SVG glyph generation via LoRA fine-tuning of Qwen2.5-Coder-7B.

Given a style description and a target character, the model produces a single
SVG glyph rendered in the requested style. Trained on ~675k glyphs extracted
from Google Fonts with metadata + LLM-generated style captions.

## Layout

- `train.py` — SFTTrainer entrypoint with auto-resume and Hub checkpointing
- `eval.py` — generation eval against a fixed 30-prompt set
- `score_glyphs.py` — CLIP zero-shot letter-accuracy scorer
- `eval_val_loss.py` — retroactive validation loss across historical checkpoints
- `data.py` — dataset loading + length filtering
- `config.py` — training hyperparameters
- `pipeline/` — one-time data curation scripts (font extraction, caption
  generation, train/val/test split)
- `colab/` — notebooks for dry-run validation, mid-training eval, and Colab
  resume training
- `prep_data_hub.py` — uploads training parquets to a HuggingFace dataset repo

## Environment

```bash
conda create -n svg_finetune python=3.12
conda activate svg_finetune
pip install -r requirements.txt
```

## Usage

```bash
# Smoke test on CPU with tiny shapes
python train.py --dry-run

# Full training run (GPU)
python train.py

# Generate eval samples from a trained adapter
python eval.py --adapter junho5400/svg-finetune-qwen-7b-lora-resume \
               --output-dir outputs/eval/mideval_68000

# Score CLIP letter accuracy on an eval folder
python score_glyphs.py outputs/eval/mideval_68000

# Retroactive val loss across historical checkpoints on HF Hub
python eval_val_loss.py
```

## Configuration

Training hyperparameters live in `config.py`. Notebook cells can override
selected values via environment variables (see env-reading block in `train.py`).
