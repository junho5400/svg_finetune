# Gotchas

Technical issues hit during the build, with the resolution. Useful for the README's
"lessons learned" section, and for anyone reproducing the work.

---

## TTF coordinate system is y-up; SVG is y-down

Without a flip, every extracted glyph rendered upside-down. Fix: include a y-axis flip
in the path's transform — `scale(s, -s)` along with translating the origin to the
baseline.

## Initial scale (1000/em) clipped descenders

First version used `scale = VIEWBOX / em` — looked fine for the letter 'A', but extracting
'g' showed the descender getting cut off at the bottom of the viewBox. The problem: glyphs
extend below the baseline (descenders) and above the cap line (accents); using `em` as the
denominator only accounts for the em-square, not the full typographic range.

Fix: use `scale = VIEWBOX / (ascender - descender)`. That's the full range, so descenders
and ascenders both fit inside the 1000×1000 box.

## cairosvg pip install doesn't bundle libcairo

`pip install cairosvg` succeeds but `import cairosvg` fails on macOS with
`OSError: no library called "cairo-2" was found`. The Python wrapper requires the system
`libcairo` library, which Homebrew doesn't auto-install.

Fix in conda: `conda install -n <env> -c conda-forge cairo`. Conda packages the binary
inside the env, no system change needed.

## Claude CLI `--bare` flag bypasses OAuth

When testing the CLI, used `--bare` for "minimal mode". Got `"Not logged in · Please run /login"`
even though I'd just logged in via `claude` interactively. Help text explained: with `--bare`,
*"Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via --settings (OAuth and
keychain are never read)"*.

Fix: don't use `--bare` if you want the Max-plan OAuth login to apply. Drop the flag.

## `--json-schema` output goes in `structured_output`, not `result`

Spent a debug round on `JSONDecodeError: Expecting value` because I parsed
`response["result"]` expecting JSON. With `--json-schema`, the validated structured object
lives in `response["structured_output"]`; the `result` field gets a free-form
confirmation string from the model ("Done. I've returned the descriptions in the required
structured format.").

Fix: parse `structured_output`, fall back error message to `result` if missing.

## Validation must match schema (caught me twice)

Updated the JSON schema to require 4 descriptions per font, ran the script, got
"0 fonts captioned" with no error. Cause: validation check `if len(descs) == 3` was unchanged.
Then changed back to 3 descriptions later, same bug.

Fix: keep the schema, the prompt, and the validation in sync.

## PIL "execution context too long" on certain TTFs

Background augmentation crashed at batch 39/92 with
`OSError: execution context too long` from PIL's `getbbox()` while rendering text in some
specific font. Some TTFs have hinting tables that PIL's text engine can't process.

Fix: wrap `render_sample` in a try/except that catches *any* exception, log the failed
font, and skip it. Cache is incremental so the run resumes from where it died — only the
broken font is lost (and we have ~1700+ others).

## Variable fonts work fine with fontTools

A worry going in was whether `fonttools` would handle Google Fonts' variable fonts
(`Roboto[wdth,wght].ttf`). It does — the default instance is what `getGlyphSet()` returns,
no special handling needed. Same for Pillow's `ImageFont.truetype`.

## Caption-as-marketing creep

LLM-augmented captions naturally drift toward marketing language ("authoritative and
cultured with dignified presence", "carries institutional weight", "pairs beautifully
with..."). That doesn't match real user input.

Fix in prompt: explicitly forbid phrases like "perfect for", "ideal for", "pairs beautifully";
ban hyperbolic adjectives ("stunning", "remarkable"); allow direct emotion words
("warm", "calm", "friendly") but disallow evocative wrappers around them.

## Claude vision unreliable on small images without category anchor

Test run with 640×160 images had Claude calling 42dot Sans "italic" (it isn't), Abel a
"serif" (it's sans). Bigger images helped (1024×256), but the bigger fix was
**passing the font's category as ground-truth in the prompt**. With `category=SANS_SERIF`
declared, Claude focused on visual nuances within sans-serif rather than miscategorizing.

## Gradient checkpointing + PEFT/LoRA — backward fails on frozen base model

After enabling gradient_checkpointing (to fit 7B + LoRA in 24GB), the next training
attempt crashed at backward:
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**Why**: `torch.utils.checkpoint` expects at least one input to require gradients. With
PEFT, only the LoRA adapter weights require grad — the base model is frozen. So the
inputs to checkpointed sub-modules don't have `requires_grad=True` and the backward
graph doesn't reach them.

**Fix** (two parts, both needed):
1. After `get_peft_model(...)`, call `model.enable_input_require_grads()`. This forces
   gradients to flow through the input embeddings even though the embedding weights
   are frozen.
2. Set `gradient_checkpointing_kwargs={"use_reentrant": False}` in SFTConfig — the
   non-reentrant checkpoint implementation handles this case more cleanly.

This combo is the standard PEFT-with-gradient-checkpointing recipe. Without it, you
hit the error above on the first backward pass.

## torch 2.6+ weights_only default breaks RNG state resume

When resuming from a checkpoint, training crashed at `_load_rng_state`:
```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct
was not an allowed global by default.
```

**Why**: torch 2.6+ changed `torch.load` to default `weights_only=True` for security.
This rejects any non-tensor pickled objects (numpy arrays, etc.). Our saved
`rng_state.pth` file contains numpy RNG state objects. With strict mode, loading fails.

**Fix**: monkey-patch `torch.load` at the top of train.py to force
`weights_only=False`. Since we only load our own trusted checkpoints, the security
risk doesn't apply to us:
```python
_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe
```
Place it BEFORE importing transformers/peft/trl so their internal `torch.load`
calls inherit the patched version.

The "proper" alternative is `torch.serialization.add_safe_globals(...)` to allowlist
specific numpy classes, but that requires enumerating every numpy type that might
appear in any checkpoint — fragile across torch versions.

## Disk full during checkpoint save (volume too small for full checkpoints)

After ~1700 steps of training, checkpoint save crashed with:
```
RuntimeError: [enforce fail at inline_container.cc:668] . unexpected pos 268800128 vs 268800016
```
This is the torch.save partial-write signature when the filesystem rejects more bytes
mid-save (volume hard limit hit).

**Why**: full checkpoint = LoRA adapter (~320 MB) + AdamW optimizer state (~640 MB) +
scheduler/RNG/trainer state (~50 MB) ≈ **1 GB each**. With `save_total_limit=2`, peak
disk during a save = 3 checkpoints × 1 GB = 3 GB, which exceeded the volume's free
space (~3 GB free at start, dropping below 1 GB during repeated checkpoints).

**Fix that we used**: resize the network volume from 20 GB to 30 GB in RunPod
(~$0.07/GB/month → ~$2.10/month for 30 GB, trivial). With the bigger volume there's
plenty of headroom for full-state checkpoints (`save_only_model=False`,
`save_total_limit=2`) so resume picks up cleanly with Adam momentum intact.

**Lesson learned**: don't shrink checkpoints to work around disk pressure if the volume
can be resized cheaply. Saving optimizer state matters for resume robustness; shrinking
checkpoints buys ~$2/month and trades real training quality for it.

## OOM at cross_entropy_loss with long seq_len (Qwen ~150k vocab)

After fixing the gradient_checkpointing+PEFT issue, training got 6 steps in then
OOM'd at the cross-entropy loss computation:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.81 GiB. ...
File ".../loss/loss_utils.py", line 26, in fixed_cross_entropy
    loss = nn.functional.cross_entropy(source, target, ...)
```

**Why**: Qwen2.5-Coder has vocab ~150k. At batch=2 × seq_len=2048 × vocab=150k × 2
bytes (bf16) = ~1.25 GB just to materialize the logits tensor. Plus shift_logits,
gradients of the cross-entropy, intermediate buffers — all proportional to
seq_len × vocab. The OOM happened on a step where the seq_len was at the max;
shorter sequences earlier in training fit fine.

**Fix**: drop `max_length` from 2048 to 1024. Halves logits memory directly.
Per task #7's token-length measurement, 1024 still fits ~88% of glyphs (vs. 95%
at 2048). The 12% lost are mostly pathological decorative fonts (pixel-art,
layered shadow effects); the bulk of style variety is preserved.

Alternatives if 1024 is too restrictive (none used):
- `liger-kernel` fused cross-entropy — doesn't materialize full logits, but
  requires extra dep + version compatibility check.
- `batch=1` — keeps max_length=2048 but halves throughput.
- A100-class GPU (40-80 GB VRAM) — out of project budget.

## Gradient checkpointing + PEFT/LoRA — backward fails on frozen base model

First real training launch crashed at step 0 with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 136.00 MiB.
GPU 0 has a total capacity of 23.54 GiB of which 11.75 MiB is free.
```

Just barely over the 24GB ceiling. Memory budget at the limit:
- 7B model in bf16: ~14 GB
- LoRA adapter (80M params) + AdamW optimizer state + gradients: ~1 GB
- Activations at batch=4 × max_length=2048 × hidden=3584 × 32 layers: 8-10 GB
- Misc CUDA buffers / fragmentation: ~1 GB
- **Total: just over 24 GB → OOM**

**Fix**: two changes in config.py:
1. `gradient_checkpointing=True` — recomputes activations in backward pass instead
   of storing them. ~30% slower, but ~50% less activation memory.
2. `per_device_train_batch_size: 4 → 2`, `gradient_accumulation_steps: 4 → 8`. Keeps
   effective batch 16 but halves activation memory per step.

Combined, the run fits comfortably with margin. Both knobs are conservative — could
go back to batch=4 if quality demands more parallelism, but checkpointing should
stay on for any 7B training on a 24GB GPU.

## Response template tokenization mismatch (DataCollatorForCompletionOnlyLM)

Dry run on RunPod showed warnings for ~40% of examples:

```
UserWarning: Could not find response key `\nSVG:\n` in the following instance: ...
This instance will be ignored in loss calculation.
```

Loss values confirmed: roughly half of steps had `loss=0.0` and `grad_norm=0.0`,
and `eval_loss=NaN` (because *all* eval examples got ignored).

**Why**: BPE tokenizers (Qwen, Llama, etc.) tokenize differently based on context.
The collator tokenizes `"\nSVG:\n"` as a standalone string and searches for those
IDs in the tokenized full text. When a caption ends with `.`, `,`, `;`, `'`, etc.,
the leading `\n` of the template gets merged with the punctuation into a single
token → standalone IDs don't match. Result: example silently dropped from loss.

**Fix**: pass `response_template` as **token IDs** (not a string), AND strip the
leading newline:

```python
response_template_ids = tok(
    cfg.response_template.lstrip("\n"),
    add_special_tokens=False,
).input_ids
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tok,
)
```

Without this, real training would silently degrade ~40% of gradient signal — not a
failure mode you'd notice without inspecting the dry-run warnings carefully.

## RunPod dep hell — fixed by pin everything + Colab-validate first

Initial RunPod session burned ~30 min and several pod minutes debugging dependency
mismatches:

1. `requirements.txt` had loose pins (`torch>=2.1`, `transformers>=4.42`, etc.). pip
   skips already-installed packages that satisfy the constraint, so we got
   pod-template-shipped torch 2.4 + latest transformers, which was incompatible
   (transformers needed torch 2.5+ for `torch.library.custom_op` schema inference).
2. `pip install --upgrade torch` pulled torch 2.11, but left torchvision 0.19 (built
   for torch 2.4) installed — torchvision then crashed transformers' image_utils
   import chain.
3. trl 0.13+ removed `DataCollatorForCompletionOnlyLM` from top-level export.
4. peft 0.19 referenced `torch.distributed.tensor.DTensor` at a path that didn't exist
   in torch 2.11.
5. HF_HOME defaulted to container disk (~20 GB) which filled up downloading the 7B
   model.

**Fix**: validate full pipeline on Colab Pro first using
`colab/dryrun_validate.ipynb` (free, A100 or T4), capture working pip freeze, lock in
exact pins in `requirements.txt`. Document `HF_HOME=/workspace/.hf_cache` so HF cache
goes to the network volume not the container disk.

**Validated combo (2026-05-08)**:
- torch==2.10.0 (cu128 wheel)
- transformers==4.46.3
- peft==0.13.2
- trl==0.12.2
- accelerate==1.13.0
- datasets==3.6.0
- wandb==0.26.1

If any of these need bumping in the future, re-run `colab/dryrun_validate.ipynb` with
new pin ranges before touching RunPod.

## 7B few-shot prompting: model copies, doesn't generalize

Tested 0/1/3-shot prompting on 5 fresh test prompts (B sans, M serif, p handwriting,
X monospace, g display) with in-context examples = (Roboto O, Lora A, Pacifico h).

What we found:
- 1-shot for letter 'p' (with Roboto 'O' as the example) → model output looks like 'O'.
- 3-shot for letter 'M' (with O/A/h examples) → model output is the Lora 'A'.
- 3-shot for letter 'p' (with O/A/h) → model output is the Pacifico 'h'.
- 3-shot for letter 'X' (no monospace example) → blank.

The model is doing **nearest-neighbor copy** of in-context examples, not letter-shape
generalization. It can produce path-based output that LOOKS like a letter when copying,
but cannot invent letter shapes for unseen (letter, style) combinations.

**Implications:**
- Fine-tuning has to teach letter-shape mapping from scratch, parametrically. We're not
  reinforcing existing in-context priors.
- Memorization risk is real — model might memorize specific (font, letter) shapes rather
  than abstract letter prototypes. Watch held-out test set carefully.
- May need higher LoRA rank (32 or 64) and more epochs (2-3) than the original defaults
  to give parametric memory enough capacity.

`gen_tokens=2048` (max budget) on 1-shot/3-shot is also a tell: model tries to mimic the
in-context example length but can't produce coherent letter-shaped paths, so it runs to
the budget without closing the SVG properly.

## 7B baseline: model uses `<text>` shortcut for letter prompts

When asked "Generate an SVG glyph for: the letter 'r' in Times New Roman style", the
untrained Qwen2.5-Coder-7B sometimes produces:

```svg
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <text x="10" y="50" font-family="Times New Roman" font-size="40">r</text>
</svg>
```

This is the model being clever, not failing — it knows SVG has a `<text>` element with
font-family lookup, so it lets the renderer's font-loading do the work. It's a perfectly
reasonable interpretation of the task; just not what we want.

**Why this is OK for our training:**
- Our training data is 100% `<path>` based (extracted from TTFs via fonttools).
- LoRA training will pull the model toward path output; the `<text>` shortcut becomes a
  format the model learns to avoid because no training example uses it.

**What to watch for post-training:**
- If trained model still emits `<text>` for some prompts, training underfit (too few
  steps, or LoRA rank too low to overwrite this prior).
- 7B baseline numbers as reference: 27/30 valid SVG syntax, ~2/30 letter-shaped (and
  those 2 used the `<text>` shortcut). Anything post-training that produces actual
  path-based letter shapes for a recognizable fraction of prompts is a win.

## Aboreto and quality-tag philosophy

Caught myself wanting to filter "low quality" fonts using Google's `/Quality/Drawing`,
`/Quality/Spacing`, etc. tags. Per-category pass-rate at threshold 70 showed: sans-serifs
~53% pass, but display ~26% and **handwriting only 13%**. The "quality" scores were
measuring *polish/conventionality*, not goodness — handwriting fonts that LOOK
intentionally hand-drawn score low because they have variable spacing.

Lesson: when filtering datasets, check per-category effects before applying a global
threshold.
