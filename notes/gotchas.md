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
