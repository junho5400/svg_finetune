# Decisions

A log of non-obvious decisions made during the project, with the reasoning. Useful for the
README, for explaining the work to others, and for picking the project back up after a break.
Organized by topic, not strictly chronological.

---

## 1. Project scope: novel font design (not generic SVG)

**Decision**: Narrow the project from "text → any SVG" to "text → single SVG glyph,
conditioned on font style description". Two features originally in scope; only Feature 1
(style → glyph) ships in the current budget.

**Reasoning**:
- A 0.5B model with LoRA can't learn "all of SVG" well — it has to over-generalize.
- Narrowing to one consistent style (font glyphs) shrinks the learning surface and gives
  every training example the same canonical structure.
- Font-style demo is a sharper pitch than "generic SVG generator" and easier for a viewer
  to evaluate.
- Feature 2 (one-shot full-alphabet completion from a single example) was scoped out as
  too hard for 0.5B + LoRA in the original $5 budget.

---

## 2. Custom-curate from Google Fonts (not HuggingFace datasets)

**Decision**: Build the training set ourselves from `github.com/google/fonts` instead of
using any of the public SVG datasets on HuggingFace.

**Reasoning**:
- `starvector/svg-icons-simple` has only `(Filename int, Svg)` — no captions at all.
- `starvector/text2svg-stack`, `JackertheHacker/svg-stack-qwen-captioned-subset`, and the
  other VLM-captioned alternatives have caption text like *"a minimalist light gray icon
  on a white background depicts a location pin with a circular center..."* — VLM image
  descriptions, not user-prompt vocabulary. Training on those would teach the model to
  expect verbose paragraph inputs that don't match how a real user would type a prompt.
- Google Fonts gives us ~2000 OFL-licensed fonts, each with metadata (category + tag scores)
  we control entirely. We can choose the caption format ourselves.

---

## 3. Caption design: include font names

**Decision**: Each glyph gets multiple caption variants — descriptive-only, name-anchored,
and combined — so the trained model handles both kinds of prompts.

**Reasoning**:
- Initial instinct was "users won't type 'Roboto' so don't include font names". That was
  too binary.
- Reality: famous fonts (Roboto, Comic Sans, Helvetica, Pacifico) ARE vocabulary even for
  non-typographers. *"Like Helvetica but warmer"* is a real prompt people type.
- Solution: train on (descriptive caption) + (name-anchored caption) + (combined). At
  inference, both work; neither is forced.
- For famous fonts not in our corpus (Helvetica, Times) — task #5 LLM augmentation can
  bridge by writing "Helvetica-like" tags for similar in-corpus fonts.

---

## 4. Don't filter by Google's `/Quality/` scores

**Decision**: Skip filtering fonts by Google Fonts' quality tags (Drawing, Spacing,
Wordspace, Concept). Keep all 1859 fonts.

**Reasoning**:
- Initially proposed threshold 70 on min(quality scores). Looked clean on aggregate (40%
  of fonts pass).
- Per-category breakdown showed brutal asymmetry: 53% of sans-serifs passed, but only
  **13% of handwriting fonts** and **26% of display fonts**. The filter would silently
  homogenize the dataset toward polished workhorses.
- Google's "quality" scores conflate "polished/conformant" with "good". A handwriting font
  with variable spacing isn't broken — variable spacing IS the style. A display font with
  rough drawing isn't a defect — rough is the design.
- More variety also reduces memorization risk: with 1859 fonts the model has to learn
  abstract style patterns rather than memorize one shape per font.
- Real broken/corrupt fonts already fail at extraction and get caught there.

**What we DO filter**:
- Fonts with <30 glyph coverage (degenerate, mostly Indic-script with sparse Latin).
- Individual glyphs over 20,000 chars (top ~0.5%, pathological layered/decorative outliers).

---

## 5. Long glyphs stay; `max_length` becomes the budget knob

**Decision**: Don't drop long decorative SVGs at the dataset level. Defer the
length-cap decision to the training config (`max_length`).

**Reasoning**:
- "Drop fonts with long glyphs" was solving the wrong problem. Long SVGs aren't bad data;
  they're just expensive to train on at fixed `max_length`.
- Decorative styles inherently produce long SVGs. Filtering them out would lose a whole
  visual category from the dataset.
- Right framing: keep all glyphs in the data, set `max_length` based on tokenized
  distribution at training time, skip examples above the threshold rather than truncate
  them (truncation teaches the model to produce broken output).

---

## 6. Caption augmentation via vision, not text-only

**Decision**: Render a sample image per font and ask Claude (Haiku, vision-capable) to
write descriptions based on what it actually sees, with the font's category passed as a
ground-truth anchor.

**Reasoning**:
- First attempt: text-only metadata → Claude. Output was just paraphrased tags
  ("humanist sans, calm" → "calm humanist sans"). Generic, fails for the 67 fonts with
  no tags.
- Vision-based: Claude sees the actual font, can describe stroke contrast, x-height,
  terminals, axis tilt — visual properties no tag captures.
- Without category anchor, Claude misread categories on small images (called Abel a
  serif, called ABeeZee bold). Passing `category=SANS_SERIF` as ground truth fixed that
  immediately and let Claude focus on visual nuances within the category.
- 1024×256 image at font size 120px gives Claude enough resolution.

---

## 7. Caption tone: factual not promotional, but allow emotion words

**Decision**: LLM captions describe what the font IS, not where it'd be "perfect to use".
Marketing phrasing banned ("perfect for", "pairs beautifully with", "carries weight").
Direct emotion words allowed ("warm", "friendly", "harsh", "calm").

**Reasoning**:
- First-pass output sounded like font-foundry marketing copy. *"Authoritative and cultured
  with dignified presence; carries institutional weight"* — that's selling, not describing.
- Real user prompts are factual or directly emotional, not evocative. *"warm humanist sans"*
  yes; *"feels remarkably warm with a quietly confident voice"* no.
- The distinction is: emotion as a direct adjective is fine; emotion wrapped in marketing
  fluff isn't.

---

## 8. Use Claude CLI (Max plan), not the API

**Decision**: Run caption augmentation via the local `claude` CLI rather than direct API
calls.

**Reasoning**:
- User has Claude Max subscription; API would cost ~$60+ at $0.034/font × 1859 fonts;
  CLI uses subscription quota (no extra dollar cost).
- CLI supports non-interactive mode (`--print`), JSON schema enforcement (`--json-schema`),
  and the Read tool (so Claude loads images itself).
- Trade-off: slower (~2.5s/font) vs API (~1s/font). On Max plan that's free time, so OK.

---

## 9. Splits: 90/5/5 by FONT (not random)

**Decision**: Hold out entire fonts for val/test, not random rows. Val for training-stability
monitoring + memorization detection; test reserved for final qualitative inspection.

**Reasoning**:
- Random splits let the model memorize specific glyphs in train and trivially "succeed"
  on val (since val fonts also appear in train). Loss drops but generalization isn't real.
- By-font splits force the model to generalize style → shape patterns. Lower val loss
  means "the model produces *a* plausible glyph for an unseen-font's caption" — which is
  exactly what novel generation means.
- 90/5/5 leaves plenty of training data (~93 val fonts is enough for stable loss curves).
- Important framing: there is no unique "correct" SVG for a style description — many fonts
  satisfy "humanist sans". So val loss isn't a "correctness" metric; it's a memorization
  detector + early-stopping signal. Real eval is qualitative on the test set.

---

## 10. Training: SFT first, then maybe DPO

**Decision**: Plan to start with standard supervised fine-tuning (NTP) on the captioned
dataset. DPO only if SFT outputs reveal specific behavioral failures.

**Reasoning**:
- Pure NTP doesn't directly train behavioral goals (syntactic validity, style match,
  diversity, edit fidelity). Preference-based methods like DPO do.
- BUT: SFT is the foundation. DPO from random init doesn't help. Need SFT first regardless.
- SFT might be sufficient: our SVG format is heavily canonicalized (single path, fixed
  viewBox, integer coords), so the model has a much smaller surface than freeform SVG —
  NTP might just work.
- The progressive-render sanitizer at inference already mitigates the "invalid SVG"
  failure mode.
- Sequencing: SFT → look at outputs → if quality lacks → build a reward function (cheapest:
  syntactic validity + letter-recog OCR) → DPO on (preferred, rejected) pairs from that
  reward. Estimated ~$5–10 added cost, fits under expanded $30 budget.

---

## 11. Stage 2 (editing) — stretch goal, not in primary scope

**Decision**: Document and design-for-future, but ship Stage 1 first.

**Reasoning**:
- The big product reason for choosing SVG is editability. A real designer iterates: generate,
  then say *"make the curves rounder"* → surgical edit. That's a different task from
  initial generation.
- Stage 2 needs its own dataset of `(original_svg, edit_instruction, modified_svg)` triples
  (programmatic transforms + Google Fonts weight/style variant pairs + synthetic style edits)
  and its own training run.
- At the original $5 budget: clearly out of scope.
- At the expanded $30 + Colab budget: plausibly in scope as a follow-up after Stage 1 ships.
- Stage 1 design choices that keep Stage 2 enabled: canonical SVG format (diffable), retain
  `font_family` (pair variants later), test fonts untouched.

---

## 12. Tokenization: keep per-digit (status quo), don't add custom number tokens

**Decision**: Use Qwen2.5-Coder's tokenizer as-is. Don't add special tokens for the
0–999 coordinate range. Don't resize embeddings.

**Reasoning** (after looking at the literature):
- We measured: 0% of coordinates are single tokens, mean 3.89 tokens per number, dominantly
  per-digit. Initial reaction: "this is wasteful, add atomic number tokens."
- That instinct is wrong. "Tokenization counts" (arXiv 2402.14903) and follow-up work show
  **per-digit tokenization OUTPERFORMS whole-number tokenization on numerical tasks** —
  it's why LLaMA-3 / Qwen-Coder / similar coder models adopted it deliberately.
- Atomic number tokens (`<n295>`) destroy place-value structure: the model treats `<n295>`
  and `<n296>` as unrelated symbols, killing magnitude generalization. Per-digit keeps that
  structure.
- StarVector (CVPR 2025), the closest published analog (LLM-based SVG generation), uses
  the base code LLM tokenizer unchanged. IconShop uses domain-specific path tokenization
  but drops the pretrained LLM — a fundamentally different architectural commitment.
- The 4× per-glyph token count is the price for keeping the model's numerical
  generalization. Cheaper alternatives exist (atomic tokens) but the empirical evidence
  says they degrade quality.

**How to apply**:
- Training config keeps `max_length=2048` (95% of glyphs fit; matches the cost).
- No `tokenizer.add_tokens(...)` calls. No embedding-layer resize. Standard LoRA on
  standard Qwen2.5-Coder-0.5B.
- If at training time we see numerical-coherence failures (model predicting `295` → `2X5`),
  the answer is more training data / longer training, not changing the tokenizer.

## 13. Eval framework + baseline measurement BEFORE training

**Decision**: Write `eval.py` and run baseline (untuned Qwen2.5-Coder-0.5B) before
running the first training pass. Same fixed prompt set will be reused after every
training run for apples-to-apples comparison.

**Reasoning**:
- Without a baseline number, "training improved the model" is unprovable. We
  don't know what the floor is.
- Without an end-to-end eval pipeline, training could "succeed" (loss drops) but
  we'd discover at the very end that some generation-time issue (prompt format
  mismatch, broken decoding, etc.) makes outputs unusable. Better to catch wiring
  bugs in eval before spending GPU money.
- The fixed-prompt set (12 prompts spanning all 5 categories + name references +
  punctuation) is small enough to inspect by eye but covers the demo's intended
  use cases. Gives us a consistent comparison surface across training experiments.
- Aligns with the user's earlier validation framing: val loss is just a memorization
  detector; real signal comes from looking at outputs on a fixed prompt set.

**How to apply**:
- `python eval.py --baseline` runs the untuned model.
- `python eval.py --adapter <path>` runs after training.
- Outputs go to `outputs/eval/{baseline,adapter_name}/` with raw generations,
  extracted SVGs, and PNG renders side-by-side, plus `_summary.json` of stats.
- Don't change the EVAL_PROMPTS list once we've recorded a baseline — the comparison
  loses its meaning.

## 14. Prompt format: "Generate an SVG glyph for: ...\nSVG:\n" — explicit instruction

**Decision**: Wrap every training pair as
```
Generate an SVG glyph for: {caption}
SVG:
{svg}
```
The `\nSVG:\n` separator is the response template for SFTTrainer's loss masking.

**Reasoning** (from the baseline-eval result):
- First baseline run used the bare format `{caption}\n{svg}` with response_template
  `\n<svg`. Untrained Qwen-Coder-0.5B produced **0/12 SVGs** — it just continued the
  prompt as a typography essay ("the letter 'A' is a classic example of humanist...").
- Inline experiment showed the model knows SVG syntax fine when given any SVG starter
  (`<svg`, `<!--SVG-->`, etc.), but doesn't volunteer it from a natural-language
  caption.
- Adding "Generate an SVG glyph for:" prefix + "SVG:" separator immediately fixes
  this: untrained 0.5B baseline produces SVG syntax (the path data is degenerate
  because the model doesn't know specific letter shapes, but the format is
  there — exactly the part training will fix).
- Bigger model (1.5B / 3B) was discussed as the alternative; not necessary once
  prompt format is fixed.

**How to apply**:
- `data.py` builds training text via `format_text(caption, svg, cfg)` which uses
  `cfg.instruction_prefix` and `cfg.response_template`.
- `eval.py` calls `format_prompt(caption, cfg)` for inference, ending with
  `SVG:\n` — model continues with SVG content.
- `cfg.response_template = "\nSVG:\n"` is what `DataCollatorForCompletionOnlyLM`
  uses to mask loss on caption tokens.
- Don't change this format mid-training — re-baseline if you do.

## 15. OCR-based letter accuracy as quantitative eval metric

**Decision**: Use EasyOCR on rendered samples (after re-rendering at 1024×1024 + tight
crop + center on white) as a quantitative letter-recognition metric for any model
checkpoint. Off-the-shelf, no need to train a custom classifier.

**Reasoning**:
- Pure SFT loss tells us the model is converging on training distribution but not
  whether outputs are recognizable letters at inference.
- Eyeballing 30 rendered PNGs is subjective and slow.
- Off-the-shelf OCR (EasyOCR, Tesseract) recognizes single letters reliably enough
  to give an objective "did the SVG render as the requested letter?" pass/fail.
- Lets us track progress: baseline 7B → step 2500 → step 7500 → … with one number.

**Pre-processing pitfall** (resolved): cairosvg outputs RGBA with transparent
background. PIL's default `convert("RGB")` composites on black, so black-ink
glyphs become all-black squares. Fix: explicitly composite on white via
`Image.paste(rgba_img, mask=rgba_img.split()[3])`.

**Baselines (full eval-set of 30 prompts)**:
- 7B no training: 0/30 exact match (random shapes)
- Step 2500 (3% of training): 2/30 exact, 6/30 letter-shaped (B, r, h→H, q→0, W→IN, E→F)
- Future targets: step ~85k completion → aiming for 20+/30 exact

**How to apply**:
- Run `python score_glyphs.py outputs/eval/<run_dir>` after generating any new eval batch.
- Track the exact-match number across checkpoints; flag plateau if no improvement
  over multiple evals (might trigger DPO follow-up).
- The script saves preprocessed images to `<run_dir>/preprocessed/` for visual
  spot-check of OCR inputs.

## 16. Base model: upgrade 0.5B → Qwen2.5-Coder-7B

**Decision**: Switch from `Qwen/Qwen2.5-Coder-0.5B` to `Qwen/Qwen2.5-Coder-7B`.

**Reasoning**:
- 0.5B baseline (with explicit "Generate an SVG" prompt) showed: 6/12 syntactic
  success, **0/12 letter-shape success**. Model knew SVG syntax but had no idea
  what specific letters look like — produced random shapes for "letter B", "letter h",
  etc.
- That gap is foundational knowledge (what letters look like in vector form), not
  style. LoRA on 0.5B would have to teach BOTH letter shapes AND style mapping
  from scratch — too much for limited adapter capacity.
- 7B has letter-shape priors baked in from pretraining, so LoRA only has to teach
  the style → glyph variation mapping — which is what fine-tuning is actually
  good at.
- 2026 research confirms Qwen2.5-Coder is still the recommended small-code-LLM
  family ("Qwen2.5-Coder family dominates coding benchmarks at every size tier
  in 2026"). Qwen3 is general-purpose, Qwen3.6 starts at 27B (too big to LoRA-train
  on a 4090 cleanly), Qwen3-Coder went MoE-only (80B total).

**How to apply**:
- `cfg.base_model = "Qwen/Qwen2.5-Coder-7B"`.
- Same tokenizer family as 0.5B → no pipeline changes downstream.
- Memory: bf16 weights ~14 GB, LoRA adapter ~50 MB, activations 5-8 GB → ~22 GB
  total. Fits on 24 GB 4090 with bf16 LoRA. QLoRA fallback if OOM.
- Training: ~10-15h per epoch on 4090 at our 770k pair scale → ~$5-8/epoch on
  RunPod spot pricing. ~$15 across 2 epochs leaves ~$15 budget reserve.
- Validate the upgrade with the 7B baseline test in `colab/baseline_7b.py`
  before training — should produce something letter-shaped (not perfect but
  recognizable) for at least some prompts.

## 16. Budget revision: $5 → $30 + Colab Pro + $10 RunPod

**Decision**: Expanded budget, allocated as: Colab Pro for dev/prototype/eval (free against
this project), $10 RunPod for the real Stage 1 training run, ~$20 personal reserve for
Stage 2 / A100 upgrade / rerun budget.

**Reasoning**:
- Original $5 was barely enough for a single 4090 LoRA training run. Couldn't experiment
  with hyperparameters, couldn't afford a rerun if something went wrong.
- $30 buys ~75 hours of 4090 spot time — comfortable for multiple experiments + a real run
  + eval.
- Colab Pro provides free A100 access for prototyping; saves RunPod credit for the actual
  training run where uninterrupted compute matters.
- Auto-terminate on every pod still mandatory regardless of budget.
