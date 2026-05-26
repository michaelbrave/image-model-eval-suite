# image-model-eval-suite

Portable evaluation suites and tooling for characterizing image-generation models.

This repository intentionally stores evaluation inputs and outputs as text files, not a runtime database. Prompt concepts can be exported from `prompt-library`, but each suite is self-contained once exported.

## Goals

- Identify which prompt styles work best for a model.
- Measure broad strengths and weaknesses across subject, style, lighting, aspect ratio, and difficulty probes.
- Generate reproducible image batches through ComfyUI.
- Score generated images with real scorer outputs only.
- Produce model cards from recorded evaluation data, without placeholder rankings.

## Default Suite Shape

The default suite is `core-75`:

- 25 prompt concepts
- 3 prompt styles per concept
- 75 generated images total

This gives enough signal to compare prompt style compatibility while keeping runtime manageable. A larger `core-100` variant can be created by adding a fourth prompt style to each concept.

## Repository Layout

```text
suites/core-75/          Portable suite definition
src/model_eval_suite/   CLI, runner, scorer, and report code
schemas/                JSON schemas for outputs and model cards
docs/                   Workflow notes and prompt-library export guidance
outputs/                Local generated outputs, ignored by Git
model_cards/            Real model cards generated from eval data
```

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Optional scorer extras are installed separately because they can be large and hardware-specific.

## Commands

Validate a suite:

```bash
image-model-eval validate-suite suites/core-75/suite.yaml
```

Render the planned cases without generating images:

```bash
image-model-eval render-plan suites/core-75/suite.yaml --out outputs/core-75-plan.jsonl
```

Generate images with ComfyUI:

```bash
image-model-eval run-comfy suites/core-75/suite.yaml \
  --model-id my-model \
  --checkpoint my_model.safetensors \
  --comfy-url http://127.0.0.1:8188 \
  --out outputs/my-model/core-75
```

Run the next untested model found in Stability Matrix's model directory:

```bash
scripts/run_models.sh --resolution-scale 0.5
```

Run every untested model sequentially:

```bash
scripts/run_models.sh --overnight --resolution-scale 0.5
```

For a quick smoke test:

```bash
scripts/run_models.sh --case-limit 3 --resolution-scale 0.5
```

The batch runner treats `outputs/<model-id>/<suite-id>/run.json` with all cases completed as already tested. Without `--overnight`, it runs one untested model and exits.

Score generated images:

```bash
image-model-eval score-run outputs/my-model/core-75/run.json \
  --scorer brightness-contrast \
  --scorer improved-aesthetic-predictor
```

Build a model card from real run data:

```bash
image-model-eval build-card outputs/my-model/core-75/run.json \
  --out model_cards/my-model.md
```

`build-card` refuses to produce ranked claims unless score data exists.

## Data Policy

Do not commit invented model rankings, fake scores, or example cards that look like real evaluations. Keep templates and schemas in Git; commit real model cards only after actual images have been generated and scored.
