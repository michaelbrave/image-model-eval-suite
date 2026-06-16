# Scorers

Scorers must return real measurements from actual images. Do not add placeholder model valuations.

## Included

### brightness-contrast

Objective image-statistics scorer using Pillow. It records brightness and contrast metadata and returns a simple normalized technical score. This is not an aesthetic judgment.

### dinov3-aesthetic-v1

Current generalist aesthetic scorer. It uses frozen `timm:vit_base_patch16_dinov3.lvd1689m` embeddings plus the locally trained V1 preference head at `aesthetic-scorer-trainer/checkpoints/v1_dinov3_vitb.pt`. Use raw-score differences for paired model comparisons; the normalized score is `sigmoid(raw_score)` and is not a calibrated human rating.

### improved-aesthetic-predictor

Legacy optional CLIP + MLP aesthetic predictor adapter. It requires real local weights via `--weight-path` and fails if weights or dependencies are missing.

### image-reward

Optional prompt-aware scorer. It requires `ImageReward` to be installed and requires the rendered prompt.

## Interpreting Scores

Do not blindly average unrelated scorer families. Aesthetic, prompt-reward, style-specific, and technical scores answer different questions.

The current card builder aggregates available normalized scores only. Scorers without normalization are preserved in run JSON until a calibration policy is added.
