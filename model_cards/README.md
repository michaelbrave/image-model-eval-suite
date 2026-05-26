# Model Cards

Model cards in this directory are generated from actual evaluation runs using `image-model-eval build-card`. Each card documents a model's performance across prompt styles, domains, subjects, lighting conditions, aspect ratios, and difficulty probes.

## Scoring and Ranking

Cards include scores from objective scorers applied to generated images:

- **brightness-contrast** — measures basic image quality (luminance distribution, contrast range)
- **improved-aesthetic-predictor** — predicts aesthetic quality based on human preference data
- **image-reward** (optional, prompt-aware) — scores alignment between image and prompt

Models are ranked by aggregate scores within each grouping (e.g., best prompt style, strongest subject domain). Rankings are computed from recorded score data only — no subjective or placeholder rankings appear in cards.
