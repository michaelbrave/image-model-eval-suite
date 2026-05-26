# Core 75 — Quick Smoke Test

`core-75` is a lightweight suite for fast prompt-style compatibility checks.

It uses 25 concepts and renders each concept with 3 prompt styles:

- `natural_descriptive`
- `cinematic_structured`
- `sdxl_tagged`

75 total cases make this ideal as a quick smoke test before running the full `core-100` or `publish-1000` evaluations.

The suite is designed to answer two questions:

1. Which prompt style works best for this model?
2. Which domains, subjects, lighting conditions, aspect ratios, and difficulty probes are strengths or weaknesses?

No model rankings are stored in this suite. Rankings belong in generated run outputs and model cards.
