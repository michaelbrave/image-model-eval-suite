# Exporting From prompt-library

This eval repository is portable. It should not require the prompt-library SQLite database at runtime.

Use the database only as a source for curated prompt concepts and style templates, then commit the exported YAML files into `suites/`.

## Recommended Flow

1. Build or refresh prompt-library eval candidates.
2. Review selected prompt concepts manually.
3. Export concepts into `suites/<suite-id>/concepts.yaml`.
4. Export or hand-author prompt styles into `suites/<suite-id>/styles.yaml`.
5. Validate the suite with `image-model-eval validate-suite`.

## Export Principles

- Keep concepts stable once a suite is published.
- Freeze seeds in suite policy so repeated runs are comparable.
- Use explicit labels for domains, subjects, probes, difficulty, and aspect ratio.
- Prefer fewer, clearer concepts over many near-duplicates.
- Do not export NSFW concepts into the core suite unless the suite is explicitly marked for that purpose.

## Prompt Style Policy

Prompt style is a measured variable. The same concept should be rendered through each style so the report can say which style works best for a model.

For `core-75`, the intended structure is:

```text
25 concepts x 3 styles = 75 generated images
```

For `core-100`, use:

```text
25 concepts x 4 styles = 100 generated images
```
