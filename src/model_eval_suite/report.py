from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import statistics


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    cases = run.get("cases", [])
    scores_by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    style_scores: dict[str, list[float]] = defaultdict(list)
    domain_scores: dict[str, list[float]] = defaultdict(list)
    probe_scores: dict[str, list[float]] = defaultdict(list)

    for case in cases:
        scores = case.get("scores", [])
        usable = [s.get("normalized_score") for s in scores if isinstance(s.get("normalized_score"), (int, float))]
        if not usable:
            continue
        value = float(statistics.mean(usable))
        style_scores[case["style_id"]].append(value)
        domain_scores[case["domain"]].append(value)
        for probe in case.get("probes", []):
            probe_scores[probe].append(value)

    return {
        "case_count": len(cases),
        "scored_case_count": sum(1 for case in cases if case.get("scores")),
        "style_scores": {key: _mean(value) for key, value in sorted(style_scores.items())},
        "domain_scores": {key: _mean(value) for key, value in sorted(domain_scores.items())},
        "probe_scores": {key: _mean(value) for key, value in sorted(probe_scores.items())},
    }


def build_markdown_card(run: dict[str, Any]) -> str:
    summary = summarize_run(run)
    if summary["scored_case_count"] == 0:
        raise ValueError("Cannot build model card rankings without real score data")

    model = run.get("model", {})
    lines = [
        f"# {model.get('model_id', 'Unknown Model')}",
        "",
        f"Suite: `{run.get('suite_id')}`",
        f"Checkpoint: `{model.get('checkpoint', '')}`",
        f"Cases: {summary['case_count']}",
        f"Scored cases: {summary['scored_case_count']}",
        "",
        "## Prompt Style Scores",
        "",
    ]
    for key, value in sorted(summary["style_scores"].items(), key=lambda item: (item[1] is None, -(item[1] or 0))):
        lines.append(f"- `{key}`: {value:.4f}" if value is not None else f"- `{key}`: unscored")

    lines.extend(["", "## Domain Scores", ""])
    for key, value in sorted(summary["domain_scores"].items(), key=lambda item: (item[1] is None, -(item[1] or 0))):
        lines.append(f"- `{key}`: {value:.4f}" if value is not None else f"- `{key}`: unscored")

    lines.extend(["", "## Probe Scores", ""])
    for key, value in sorted(summary["probe_scores"].items(), key=lambda item: (item[1] is None, -(item[1] or 0))):
        lines.append(f"- `{key}`: {value:.4f}" if value is not None else f"- `{key}`: unscored")

    lines.extend([
        "",
        "## Notes",
        "",
        "This card was generated from recorded scorer outputs. Interpret scorer-specific results according to the scorer metadata in the run JSON.",
    ])
    return "\n".join(lines) + "\n"


def write_card(run_path: Path, out_path: Path) -> None:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown_card(run), encoding="utf-8")
