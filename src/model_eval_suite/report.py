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


def _ranked_means(groups: dict[str, list[float]]) -> dict[str, float | None]:
    return {key: _mean(value) for key, value in sorted(groups.items())}


def _case_kind_value(case: dict[str, Any], score_kind: str) -> float | None:
    values: list[float] = []
    for score in case.get("scores", []):
        if score.get("score_kind") != score_kind:
            continue
        value = score.get("normalized_score")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return _mean(values)


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    cases = run.get("cases", [])
    style_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    domain_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    probe_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    scorer_scores: dict[str, list[float]] = defaultdict(list)
    score_kind_scores: dict[str, list[float]] = defaultdict(list)
    score_counts: dict[str, int] = defaultdict(int)

    for case in cases:
        scores = case.get("scores", [])
        score_kinds: set[str] = set()
        for score in scores:
            value = score.get("normalized_score")
            if not isinstance(value, (int, float)):
                continue
            score_kind = str(score.get("score_kind", "unknown"))
            scorer_name = str(score.get("scorer_name", "unknown"))
            numeric = float(value)
            score_counts[score_kind] += 1
            score_kind_scores[score_kind].append(numeric)
            scorer_scores[scorer_name].append(numeric)
            score_kinds.add(score_kind)

        for score_kind in score_kinds:
            value = _case_kind_value(case, score_kind)
            if value is None:
                continue
            style_scores[score_kind][case["style_id"]].append(value)
            domain_scores[score_kind][case["domain"]].append(value)
            for probe in case.get("probes", []):
                probe_scores[score_kind][probe].append(value)

    return {
        "case_count": len(cases),
        "scored_case_count": sum(1 for case in cases if case.get("scores")),
        "score_counts": dict(sorted(score_counts.items())),
        "score_kind_scores": _ranked_means(score_kind_scores),
        "scorer_scores": _ranked_means(scorer_scores),
        "style_scores": {kind: _ranked_means(values) for kind, values in sorted(style_scores.items())},
        "domain_scores": {kind: _ranked_means(values) for kind, values in sorted(domain_scores.items())},
        "probe_scores": {kind: _ranked_means(values) for kind, values in sorted(probe_scores.items())},
    }


def _append_score_table(lines: list[str], scores: dict[str, float | None], limit: int | None = None) -> None:
    rows = sorted(scores.items(), key=lambda item: (item[1] is None, -(item[1] or 0)))
    if limit is not None:
        rows = rows[:limit]
    for key, value in rows:
        lines.append(f"- `{key}`: {value:.4f}" if value is not None else f"- `{key}`: unscored")


def _append_grouped_scores(lines: list[str], grouped: dict[str, dict[str, float | None]], limit: int | None = None) -> None:
    for kind, scores in sorted(grouped.items()):
        lines.extend(["", f"### {kind}", ""])
        _append_score_table(lines, scores, limit=limit)


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
        "## Score Counts",
        "",
    ]
    for key, value in summary["score_counts"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Score Families", ""])
    _append_score_table(lines, summary["score_kind_scores"])

    lines.extend(["", "## Scorer Scores", ""])
    _append_score_table(lines, summary["scorer_scores"])

    lines.extend(["", "## Prompt Style Scores"])
    _append_grouped_scores(lines, summary["style_scores"])

    lines.extend(["", "## Domain Scores"])
    _append_grouped_scores(lines, summary["domain_scores"])

    lines.extend(["", "## Top Probe Scores"])
    _append_grouped_scores(lines, summary["probe_scores"], limit=20)

    lines.extend([
        "",
        "## Notes",
        "",
        "This card was generated from recorded scorer outputs. Score families are kept separate because aesthetic, prompt reward, classifier, and technical image-stat scores are not interchangeable.",
    ])
    return "\n".join(lines) + "\n"


def write_card(run_path: Path, out_path: Path) -> None:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown_card(run), encoding="utf-8")
