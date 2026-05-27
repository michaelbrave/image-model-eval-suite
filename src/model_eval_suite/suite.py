from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import yaml


@dataclass(frozen=True)
class Concept:
    id: str
    domain: str
    subject: str
    aspect_ratio: str
    probes: list[str]
    difficulty: str
    concept: str = ""
    details: str = ""
    lighting: str = ""
    composition: str = ""


@dataclass(frozen=True)
class PromptStyle:
    id: str
    family: str
    description: str
    positive_template: str
    negative_template: str


ASPECT_DIMENSIONS: dict[str, tuple[int, int]] = {
    "square": (1024, 1024),
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "wide": (1344, 768),
    "tall": (768, 1344),
}


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    concept_id: str
    style_id: str
    domain: str
    subject: str
    probes: list[str]
    difficulty: str
    aspect_ratio: str
    image_seed: int
    wildcard_seed: int
    positive_prompt: str
    negative_prompt: str
    variant: int = 0

    @property
    def width(self) -> int:
        dims = ASPECT_DIMENSIONS.get(self.aspect_ratio)
        if dims is None:
            return 1024
        return dims[0]

    @property
    def height(self) -> int:
        dims = ASPECT_DIMENSIONS.get(self.aspect_ratio)
        if dims is None:
            return 1024
        return dims[1]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Suite:
    path: Path
    suite_id: str
    version: int
    name: str
    description: str
    cases: list[EvalCase]
    concepts: list[Concept]
    case_policy: dict[str, Any]
    scoring: dict[str, Any]
    aggregation: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain an object: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _expand_legacy(concepts: list[Concept], styles: list[PromptStyle], policy: dict[str, Any]) -> list[EvalCase]:
    if policy.get("expand") != "all_concepts_x_styles":
        raise ValueError(f"Unsupported expand: {policy.get('expand')}")
    ratios = policy["aspect_ratios"]
    img_start = int(policy.get("image_seed_start", 1))
    wc_start = int(policy.get("wildcard_seed_start", 100000))

    cases: list[EvalCase] = []
    idx = 0
    for c in concepts:
        w, h = ratios[c.aspect_ratio]
        vals = asdict(c)
        for s in styles:
            idx += 1
            cases.append(EvalCase(
                case_id=f"{c.id}__{s.id}",
                concept_id=c.id,
                style_id=s.id,
                domain=c.domain,
                subject=c.subject,
                probes=c.probes,
                difficulty=c.difficulty,
                aspect_ratio=c.aspect_ratio,
                image_seed=img_start + idx,
                wildcard_seed=wc_start + idx,
                positive_prompt=s.positive_template.format(**vals),
                negative_prompt=s.negative_template.format(**vals),
            ))
    return cases


def load_suite(path: str | Path) -> Suite:
    suite_path = Path(path)
    root = suite_path.parent
    data = _read_yaml(suite_path)
    policy = data.get("case_policy", {})

    if "cases_file" in policy:
        cases_data = _load_jsonl(root / policy["cases_file"])
        cases = [EvalCase(**item) for item in cases_data]
        seen: set[str] = set()
        concepts: list[Concept] = []
        for c in cases:
            if c.concept_id not in seen:
                seen.add(c.concept_id)
                concepts.append(Concept(
                    id=c.concept_id,
                    domain=c.domain,
                    subject=c.subject,
                    aspect_ratio=c.aspect_ratio,
                    probes=c.probes,
                    difficulty=c.difficulty,
                ))
    else:
        concepts_data = _read_yaml(root / data["concepts"])["concepts"]
        concepts = [Concept(**item) for item in concepts_data]
        styles_data = _read_yaml(root / data["styles"])["styles"]
        styles = [PromptStyle(**item) for item in styles_data]
        cases = _expand_legacy(concepts, styles, policy)

    return Suite(
        path=suite_path,
        suite_id=data["id"],
        version=int(data["version"]),
        name=data["name"],
        description=data.get("description", ""),
        cases=cases,
        concepts=concepts,
        case_policy=policy,
        scoring=data.get("scoring", {}),
        aggregation=data.get("aggregation", {}),
    )


def validate_suite(suite: Suite) -> list[str]:
    errors: list[str] = []
    if not suite.cases:
        errors.append("suite must contain at least one case")
    if not suite.concepts:
        errors.append("suite must contain at least one concept")
    return errors


def expand_cases(suite: Suite) -> list[EvalCase]:
    errors = validate_suite(suite)
    if errors:
        raise ValueError("Invalid suite: " + "; ".join(errors))
    return list(suite.cases)


def expand_cases_for_styles(suite: Suite, style_ids: list[str]) -> list[EvalCase]:
    return [c for c in suite.cases if c.style_id in style_ids]


def expand_cases_for_concepts(suite: Suite, concept_ids: list[str]) -> list[EvalCase]:
    return [c for c in suite.cases if c.concept_id in set(concept_ids)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
