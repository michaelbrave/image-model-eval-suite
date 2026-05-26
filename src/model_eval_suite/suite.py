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
    concept: str
    details: str
    lighting: str
    composition: str


@dataclass(frozen=True)
class PromptStyle:
    id: str
    family: str
    description: str
    positive_template: str
    negative_template: str


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
    width: int
    height: int
    image_seed: int
    wildcard_seed: int
    positive_prompt: str
    negative_prompt: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Suite:
    path: Path
    suite_id: str
    version: int
    name: str
    description: str
    concepts: list[Concept]
    styles: list[PromptStyle]
    case_policy: dict[str, Any]
    scoring: dict[str, Any]
    aggregation: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain an object: {path}")
    return data


def load_suite(path: str | Path) -> Suite:
    suite_path = Path(path)
    root = suite_path.parent
    data = _read_yaml(suite_path)
    concepts_data = _read_yaml(root / data["concepts"])["concepts"]
    styles_data = _read_yaml(root / data["styles"])["styles"]
    concepts = [Concept(**item) for item in concepts_data]
    styles = [PromptStyle(**item) for item in styles_data]
    return Suite(
        path=suite_path,
        suite_id=data["id"],
        version=int(data["version"]),
        name=data["name"],
        description=data.get("description", ""),
        concepts=concepts,
        styles=styles,
        case_policy=data.get("case_policy", {}),
        scoring=data.get("scoring", {}),
        aggregation=data.get("aggregation", {}),
    )


def validate_suite(suite: Suite) -> list[str]:
    errors: list[str] = []
    concept_ids = [item.id for item in suite.concepts]
    style_ids = [item.id for item in suite.styles]
    if len(concept_ids) != len(set(concept_ids)):
        errors.append("concept ids must be unique")
    if len(style_ids) != len(set(style_ids)):
        errors.append("style ids must be unique")
    if not suite.concepts:
        errors.append("suite must contain at least one concept")
    if not suite.styles:
        errors.append("suite must contain at least one style")
    ratios = suite.case_policy.get("aspect_ratios", {})
    for concept in suite.concepts:
        if concept.aspect_ratio not in ratios:
            errors.append(f"concept {concept.id} references unknown aspect_ratio {concept.aspect_ratio}")
    for style in suite.styles:
        for required in ("{concept}", "{details}", "{lighting}", "{composition}"):
            if required not in style.positive_template:
                errors.append(f"style {style.id} positive_template missing {required}")
    return errors


def expand_cases(suite: Suite) -> list[EvalCase]:
    errors = validate_suite(suite)
    if errors:
        raise ValueError("Invalid suite: " + "; ".join(errors))

    policy = suite.case_policy
    if policy.get("expand") != "all_concepts_x_styles":
        raise ValueError(f"Unsupported case_policy.expand: {policy.get('expand')}")
    ratios = policy["aspect_ratios"]
    image_seed_start = int(policy.get("image_seed_start", 1))
    wildcard_seed_start = int(policy.get("wildcard_seed_start", 100000))

    cases: list[EvalCase] = []
    index = 0
    for concept in suite.concepts:
        width, height = ratios[concept.aspect_ratio]
        values = asdict(concept)
        for style in suite.styles:
            index += 1
            positive = style.positive_template.format(**values)
            negative = style.negative_template.format(**values)
            cases.append(
                EvalCase(
                    case_id=f"{concept.id}__{style.id}",
                    concept_id=concept.id,
                    style_id=style.id,
                    domain=concept.domain,
                    subject=concept.subject,
                    probes=concept.probes,
                    difficulty=concept.difficulty,
                    aspect_ratio=concept.aspect_ratio,
                    width=int(width),
                    height=int(height),
                    image_seed=image_seed_start + index,
                    wildcard_seed=wildcard_seed_start + index,
                    positive_prompt=positive,
                    negative_prompt=negative,
                )
            )
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
