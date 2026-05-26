from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def load_scorer_groups(path: str | Path) -> dict[str, Any]:
    group_path = Path(path)
    with group_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"Invalid scorer group file: {group_path}")
    return data["groups"]


def _matches(case: dict[str, Any], match: dict[str, Any]) -> bool:
    if "domain" in match and case.get("domain") != match["domain"]:
        return False
    if "subject" in match and case.get("subject") != match["subject"]:
        return False
    if "style_id" in match and case.get("style_id") != match["style_id"]:
        return False
    probes = _as_set(case.get("probes"))
    if "probes_any" in match and not (probes & _as_set(match["probes_any"])):
        return False
    if "probes_all" in match and not _as_set(match["probes_all"]).issubset(probes):
        return False
    return True


def scorers_for_case(case: dict[str, Any], group: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for name in group.get("always", []):
        if name not in names:
            names.append(name)
    for route in group.get("routes", []):
        if _matches(case, route.get("match", {})):
            for name in route.get("scorers", []):
                if name not in names:
                    names.append(name)
    for name in group.get("prompt_style_fit", []):
        if name not in names:
            names.append(name)
    return names
