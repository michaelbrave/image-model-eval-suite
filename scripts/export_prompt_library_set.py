#!/usr/bin/env python3
"""Export a prompt-library prompt_set into a portable concepts YAML draft.

This is an optional bridge for the existing prompt-library SQLite database. The
result should be manually reviewed before becoming a stable eval suite.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def _metadata(row: sqlite3.Row) -> dict[str, Any]:
    raw = row["metadata"] if "metadata" in row.keys() else row["metadata_json"]
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def export_concepts(db_path: Path, prompt_set: str, limit: int | None) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT p.identifier, p.concept, p.metadata, psm.position
        FROM prompt_sets ps
        JOIN prompt_set_members psm ON psm.prompt_set_id = ps.id
        JOIN prompts p ON p.id = psm.prompt_id
        WHERE ps.name = ? AND psm.enabled = 1
        ORDER BY psm.position ASC
    """
    rows = conn.execute(sql, (prompt_set,)).fetchall()
    if limit is not None:
        rows = rows[:limit]
    concepts: list[dict[str, Any]] = []
    for row in rows:
        metadata = _metadata(row)
        eval_meta = metadata.get("eval_selection", {})
        tags = eval_meta.get("eval_tags", [])
        domain = eval_meta.get("eval_domain", "mixed_general").replace("-", "_")
        subject = tags[0] if tags else "general"
        probes = list(dict.fromkeys([*tags, *eval_meta.get("difficulty", [])]))
        concepts.append(
            {
                "id": row["identifier"],
                "domain": domain,
                "subject": subject,
                "aspect_ratio": "square",
                "probes": probes or ["general"],
                "difficulty": "medium",
                "concept": row["concept"],
                "details": "review and replace with concise visual details",
                "lighting": "review and replace with explicit lighting intent",
                "composition": "review and replace with explicit composition intent",
            }
        )
    return concepts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prompt-library prompt_set to concepts YAML draft")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    concepts = export_concepts(args.db, args.prompt_set, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({"concepts": concepts}, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(json.dumps({"prompt_set": args.prompt_set, "concepts": len(concepts), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
