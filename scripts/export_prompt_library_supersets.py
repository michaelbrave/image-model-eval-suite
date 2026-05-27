#!/usr/bin/env python3
"""Export prompt-library supersets (eval-core-100, publish-curated-styles-1000)
into image-model-eval-suite-check as portable text files with 6 style variations.

For each concept we export:
  - Metadata (id, domain, subject, aspect_ratio, probes, difficulty)
  - 6 style templates with wildcard placeholders preserved
  - Wildcard definition values for runtime resolution

Usage:
  python scripts/export_prompt_library_supersets.py --db ../prompt-library/data/prompts.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml


STYLE_PROFILES = [
    "everyday-speech",
    "comma-separated",
    "booru-tags",
    "enhanced-prompt",
    "lisp-like",
    "structured-fields",
]

ASPECT_MAP = {
    "landscape": "landscape",
    "architecture": "wide",
    "environment": "landscape",
    "cityscape": "wide",
    "vehicle": "landscape",
    "action": "tall",
    "motion": "landscape",
    "full_body": "portrait",
    "portrait": "portrait",
    "product": "square",
    "food": "portrait",
    "graphic_design": "portrait",
    "poster": "portrait",
    "surreal": "tall",
    "hands": "square",
    "macro": "square",
    "closeup": "square",
    "interior": "landscape",
    "botanical": "portrait",
    "narrative_scene": "landscape",
    "creature": "square",
    "character": "portrait",
    "people_scene": "landscape",
    "urban_scene": "landscape",
}


def _aspect_ratio(subject: str, probes: list[str]) -> str:
    for keyword in probes + [subject]:
        if keyword in ASPECT_MAP:
            return ASPECT_MAP[keyword]
    return "square"


DOMAIN_MAP = {
    "photography": "photography",
    "anime-cartoon": "anime_cartoon",
    "illustration-painting": "illustration_painting",
    "cgi-render": "cgi_render",
    "mixed-general": "mixed_general",
    "mixed-hard-general": "mixed_general",
    "design-product": "design_product",
    "environment-world": "environment_world",
}


def get_active_style_profiles(cur: sqlite3.Cursor) -> dict[str, int]:
    cur.execute(
        "SELECT id, identifier FROM prompt_style_profiles WHERE identifier IN ({})".format(
            ",".join("?" for _ in STYLE_PROFILES)
        ),
        STYLE_PROFILES,
    )
    return {row[1]: row[0] for row in cur.fetchall()}


def extract_wildcard_values(cur: sqlite3.Cursor) -> dict[str, list[str]]:
    """Export all active wildcard values keyed by wildcard_key."""
    cur.execute("""
        SELECT wd.wildcard_key, wv.value
        FROM wildcard_values wv
        JOIN wildcard_definitions wd ON wv.wildcard_definition_id = wd.id
        WHERE wd.status = 'active'
        ORDER BY wd.wildcard_key, wv.id
    """)
    result: dict[str, list[str]] = {}
    for key, value in cur.fetchall():
        result.setdefault(key, []).append(value)
    return result


def extract_prompt_set(
    cur: sqlite3.Cursor,
    set_name: str,
    style_ids: dict[str, int],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Extract all concepts and their style templates for a given prompt set."""
    cur.execute(
        "SELECT id FROM prompt_sets WHERE name = ?",
        (set_name,),
    )
    set_row = cur.fetchone()
    if not set_row:
        print(f"  WARNING: prompt set '{set_name}' not found")
        return []

    cur.execute(
        """
        SELECT p.id, p.identifier, p.concept, p.metadata
        FROM prompt_set_members psm
        JOIN prompts p ON psm.prompt_id = p.id
        WHERE psm.prompt_set_id = ?
          AND p.status = 'active'
          AND psm.enabled = 1
        ORDER BY psm.position
        """,
        (set_row[0],),
    )
    rows = cur.fetchall()
    if limit is not None:
        rows = rows[:limit]

    print(f"  Found {len(rows)} active prompts in set '{set_name}'")

    concepts = []
    for row in rows:
        pid, identifier, concept_text, metadata_json = row
        metadata = json.loads(metadata_json or "{}")
        eval_meta = metadata.get("eval_selection") or metadata.get("publish_curation") or metadata.get("publish_selection") or {}
        tags = eval_meta.get("eval_tags", eval_meta.get("publish_tags", eval_meta.get("tags", [])))
        domain_raw = eval_meta.get("eval_domain", eval_meta.get("publish_domain", eval_meta.get("domain", "mixed_general")))
        domain = DOMAIN_MAP.get(domain_raw, "mixed_general")
        subject = tags[0] if tags else "general"
        difficulty = eval_meta.get("difficulty", [])
        probes = list(dict.fromkeys([*tags, *difficulty]))

        templates = {}
        for style_name, style_id in style_ids.items():
            cur.execute(
                """
                SELECT positive_template, negative_template
                FROM prompt_templates
                WHERE prompt_id = ? AND style_profile_id = ? AND enabled = 1
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (pid, style_id),
            )
            t = cur.fetchone()
            if t:
                templates[style_name] = {"pos": t[0].replace("\n", " ").replace("\r", " "), "neg": (t[1] or "").replace("\n", " ").replace("\r", " ")}

        concepts.append(
            {
                "id": identifier,
                "domain": domain,
                "subject": subject,
                "aspect_ratio": _aspect_ratio(subject, probes),
                "probes": probes or ["general"],
                "difficulty": "medium",
                "concept": concept_text.replace("\n", " ").replace("\r", " "),
                "templates": templates,
            }
        )
    return concepts


def write_suite(
    out_dir: Path,
    suite_id: str,
    name: str,
    description: str,
    concepts: list[dict[str, Any]],
    wildcard_values: dict[str, list[str]],
) -> None:
    if not concepts:
        print(f"  SKIPPING {suite_id}: no concepts to export")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    styles_dir = out_dir / "styles"
    styles_dir.mkdir(exist_ok=True)

    # concepts.yaml - metadata only
    concepts_meta = []
    for c in concepts:
        concepts_meta.append({
            "id": c["id"],
            "domain": c["domain"],
            "subject": c["subject"],
            "aspect_ratio": c["aspect_ratio"],
            "probes": c["probes"],
            "difficulty": c["difficulty"],
        })
    (out_dir / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": concepts_meta}, sort_keys=False, allow_unicode=False)
    )

    # One text file per style with wildcard templates (one line per concept)
    style_file_count = 0
    for style_name in STYLE_PROFILES:
        style_file = styles_dir / f"{style_name}.txt"
        lines = []
        for c in concepts:
            t = c["templates"].get(style_name)
            if t:
                lines.append(t["pos"])
            else:
                lines.append(c["concept"])
        style_file.write_text("\n".join(lines) + "\n")
        style_file_count += 1

    # Negative prompts: one file per style
    neg_dir = styles_dir / "negatives"
    neg_dir.mkdir(exist_ok=True)
    for style_name in STYLE_PROFILES:
        neg_file = neg_dir / f"{style_name}.txt"
        lines = []
        for c in concepts:
            t = c["templates"].get(style_name)
            if t and t["neg"]:
                lines.append(t["neg"])
            else:
                lines.append("low quality, blurry, distorted")
        neg_file.write_text("\n".join(lines) + "\n")

    # Wildcard values for runtime resolution
    (styles_dir / "wildcards.yaml").write_text(
        yaml.safe_dump({"wildcards": wildcard_values}, sort_keys=False, allow_unicode=False)
    )

    # Reference file: concepts with their prompts (for review)
    ref_entries = []
    for c in concepts:
        entry = {
            "id": c["id"],
            "domain": c["domain"],
            "subject": c["subject"],
            "concept": c["concept"],
            "templates": {},
        }
        for sn in STYLE_PROFILES:
            t = c["templates"].get(sn)
            if t:
                entry["templates"][sn] = {"pos": t["pos"]}
        ref_entries.append(entry)
    (out_dir / "concepts_with_prompts.yaml").write_text(
        yaml.safe_dump({"concepts": ref_entries}, sort_keys=False, allow_unicode=False)
    )

    # suite.yaml
    policy = {
        "expand": "all_concepts_x_styles",
        "image_seed_start": 410000,
        "wildcard_seed_start": 510000,
        "default_width": 1024,
        "default_height": 1024,
        "aspect_ratios": {
            "square": [1024, 1024],
            "portrait": [832, 1216],
            "landscape": [1216, 832],
            "wide": [1344, 768],
            "tall": [768, 1344],
        },
    }

    suite = {
        "id": suite_id,
        "version": 1,
        "name": name,
        "description": description,
        "concepts": "concepts.yaml",
        "styles": "styles/",
        "case_policy": policy,
        "scoring": {
            "recommended": ["brightness-contrast", "improved-aesthetic-predictor"],
            "optional_prompt_aware": ["image-reward"],
        },
        "aggregation": {
            "primary_groupings": ["domain", "subject", "prompt_style", "probes", "aspect_ratio"],
        },
    }
    (out_dir / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=False))

    print(f"  Wrote {len(concepts)} concepts to {out_dir}")
    print(f"  Wrote {style_file_count} style files to {styles_dir}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prompt-library supersets to eval-suite text files")
    parser.add_argument("--db", required=True, type=Path, help="Path to prompt-library/data/prompts.db")
    parser.add_argument("--limit-100", type=int, default=None, help="Limit core-100 prompt count")
    parser.add_argument("--limit-1000", type=int, default=None, help="Limit publish-1000 prompt count")
    parser.add_argument("--out", type=Path, default=Path("suites"), help="Output directory under eval-suite")
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    style_ids = get_active_style_profiles(cur)
    print(f"Found {len(style_ids)} style profiles: {', '.join(style_ids.keys())}\n")

    wildcard_values = extract_wildcard_values(cur)
    print(f"Exported {len(wildcard_values)} wildcard categories with values\n")

    # --- eval-core-100 ---
    print("=== Exporting eval-core-100 ===")
    core100 = extract_prompt_set(cur, "eval-core-100", style_ids, limit=args.limit_100)
    write_suite(
        out_dir=args.out / "core-100",
        suite_id="core-100",
        name="Core 100 Prompt Model Evaluation Suite",
        description=f"{len(core100)} concepts from eval-core-100 with 6 style variations using wildcard-based templates.",
        concepts=core100,
        wildcard_values=wildcard_values,
    )

    # --- publish-curated-styles-1000 ---
    print("=== Exporting publish-curated-styles-1000 ===")
    publish1000 = extract_prompt_set(cur, "publish-curated-styles-1000", style_ids, limit=args.limit_1000)
    write_suite(
        out_dir=args.out / "publish-1000",
        suite_id="publish-1000",
        name="Publish 1000 Prompt Model Evaluation Suite",
        description=f"{len(publish1000)} concepts from publish-curated-styles-1000 with 6 style variations using wildcard-based templates.",
        concepts=publish1000,
        wildcard_values=wildcard_values,
    )

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
