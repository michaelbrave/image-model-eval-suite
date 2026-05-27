#!/usr/bin/env python3
"""Build model evaluation suites from prompt-library supersets.

Pre-resolves all wildcards to produce concrete, repeatable prompts.
Each concept × style combination generates N seed variants so scores
can be averaged across seeds for a fair measurement.

Usage:
  python scripts/build_model_eval_suites.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import re
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

DOMAIN_MAP: dict[str, str] = {
    "photography": "photography",
    "anime-cartoon": "anime_cartoon",
    "illustration-painting": "illustration_painting",
    "cgi-render": "cgi_render",
    "mixed-general": "mixed_general",
    "mixed-hard-general": "mixed_general",
    "design-product": "design_product",
    "environment-world": "environment_world",
}

ASPECT_MAP: dict[str, str] = {
    "landscape": "landscape",
    "architecture": "wide",
    "environment": "landscape",
    "cityscape": "wide",
    "vehicle": "landscape",
    "action": "tall",
    "motion": "landscape",
    "portrait": "portrait",
    "product": "square",
    "food": "portrait",
    "graphic_design": "portrait",
    "poster": "portrait",
    "surreal": "tall",
    "hands": "square",
    "macro": "square",
    "interior": "landscape",
    "botanical": "portrait",
    "narrative_scene": "landscape",
    "creature": "square",
    "character": "portrait",
    "people_scene": "landscape",
    "urban_scene": "landscape",
}


WILDCARD_RE = re.compile(r"\{(\w+)\}")


def resolve_wildcards(template: str, wildcard_values: dict[str, list[str]], seed: int) -> str:
    rng = random.Random(seed)
    def _replacer(m: re.Match) -> str:
        key = m.group(1)
        values = wildcard_values.get(key)
        if not values:
            return m.group(0)
        return rng.choice(values)
    return WILDCARD_RE.sub(_replacer, template)


def _aspect_ratio(subject: str, probes: list[str]) -> str:
    for keyword in probes + [subject]:
        if keyword in ASPECT_MAP:
            return ASPECT_MAP[keyword]
    return "square"


def load_wildcard_values(cur: sqlite3.Cursor) -> dict[str, list[str]]:
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


def build_suite(
    db_path: Path,
    set_name: str,
    suite_id: str,
    suite_name: str,
    suite_desc: str,
    out_dir: Path,
    seed_variants: int = 4,
    limit: int | None = None,
    target_concepts: int | None = None,
    supplement_set: str | None = None,
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Load style profile IDs
    style_ids: dict[str, int] = {}
    cur.execute(
        "SELECT id, identifier FROM prompt_style_profiles WHERE identifier IN ({})".format(
            ",".join("?" for _ in STYLE_PROFILES)
        ),
        STYLE_PROFILES,
    )
    for row in cur.fetchall():
        style_ids[row["identifier"]] = row["id"]

    # Load wildcard values
    wildcard_values = load_wildcard_values(cur)

    # Get prompt set
    cur.execute("SELECT id FROM prompt_sets WHERE name = ?", (set_name,))
    set_row = cur.fetchone()
    if not set_row:
        print(f"  SKIPPING '{set_name}': not found")
        return
    prompt_set_id = set_row["id"]

    # Get members
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
        (prompt_set_id,),
    )
    members = cur.fetchall()
    if limit is not None:
        members = members[:limit]

    # Supplement from another set if we need more concepts
    used_identifiers: set[str] = set()
    if target_concepts is not None and len(members) < target_concepts and supplement_set:
        needed = target_concepts - len(members)
        cur.execute("SELECT id FROM prompt_sets WHERE name = ?", (supplement_set,))
        supp_row = cur.fetchone()
        if supp_row:
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
                (supp_row["id"],),
            )
            existing_identifiers = {m["identifier"] for m in members}
            candidates: list[Any] = []
            for row in cur.fetchall():
                if row["identifier"] not in existing_identifiers:
                    candidates.append(row)

            # Balance domains: count current domains, then round-robin from underrepresented
            current_domain_counts: dict[str, int] = defaultdict(int)
            for mi, member in enumerate(members):
                pid = member["id"]
                metadata_raw = json.loads(member["metadata"] or "{}")
                em = (
                    metadata_raw.get("eval_selection")
                    or metadata_raw.get("publish_curation")
                    or metadata_raw.get("publish_selection")
                    or {}
                )
                domain_raw = em.get("eval_domain", em.get("publish_domain", em.get("domain", "mixed_general")))
                current_domain_counts[DOMAIN_MAP.get(domain_raw, "mixed_general")] += 1

            candidates_by_domain: dict[str, list[Any]] = defaultdict(list)
            for row in candidates:
                md = json.loads(row["metadata"] or "{}")
                em = md.get("publish_curation") or md.get("publish_selection") or md.get("eval_selection") or {}
                domain_raw = em.get("publish_domain", em.get("eval_domain", em.get("domain", "mixed_general")))
                dom = DOMAIN_MAP.get(domain_raw, "mixed_general")
                candidates_by_domain[dom].append(row)

            all_domains = sorted(candidates_by_domain.keys())
            domains_needed = list(all_domains)
            idx = 0
            while needed > 0 and any(candidates_by_domain.values()):
                if not domains_needed:
                    domains_needed = list(all_domains)
                dom = domains_needed[idx % len(domains_needed)]
                idx += 1
                candidates_list = candidates_by_domain.get(dom)
                if not candidates_list:
                    domains_needed = [d for d in domains_needed if d != dom]
                    continue
                row = candidates_list.pop(0)
                members.append(row)
                current_domain_counts[dom] = current_domain_counts.get(dom, 0) + 1
                needed -= 1

        if needed > 0:
            print(f"  Could only supplement to {len(members)} concepts (needed {needed} more from '{supplement_set}')")

    print(f"  {len(members)} concepts from '{set_name}'" + (f" (supplemented from '{supplement_set}')" if supplement_set and target_concepts else ""))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build all cases
    concepts_meta: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    case_index = 0
    image_seed_base = 410000
    wc_seed_base = 510000

    for mi, member in enumerate(members):
        pid = member["id"]
        identifier = member["identifier"]
        concept_text = (member["concept"] or "").replace("\n", " ").replace("\r", " ").strip()
        metadata = json.loads(member["metadata"] or "{}")

        # Extract domain/tags
        eval_meta = (
            metadata.get("eval_selection")
            or metadata.get("publish_curation")
            or metadata.get("publish_selection")
            or {}
        )
        tags = eval_meta.get("eval_tags", eval_meta.get("publish_tags", eval_meta.get("tags", [])))
        domain_raw = eval_meta.get("eval_domain", eval_meta.get("publish_domain", eval_meta.get("domain", "mixed_general")))
        domain = DOMAIN_MAP.get(domain_raw, "mixed_general")
        difficulty_list = eval_meta.get("difficulty", [])
        subject = tags[0] if tags else "general"
        probes = list(dict.fromkeys([*tags, *difficulty_list]))
        ar = _aspect_ratio(subject, probes)

        concepts_meta.append({
            "id": identifier,
            "domain": domain,
            "subject": subject,
            "aspect_ratio": ar,
            "probes": probes,
            "difficulty": difficulty_list[0] if difficulty_list else "medium",
        })

        # Get templates for all styles
        templates: dict[str, dict[str, str]] = {}
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
                templates[style_name] = {
                    "pos": t["positive_template"].replace("\n", " ").replace("\r", " ").strip(),
                    "neg": (t["negative_template"] or "").replace("\n", " ").replace("\r", " ").strip(),
                }

        if not templates:
            # No style templates at all - use concept text
            for style_name in STYLE_PROFILES:
                templates[style_name] = {"pos": concept_text, "neg": ""}

        # Generate seed variants for each style
        for style_name in STYLE_PROFILES:
            t = templates.get(style_name)
            if not t:
                continue

            for vi in range(seed_variants):
                case_index += 1
                wc_seed = wc_seed_base + case_index
                pos_prompt = resolve_wildcards(t["pos"], wildcard_values, wc_seed) if wildcard_values else t["pos"]
                neg_prompt = resolve_wildcards(t["neg"], wildcard_values, wc_seed) if wildcard_values else t["neg"]

                # Also resolve any stray {concept} wildcards
                pos_prompt = pos_prompt.replace("{concept}", concept_text)
                neg_prompt = neg_prompt.replace("{concept}", concept_text)
                # Strip any remaining template marker artifacts (unresolved wildcards, malformed brackets)
                pos_prompt = re.sub(r"\{\s*\w+\s*\}", "", pos_prompt)
                neg_prompt = re.sub(r"\{\s*\w+\s*\}", "", neg_prompt)
                pos_prompt = re.sub(r"\{\s*[^}]*\}", "", pos_prompt)
                neg_prompt = re.sub(r"\{\s*[^}]*\}", "", neg_prompt)
                # Clean up double spaces
                pos_prompt = re.sub(r" +", " ", pos_prompt).strip()
                neg_prompt = re.sub(r" +", " ", neg_prompt).strip()

                cases.append({
                    "case_id": f"{identifier}__{style_name}__v{vi}",
                    "concept_id": identifier,
                    "style_id": style_name,
                    "variant": vi,
                    "domain": domain,
                    "subject": subject,
                    "probes": probes,
                    "difficulty": difficulty_list[0] if difficulty_list else "medium",
                    "aspect_ratio": ar,
                    "image_seed": image_seed_base + case_index,
                    "wildcard_seed": wc_seed,
                    "positive_prompt": pos_prompt,
                    "negative_prompt": neg_prompt,
                })

    # Write concepts.yaml (metadata)
    (out_dir / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": concepts_meta}, sort_keys=False, allow_unicode=False)
    )
    concept_count = len(concepts_meta)
    suite_desc_used = suite_desc.format(concepts=concept_count, seed_variants=seed_variants)
    print(f"  Wrote {concept_count} concepts to concepts.yaml")

    # Write cases.jsonl (fully resolved test cases)
    cases_path = out_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    print(f"  Wrote {len(cases)} cases to cases.jsonl ({len(cases) // (len(concepts_meta) or 1)} variants/concept)")

    # Write suite.yaml
    suite = {
        "id": suite_id,
        "version": 1,
        "name": suite_name,
        "description": suite_desc_used,
        "case_policy": {
            "cases_file": "cases.jsonl",
            "seed_variants": seed_variants,
            "aspect_ratios": {
                "square": [1024, 1024],
                "portrait": [832, 1216],
                "landscape": [1216, 832],
                "wide": [1344, 768],
                "tall": [768, 1344],
            },
        },
        "scoring": {
            "recommended": ["brightness-contrast", "improved-aesthetic-predictor"],
            "optional_prompt_aware": ["image-reward"],
        },
        "aggregation": {
            "primary_groupings": ["domain", "subject", "prompt_style", "probes", "aspect_ratio"],
        },
    }
    (out_dir / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=False))
    print(f"  Wrote suite.yaml")

    conn.close()
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model eval suites from prompt-library supersets")
    parser.add_argument("--db", type=Path, default=Path("../prompt-library/data/prompts.db"))
    parser.add_argument("--out", type=Path, default=Path("suites"))
    parser.add_argument("--seed-variants", type=int, default=4, help="Number of seed variants per concept×style")
    parser.add_argument("--limit-100", type=int, default=None)
    parser.add_argument("--limit-1000", type=int, default=None)
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"Building suites from {db_path} with {args.seed_variants} seed variant(s) each\n")

    build_suite(
        db_path=db_path,
        set_name="eval-core-100",
        suite_id="core-100",
        suite_name="Core 100 Model Evaluation Suite",
        suite_desc="{concepts} pre-resolved concrete prompts from eval-core-100 (supplemented from publish-curated-styles-1000) with 6 style variations and {seed_variants} seed variants each.",
        out_dir=args.out / "core-100",
        seed_variants=args.seed_variants,
        limit=args.limit_100,
        target_concepts=100,
        supplement_set="publish-curated-styles-1000",
    )

    build_suite(
        db_path=db_path,
        set_name="publish-curated-styles-1000",
        suite_id="publish-1000",
        suite_name="Publish 1000 Model Evaluation Suite",
        suite_desc="{concepts} pre-resolved concrete prompts from publish-curated-styles-1000 with 6 style variations and {seed_variants} seed variants each.",
        out_dir=args.out / "publish-1000",
        seed_variants=args.seed_variants,
        limit=args.limit_1000,
    )

    print("Done.")


if __name__ == "__main__":
    main()
