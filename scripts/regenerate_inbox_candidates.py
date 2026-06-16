#!/usr/bin/env python3
"""Build an eval suite from most viable standalone prompts in prompt-inbox."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any

import yaml

from regenerate_core100_from_csv import (
    ASPECT_BY_SUBJECT,
    BOORU_NEGATIVE,
    IMAGE_SEED_BASE,
    SEED_VARIANTS,
    STANDARD_NEGATIVE,
    STYLE_PROFILES,
    WILDCARD_SEED_BASE,
    clean_spaces,
    infer_domain,
    infer_probes,
    infer_subject,
    split_negative_clauses,
    style_prompt,
)


SUITE_ID = "inbox-candidates"
CONCEPT_PREFIX = "inbox_candidate"
IMAGE_SEED_OFFSET = 30000
WILDCARD_SEED_OFFSET = 30000

MJ_PARAM_PATTERN = re.compile(
    r"--(?:ar|chaos|stylize|style|stylea|v|q|niji|raw|quality|personalize|profile|s|c)"
    r"(?:\s+[\S]+)?",
    re.IGNORECASE,
)
LORA_SYNTAX = re.compile(r"\s*<lora:[^>]+>", re.IGNORECASE)
WEIGHTED_PARENS = re.compile(r"\(([^:)]+):\d+(?:\.\d+)?\)")
BRACKET_WEIGHT = re.compile(r"\[([^:\]]+):\d+(?:\.\d+)?\]")
ALT_SYNTAX = re.compile(r"\[([^\]|]+)\|[^\]]+\]")
DOUBLE_BRACE = re.compile(r"\{\{\s*([^}]+)\s*\}\}")
SCORE_TAG = re.compile(r"\bscore_\d+\b", re.IGNORECASE)
ARTIST_TAG = re.compile(r"@\w+(?:[\s_]\w+)*")
BREAK_KEYWORD = re.compile(r"\bBREAK\b")
MODEL_MARKER = re.compile(r"^(?:ye-pop|ye_pop)\s*", re.IGNORECASE)
NEGATIVE_MARKER = re.compile(r"\bNegative prompt\s*:", re.IGNORECASE)
JSON_BLOCK = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)
FIELD_LINE = re.compile(
    r"^\s*(Subject|Clothing|Action|Environment|Camera|Lighting|Objects|"
    r"Style Details|Hair|Accessories)\s*:",
    re.IGNORECASE,
)
VISUAL_TERMS = re.compile(
    r"\b("
    r"photo|photograph|photography|portrait|painting|illustration|render|art|"
    r"artwork|scene|landscape|cityscape|character|anime|cinematic|macro|studio|"
    r"product|poster|vector|watercolor|oil painting|pixel art|fashion|food|"
    r"interior|architecture|mecha|robot|fantasy|travel|shot|camera|image|"
    r"view|lighting|background|composition|style|photorealistic|detailed|"
    r"masterpiece|colors|outdoors|indoors|sky|close-up|high quality"
    r")\b",
    re.IGNORECASE,
)
REFERENCE_EDIT_TERMS = re.compile(
    r"\b("
    r"attached image|use as reference|reference image|transform the photo|"
    r"redraw the attached|redraw this|replace the background|remove the background|"
    r"upscale|inpaint|outpaint|line stickers|sticker pack"
    r")\b",
    re.IGNORECASE,
)
TEMPLATE_TERMS = re.compile(r"\{argument\b|\{[^{}]*default=", re.IGNORECASE)
UNSAFE_TERMS = re.compile(
    r"\b("
    r"underage|teen girl|young teen|child|loli|pussy|pubic|penis|erection|"
    r"porn|av\b|gravure|nsfw|nudity|nude|topless|bra too small|panties|"
    r"tiny bra|sexy mini skirt|seductive cute young lady|photo of unaware female"
    r")\b",
    re.IGNORECASE,
)
SOURCE_ORDER = [
    "gpt-image2-prompts.txt",
    "promptdexter.txt",
    "civitai-new-prompts.csv",
    "Lexica.txt",
    "PromptHero.txt",
    "civitai.txt",
]


def normalize_text(text: str) -> str:
    text = html.unescape(text).replace("\x00", "")
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "â": "'",
        "â": '"',
        "â": '"',
        "â": "-",
        "â": "-",
        "\ufffd": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def is_json_block(text: str) -> bool:
    stripped = text.strip()
    if not JSON_BLOCK.match(stripped):
        return False
    try:
        return isinstance(json.loads(stripped), dict)
    except json.JSONDecodeError:
        return False


def split_text_blocks(text: str) -> list[str]:
    raw_blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", normalize_text(text))
        if block.strip()
    ]
    blocks: list[str] = []
    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("###"):
            continue
        if FIELD_LINE.match(lines[0]) and blocks and not any(FIELD_LINE.match(item) for item in blocks[-1].splitlines()):
            blocks[-1] = blocks[-1] + "\n" + "\n".join(lines)
            continue
        blocks.append("\n".join(lines))
    return blocks

def iter_source_blocks(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="", errors="ignore") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample) if "," in sample else csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            rows = []
            for index, row in enumerate(reader, start=1):
                raw = row.get("Prompt") or row.get("prompt") or next(iter(row.values()), "")
                if raw:
                    rows.append({"source_file": path.name, "source_index": index, "raw": raw})
            return rows

    return [
        {"source_file": path.name, "source_index": index, "raw": block}
        for index, block in enumerate(split_text_blocks(path.read_text(encoding="utf-8", errors="ignore")), start=1)
    ]


def extract_negative(raw: str) -> tuple[str, str]:
    match = NEGATIVE_MARKER.search(raw)
    if not match:
        return raw, ""
    return raw[: match.start()].strip(), raw[match.end() :].strip()


def clean_prompt(raw: str) -> tuple[str, str]:
    raw = normalize_text(raw)
    positive, explicit_negative = extract_negative(raw)
    positive = LORA_SYNTAX.sub("", positive)
    positive = MJ_PARAM_PATTERN.sub("", positive)
    positive = WEIGHTED_PARENS.sub(r"\1", positive)
    positive = BRACKET_WEIGHT.sub(r"\1", positive)
    positive = ALT_SYNTAX.sub(r"\1", positive)
    positive = DOUBLE_BRACE.sub(r"\1", positive)
    positive = SCORE_TAG.sub("", positive)
    positive = ARTIST_TAG.sub("", positive)
    positive = BREAK_KEYWORD.sub(" ", positive)
    positive = MODEL_MARKER.sub("", positive)
    positive = re.sub(r"\s+", " ", positive)
    positive = re.sub(r"\s*,\s*", ", ", positive)
    positive = re.sub(r"(?:,\s*){2,}", ", ", positive)
    positive = positive.strip(" ,.-")

    positive, clause_negative = split_negative_clauses(positive)
    negative = clean_spaces(", ".join(x for x in [explicit_negative, clause_negative] if x))
    negative = LORA_SYNTAX.sub("", negative)
    negative = MJ_PARAM_PATTERN.sub("", negative)
    negative = SCORE_TAG.sub("", negative)
    negative = ARTIST_TAG.sub("", negative)
    negative = BREAK_KEYWORD.sub(" ", negative)
    negative = clean_spaces(negative)
    return positive, negative


def rejection_reason(raw: str, positive: str) -> str | None:
    if is_json_block(raw):
        return "json/style-recipe block"
    if TEMPLATE_TERMS.search(raw):
        return "template with argument placeholders"
    if REFERENCE_EDIT_TERMS.search(raw):
        return "image-edit/reference prompt"
    if UNSAFE_TERMS.search(positive):
        return "unsafe or overly sexualized prompt"
    if len(positive) < 35:
        return "too short after cleaning"
    if len(positive) > 1800:
        return "too long for this eval suite"
    if len(positive.split()) < 5:
        return "too few words after cleaning"
    if not VISUAL_TERMS.search(positive):
        return "no clear visual-generation terms"
    if not any(char.isalpha() for char in positive):
        return "no alphabetic content"
    return None


def style_negative(style_id: str, source_negative: str) -> str:
    if style_id == "booru-tags":
        return clean_spaces(", ".join(x for x in [BOORU_NEGATIVE, source_negative] if x))
    if style_id == "lisp-like":
        negative = clean_spaces(", ".join(x for x in [STANDARD_NEGATIVE, source_negative] if x))
        return f"(negative {negative})"
    return source_negative


def inbox_style_prompt(style_id: str, positive: str) -> str:
    if style_id == "everyday-speech" and re.match(
        r"^(create|generate|imagine|draw|paint|render|capture|make|produce)\b",
        positive,
        flags=re.IGNORECASE,
    ):
        return positive.rstrip(".") + "."
    return style_prompt(style_id, positive)


def source_paths(root: Path) -> list[Path]:
    inbox = root / "prompt-inbox"
    paths = [inbox / name for name in SOURCE_ORDER if (inbox / name).exists()]
    extras = sorted(path for path in inbox.iterdir() if path.is_file() and path not in paths)
    return paths + extras


def collect_prompts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in source_paths(root):
        for block in iter_source_blocks(path):
            raw = block["raw"]
            positive, negative = clean_prompt(raw)
            key = re.sub(r"\W+", " ", positive).strip().lower()
            reason = rejection_reason(raw, positive)
            if not reason and key in seen:
                reason = "duplicate prompt"
            if reason:
                rejected.append({
                    "source_file": block["source_file"],
                    "source_index": block["source_index"],
                    "reason": reason,
                    "raw": clean_spaces(normalize_text(raw))[:500],
                    "clean_prompt": positive[:500],
                })
                continue
            seen.add(key)
            accepted.append({
                "source_file": block["source_file"],
                "source_index": block["source_index"],
                "source_prompt": clean_spaces(normalize_text(raw)),
                "clean_prompt": positive,
                "source_negative": negative,
            })
    return accepted, rejected


def build(root: Path, suite_dir: Path) -> dict[str, int]:
    prompts, rejected = collect_prompts(root)
    concepts: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    case_index = 0

    for prompt_index, item in enumerate(prompts, start=1):
        positive = item["clean_prompt"]
        concept_id = f"{CONCEPT_PREFIX}_{prompt_index:03d}"
        subject = infer_subject(positive)
        domain = infer_domain(positive)
        probes = infer_probes(positive, subject)
        aspect_ratio = ASPECT_BY_SUBJECT.get(subject, "square")
        difficulty = "hard" if len(positive) > 300 else "medium"

        concepts.append({
            "id": concept_id,
            "domain": domain,
            "subject": subject,
            "aspect_ratio": aspect_ratio,
            "probes": probes,
            "difficulty": difficulty,
            "source_file": item["source_file"],
            "source_index": item["source_index"],
            "source_prompt": item["source_prompt"],
            "clean_prompt": positive,
        })

        for style_id in STYLE_PROFILES:
            for variant in range(SEED_VARIANTS):
                case_index += 1
                cases.append({
                    "case_id": f"{concept_id}__{style_id}__v{variant}",
                    "concept_id": concept_id,
                    "style_id": style_id,
                    "variant": variant,
                    "domain": domain,
                    "subject": subject,
                    "probes": probes,
                    "difficulty": difficulty,
                    "aspect_ratio": aspect_ratio,
                    "image_seed": IMAGE_SEED_BASE + IMAGE_SEED_OFFSET + case_index,
                    "wildcard_seed": WILDCARD_SEED_BASE + WILDCARD_SEED_OFFSET + case_index,
                    "positive_prompt": inbox_style_prompt(style_id, positive),
                    "negative_prompt": style_negative(style_id, item["source_negative"]),
                })

    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": concepts}, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    with (suite_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    with (suite_dir / "rejections.jsonl").open("w", encoding="utf-8") as handle:
        for rejection in rejected:
            handle.write(json.dumps(rejection, sort_keys=True) + "\n")
    return {"concepts": len(concepts), "cases": len(cases), "rejected": len(rejected)}


def write_suite_yaml(suite_dir: Path) -> None:
    data = {
        "id": SUITE_ID,
        "version": 1,
        "name": "Inbox Candidate Prompt Evaluation Suite",
        "description": (
            "Candidate prompts extracted from prompt-inbox, cleaned for standalone "
            "text-to-image evaluation, and translated into 6 prompt styles with 4 "
            "seed variants each. This suite is intended for broad screening before "
            "selecting replacements for weak core-100 prompts."
        ),
        "case_policy": {
            "cases_file": "cases.jsonl",
            "seed_variants": SEED_VARIANTS,
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
            "primary_groupings": [
                "domain",
                "subject",
                "prompt_style",
                "probes",
                "aspect_ratio",
                "source_file",
            ],
        },
    }
    (suite_dir / "suite.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    suite_dir = root / "suites" / SUITE_ID
    stats = build(root, suite_dir)
    write_suite_yaml(suite_dir)
    print(json.dumps({"suite_id": SUITE_ID, **stats}, indent=2))


if __name__ == "__main__":
    main()
