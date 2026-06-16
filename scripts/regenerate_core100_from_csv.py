#!/usr/bin/env python3
"""Regenerate core-100 cases from a one-column prompt CSV."""

from __future__ import annotations

import csv
import json
import re
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

SEED_VARIANTS = 4
IMAGE_SEED_BASE = 410000
WILDCARD_SEED_BASE = 510000

STANDARD_NEGATIVE = (
    "low quality, blurry, distorted, malformed anatomy, extra limbs, "
    "text artifacts, watermark"
)
BOORU_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, "
    "watermark, username, blurry"
)


SUBJECT_KEYWORDS = [
    ("portrait", ["portrait", "woman", "man", "girl", "model", "freddie mercury", "loki"]),
    ("character", ["character", "mermaid", "warrior", "reploid", "goddess", "monster", "ghoul"]),
    ("animal", ["cat", "fox", "owl"]),
    ("product", ["product", "sculpture", "seal", "logo"]),
    ("interior", ["interior", "room", "bar"]),
    ("landscape", ["seascape", "lighthouse", "garden", "river", "rapids", "forest", "ocean"]),
    ("vehicle", ["sportscar", "ship"]),
    ("graphic_design", ["poster", "vector", "pixel art", "circles"]),
]

ASPECT_BY_SUBJECT = {
    "portrait": "portrait",
    "character": "portrait",
    "animal": "square",
    "product": "square",
    "interior": "landscape",
    "landscape": "landscape",
    "vehicle": "landscape",
    "graphic_design": "square",
}


def clean_spaces(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip(" ,")


def split_negative_clauses(prompt: str) -> tuple[str, str]:
    """Move Midjourney-style 'no ...' clauses to negative text."""
    negative_parts: list[str] = []

    def parenthetical_replacer(match: re.Match[str]) -> str:
        negative_parts.append(clean_spaces(match.group(1)))
        return ""

    prompt = re.sub(
        r"\(\s*no\s+([^)]+)\)",
        parenthetical_replacer,
        prompt,
        flags=re.IGNORECASE,
    )

    no_arg = re.search(r"\s--no\s+(.+)$", prompt, flags=re.IGNORECASE)
    if no_arg:
        negative_parts.append(clean_spaces(no_arg.group(1)))
        prompt = prompt[:no_arg.start()]

    def sentence_no_replacer(match: re.Match[str]) -> str:
        prefix = match.group(1)
        negative_parts.append(clean_spaces(match.group(2)))
        return prefix

    prompt = re.sub(
        r"(^|[.;]\s+)[Nn]o\s+([^,.;]+)",
        sentence_no_replacer,
        prompt,
    )

    prompt = re.sub(
        r"\bno signs of impatience\b",
        lambda _match: (negative_parts.append("signs of impatience") or ""),
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\bNo harsh shadows\b",
        lambda _match: (negative_parts.append("harsh shadows") or ""),
        prompt,
    )


    parts = [p.strip() for p in re.split(r"\s*,\s*", prompt)]
    positive_parts: list[str] = []

    for part in parts:
        match = re.match(r"^(?:--)?no\s+(.+)$", part, flags=re.IGNORECASE)
        if match:
            negative_text = match.group(1)
            tail = ""
            sentence_boundary = re.search(r"([.;])\s+([A-Z0-9].*)$", negative_text)
            if sentence_boundary:
                tail = sentence_boundary.group(2)
                negative_text = negative_text[:sentence_boundary.start(1)]
            negative_parts.append(clean_spaces(negative_text))
            if tail:
                positive_parts.append(tail)
            continue
        positive_parts.append(part)

    positive = clean_spaces(", ".join(p for p in positive_parts if p))
    negative = clean_spaces(", ".join(n for n in negative_parts if n))
    return positive, negative


def csv_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prompts = [clean_spaces(row["Prompt"]) for row in rows if clean_spaces(row.get("Prompt", ""))]
    if len(prompts) != 100:
        raise ValueError(f"Expected 100 prompts in {path}, found {len(prompts)}")
    return prompts


def infer_subject(prompt: str) -> str:
    lowered = prompt.lower()
    for subject, keywords in SUBJECT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return subject
    return "general"


def infer_domain(prompt: str) -> str:
    lowered = prompt.lower()
    if any(word in lowered for word in ["photo", "photography", "70mm", "instagram model"]):
        return "photography"
    if any(word in lowered for word in ["anime", "manga", "reploid"]):
        return "anime_cartoon"
    if any(word in lowered for word in ["product", "logo", "vector", "interior"]):
        return "design_product"
    if any(word in lowered for word in ["landscape", "seascape", "ocean", "forest", "garden", "river"]):
        return "environment_world"
    if any(word in lowered for word in ["painting", "illustration", "comic", "art style", "gouache"]):
        return "illustration_painting"
    return "mixed_general"


def infer_probes(prompt: str, subject: str) -> list[str]:
    lowered = prompt.lower()
    probes = [subject]
    checks = [
        ("portrait", ["portrait", "face", "eyes", "model", "woman", "man", "girl"]),
        ("character", ["character", "mermaid", "warrior", "goddess", "monster"]),
        ("lighting", ["light", "lighting", "shadow", "glow", "neon", "noir", "golden hour"]),
        ("composition", ["composition", "centered", "wide shot", "close up", "framing", "perspective"]),
        ("materials", ["crystal", "glass", "fabric", "leather", "wax", "paint", "texture"]),
        ("motion", ["dancing", "dives", "driving", "splashing", "dynamic", "motion"]),
        ("landscape", ["landscape", "ocean", "forest", "garden", "river", "seascape"]),
        ("product", ["product", "sculpture", "logo", "seal"]),
        ("text-detail", ["text", "markings", "runes", "poster"]),
        ("style-transfer", ["style of", "by ", "inspired", "picasso", "anime", "comic"]),
        ("color", ["color", "palette", "cyan", "neon", "pastel", "saturated"]),
    ]
    for probe, keywords in checks:
        if probe not in probes and any(keyword in lowered for keyword in keywords):
            probes.append(probe)
    return probes


def style_prompt(style_id: str, positive: str) -> str:
    comma_text = ", ".join(p.strip(" .") for p in positive.split(",") if p.strip(" ."))
    if style_id == "everyday-speech":
        return f"Create an image of {positive}."
    if style_id == "comma-separated":
        return comma_text
    if style_id == "booru-tags":
        tags = re.sub(r"[^a-zA-Z0-9, #:+.'-]+", " ", comma_text.lower())
        tags = re.sub(r"\s+", " ", tags).strip()
        return f"masterpiece, best quality, highly detailed, {tags}"
    if style_id == "enhanced-prompt":
        return f"(high quality:1.2), ({positive}:1.1), coherent composition, detailed rendering"
    if style_id == "lisp-like":
        escaped = positive.replace("\\", "\\\\").replace('"', '\\"')
        return f'(prompt (subject "{escaped}") (style "lisp-like"))'
    if style_id == "structured-fields":
        return (
            f"Subject: {positive} "
            "Requirements: preserve the listed visual details, style references, colors, and mood. "
            "Composition: coherent high-quality image."
        )
    raise ValueError(f"Unknown style: {style_id}")


def style_negative(style_id: str, source_negative: str) -> str:
    if style_id == "booru-tags":
        return clean_spaces(", ".join(x for x in [BOORU_NEGATIVE, source_negative] if x))
    if style_id == "lisp-like":
        negative = clean_spaces(", ".join(x for x in [STANDARD_NEGATIVE, source_negative] if x))
        return f"(negative {negative})"
    return source_negative


def build(csv_path: Path, suite_dir: Path) -> None:
    prompts = csv_prompts(csv_path)
    concepts: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    case_index = 0

    for prompt_index, raw_prompt in enumerate(prompts, start=1):
        positive, source_negative = split_negative_clauses(raw_prompt)
        concept_id = f"csv_prompt_{prompt_index:03d}"
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
            "source_prompt": raw_prompt,
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
                    "image_seed": IMAGE_SEED_BASE + case_index,
                    "wildcard_seed": WILDCARD_SEED_BASE + case_index,
                    "positive_prompt": style_prompt(style_id, positive),
                    "negative_prompt": style_negative(style_id, source_negative),
                })

    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": concepts}, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    with (suite_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build(root / "100-prompts.csv", root / "suites" / "core-100")


if __name__ == "__main__":
    main()
