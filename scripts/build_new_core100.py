#!/usr/bin/env python3
"""Build new core-100 archive and suite after user preference labeling."""

import json
import shutil
import sqlite3
from pathlib import Path
import yaml

ROOT = Path("/home/mike/image-workflow/image-model-eval-suite-check")
DB = "/home/mike/image-workflow/image-preference-labeler/data/preferences.sqlite"
SUITES = ROOT / "suites"
ARCHIVE = SUITES / "archive"
OLD_CORE = SUITES / "core-100"
OLD_INBOX = SUITES / "inbox-candidates"
NEW_CORE = SUITES / "core-100"

conn = sqlite3.connect(DB)

ASPECT_MAP = {
    "landscape": "landscape", "architecture": "wide", "environment": "landscape",
    "cityscape": "wide", "vehicle": "landscape", "action": "tall", "motion": "landscape",
    "portrait": "portrait", "product": "square", "food": "portrait",
    "graphic_design": "portrait", "poster": "portrait", "surreal": "tall",
    "hands": "square", "macro": "square", "interior": "landscape", "botanical": "portrait",
    "narrative_scene": "landscape", "creature": "square", "character": "portrait",
    "people_scene": "landscape", "urban_scene": "landscape",
    "general": "square", "animal": "square", "portrait": "portrait",
}

def aspect_ratio(subject, probes):
    for kw in probes + [subject]:
        if kw in ASPECT_MAP:
            return ASPECT_MAP[kw]
    return "square"

# ===== 1. ARCHIVE OLD SUITES =====
if ARCHIVE.exists():
    shutil.rmtree(ARCHIVE)
if OLD_CORE.exists():
    shutil.copytree(OLD_CORE, ARCHIVE / "core-100")
    shutil.rmtree(OLD_CORE)
if OLD_INBOX.exists():
    shutil.copytree(OLD_INBOX, ARCHIVE / "inbox-candidates")
    shutil.rmtree(OLD_INBOX)
print("Archived old suites to suites/archive/")

# ===== 2. LOAD OLD CORE-100 CONCEPTS.YAML =====
with open(ARCHIVE / "core-100" / "concepts.yaml") as f:
    old_yaml = yaml.safe_load(f)

old_concepts = {c["id"]: c for c in old_yaml["concepts"]}

# Which old core-100 concepts to keep (not both_bad, not both_good>1)
cur = conn.execute("SELECT DISTINCT c.prompt FROM comparisons c JOIN decisions d ON c.id = d.comparison_id WHERE d.session_id = 8")
all_prompts = {r[0] for r in cur.fetchall()}
cur = conn.execute("SELECT DISTINCT c.prompt FROM decisions d JOIN comparisons c ON d.comparison_id = c.id WHERE d.session_id = 8 AND d.label = 'both_bad'")
bad_prompts = {r[0] for r in cur.fetchall()}
cur = conn.execute("SELECT c.prompt, COUNT(*) as cnt FROM decisions d JOIN comparisons c ON d.comparison_id = c.id WHERE d.session_id = 8 AND d.label = 'both_good' GROUP BY c.prompt HAVING cnt > 1")
good_exclude = {r[0] for r in cur.fetchall()}

keep_ids = set()
cur = conn.execute("SELECT DISTINCT c.case_id, c.prompt FROM comparisons c JOIN decisions d ON c.id = d.comparison_id WHERE d.session_id = 8")
for case_id, prompt in cur.fetchall():
    cid = case_id.split("__")[0]
    if prompt not in bad_prompts and prompt not in good_exclude:
        keep_ids.add(cid)

# ===== 3. SELECT 24 FROM INBOX-CANDIDATES =====
inbox_cases = []
with open(ARCHIVE / "inbox-candidates" / "cases.jsonl") as f:
    for line in f:
        inbox_cases.append(json.loads(line.strip()))

inbox_concepts = {}
for c in inbox_cases:
    cid = c["concept_id"]
    if cid not in inbox_concepts:
        inbox_concepts[cid] = {
            "domain": c["domain"],
            "subject": c["subject"],
            "probes": c["probes"],
            "difficulty": c["difficulty"],
            "prompt": c["positive_prompt"],
        }

# Which inbox concepts to keep
cur = conn.execute("SELECT DISTINCT c.case_id, c.prompt FROM comparisons c JOIN decisions d ON c.id = d.comparison_id WHERE d.session_id = 9")
all_inbox = {}
for case_id, prompt in cur.fetchall():
    cid = case_id.rsplit("__", 2)[0]
    if cid not in all_inbox:
        all_inbox[cid] = prompt

cur = conn.execute("SELECT c.prompt, COUNT(*) as cnt FROM decisions d JOIN comparisons c ON d.comparison_id = c.id WHERE d.session_id = 9 AND d.label = 'both_bad' GROUP BY c.prompt HAVING cnt > 1")
inbox_bad = {r[0] for r in cur.fetchall()}
cur = conn.execute("SELECT c.prompt, COUNT(*) as cnt FROM decisions d JOIN comparisons c ON d.comparison_id = c.id WHERE d.session_id = 9 AND d.label = 'both_good' GROUP BY c.prompt HAVING cnt > 1")
inbox_good_ex = {r[0] for r in cur.fetchall()}

remaining_inbox = {}
for cid, prompt in all_inbox.items():
    if prompt not in inbox_bad and prompt not in inbox_good_ex:
        remaining_inbox[cid] = inbox_concepts.get(cid, {})

# Group by domain for diversity picking
by_domain = {}
for cid, info in remaining_inbox.items():
    d = info.get("domain", "mixed_general")
    by_domain.setdefault(d, []).append(cid)

# My diversity-biased picks
inbox_picks = []
# design_product: 5
dp = sorted(by_domain.get("design_product", []))
inbox_picks.extend(dp[:5])
# environment_world: 5
env = sorted(by_domain.get("environment_world", []))
inbox_picks.extend(env[:5])
# anime_cartoon: 5
anime = sorted(by_domain.get("anime_cartoon", []))
inbox_picks.extend(anime[:5])
# photography: 4
photo = sorted(by_domain.get("photography", []))
inbox_picks.extend(photo[:4])
# illustration_painting: 3
illus = sorted(by_domain.get("illustration_painting", []))
inbox_picks.extend(illus[:3])
# mixed_general: 2
mixed = sorted(by_domain.get("mixed_general", []))
inbox_picks.extend(mixed[:2])

print(f"Inbox picks: {len(inbox_picks)}")

# ===== 4. BUILD NEW CONCEPTS.YAML =====
new_concepts = []

# Add 76 remaining core-100 in original order
for c in old_yaml["concepts"]:
    if c["id"] in keep_ids:
        entry = {k: v for k, v in c.items()}
        new_concepts.append(entry)

# Add 24 inbox-candidates picks
for i, cid in enumerate(inbox_picks):
    info = inbox_concepts[cid]
    prompt = info["prompt"]
    # Clean: remove "Create an image of " prefix (everyday-speech style artifact)
    for prefix in ["Create an image of ", "Create a "]:
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):]
            break
    entry = {
        "id": cid,
        "domain": info["domain"],
        "subject": info["subject"],
        "aspect_ratio": aspect_ratio(info["subject"], info["probes"]),
        "probes": info["probes"],
        "difficulty": info["difficulty"],
        "source_prompt": prompt,
    }
    new_concepts.append(entry)

NEW_CORE.mkdir(parents=True, exist_ok=True)
(NEW_CORE / "concepts.yaml").write_text(
    yaml.safe_dump({"concepts": new_concepts}, sort_keys=False, allow_unicode=False)
)
print(f"Wrote {len(new_concepts)} concepts to suites/core-100/concepts.yaml")

# ===== 5. GENERATE CASES.JSONL =====
# Simple generation: one case per concept using "everyday-speech" style (just the prompt)
# The full suite with 6 styles × 4 seeds will be regenerated from the DB later.
# For now, create minimal cases for eval compatibility (4 seed variants with the prompt unchanged)
cases = []
image_seed_base = 410000
for mi, entry in enumerate(new_concepts):
    for vi in range(4):
        case_id = f"{entry['id']}__plain__v{vi}"
        cases.append({
            "case_id": case_id,
            "concept_id": entry["id"],
            "style_id": "plain",
            "variant": vi,
            "domain": entry["domain"],
            "subject": entry["subject"],
            "probes": entry["probes"],
            "difficulty": entry["difficulty"],
            "aspect_ratio": entry["aspect_ratio"],
            "image_seed": image_seed_base + mi * 4 + vi,
            "wildcard_seed": image_seed_base + mi * 4 + vi,
            "positive_prompt": entry["source_prompt"],
            "negative_prompt": "",
        })

with open(NEW_CORE / "cases.jsonl", "w") as f:
    for c in cases:
        f.write(json.dumps(c, sort_keys=True) + "\n")
print(f"Wrote {len(cases)} cases to suites/core-100/cases.jsonl")

# ===== 6. WRITE SUITE.YAML =====
suite = {
    "id": "core-100",
    "version": 2,
    "name": "Core 100 Model Evaluation Suite",
    "description": "100 curated prompts for model evaluation, selected from user preference labeling. 76 from original core-100 (minus both_bad and both_good duplicates), 24 from inbox-candidates for diversity.",
    "case_policy": {
        "cases_file": "cases.jsonl",
        "seed_variants": 4,
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
(NEW_CORE / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=False))
print("Wrote suites/core-100/suite.yaml")

# ===== 7. SAVE TENTATIVE BACKUP LIST =====
# Remaining inbox-candidates = all_remaining - picked
remaining_ids = sorted(set(remaining_inbox.keys()) - set(inbox_picks))
tentative = []
for cid in remaining_ids:
    info = inbox_concepts[cid]
    prompt = info["prompt"]
    for prefix in ["Create an image of ", "Create a "]:
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):]
            break
    tentative.append({"id": cid, "domain": info["domain"], "prompt": prompt})

with open(NEW_CORE / "tentative-prompts.yaml", "w") as f:
    f.write("# Tentative backup prompts (not in core-100, saved for future substitution)\n")
    f.write(yaml.safe_dump({"concepts": tentative}, sort_keys=False, allow_unicode=False))
print(f"Wrote {len(tentative)} tentative prompts to suites/core-100/tentative-prompts.yaml")

conn.close()
print("\nDone!")
