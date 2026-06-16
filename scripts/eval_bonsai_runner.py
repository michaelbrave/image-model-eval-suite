#!/usr/bin/env python3
"""Generate images for the eval suite using the official Bonsai Image backend API."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model_eval_suite.suite import (
    expand_cases_for_concepts,
    expand_cases_for_styles,
    load_suite,
)
from model_eval_suite.scorers import load_run, make_scorer, write_run


def _scaled_dimension(value: int, scale: float) -> int:
    return max(64, int(round(value * scale / 8) * 8))


class BonsaiClient:
    """Client for the official Bonsai-Image-Demo FastAPI backend.

    The backend (started by scripts/serve.sh) keeps the gemlite-quantized model
    resident, so each call is fast after the initial ~30s cold start.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        seed: int,
        steps: int,
        width: int,
        height: int,
        timeout_seconds: int = 600,
    ) -> bytes:
        payload = {
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "backend": "bonsai-ternary-gemlite",
        }
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.content


def generate_single_case(
    suite,
    case,
    args,
    out_dir: Path,
    record: dict[str, Any],
    client: BonsaiClient,
) -> dict[str, Any]:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    width = _scaled_dimension(case.width, args.resolution_scale)
    height = _scaled_dimension(case.height, args.resolution_scale)
    output_case_id = record.get("case_id", case.case_id)
    record["generated_width"] = width
    record["generated_height"] = height
    record["status"] = "queued"
    try:
        image_bytes = client.generate(
            prompt=case.positive_prompt,
            seed=case.image_seed,
            steps=record.get("steps", args.steps),
            width=width,
            height=height,
            timeout_seconds=args.timeout,
        )
        image_path = images_dir / f"{output_case_id}.png"
        image_path.write_bytes(image_bytes)
        record.update({
            "status": "completed",
            "image_path": str(image_path),
        })
    except Exception as exc:
        record.update({"status": "failed", "error": str(exc)})
    return record


def generate_cases(
    suite,
    cases,
    args,
    out_dir: Path,
    run: dict[str, Any],
) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    client = BonsaiClient(args.bonsai_url)

    for index, case in enumerate(cases, start=1):
        record = case.to_json()
        record["steps"] = args.steps
        record = generate_single_case(suite, case, args, out_dir, record, client)
        run["cases"].append(record)
        write_run(out_dir / "run.json", run)
        print(json.dumps({"case": index, "case_id": case.case_id, "status": record["status"]}))

    print(json.dumps({"run": str(out_dir / "run.json"), "cases": len(run["cases"])}, indent=2))


def build_calibration_plan(smoke_steps: int, step_sweep: list[int]) -> list[tuple[str, int]]:
    planned: list[tuple[str, int]] = [("load-smoke", smoke_steps)]
    planned.extend(("step-sweep", step) for step in step_sweep if step != smoke_steps)
    return planned


def recommend_steps_from_records(
    records: list[dict[str, Any]],
    step_sweep: list[int],
    plateau_delta: float,
) -> dict[str, Any]:
    scored: list[tuple[int, float]] = []
    allowed_steps = set(step_sweep)
    for rec in records:
        if rec.get("calibration_stage") not in {"load-smoke", "step-sweep"}:
            continue
        step = rec.get("steps")
        if step not in allowed_steps:
            continue
        aesthetic_vals = [
            s.get("normalized_score") for s in rec.get("scores", [])
            if s.get("score_kind") == "aesthetic" and isinstance(s.get("normalized_score"), (int, float))
        ]
        if isinstance(step, int) and aesthetic_vals:
            scored.append((step, sum(aesthetic_vals) / len(aesthetic_vals)))

    if not scored:
        return {"recommended_steps": None, "reason": "no aesthetic scores available"}

    best_step, best_score = max(scored, key=lambda item: item[1])
    threshold = best_score - plateau_delta
    recommended_step = min(step for step, score in scored if score >= threshold)
    return {
        "recommended_steps": recommended_step,
        "best_steps": best_step,
        "best_score": best_score,
        "plateau_delta": plateau_delta,
        "step_scores": {str(step): score for step, score in sorted(scored)},
        "reason": "earliest step within plateau_delta of best aesthetic score",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bonsai eval suite generation")
    parser.add_argument("mode", choices=["smoke", "style-find", "full"])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--model-id", default="bonsai-ternary-4b")
    parser.add_argument("--bonsai-url", default="http://127.0.0.1:8000", help="URL of the Bonsai-Image-Demo backend")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--style", default=None, help="Single style ID for full eval")
    parser.add_argument("--sample-concepts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=4, help="Low-step smoke image before the sweep")
    parser.add_argument("--step-sweep", default="4,6,8,10,12,16,20,24,30", help="Comma-separated step values to test")
    parser.add_argument("--scorer", action="append", default=[], help="Scorer(s) for smoke test scoring")
    parser.add_argument("--plateau-delta", type=float, default=0.02, help="Pick earliest step within this normalized aesthetic delta of best")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        # Smoke test: calibrate step count via aesthetic plateau detection
        cases = suite.cases
        if not cases:
            raise ValueError("Suite has no cases for smoke test")
        case = cases[0]
        sm = suite.suite_id

        scorer_names = args.scorer or ["improved-aesthetic-predictor"]
        step_sweep = sorted(set(int(s) for s in args.step_sweep.split(",") if s.strip()))
        planned = build_calibration_plan(args.smoke_steps, step_sweep)

        client = BonsaiClient(args.bonsai_url)
        records: list[dict[str, Any]] = []

        for planned_index, (stage, steps) in enumerate(planned, start=1):
            record = case.to_json()
            record["status"] = "queued"
            record["calibration_stage"] = stage
            record["steps"] = steps
            record["case_id"] = f"{case.case_id}__{stage}__{steps}"
            print(json.dumps({"stage": stage, "steps": steps, "case_id": record["case_id"]}))
            record = generate_single_case(suite, case, args, out_dir, record, client)

            if record["status"] == "completed" and record.get("image_path"):
                image_path = Path(record["image_path"])
                record.setdefault("scores", [])
                for name in scorer_names:
                    try:
                        scorer = make_scorer(name)
                        result = scorer.score(
                            image_path,
                            prompt=record.get("positive_prompt"),
                            style_hint=record.get("style_id"),
                        )
                        record["scores"].append(result.to_json())
                    except Exception as exc:
                        print(json.dumps({"warning": f"{name} failed for {record.get('case_id')}: {exc}"}))

            records.append(record)
            print(json.dumps({"case": planned_index, "stage": stage, "steps": steps, "status": record["status"]}))

        recommendation = recommend_steps_from_records(records, step_sweep, args.plateau_delta)
        if recommendation.get("recommended_steps") is not None:
            print(json.dumps({"recommendation": recommendation}))

        run: dict[str, Any] = {
            "suite_id": sm,
            "suite_version": suite.version,
            "model": {"model_id": args.model_id, "backend": "bonsai-image-demo"},
            "calibration": {
                "case_id": case.case_id,
                "style_id": case.style_id,
                "variant": case.variant,
                "scorers": scorer_names,
                "smoke_steps": args.smoke_steps,
                "step_sweep": step_sweep,
                "recommendation": recommendation,
            },
            "cases": records,
        }
        write_run(out_dir / "run.json", run)
        (out_dir / "calibration.json").write_text(json.dumps(run["calibration"], indent=2, sort_keys=True))
        print(json.dumps({
            "run": str(out_dir / "run.json"),
            "calibration": str(out_dir / "calibration.json"),
            "recommended_steps": recommendation.get("recommended_steps"),
        }, indent=2))

    elif args.mode == "style-find":
        concept_ids = sorted(set(c.concept_id for c in suite.cases))
        random.seed(args.seed)
        random.shuffle(concept_ids)
        selected_ids = concept_ids[: args.sample_concepts]
        all_cases = expand_cases_for_concepts(suite, selected_ids)
        cases = [c for c in all_cases if c.variant == 0]
        total = len(cases)
        print(json.dumps({
            "stage": "style-find",
            "concepts": len(selected_ids),
            "styles_per_concept": 6,
            "cases": total,
        }))
        run: dict[str, Any] = {
            "suite_id": suite.suite_id,
            "suite_version": suite.version,
            "model": {"model_id": args.model_id, "backend": "bonsai-image-demo"},
            "generation": {
                "steps": args.steps,
                "bonsai_url": args.bonsai_url,
                "resolution_scale": args.resolution_scale,
                "stage": "style-find",
                "sample_concepts": args.sample_concepts,
            },
            "cases": [],
        }
    else:
        # Full eval
        if args.style:
            cases = expand_cases_for_styles(suite, [args.style])
        else:
            cases = suite.cases
        print(json.dumps({
            "stage": "full",
            "cases": len(cases),
            "style": args.style or "all",
        }))
        run = {
            "suite_id": suite.suite_id,
            "suite_version": suite.version,
            "model": {"model_id": args.model_id, "backend": "bonsai-image-demo"},
            "generation": {
                "steps": args.steps,
                "bonsai_url": args.bonsai_url,
                "resolution_scale": args.resolution_scale,
                "style": args.style,
            },
            "cases": [],
        }

    if args.mode in ("style-find", "full"):
        generate_cases(suite, cases, args, out_dir, run)

    if args.mode == "style-find":
        # Score and determine winning style
        from model_eval_suite.routing import load_scorer_groups, scorers_for_case

        run = load_run(out_dir / "run.json")
        scorer_names = ["brightness-contrast", "improved-aesthetic-predictor"]
        scorer_cache: dict[str, Any] = {}
        for case in run.get("cases", []):
            if case.get("status") != "completed" or not case.get("image_path"):
                continue
            image_path = Path(case["image_path"])
            case.setdefault("scores", [])
            for name in scorer_names:
                try:
                    scorer = scorer_cache.setdefault(name, make_scorer(name))
                    result = scorer.score(
                        image_path,
                        prompt=case.get("positive_prompt"),
                        style_hint=case.get("style_id"),
                    )
                    case["scores"].append(result.to_json())
                except Exception as exc:
                    print(json.dumps({"warning": f"{name} failed for {case.get('case_id')}: {exc}"}))
        write_run(out_dir / "run.json", run)

        # Determine winning style per score_kind
        style_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for case in run.get("cases", []):
            style_id = case.get("style_id", "")
            for score in case.get("scores", []):
                kind = str(score.get("score_kind", "unknown"))
                val = score.get("normalized_score")
                if isinstance(val, (int, float)):
                    style_values[kind][style_id].append(float(val))

        winners = {}
        for kind, styles in style_values.items():
            means = {sid: sum(vals) / len(vals) for sid, vals in styles.items()}
            winners[kind] = max(means, key=means.get)
            print(json.dumps({
                "score_kind": kind,
                "winner": winners[kind],
                "scores": {k: round(v, 4) for k, v in sorted(means.items())},
            }))

        result = {
            "winners": winners,
            "sample_concepts": selected_ids,
            "concept_count": len(selected_ids),
            "scorers": scorer_names,
        }
        result_path = out_dir / "preferred_style.json"
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps({"preferred_style": str(result_path), "winners": winners}))


if __name__ == "__main__":
    main()
