#!/usr/bin/env python3
"""Generate images for the eval suite using LensPipeline directly."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model_eval_suite.suite import (
    expand_cases_for_concepts,
    expand_cases_for_styles,
    load_suite,
)
from model_eval_suite.scorers import load_run, make_scorer, write_run

SUITE_AR_TO_LENS = {
    "square": "1:1",
    "landscape": "3:2",
    "portrait": "2:3",
    "wide": "16:9",
}


def load_pipeline(device: str = "cuda"):
    from transformers import Mxfp4Config
    from lens import LensGptOssEncoder, LensPipeline

    print(json.dumps({"event": "loading_text_encoder"}), flush=True)
    text_encoder = LensGptOssEncoder.from_pretrained(
        "microsoft/Lens",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=False),
        device_map="auto",
        max_memory={0: "14GB"},
        offload_folder="/tmp/lens_offload",
    )

    print(json.dumps({"event": "loading_pipeline"}), flush=True)
    pipe = LensPipeline.from_pretrained(
        "microsoft/Lens",
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    print(json.dumps({"event": "moving_to_device", "device": device}), flush=True)
    text_encoder.to(device)
    pipe.transformer.to(device)
    pipe.vae.to(device)
    torch.cuda.empty_cache()

    mem = torch.cuda.memory_allocated() / 1024**3
    print(json.dumps({"event": "pipeline_ready", "cuda_gb": round(mem, 2)}), flush=True)
    return pipe


def generate_cases(suite, cases, args, out_dir: Path, run: dict[str, Any], pipe) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        start_t = time.time()
        lens_ar = SUITE_AR_TO_LENS.get(case.aspect_ratio, "1:1")
        seed = case.image_seed
        record = case.to_json()
        record["generated_aspect_ratio"] = lens_ar
        record["generated_base_resolution"] = args.base_resolution
        record["status"] = "queued"
        try:
            generator = torch.Generator("cuda").manual_seed(seed)

            out = pipe(
                prompt=[case.positive_prompt],
                negative_prompt=case.negative_prompt,
                base_resolution=args.base_resolution,
                aspect_ratio=lens_ar,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=generator,
            )

            image_path = images_dir / f"{case.case_id}.png"
            out.images[0].save(image_path)

            elapsed = time.time() - start_t
            record.update({
                "status": "completed",
                "image_path": str(image_path),
                "elapsed_seconds": round(elapsed, 2),
            })
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})

        run["cases"].append(record)
        write_run(out_dir / "run.json", run)
        print(json.dumps({
            "case": index,
            "case_id": case.case_id,
            "style": case.style_id,
            "ar": lens_ar,
            "seed": seed,
            "status": record["status"],
            "elapsed": record.get("elapsed_seconds"),
        }))
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps({
        "run": str(out_dir / "run.json"),
        "cases": len(run["cases"]),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lens eval suite generation")
    parser.add_argument("mode", choices=["style-find", "full"])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--model-id", default="microsoft/Lens")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--base-resolution", type=int, default=1024)
    parser.add_argument("--style", default=None, help="Single style ID for full eval")
    parser.add_argument("--sample-concepts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "style-find":
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
            "model": {
                "model_id": args.model_id,
                "precision": "mxfp4-text-encoder_bf16-transformer-vae",
            },
            "generation": {
                "steps": args.steps,
                "cfg": args.cfg,
                "base_resolution": args.base_resolution,
                "stage": "style-find",
                "sample_concepts": args.sample_concepts,
            },
            "cases": [],
        }
    else:
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
            "model": {
                "model_id": args.model_id,
                "precision": "mxfp4-text-encoder_bf16-transformer-vae",
            },
            "generation": {
                "steps": args.steps,
                "cfg": args.cfg,
                "base_resolution": args.base_resolution,
                "style": args.style,
            },
            "cases": [],
        }

    pipe = load_pipeline()
    generate_cases(suite, cases, args, out_dir, run, pipe)

    if args.mode == "style-find":
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
                    print(json.dumps({
                        "warning": f"{name} failed for {case.get('case_id')}: {exc}",
                    }))
        write_run(out_dir / "run.json", run)

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
