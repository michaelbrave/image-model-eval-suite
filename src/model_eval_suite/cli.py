from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .comfyui import ComfyUIClient, build_basic_workflow, first_image_ref
from .report import write_card
from .scorers import load_run, make_scorer, write_run
from .suite import expand_cases, load_suite, validate_suite, write_jsonl


def _cmd_validate_suite(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    errors = validate_suite(suite)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        raise SystemExit(1)
    cases = expand_cases(suite)
    print(json.dumps({"valid": True, "suite_id": suite.suite_id, "concepts": len(suite.concepts), "styles": len(suite.styles), "cases": len(cases)}, indent=2))


def _cmd_render_plan(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    cases = [case.to_json() for case in expand_cases(suite)]
    out = Path(args.out)
    write_jsonl(out, cases)
    print(json.dumps({"suite_id": suite.suite_id, "cases": len(cases), "out": str(out)}, indent=2))


def _cmd_run_comfy(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    cases = expand_cases(suite)
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyUIClient(args.comfy_url)

    run: dict[str, Any] = {
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "model": {"model_id": args.model_id, "checkpoint": args.checkpoint},
        "generation": {
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "comfy_url": args.comfy_url,
        },
        "cases": [],
    }

    for index, case in enumerate(cases, start=1):
        prefix = f"{args.model_id}_{suite.suite_id}_{case.case_id}"
        workflow = build_basic_workflow(
            checkpoint=args.checkpoint,
            positive=case.positive_prompt,
            negative=case.negative_prompt,
            seed=case.image_seed,
            width=case.width,
            height=case.height,
            steps=args.steps,
            cfg=args.cfg,
            sampler=args.sampler,
            scheduler=args.scheduler,
            filename_prefix=prefix,
        )
        record = case.to_json()
        record["status"] = "queued"
        try:
            queued = client.queue_prompt(workflow)
            prompt_id = queued.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI response missing prompt_id: {queued}")
            history = client.wait_for_history(prompt_id, timeout_seconds=args.timeout)
            image_ref = first_image_ref(history)
            if image_ref is None:
                raise RuntimeError("ComfyUI history did not include an image")
            image_bytes = client.download_image(
                filename=image_ref["filename"],
                subfolder=image_ref.get("subfolder", ""),
                folder_type=image_ref.get("type", "output"),
            )
            image_path = images_dir / f"{case.case_id}.png"
            image_path.write_bytes(image_bytes)
            record.update({"status": "completed", "image_path": str(image_path), "comfyui": {"prompt_id": prompt_id, "image_ref": image_ref}})
        except Exception as exc:  # noqa: BLE001 - preserve per-case failures in run data
            record.update({"status": "failed", "error": str(exc)})
        run["cases"].append(record)
        write_run(out_dir / "run.json", run)
        print(json.dumps({"case": index, "case_id": case.case_id, "status": record["status"]}))

    print(json.dumps({"run": str(out_dir / "run.json"), "cases": len(run["cases"])}, indent=2))


def _cmd_score_run(args: argparse.Namespace) -> None:
    run_path = Path(args.run_json)
    run = load_run(run_path)
    scorers = [make_scorer(name, weight_path=args.weight_path) for name in args.scorer]
    scored = 0
    for case in run.get("cases", []):
        if case.get("status") != "completed" or not case.get("image_path"):
            continue
        image_path = Path(case["image_path"])
        case.setdefault("scores", [])
        existing = {score.get("scorer_name") for score in case["scores"]}
        for scorer in scorers:
            if scorer.name in existing and not args.rescore:
                continue
            result = scorer.score(image_path, prompt=case.get("positive_prompt"), style_hint=case.get("style_id"))
            case["scores"].append(result.to_json())
            scored += 1
    write_run(run_path, run)
    print(json.dumps({"run": str(run_path), "new_scores": scored}, indent=2))


def _cmd_build_card(args: argparse.Namespace) -> None:
    write_card(Path(args.run_json), Path(args.out))
    print(json.dumps({"card": args.out}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portable image model evaluation suite")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-suite", help="Validate a suite YAML file")
    validate.add_argument("suite")
    validate.set_defaults(func=_cmd_validate_suite)

    render = sub.add_parser("render-plan", help="Render suite cases to JSONL without generating images")
    render.add_argument("suite")
    render.add_argument("--out", required=True)
    render.set_defaults(func=_cmd_render_plan)

    run = sub.add_parser("run-comfy", help="Generate suite images through ComfyUI")
    run.add_argument("suite")
    run.add_argument("--model-id", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    run.add_argument("--out", required=True)
    run.add_argument("--steps", type=int, default=20)
    run.add_argument("--cfg", type=float, default=7.0)
    run.add_argument("--sampler", default="euler")
    run.add_argument("--scheduler", default="normal")
    run.add_argument("--timeout", type=int, default=600)
    run.set_defaults(func=_cmd_run_comfy)

    score = sub.add_parser("score-run", help="Score completed images in a run JSON")
    score.add_argument("run_json")
    score.add_argument("--scorer", action="append", required=True)
    score.add_argument("--weight-path", default=None, help="Weights for improved-aesthetic-predictor")
    score.add_argument("--rescore", action="store_true")
    score.set_defaults(func=_cmd_score_run)

    card = sub.add_parser("build-card", help="Build a model card from real scored run data")
    card.add_argument("run_json")
    card.add_argument("--out", required=True)
    card.set_defaults(func=_cmd_build_card)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
