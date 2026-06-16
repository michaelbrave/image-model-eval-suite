from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .comfyui import (
    ComfyUIClient,
    build_basic_workflow,
    build_z_image_turbo_workflow,
    build_pixeldit_workflow,
    build_qwen_image_workflow,
    build_lumina2_workflow,
    first_image_ref,
    get_model_params,
    WORKFLOW_BUILDERS,
)
from .report import write_card
from .routing import load_scorer_groups, scorers_for_case
from .scorers import load_run, make_scorer, write_run
from .suite import (
    expand_cases,
    expand_cases_for_concepts,
    expand_cases_for_styles,
    load_suite,
    validate_suite,
    write_jsonl,
)


def _cmd_validate_suite(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    errors = validate_suite(suite)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        raise SystemExit(1)
    cases = expand_cases(suite)
    style_ids = sorted(set(c.style_id for c in cases))
    print(json.dumps({"valid": True, "suite_id": suite.suite_id, "concepts": len(suite.concepts), "styles": len(style_ids), "cases": len(cases)}, indent=2))


def _cmd_render_plan(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    cases = [case.to_json() for case in expand_cases(suite)]
    out = Path(args.out)
    write_jsonl(out, cases)
    print(json.dumps({"suite_id": suite.suite_id, "cases": len(cases), "out": str(out)}, indent=2))


def _scaled_dimension(value: int, scale: float) -> int:
    scaled = max(64, int(round(value * scale / 8) * 8))
    return scaled


def _resolve_negative(args: argparse.Namespace, case) -> str:
    if args.negative_prompt:
        return args.negative_prompt
    return case.negative_prompt


def _generate_cases(
    suite,
    cases,
    args,
    out_dir: Path,
    run: dict[str, Any],
) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyUIClient(args.comfy_url)

    workflow_builder = WORKFLOW_BUILDERS.get(args.workflow_type, build_basic_workflow)
    model_params = get_model_params(args.workflow_type)

    for index, case in enumerate(cases, start=1):
        width = _scaled_dimension(case.width, args.resolution_scale)
        height = _scaled_dimension(case.height, args.resolution_scale)
        prefix = f"{args.model_id}_{suite.suite_id}_{case.case_id}"
        negative = _resolve_negative(args, case)

        if args.workflow_type == "checkpoint":
            workflow = workflow_builder(
                checkpoint=args.checkpoint,
                positive=case.positive_prompt,
                negative=negative,
                seed=case.image_seed,
                width=width,
                height=height,
                steps=args.steps,
                cfg=args.cfg,
                sampler=args.sampler,
                scheduler=args.scheduler,
                filename_prefix=prefix,
            )
        elif args.workflow_type == "z-image-turbo":
            workflow = build_z_image_turbo_workflow(
                unet_name=model_params["unet_name"],
                clip_name=model_params["clip_name"],
                vae_name=model_params["vae_name"],
                positive=case.positive_prompt,
                seed=case.image_seed,
                width=width,
                height=height,
                steps=args.steps,
                cfg=args.cfg,
                sampler=args.sampler,
                scheduler=args.scheduler,
                filename_prefix=prefix,
            )
        elif args.workflow_type == "pixeldit":
            workflow = build_pixeldit_workflow(
                unet_name=model_params["unet_name"],
                clip_name=model_params["clip_name"],
                vae_name=model_params["vae_name"],
                positive=case.positive_prompt,
                negative=negative,
                seed=case.image_seed,
                width=width,
                height=height,
                steps=args.steps,
                cfg=args.cfg,
                sampler=args.sampler,
                scheduler=args.scheduler,
                filename_prefix=prefix,
            )
        elif args.workflow_type == "qwen-image":
            workflow = build_qwen_image_workflow(
                unet_name=model_params["unet_name"],
                clip_name=model_params["clip_name"],
                vae_name=model_params["vae_name"],
                positive=case.positive_prompt,
                negative=negative,
                seed=case.image_seed,
                width=width,
                height=height,
                steps=args.steps,
                cfg=args.cfg,
                sampler=args.sampler,
                scheduler=args.scheduler,
                filename_prefix=prefix,
            )
        elif args.workflow_type == "lumina2":
            workflow = build_lumina2_workflow(
                unet_name=model_params["unet_name"],
                clip_name=model_params["clip_name"],
                vae_name=model_params["vae_name"],
                positive=case.positive_prompt,
                negative=negative,
                seed=case.image_seed,
                width=width,
                height=height,
                steps=args.steps,
                cfg=args.cfg,
                sampler=args.sampler,
                scheduler=args.scheduler,
                filename_prefix=prefix,
            )
        else:
            raise ValueError(f"Unknown workflow_type: {args.workflow_type}")
        record = case.to_json()
        record["generated_width"] = width
        record["generated_height"] = height
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


def _cmd_run_comfy(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    cases = expand_cases(suite)

    if args.negative_prompt_file:
        args.negative_prompt = Path(args.negative_prompt_file).read_text(encoding="utf-8").strip()
        args.negative_prompt_file = None  # resolved, don't confuse _generate_cases

    if args.style:
        style_ids = [s.strip() for s in args.style.split(",")]
        cases = expand_cases_for_styles(suite, style_ids)

    if args.sample_concepts is not None:
        concept_ids = sorted(set(c.concept_id for c in cases))
        random.shuffle(concept_ids)
        selected = concept_ids[: args.sample_concepts]
        cases = expand_cases_for_concepts(suite, selected)

    if args.case_limit is not None:
        cases = cases[: args.case_limit]

    out_dir = Path(args.out)
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
            "case_limit": args.case_limit,
            "resolution_scale": args.resolution_scale,
            "style": args.style,
            "sample_concepts": args.sample_concepts,
            "workflow_type": args.workflow_type,
            "negative_prompt": args.negative_prompt,
        },
        "cases": [],
    }
    _generate_cases(suite, cases, args, out_dir, run)


def _parse_step_sweep(value: str) -> list[int]:
    steps: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step <= 0:
            raise ValueError("step sweep values must be positive integers")
        steps.append(step)
    if not steps:
        raise ValueError("step sweep must contain at least one step value")
    return sorted(set(steps))


def _select_calibration_case(suite, args: argparse.Namespace):
    cases = expand_cases(suite)
    if args.style:
        style_cases = [case for case in cases if case.style_id == args.style]
        if style_cases:
            cases = style_cases
    variant_cases = [case for case in cases if case.variant == args.variant]
    if variant_cases:
        cases = variant_cases
    if args.case_id:
        matches = [case for case in cases if case.case_id == args.case_id]
        if not matches:
            raise ValueError(f"Calibration case not found: {args.case_id}")
        return matches[0]
    if not cases:
        raise ValueError("Suite has no cases available for calibration")
    return sorted(cases, key=lambda case: case.case_id)[0]


def _score_calibration_record(record: dict[str, Any], scorer_names: list[str], args: argparse.Namespace) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    if record.get("status") != "completed" or not record.get("image_path"):
        return failures
    image_path = Path(record["image_path"])
    record.setdefault("scores", [])
    existing = {score.get("scorer_name") for score in record["scores"]}
    for name in scorer_names:
        if name in existing:
            continue
        try:
            scorer = make_scorer(name, weight_path=args.weight_path)
            result = scorer.score(
                image_path,
                prompt=record.get("positive_prompt"),
                style_hint=record.get("style_id"),
            )
            record["scores"].append(result.to_json())
        except Exception as exc:  # noqa: BLE001 - calibration should preserve scorer failures
            failures.append({"case_id": record.get("case_id", ""), "scorer": name, "error": str(exc)})
    return failures


def _record_score(record: dict[str, Any], score_kind: str = "aesthetic") -> float | None:
    values: list[float] = []
    for score in record.get("scores", []):
        if score.get("score_kind") != score_kind:
            continue
        value = score.get("normalized_score")
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _recommend_steps(records: list[dict[str, Any]], plateau_delta: float, step_sweep: list[int] | None = None) -> dict[str, Any]:
    scored: list[tuple[int, float]] = []
    allowed_steps = set(step_sweep) if step_sweep is not None else None
    for record in records:
        if record.get("calibration_stage") not in {"load-smoke", "step-sweep"}:
            continue
        step = record.get("generation", {}).get("steps")
        if allowed_steps is not None and step not in allowed_steps:
            continue
        score = _record_score(record, "aesthetic")
        if isinstance(step, int) and score is not None:
            scored.append((step, score))
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


def _cmd_calibrate_comfy(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    case = _select_calibration_case(suite, args)
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyUIClient(args.comfy_url)
    scorer_names = args.scorer or ["dinov3-aesthetic-v1"]
    step_sweep = _parse_step_sweep(args.step_sweep)

    width = _scaled_dimension(case.width, args.resolution_scale)
    height = _scaled_dimension(case.height, args.resolution_scale)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    planned = [("load-smoke", args.smoke_steps)]
    planned.extend(("step-sweep", step) for step in step_sweep if step != args.smoke_steps)

    for index, (stage, steps) in enumerate(planned, start=1):
        record = case.to_json()
        record["status"] = "queued"
        record["calibration_stage"] = stage
        record["generated_width"] = width
        record["generated_height"] = height
        record["generation"] = {
            "steps": steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "resolution_scale": args.resolution_scale,
        }
        prefix = f"{args.model_id}_calibration_{stage}_{steps}_{case.case_id}"
        workflow = build_basic_workflow(
            checkpoint=args.checkpoint,
            positive=case.positive_prompt,
            negative=case.negative_prompt if not args.no_negative else "",
            seed=case.image_seed,
            width=width,
            height=height,
            steps=steps,
            cfg=args.cfg,
            sampler=args.sampler,
            scheduler=args.scheduler,
            filename_prefix=prefix,
        )
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
            image_path = images_dir / f"{stage}-{steps}-{case.case_id}.png"
            image_path.write_bytes(image_bytes)
            record.update({
                "status": "completed",
                "image_path": str(image_path),
                "comfyui": {"prompt_id": prompt_id, "image_ref": image_ref},
            })
        except Exception as exc:  # noqa: BLE001 - preserve per-calibration failures
            record.update({"status": "failed", "error": str(exc)})

        failures.extend(_score_calibration_record(record, scorer_names, args))
        records.append(record)
        print(json.dumps({"case": index, "stage": stage, "steps": steps, "status": record["status"]}))

    recommendation = _recommend_steps(records, args.plateau_delta, step_sweep)
    calibration = {
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "model": {"model_id": args.model_id, "checkpoint": args.checkpoint},
        "calibration": {
            "case_id": case.case_id,
            "style_id": case.style_id,
            "variant": case.variant,
            "scorers": scorer_names,
            "smoke_steps": args.smoke_steps,
            "step_sweep": step_sweep,
            "negative_prompt_policy": "disabled" if args.no_negative else "suite-negative",
            "recommendation": recommendation,
            "failures": failures,
        },
        "cases": records,
    }
    write_run(out_dir / "run.json", calibration)
    (out_dir / "calibration.json").write_text(json.dumps(calibration["calibration"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "run": str(out_dir / "run.json"),
        "calibration": str(out_dir / "calibration.json"),
        "recommended_steps": recommendation.get("recommended_steps"),
        "failures": len(failures),
    }, indent=2))

def _scorer_names_for_case(args: argparse.Namespace, case: dict[str, Any], group: dict[str, Any] | None) -> list[str]:
    names: list[str] = []
    if args.scorer:
        names.extend(args.scorer)
    if group is not None:
        names.extend(scorers_for_case(case, group))
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _cmd_score_run(args: argparse.Namespace) -> None:
    run_path = Path(args.run_json)
    run = load_run(run_path)
    group = None
    if args.scorer_group:
        groups = load_scorer_groups(args.scorer_groups_file)
        if args.scorer_group not in groups:
            raise KeyError(f"Unknown scorer group: {args.scorer_group}")
        group = groups[args.scorer_group]
        run.setdefault("scoring", {})["scorer_group"] = args.scorer_group
        run["scoring"]["scorer_groups_file"] = args.scorer_groups_file
    scorer_cache: dict[str, Any] = {}
    scored = 0
    failures: list[dict[str, str]] = []
    for case in run.get("cases", []):
        if case.get("status") != "completed" or not case.get("image_path"):
            continue
        image_path = Path(case["image_path"])
        case.setdefault("scores", [])
        existing = {score.get("scorer_name") for score in case["scores"]}
        for name in _scorer_names_for_case(args, case, group):
            if name in existing and not args.rescore:
                continue
            try:
                scorer = scorer_cache.setdefault(name, make_scorer(name, weight_path=args.weight_path))
                result = scorer.score(image_path, prompt=case.get("positive_prompt"), style_hint=case.get("style_id"))
                payload = result.to_json()
                if group is not None:
                    payload.setdefault("metadata", {})["scorer_group"] = args.scorer_group
                case["scores"].append(payload)
                scored += 1
            except Exception as exc:  # noqa: BLE001 - keep scoring batch moving
                failures.append({"case_id": case.get("case_id", ""), "scorer": name, "error": str(exc)})
                if not args.keep_going:
                    write_run(run_path, run)
                    raise
    if failures:
        run.setdefault("scoring", {})["failures"] = failures
    write_run(run_path, run)
    print(json.dumps({"run": str(run_path), "new_scores": scored, "failures": len(failures)}, indent=2))


def _cmd_build_card(args: argparse.Namespace) -> None:
    preferred_style = None
    if args.preferred_style:
        preferred_style = json.loads(args.preferred_style)
    write_card(Path(args.run_json), Path(args.out), preferred_style=preferred_style)
    print(json.dumps({"card": args.out}, indent=2))


def _cmd_style_find(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)

    if args.negative_prompt_file:
        args.negative_prompt = Path(args.negative_prompt_file).read_text(encoding="utf-8").strip()
        args.negative_prompt_file = None

    concept_ids = sorted(set(c.concept_id for c in suite.cases))
    random.seed(args.seed)
    random.shuffle(concept_ids)
    selected_ids = concept_ids[: args.sample_concepts]

    # Variant 0 only (1 seed per concept x style)
    all_cases = expand_cases_for_concepts(suite, selected_ids)
    cases = [c for c in all_cases if c.variant == 0]

    total = len(cases)
    print(json.dumps({"stage": "style-find", "concepts": len(selected_ids), "styles_per_concept": 6, "cases": total}))

    out_dir = Path(args.out)
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
            "resolution_scale": args.resolution_scale,
            "stage": "style-find",
            "sample_concepts": args.sample_concepts,
            "workflow_type": args.workflow_type,
            "negative_prompt": args.negative_prompt,
        },
        "cases": [],
    }

    _generate_cases(suite, cases, args, out_dir, run)

    # Score the generated images
    group = None
    scorer_names: list[str] = list(args.scorer)
    if not scorer_names:
        groups = load_scorer_groups(args.scorer_groups_file)
        group = groups.get(args.scorer_group, {}) if args.scorer_group in groups else None
        if group:
            scorer_names = list(group.get("always", []))
    if not scorer_names:
        scorer_names = ["brightness-contrast", "dinov3-aesthetic-v1"]

    run = load_run(out_dir / "run.json")
    scorer_cache: dict[str, Any] = {}
    for case in run.get("cases", []):
        if case.get("status") != "completed" or not case.get("image_path"):
            continue
        image_path = Path(case["image_path"])
        case.setdefault("scores", [])
        for name in scorer_names:
            try:
                scorer = scorer_cache.setdefault(name, make_scorer(name, weight_path=args.weight_path))
                result = scorer.score(image_path, prompt=case.get("positive_prompt"), style_hint=case.get("style_id"))
                case["scores"].append(result.to_json())
            except Exception as exc:  # noqa: BLE001 - keep batch moving
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
        print(json.dumps({"score_kind": kind, "winner": winners[kind], "scores": {k: round(v, 4) for k, v in sorted(means.items())}}))

    result = {
        "winners": winners,
        "sample_concepts": selected_ids,
        "concept_count": len(selected_ids),
        "scorers": scorer_names,
    }
    result_path = out_dir / "preferred_style.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"preferred_style": str(result_path), "winners": winners}))


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
    run.add_argument("--case-limit", type=int, default=None, help="Run only the first N expanded cases")
    run.add_argument("--resolution-scale", type=float, default=1.0, help="Scale suite dimensions, e.g. 0.5 for SD1.5")
    run.add_argument("--style", default=None, help="Comma-separated style IDs to filter (e.g. 'everyday-speech,comma-separated')")
    run.add_argument("--sample-concepts", type=int, default=None, help="Random sample N concepts from the suite")
    run.add_argument("--negative-prompt", default=None, help="Global negative prompt override applied to all cases")
    run.add_argument("--negative-prompt-file", default=None, help="Read global negative prompt from file")
    run.add_argument("--workflow-type", default="checkpoint", choices=["checkpoint", "z-image-turbo", "pixeldit", "qwen-image", "lumina2"], help="Workflow template to use")
    run.set_defaults(func=_cmd_run_comfy)

    calibrate = sub.add_parser("calibrate-comfy", help="Run a cheap smoke + step-sweep calibration through ComfyUI")
    calibrate.add_argument("suite")
    calibrate.add_argument("--model-id", required=True)
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    calibrate.add_argument("--out", required=True)
    calibrate.add_argument("--smoke-steps", type=int, default=4, help="Low-step smoke image before the sweep")
    calibrate.add_argument("--step-sweep", default="4,6,8,10,12,16,20,24,30", help="Comma-separated step values to test")
    calibrate.add_argument("--cfg", type=float, default=7.0)
    calibrate.add_argument("--sampler", default="euler")
    calibrate.add_argument("--scheduler", default="normal")
    calibrate.add_argument("--timeout", type=int, default=600)
    calibrate.add_argument("--resolution-scale", type=float, default=1.0, help="Scale suite dimensions, e.g. 0.5 for SD1.5")
    calibrate.add_argument("--style", default=None, help="Preferred style id for the calibration prompt")
    calibrate.add_argument("--variant", type=int, default=0, help="Prompt variant to calibrate with")
    calibrate.add_argument("--case-id", default=None, help="Specific suite case_id to calibrate with")
    calibrate.add_argument("--scorer", action="append", default=[], help="Scorer(s) to evaluate (default: dinov3-aesthetic-v1)")
    calibrate.add_argument("--weight-path", default=None, help="Override scorer weights/checkpoint where supported")
    calibrate.add_argument("--plateau-delta", type=float, default=0.02, help="Pick earliest step within this normalized aesthetic delta of best")
    calibrate.add_argument("--no-negative", action="store_true", help="Disable suite negative prompt during calibration")
    calibrate.add_argument("--workflow-type", default="checkpoint", choices=["checkpoint", "z-image-turbo", "pixeldit", "qwen-image", "lumina2"], help="Workflow template to use")
    calibrate.set_defaults(func=_cmd_calibrate_comfy)

    score = sub.add_parser("score-run", help="Score completed images in a run JSON")
    score.add_argument("run_json")
    score.add_argument("--scorer", action="append", default=[])
    score.add_argument("--scorer-group", default=None, help="Named scorer group from scorer_groups.yaml")
    score.add_argument("--scorer-groups-file", default="scorer_groups.yaml")
    score.add_argument("--weight-path", default=None, help="Override scorer weights/checkpoint where supported")
    score.add_argument("--rescore", action="store_true")
    score.add_argument("--keep-going", action="store_true", help="Record scorer failures and continue")
    score.set_defaults(func=_cmd_score_run)

    card = sub.add_parser("build-card", help="Build a model card from real scored run data")
    card.add_argument("run_json")
    card.add_argument("--out", required=True)
    card.add_argument("--preferred-style", default=None, help="JSON string with preferred style per score_kind (from style-find)")
    card.set_defaults(func=_cmd_build_card)

    style_find = sub.add_parser("style-find", help="Find preferred prompt style for a model via small test sample")
    style_find.add_argument("suite")
    style_find.add_argument("--model-id", required=True)
    style_find.add_argument("--checkpoint", required=True)
    style_find.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    style_find.add_argument("--out", required=True)
    style_find.add_argument("--steps", type=int, default=20)
    style_find.add_argument("--cfg", type=float, default=7.0)
    style_find.add_argument("--sampler", default="euler")
    style_find.add_argument("--scheduler", default="normal")
    style_find.add_argument("--timeout", type=int, default=600)
    style_find.add_argument("--resolution-scale", type=float, default=1.0, help="Scale suite dimensions, e.g. 0.5 for SD1.5")
    style_find.add_argument("--sample-concepts", type=int, default=5, help="Number of concepts to test (each run with all 6 styles)")
    style_find.add_argument("--seed", type=int, default=0, help="Random seed for concept sampling")
    style_find.add_argument("--scorer", action="append", default=[], help="Scorer(s) to evaluate (default: brightness-contrast + dinov3-aesthetic-v1)")
    style_find.add_argument("--scorer-group", default="routed-default", help="Scorer group from scorer_groups.yaml")
    style_find.add_argument("--scorer-groups-file", default="scorer_groups.yaml")
    style_find.add_argument("--weight-path", default=None)
    style_find.add_argument("--negative-prompt", default=None, help="Global negative prompt override applied to all cases")
    style_find.add_argument("--negative-prompt-file", default=None, help="Read global negative prompt from file")
    style_find.add_argument("--workflow-type", default="checkpoint", choices=["checkpoint", "z-image-turbo", "pixeldit", "qwen-image", "lumina2"], help="Workflow template to use")
    style_find.set_defaults(func=_cmd_style_find)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
