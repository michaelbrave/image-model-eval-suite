from __future__ import annotations

import importlib.util
from pathlib import Path

from model_eval_suite.cli import _recommend_steps


def _score(value: float) -> dict:
    return {
        "score_kind": "aesthetic",
        "normalized_score": value,
    }


def _load_bonsai_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_bonsai_runner.py"
    spec = importlib.util.spec_from_file_location("eval_bonsai_runner", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bonsai_calibration_plan_does_not_duplicate_smoke_step():
    runner = _load_bonsai_runner()

    assert runner.build_calibration_plan(4, [4, 6, 8]) == [
        ("load-smoke", 4),
        ("step-sweep", 6),
        ("step-sweep", 8),
    ]


def test_bonsai_recommendation_uses_successful_smoke_step():
    runner = _load_bonsai_runner()
    records = [
        {"calibration_stage": "load-smoke", "steps": 4, "scores": [_score(0.45)]},
        {"calibration_stage": "step-sweep", "steps": 6, "scores": [_score(0.36)]},
        {"calibration_stage": "step-sweep", "steps": 8, "scores": [_score(0.37)]},
    ]

    recommendation = runner.recommend_steps_from_records(records, [4, 6, 8], 0.02)

    assert recommendation["recommended_steps"] == 4
    assert recommendation["best_steps"] == 4
    assert recommendation["step_scores"]["4"] == 0.45


def test_generic_recommendation_uses_successful_smoke_step():
    records = [
        {"calibration_stage": "load-smoke", "generation": {"steps": 4}, "scores": [_score(0.45)]},
        {"calibration_stage": "step-sweep", "generation": {"steps": 6}, "scores": [_score(0.36)]},
    ]

    recommendation = _recommend_steps(records, 0.02, [4, 6])

    assert recommendation["recommended_steps"] == 4
    assert recommendation["best_steps"] == 4
