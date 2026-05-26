#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="$ROOT_DIR/suites/core-75/suite.yaml"
MODELS_DIR="/home/mike/Applications/StabilityMatrix/Data/Models/StableDiffusion"
OUT_ROOT="$ROOT_DIR/outputs"
COMFY_URL="http://127.0.0.1:8188"
PYTHON_BIN="python3"
STEPS="20"
CFG="7"
SAMPLER="euler"
SCHEDULER="normal"
TIMEOUT="900"
CASE_LIMIT=""
RESOLUTION_SCALE="1.0"
SCORER="brightness-contrast"
OVERNIGHT=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_models.sh [options]

Runs the eval suite against checkpoint files that have not already produced a
completed run.json under outputs/<model-id>/<suite-id>/.

Default behavior tests only the next untested model and exits. Use --overnight
to run all untested models sequentially.

Options:
  --overnight                 Run every untested model, one after another.
  --suite PATH                Suite YAML path. Default: suites/core-75/suite.yaml
  --models-dir PATH           Directory containing .safetensors/.ckpt files.
  --out-root PATH             Output root. Default: outputs/
  --comfy-url URL             ComfyUI URL. Default: http://127.0.0.1:8188
  --python PATH               Python executable. Default: python3
  --steps N                   Sampling steps. Default: 20
  --cfg N                     CFG scale. Default: 7
  --sampler NAME              Sampler. Default: euler
  --scheduler NAME            Scheduler. Default: normal
  --timeout SECONDS           Per-case ComfyUI timeout. Default: 900
  --case-limit N              Run only first N cases, useful for smoke tests.
  --resolution-scale N        Scale suite dimensions, e.g. 0.5 for SD1.5.
  --scorer NAME               Scorer to run after generation. Default: brightness-contrast
  --dry-run                   Print models that would run, without generating.
  -h, --help                  Show this help.

Examples:
  scripts/run_models.sh --resolution-scale 0.5 --case-limit 3
  scripts/run_models.sh --overnight --resolution-scale 0.5
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overnight)
      OVERNIGHT=1
      shift
      ;;
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --comfy-url)
      COMFY_URL="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --cfg)
      CFG="$2"
      shift 2
      ;;
    --sampler)
      SAMPLER="$2"
      shift 2
      ;;
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --case-limit)
      CASE_LIMIT="$2"
      shift 2
      ;;
    --resolution-scale)
      RESOLUTION_SCALE="$2"
      shift 2
      ;;
    --scorer)
      SCORER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$SUITE" ]]; then
  echo "Suite not found: $SUITE" >&2
  exit 1
fi

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Models directory not found: $MODELS_DIR" >&2
  exit 1
fi

suite_id="$($PYTHON_BIN - <<'PY' "$SUITE"
from pathlib import Path
import sys, yaml
with Path(sys.argv[1]).open('r', encoding='utf-8') as handle:
    print(yaml.safe_load(handle)['id'])
PY
)"

sanitize_model_id() {
  local name="$1"
  name="${name%.*}"
  printf '%s' "$name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

run_is_complete() {
  local run_json="$1"
  [[ -f "$run_json" ]] || return 1
  "$PYTHON_BIN" - <<'PY' "$run_json"
from pathlib import Path
import json, sys
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(1)
cases = data.get('cases', [])
if cases and all(case.get('status') == 'completed' for case in cases):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

mapfile -t models < <(find "$MODELS_DIR" -type f \( -iname '*.safetensors' -o -iname '*.ckpt' \) | sort)

if [[ ${#models[@]} -eq 0 ]]; then
  echo "No checkpoint files found under $MODELS_DIR" >&2
  exit 1
fi

pending=()
for checkpoint in "${models[@]}"; do
  model_id="$(sanitize_model_id "$(basename "$checkpoint")")"
  run_json="$OUT_ROOT/$model_id/$suite_id/run.json"
  if run_is_complete "$run_json"; then
    echo "Already tested: $model_id"
  else
    pending+=("$checkpoint")
  fi
done

if [[ ${#pending[@]} -eq 0 ]]; then
  echo "No untested models found."
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Untested models: ${#pending[@]}"
  for checkpoint in "${pending[@]}"; do
    echo "$(sanitize_model_id "$(basename "$checkpoint")") :: $checkpoint"
    if [[ "$OVERNIGHT" -eq 0 ]]; then
      break
    fi
  done
  exit 0
fi

run_one() {
  local checkpoint="$1"
  local model_id
  local out_dir
  model_id="$(sanitize_model_id "$(basename "$checkpoint")")"
  out_dir="$OUT_ROOT/$model_id/$suite_id"

  echo "Testing model: $model_id"
  echo "Checkpoint: $checkpoint"
  mkdir -p "$out_dir"

  cmd=(
    "$PYTHON_BIN" -m model_eval_suite.cli run-comfy "$SUITE"
    --model-id "$model_id"
    --checkpoint "$checkpoint"
    --comfy-url "$COMFY_URL"
    --out "$out_dir"
    --steps "$STEPS"
    --cfg "$CFG"
    --sampler "$SAMPLER"
    --scheduler "$SCHEDULER"
    --timeout "$TIMEOUT"
    --resolution-scale "$RESOLUTION_SCALE"
  )

  if [[ -n "$CASE_LIMIT" ]]; then
    cmd+=(--case-limit "$CASE_LIMIT")
  fi

  PYTHONPATH="$ROOT_DIR/src" "${cmd[@]}"

  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m model_eval_suite.cli score-run \
    "$out_dir/run.json" \
    --scorer "$SCORER"

  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m model_eval_suite.cli build-card \
    "$out_dir/run.json" \
    --out "$out_dir/$model_id.md"

  echo "Finished: $model_id"
  echo "Run: $out_dir/run.json"
  echo "Card: $out_dir/$model_id.md"
}

if [[ "$OVERNIGHT" -eq 1 ]]; then
  for checkpoint in "${pending[@]}"; do
    run_one "$checkpoint"
  done
else
  run_one "${pending[0]}"
  remaining=$((${#pending[@]} - 1))
  if [[ "$remaining" -gt 0 ]]; then
    echo "Stopped after one model. $remaining untested model(s) remain. Use --overnight to continue through all."
  fi
fi
