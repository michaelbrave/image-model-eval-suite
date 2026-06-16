#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="$ROOT_DIR/suites/core-100/suite.yaml"
MODELS_DIR="/home/mike/Applications/StabilityMatrix/Data/Models/StableDiffusion"
OUT_ROOT="$ROOT_DIR/outputs"
COMFY_URL="http://127.0.0.1:8188"
PYTHON_BIN="python3"
STEPS="20"
SMOKE_STEPS="4"
STEP_SWEEP="4,6,8,10,12,16,20,24,30"
CFG="7"
SAMPLER="euler"
SCHEDULER="normal"
TIMEOUT="900"
CASE_LIMIT=""
RESOLUTION_SCALE="1.0"
SCORER="brightness-contrast"
STYLE=""
STAGE="full"
OVERNIGHT=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_models.sh [options]

Runs the eval suite against checkpoint files that have not already produced a
completed run.json under outputs/<model-id>/<suite-id>/.

Default behavior tests only the next untested model and exits. Use --overnight
to run all untested models sequentially.

Stages (--stage):
  calibrate   Run smoke + single-prompt step sweep and write calibration.json
  full        Run generation + scoring + card (default)
  style-find  Run style preference test (5 concepts x 6 styles, 1 seed each)
  both        Run calibration, then style-find, then full evaluation with winning style

Options:
  --overnight                 Run every untested model, one after another.
  --suite PATH                Suite YAML path. Default: suites/core-100/suite.yaml
  --models-dir PATH           Directory containing .safetensors/.ckpt files.
  --out-root PATH             Output root. Default: outputs/
  --comfy-url URL             ComfyUI URL. Default: http://127.0.0.1:8188
  --python PATH               Python executable. Default: python3
  --steps N                   Sampling steps if no calibration result is used. Default: 20
  --smoke-steps N             Low-step calibration smoke image. Default: 4
  --step-sweep LIST           Comma-separated calibration steps. Default: 4,6,8,10,12,16,20,24,30
  --cfg N                     CFG scale. Default: 7
  --sampler NAME              Sampler. Default: euler
  --scheduler NAME            Scheduler. Default: normal
  --timeout SECONDS           Per-case ComfyUI timeout. Default: 900
  --case-limit N              Run only first N cases, useful for smoke tests.
  --resolution-scale N        Scale suite dimensions, e.g. 0.5 for SD1.5.
  --scorer NAME               Scorer to run after generation. Default: brightness-contrast
  --style STYLE               Preferred prompt style for full run (e.g. everyday-speech).
                              If not set but preferred_style.json exists, uses that.
  --stage STAGE               Workflow stage: calibrate, full, style-find, or both (default: full)
  --dry-run                   Print models that would run, without generating.
  -h, --help                  Show this help.

Examples:
  scripts/run_models.sh --resolution-scale 0.5 --case-limit 3
  scripts/run_models.sh --overnight --resolution-scale 0.5
  scripts/run_models.sh --stage calibrate --resolution-scale 0.5
  scripts/run_models.sh --stage style-find --resolution-scale 0.5
  scripts/run_models.sh --stage both --resolution-scale 0.5
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
    --smoke-steps)
      SMOKE_STEPS="$2"
      shift 2
      ;;
    --step-sweep)
      STEP_SWEEP="$2"
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
    --style)
      STYLE="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
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

case "$STAGE" in
  calibrate|full|style-find|both) ;;
  *) echo "Unknown stage: $STAGE (use calibrate, full, style-find, or both)" >&2; exit 2 ;;
esac

suite_id="$($PYTHON_BIN - <<'PY' "$SUITE"
from pathlib import Path
import sys, yaml
with Path(sys.argv[1]).open('r', encoding='utf-8') as handle:
    print(yaml.safe_load(handle)['id'])
PY
)"

CLI="$PYTHON_BIN -m model_eval_suite.cli"
CLIENV="PYTHONPATH=$ROOT_DIR/src"

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

preferred_style_from_file() {
  local pref_file="$1"
  if [[ -f "$pref_file" ]]; then
    "$PYTHON_BIN" -c "
import json, sys
d = json.loads(open(sys.argv[1]).read())
print(json.dumps(d.get('winners', {})))
" "$pref_file"
  fi
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

run_calibration() {
  local checkpoint="$1"
  local model_id="$2"
  local calibration_dir="$3"

  echo "=== Stage 0: Calibration ==="
  echo "Model: $model_id"
  echo "Checkpoint: $checkpoint"
  mkdir -p "$calibration_dir"

  local calibration_cmd=(
    $CLI calibrate-comfy "$SUITE"
    --model-id "$model_id"
    --checkpoint "$checkpoint"
    --comfy-url "$COMFY_URL"
    --out "$calibration_dir"
    --smoke-steps "$SMOKE_STEPS"
    --step-sweep "$STEP_SWEEP"
    --cfg "$CFG"
    --sampler "$SAMPLER"
    --scheduler "$SCHEDULER"
    --timeout "$TIMEOUT"
    --resolution-scale "$RESOLUTION_SCALE"
  )

  env $CLIENV "${calibration_cmd[@]}"
}

recommended_steps_from_calibration() {
  local calibration_file="$1"
  if [[ -f "$calibration_file" ]]; then
    "$PYTHON_BIN" -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    rec = d.get('recommendation', {})
    value = rec.get('recommended_steps')
    print(value if value is not None else '')
except Exception:
    print('')
" "$calibration_file"
  fi
}

run_style_find() {
  local checkpoint="$1"
  local model_id="$2"
  local out_dir="$3"

  echo "=== Stage 1: Style Preference Test ==="
  echo "Model: $model_id"
  echo "Checkpoint: $checkpoint"
  mkdir -p "$out_dir"

  local style_cmd=(
    $CLI style-find "$SUITE"
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
    --scorer "$SCORER"
  )

  env $CLIENV "${style_cmd[@]}"
}

run_full_eval() {
  local checkpoint="$1"
  local model_id="$2"
  local out_dir="$3"
  local style_arg="$4"

  echo "=== Stage 2: Full Evaluation ==="
  echo "Model: $model_id"
  echo "Style: ${style_arg:-none (all styles)}"
  echo "Checkpoint: $checkpoint"
  mkdir -p "$out_dir"

  local run_cmd=(
    $CLI run-comfy "$SUITE"
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
    run_cmd+=(--case-limit "$CASE_LIMIT")
  fi

  if [[ -n "$style_arg" ]]; then
    run_cmd+=(--style "$style_arg")
  fi

  env $CLIENV "${run_cmd[@]}"

  echo "=== Scoring ==="
  env $CLIENV $CLI score-run \
    "$out_dir/run.json" \
    --scorer "$SCORER"

  echo "=== Building Card ==="
  local pref_flag=""
  local pref_file="$out_dir/preferred_style.json"
  if [[ -f "$pref_file" ]]; then
    local pref_json
    pref_json="$(preferred_style_from_file "$pref_file")"
    if [[ -n "$pref_json" && "$pref_json" != "{}" ]]; then
      pref_flag="--preferred-style $(printf '%s' "$pref_json" | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin)))')"
    fi
  fi

  if [[ -n "$pref_flag" ]]; then
    env $CLIENV $CLI build-card "$out_dir/run.json" --out "$out_dir/$model_id.md" $pref_flag
  else
    env $CLIENV $CLI build-card "$out_dir/run.json" --out "$out_dir/$model_id.md"
  fi

  echo "Card: $out_dir/$model_id.md"
}

run_one() {
  local checkpoint="$1"
  local model_id
  local out_dir
  local calibration_dir
  model_id="$(sanitize_model_id "$(basename "$checkpoint")")"
  out_dir="$OUT_ROOT/$model_id/$suite_id"
  calibration_dir="$OUT_ROOT/$model_id/calibration"

  echo "================================================"
  echo " Model: $model_id"
  echo "================================================"

  case "$STAGE" in
    calibrate)
      run_calibration "$checkpoint" "$model_id" "$calibration_dir"
      echo "Recommended steps: $(recommended_steps_from_calibration "$calibration_dir/calibration.json")"
      ;;
    style-find)
      run_style_find "$checkpoint" "$model_id" "$out_dir"
      echo "Style preference result: $(preferred_style_from_file "$out_dir/preferred_style.json")"
      ;;
    full)
      local style="$STYLE"
      if [[ -z "$style" ]]; then
        local pref_file="$OUT_ROOT/$model_id/$suite_id/preferred_style.json"
        if [[ -f "$pref_file" ]]; then
          style="$("$PYTHON_BIN" -c "import json,sys; d=json.load(open(sys.argv[1])); print(list(d.get('winners',{}).values())[0] if d.get('winners') else '')" "$pref_file")"
        fi
      fi
      run_full_eval "$checkpoint" "$model_id" "$out_dir" "$style"
      ;;
    both)
      run_calibration "$checkpoint" "$model_id" "$calibration_dir"
      local calibrated_steps
      calibrated_steps="$(recommended_steps_from_calibration "$calibration_dir/calibration.json")"
      if [[ -n "$calibrated_steps" ]]; then
        echo "Using calibrated steps: $calibrated_steps"
        STEPS="$calibrated_steps"
      fi
      run_style_find "$checkpoint" "$model_id" "$out_dir"
      local style
      style="$("$PYTHON_BIN" -c "import json,sys; d=json.load(open(sys.argv[1])); print(list(d.get('winners',{}).values())[0] if d.get('winners') else '')" "$out_dir/preferred_style.json")"
      echo "Winning style: $style"
      # Move style-find run aside and do full run
      if [[ -f "$out_dir/run.json" ]]; then
        mv "$out_dir/run.json" "$out_dir/run.style-find.json"
      fi
      run_full_eval "$checkpoint" "$model_id" "$out_dir" "$style"
      ;;
  esac

  echo "Finished: $model_id"
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
