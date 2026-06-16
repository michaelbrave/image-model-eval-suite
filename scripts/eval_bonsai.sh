#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${SUITE:-$ROOT/suites/core-100/suite.yaml}"
BONSAI_URL="${BONSAI_URL:-http://127.0.0.1:8000}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
VENV_PYTHON="$ROOT/.venv/bin/python"
SCORER_GROUP="${SCORER_GROUP:-routed-default}"
RES_SCALE="${RES_SCALE:-1.0}"
BONSAI_DEMO_DIR="${BONSAI_DEMO_DIR:-/home/mike/Bonsai-Image-Demo}"

MODEL_ID="bonsai-ternary-4b"
OUTDIR="${OUTDIR:-$ROOT/outputs/$MODEL_ID/core-100}"
STYLE_FILE="$OUTDIR/preferred_style.json"
RUN_FILE="$OUTDIR/run.json"
CARD_FILE="$OUTDIR/$MODEL_ID.md"

# Find the python to use (try venv first, then system)
if [[ -f "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
fi

mkdir -p "$OUTDIR"

echo "================================================"
echo " Bonsai Image Eval Suite"
echo " Model: $MODEL_ID"
echo " Suite: $SUITE"
echo " Backend: $BONSAI_URL"
echo " Output: $OUTDIR"
echo "================================================"

usage() {
  echo "Usage: eval_bonsai.sh [--stage smoke|style-find|full|score|card|all]"
  echo ""
  echo "Stages:"
  echo "  smoke        Run step calibration smoke test to determine optimal step count"
  echo "  style-find   Run 5 concepts x 6 styles (30 images) to find preferred prompt style"
  echo "  full         Run full 100 concepts x 4 seeds (400 images) with winning style"
  echo "  score        Score completed run images"
  echo "  card         Build model card from scored run"
  echo "  all          Run all stages (default)"
  exit 0
}

STAGE="${1:-all}"

SMOKE_OUTDIR="$ROOT/outputs/$MODEL_ID/smoke-core-100"
CALIBRATION_FILE="$SMOKE_OUTDIR/calibration.json"

ensure_backend() {
    if ! curl -sf "$BONSAI_URL/backends" > /dev/null 2>&1; then
        echo ""
        echo "WARNING: Bonsai backend not running at $BONSAI_URL"
        echo ""
        echo "Start it with:"
        echo "  cd $BONSAI_DEMO_DIR && bash scripts/serve.sh"
        echo ""
        echo "The first startup takes ~30s (cold start + JIT compile)."
        echo "After it's ready, images generate in ~1-3s each."
        echo ""
        if [[ -f "$BONSAI_DEMO_DIR/.venv/bin/python" ]]; then
            echo "Attempting to start backend automatically..."
            BACKEND_PORT="${BONSAI_URL##*:}"  # extract port
            BACKEND_PORT="${BACKEND_PORT:-8000}"
            cd "$BONSAI_DEMO_DIR"
            nohup bash scripts/serve.sh > /tmp/bonsai-server.log 2>&1 &
            echo "  PID: $! (logs: /tmp/bonsai-server.log)"
            echo "  Waiting for backend to be ready..."
            for i in $(seq 1 120); do
                if curl -sf "$BONSAI_URL/backends" > /dev/null 2>&1; then
                    echo "  Backend ready after ${i}s!"
                    break
                fi
                sleep 2
            done
            cd "$ROOT"
        else
            echo "Bonsai-Image-Demo not set up at $BONSAI_DEMO_DIR"
            echo "Please set BONSAI_DEMO_DIR to the correct path."
            exit 1
        fi
    fi
}

generate() {
    local MODE="$1"  # "smoke", "style-find", or "full"
    shift
    ensure_backend
    env PYTHONPATH="$ROOT/src" "$PYTHON" "$ROOT/scripts/eval_bonsai_runner.py" \
        "$MODE" \
        --suite "$SUITE" \
        --out "$OUTDIR" \
        --bonsai-url "$BONSAI_URL" \
        "$@"
}

# Resolve steps from calibration or use default
RESOLVED_STEPS=4
if [[ -f "$CALIBRATION_FILE" ]]; then
    RESOLVED_STEPS=$("$PYTHON" -c "
import json
d = json.load(open('$CALIBRATION_FILE'))
rec = d.get('recommendation', {})
print(rec.get('recommended_steps') or 4)
" 2>/dev/null || echo 4)
fi

if [[ "$STAGE" == "smoke" || "$STAGE" == "all" ]]; then
    mkdir -p "$SMOKE_OUTDIR"
    echo ""
    echo "=== Stage 1: Quality Smoke Test (Step Calibration) ==="
    echo "  Running step sweep to determine optimal step count..."
    ensure_backend
    env PYTHONPATH="$ROOT/src" "$PYTHON" "$ROOT/scripts/eval_bonsai_runner.py" \
        "smoke" \
        --suite "$SUITE" \
        --out "$SMOKE_OUTDIR" \
        --bonsai-url "$BONSAI_URL" \
        --steps 4
    # Re-resolve steps after smoke test
    if [[ -f "$CALIBRATION_FILE" ]]; then
        RESOLVED_STEPS=$("$PYTHON" -c "
import json
d = json.load(open('$CALIBRATION_FILE'))
rec = d.get('recommendation', {})
print(rec.get('recommended_steps') or 4)
" 2>/dev/null || echo 4)
    fi
    echo "  Recommended steps: $RESOLVED_STEPS"
fi

if [[ "$STAGE" == "style-find" || "$STAGE" == "all" ]]; then
    echo ""
    echo "=== Stage 2: Style Preference Test ==="
    echo "  Using $RESOLVED_STEPS steps"
    echo "  Generating 30 images (5 concepts x 6 styles x 1 seed)..."
    generate "style-find" --sample-concepts 5 --steps "$RESOLVED_STEPS"
fi

if [[ "$STAGE" == "full" || "$STAGE" == "all" ]]; then
    # Determine winning style
    WINNING_STYLE=""
    if [[ -f "$STYLE_FILE" ]]; then
        WINNING_STYLE=$("$PYTHON" -c "
import json
d = json.load(open('$STYLE_FILE'))
wins = d.get('winners', {})
if wins:
    print(wins.get('aesthetic') or wins.get('technical_image_stats') or list(wins.values())[0])
" 2>/dev/null || true)
    fi


    echo ""
    echo "=== Stage 3: Full Evaluation ==="
    echo "  Using $RESOLVED_STEPS steps"
    if [[ -n "$WINNING_STYLE" ]]; then
        echo "  Winning style: $WINNING_STYLE"
        echo "  Generating 400 images (100 concepts x 4 seeds)..."
        generate "full" --style "$WINNING_STYLE" --steps "$RESOLVED_STEPS"
    else
        echo "  No winning style found, generating all styles..."
        generate "full" --steps "$RESOLVED_STEPS"
    fi
fi

if [[ "$STAGE" == "score" || "$STAGE" == "all" ]]; then
    echo ""
    echo "=== Stage 4: Scoring ==="
    if [[ ! -f "$RUN_FILE" ]]; then
        echo "ERROR: No run.json found at $RUN_FILE. Run generation first." >&2
        exit 1
    fi
    echo "  Scoring with group: $SCORER_GROUP"
    env PYTHONPATH="$ROOT/src" "$PYTHON" -m model_eval_suite.cli score-run \
        "$RUN_FILE" \
        --scorer-group "$SCORER_GROUP" \
        --keep-going
fi

if [[ "$STAGE" == "card" || "$STAGE" == "all" ]]; then
    echo ""
    echo "=== Stage 5: Model Card ==="
    if [[ ! -f "$RUN_FILE" ]]; then
        echo "ERROR: No run.json found at $RUN_FILE. Run scoring first." >&2
        exit 1
    fi
    PREF_JSON=""
    if [[ -f "$STYLE_FILE" ]]; then
        PREF_JSON=$("$PYTHON" -c "
import json
d = json.load(open('$STYLE_FILE'))
print(json.dumps(d.get('winners', {})))
")
    fi
    if [[ -n "$PREF_JSON" && "$PREF_JSON" != "{}" ]]; then
        env PYTHONPATH="$ROOT/src" "$PYTHON" -m model_eval_suite.cli build-card \
            "$RUN_FILE" \
            --out "$CARD_FILE" \
            --preferred-style "$PREF_JSON"
    else
        env PYTHONPATH="$ROOT/src" "$PYTHON" -m model_eval_suite.cli build-card \
            "$RUN_FILE" \
            --out "$CARD_FILE"
    fi
    echo ""
    echo "================================================"
    echo " Done! Card written to: $CARD_FILE"
    echo "================================================"
fi
