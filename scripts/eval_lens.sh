#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LENS_REPO="${LENS_REPO:-/home/mike/image-workflow/Lens}"
SUITE="${SUITE:-$ROOT/suites/core-100/suite.yaml}"
PYTHON="${PYTHON:-/home/mike/image-workflow/.venv/bin/python3}"
SCORER_GROUP="${SCORER_GROUP:-routed-default}"

MODEL_ID="microsoft/Lens"
OUTDIR="${OUTDIR:-$ROOT/outputs/lens/core-100}"
STYLE_FILE="$OUTDIR/preferred_style.json"
RUN_FILE="$OUTDIR/run.json"
CARD_FILE="$OUTDIR/lens.md"

mkdir -p "$OUTDIR"

echo "================================================"
echo " Lens Eval Suite"
echo " Model: $MODEL_ID"
echo " Lens repo: $LENS_REPO"
echo " Suite: $SUITE"
echo " Output: $OUTDIR"
echo "================================================"

usage() {
  echo "Usage: eval_lens.sh [--stage style-find|full|score|card|all]"
  echo ""
  echo "Stages:"
  echo "  style-find   Run 5 concepts x 6 styles (30 images) to find preferred prompt style"
  echo "  full         Run full 100 concepts x 4 seeds (400 images) with winning style"
  echo "  score        Score completed run images"
  echo "  card         Build model card from scored run"
  echo "  all          Run all stages (default)"
  exit 0
}

STAGE="${1:-all}"

generate() {
    local MODE="$1"
    shift
    PYTHONPATH="$ROOT/src:$LENS_REPO" "$PYTHON" "$ROOT/scripts/eval_lens_runner.py" \
        "$MODE" \
        --suite "$SUITE" \
        --out "$OUTDIR" \
        "$@"
}

if [[ "$STAGE" == "style-find" || "$STAGE" == "all" ]]; then
    echo ""
    echo "=== Stage 1: Style Preference Test ==="
    echo "  Generating 30 images (5 concepts x 6 styles x 1 seed)..."
    generate "style-find" --sample-concepts 5
fi

if [[ "$STAGE" == "full" || "$STAGE" == "all" ]]; then
    WINNING_STYLE=""
    if [[ -f "$STYLE_FILE" ]]; then
        WINNING_STYLE=$("$PYTHON" -c "
import json
d = json.load(open('$STYLE_FILE'))
wins = d.get('winners', {})
if wins:
    print(list(wins.values())[0])
" 2>/dev/null || true)
    fi

    echo ""
    echo "=== Stage 2: Full Evaluation ==="
    if [[ -n "$WINNING_STYLE" ]]; then
        echo "  Winning style: $WINNING_STYLE"
        echo "  Generating 400 images (100 concepts x 4 seeds)..."
        generate "full" --style "$WINNING_STYLE"
    else
        echo "  No winning style found, generating all styles..."
        generate "full"
    fi
fi

if [[ "$STAGE" == "score" || "$STAGE" == "all" ]]; then
    echo ""
    echo "=== Stage 3: Scoring ==="
    if [[ ! -f "$RUN_FILE" ]]; then
        echo "ERROR: No run.json found at $RUN_FILE. Run generation first." >&2
        exit 1
    fi
    echo "  Scoring with group: $SCORER_GROUP"
    PYTHONPATH="$ROOT/src:$LENS_REPO" "$PYTHON" -m model_eval_suite.cli score-run \
        "$RUN_FILE" \
        --scorer-group "$SCORER_GROUP" \
        --keep-going
fi

if [[ "$STAGE" == "card" || "$STAGE" == "all" ]]; then
    echo ""
    echo "=== Stage 4: Model Card ==="
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
        PYTHONPATH="$ROOT/src:$LENS_REPO" "$PYTHON" -m model_eval_suite.cli build-card \
            "$RUN_FILE" \
            --out "$CARD_FILE" \
            --preferred-style "$PREF_JSON"
    else
        PYTHONPATH="$ROOT/src:$LENS_REPO" "$PYTHON" -m model_eval_suite.cli build-card \
            "$RUN_FILE" \
            --out "$CARD_FILE"
    fi
    echo ""
    echo "================================================"
    echo " Done! Card written to: $CARD_FILE"
    echo "================================================"
fi
