#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${SUITE:-$ROOT/suites/core-100/suite.yaml}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
PYTHON="${PYTHON:-python3}"
SCORER_GROUP="${SCORER_GROUP:-routed-default}"

usage() {
  cat <<'EOF'
Usage: eval_model.sh <checkpoint> [model_id]

Runs the full 3-stage model evaluation:
  1. style-find — 5 concepts × 6 styles (30 images) to find preferred prompt style
  2. Full eval — 100 concepts × 1 style × 4 seeds (400 images) using winning style
  3. Score + Card — runs routed-default scorer group, generates card

Environment variables (optional):
  SUITE=suites/core-100/suite.yaml       Suite to use (default: core-100)
  COMFY_URL=http://127.0.0.1:8188         ComfyUI endpoint
  SCORER_GROUP=routed-default             Scorer group from scorer_groups.yaml
  PYTHON=python3                          Python interpreter
  RES_SCALE=<float>                       Resolution scale (auto-detected if unset)

Examples:
  ./scripts/eval_model.sh ~/models/my-model.safetensors
  SCORER_GROUP=routed-default ./scripts/eval_model.sh ~/models/my-model.safetensors my-model
EOF
  exit 0
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
fi

CHECKPOINT="$(realpath "$1")"
MODEL_ID="${2:-}"
if [[ -z "$MODEL_ID" ]]; then
  name="$(basename "$CHECKPOINT")"
  name="${name%.*}"
  MODEL_ID="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
fi

# Auto-detect resolution scale from checkpoint name conventions
RES_SCALE="${RES_SCALE:-}"
if [[ -z "$RES_SCALE" ]]; then
  lowermodel="$(printf '%s' "$MODEL_ID" | tr '[:upper:]' '[:lower:]')"
  case "$lowermodel" in
    *xl*|*sdxl*|*pony*|*playground*|*sd3*|*flux*|*pixart*)
      RES_SCALE=1.0 ;;
    *)
      RES_SCALE=0.5 ;;
  esac
  echo "Detected resolution scale: $RES_SCALE"
fi

OUTDIR="$ROOT/outputs/$MODEL_ID/core-100"
STYLE_FILE="$OUTDIR/preferred_style.json"
RUN_FILE="$OUTDIR/run.json"
CARD_FILE="$OUTDIR/$MODEL_ID.md"

echo "================================================"
echo " Model: $MODEL_ID"
echo " Checkpoint: $CHECKPOINT"
echo " Suite: $SUITE"
echo " ComfyUI: $COMFY_URL"
echo " Scale: $RES_SCALE"
echo " Output: $OUTDIR"
echo "================================================"

mkdir -p "$OUTDIR"

# ---- Stage 1: Style-find ----
echo ""
echo "=== Stage 1: Style Preference Test ==="
echo "  Generating 30 images (5 concepts × 6 styles × 1 seed)..."
env PYTHONPATH="$ROOT/src" "$PYTHON" -m model_eval_suite.cli style-find \
  "$SUITE" \
  --model-id "$MODEL_ID" \
  --checkpoint "$CHECKPOINT" \
  --comfy-url "$COMFY_URL" \
  --out "$OUTDIR" \
  --steps 20 \
  --cfg 7 \
  --sampler euler \
  --scheduler normal \
  --timeout 900 \
  --resolution-scale "$RES_SCALE" \
  --sample-concepts 5 \
  --scorer-group "$SCORER_GROUP"

# ---- Stage 2: Full eval with winning style ----
WINNING_STYLE=""
if [[ -f "$STYLE_FILE" ]]; then
  WINNING_STYLE="$("$PYTHON" -c "
import json
d = json.load(open('$STYLE_FILE'))
wins = d.get('winners', {})
# Pick the first winner across all score kinds
if wins:
    print(list(wins.values())[0])
" 2>/dev/null || true)"
fi

if [[ -z "$WINNING_STYLE" ]]; then
  echo "WARNING: No preferred style found. Using all 6 styles." >&2
  STYLE_FLAG=""
else
  echo ""
  echo "=== Stage 2: Full Evaluation ==="
  echo "  Winning style: $WINNING_STYLE"
  echo "  Generating 400 images (100 concepts × 4 seeds)..."
  STYLE_FLAG="--style $WINNING_STYLE"
fi

# Backup style-find run if it exists
if [[ -f "$RUN_FILE" ]]; then
  mv "$RUN_FILE" "$RUN_FILE.style-find.bak"
fi

env PYTHONPATH="$ROOT/src" "$PYTHON" -m model_eval_suite.cli run-comfy \
  "$SUITE" \
  --model-id "$MODEL_ID" \
  --checkpoint "$CHECKPOINT" \
  --comfy-url "$COMFY_URL" \
  --out "$OUTDIR" \
  --steps 20 \
  --cfg 7 \
  --sampler euler \
  --scheduler normal \
  --timeout 900 \
  --resolution-scale "$RES_SCALE" \
  $STYLE_FLAG

# ---- Stage 3: Score ----
echo ""
echo "=== Stage 3: Scoring (group: $SCORER_GROUP) ==="
env PYTHONPATH="$ROOT/src" "$PYTHON" -m model_eval_suite.cli score-run \
  "$RUN_FILE" \
  --scorer-group "$SCORER_GROUP" \
  --keep-going

# ---- Stage 4: Build Card ----
echo ""
echo "=== Stage 4: Model Card ==="
PREF_JSON=""
if [[ -f "$STYLE_FILE" ]]; then
  PREF_JSON="$("$PYTHON" -c "
import json
d = json.load(open('$STYLE_FILE'))
print(json.dumps(d.get('winners', {})))
")"
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
echo " Done!"
echo " Model: $MODEL_ID"
echo " Style-find: $STYLE_FILE"
echo " Run data: $RUN_FILE"
echo " Card: $CARD_FILE"
echo "================================================"
