#!/usr/bin/env bash
set -euo pipefail

# General transfer runner: one command triggers both MF-style and LightGCN-style runs.
# Uses rule-based coordinator for determinism and avoids in-loop target refits.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/experiments"
mkdir -p "$OUT_DIR"

DATASET="ml-latest-small"
PROCESSED_ROOT="processed"
MAX_INTERACTIONS="10000"
TARGET_ITEM_ID="1215"
TARGET_KEYWORD="horror"
NUM_STEPS="20"
GOAL_RANK="3"
NUM_AGENTS="4"
CANDIDATE_SET="cluster"

echo "Running MF/NeuMF-style transfer (direct target signal)..."
OUT_MF="${OUT_DIR}/transfer_general_mf_${TS}.json"
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root "${PROCESSED_ROOT}" \
  --dataset "${DATASET}" \
  --max-interactions "${MAX_INTERACTIONS}" \
  --target-item-id "${TARGET_ITEM_ID}" \
  --target-keyword "${TARGET_KEYWORD}" \
  --num-steps "${NUM_STEPS}" \
  --goal-rank "${GOAL_RANK}" \
  --num-agents "${NUM_AGENTS}" \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --episode-model neumf \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --target-models neumf \
  --transfer-candidate-set "${CANDIDATE_SET}" \
  --output "${OUT_MF}"
echo "SAVED:${OUT_MF}"

echo "Running LightGCN-style transfer (real users + direct target signal)..."
OUT_LGCN="${OUT_DIR}/transfer_general_lightgcn_${TS}.json"
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root "${PROCESSED_ROOT}" \
  --dataset "${DATASET}" \
  --max-interactions "${MAX_INTERACTIONS}" \
  --target-item-id "${TARGET_ITEM_ID}" \
  --target-keyword "${TARGET_KEYWORD}" \
  --num-steps "${NUM_STEPS}" \
  --goal-rank "${GOAL_RANK}" \
  --num-agents "${NUM_AGENTS}" \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --use-segment-users-as-agents \
  --rule-max-snipers 1 \
  --target-models lightgcn \
  --transfer-candidate-set "${CANDIDATE_SET}" \
  --output "${OUT_LGCN}"
echo "SAVED:${OUT_LGCN}"

echo "Done."
