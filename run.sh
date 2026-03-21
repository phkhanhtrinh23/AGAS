#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

DATASET="${DATASET:-ml-latest-small}"
TARGET_ITEM_ID="${TARGET_ITEM_ID:-101}"
TARGET_KEYWORD="${TARGET_KEYWORD:-horror}"
GOAL_RANK="${GOAL_RANK:-3}"
NUM_STEPS="${NUM_STEPS:-16}"
NUM_TRIALS="${NUM_TRIALS:-10}"
START_SEED="${START_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/batch_${DATASET}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_ROOT}"

for ((offset=0; offset<NUM_TRIALS; offset++)); do
  seed=$((START_SEED + offset))
  echo
  echo "=== Trial ${offset} | seed=${seed} ==="
  PYTHONPATH=src python -m agas.cli run-episode \
    --processed-root processed \
    --dataset "${DATASET}" \
    --max-interactions 20000 \
    --n-factors 32 \
    --target-item-id "${TARGET_ITEM_ID}" \
    --target-keyword "${TARGET_KEYWORD}" \
    --goal-rank "${GOAL_RANK}" \
    --num-agents 4 \
    --num-steps "${NUM_STEPS}" \
    --seed "${seed}" \
    --coordinator-policy openai \
    --worker-policy openai \
    --llm-model gpt-5-mini \
    --openai-api-key "${OPENAI_API_KEY}" \
    --prompt-root prompts \
    --output "${OUTPUT_ROOT}/episode_seed_${seed}.json"
done

echo
echo "Saved episode traces to ${OUTPUT_ROOT}"

# python -m agas.cli run-transfer \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 50000 \
#   --num-steps 4 \
#   --num-agents 4 \
#   --coordinator-policy rule \
#   --worker-policy rule \
#   --transfer-mode both \
#   --target-epochs 1 \
#   --target-batch-size 2048 \
#   --output outputs/transfer_result.json
