#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

PYTHONPATH=src python -m agas.cli run-episode \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 5000 \
  --n-factors 24 \
  --target-item-id 250 \
  --goal-rank 3 \
  --num-steps 20 \
  --num-agents 4 \
  --coordinator-policy openai \
  --worker-policy openai \
  --llm-model gpt-5-mini \
  --openai-api-key "${OPENAI_API_KEY}" \
  --prompt-root prompts \
  --stop-on-goal \
  --output outputs/episode_result_blackbox.json
