#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

PYTHONPATH=src python -m agas.cli run-episode \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 2000 \
  --n-factors 32 \
  --target-item-id 101 \
  --target-keyword horror \
  --goal-rank 3 \
  --num-steps 30 \
  --num-agents 4 \
  --coordinator-policy openai \
  --worker-policy openai \
  --llm-model gpt-5-mini \
  --openai-api-key "${OPENAI_API_KEY}" \
  --prompt-root prompts \
  --output outputs/episode_result_blackbox.json
