#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

# Online attack
# PYTHONPATH=src python -m agas.cli run-episode \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --n-factors 24 \
#   --target-item-id 1215 \
#   --goal-rank 3 \
#   --num-steps 20 \
#   --num-agents 4 \
#   --coordinator-policy openai \
#   --worker-policy openai \
#   --llm-model gpt-5-mini \
#   --openai-api-key "${OPENAI_API_KEY}" \
#   --prompt-root prompts \
#   --stop-on-goal \
#   --output outputs/episode_result_blackbox.json

# ## Offline attack
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --n-factors 24 \
#   --target-item-id 1215  \
#   --target-keyword horror \
#   --goal-rank 3 \
#   --num-steps 100 \
#   --num-agents 4 \
#   --transfer-mode option-a \
#   --target-models neumf,lightgcn \
#   --target-epochs 3 \
#   --target-batch-size 2048 \
#   --coordinator-policy openai \
#   --worker-policy openai \
#   --llm-model gpt-5-mini \
#   --openai-api-key "${OPENAI_API_KEY}" \
#   --prompt-root prompts \
#   --stop-on-goal \
#   --target-implicit-only \
#   --output outputs/transfer_top50_implicit.json \
#   --transfer-candidate-set all_items

  # PYTHONPATH=src python -m agas.cli run-transfer \
  # --processed-root processed \
  # --dataset ml-latest-small \
  # --max-interactions 5000 \
  # --n-factors 24 \
  # --target-item-id 1215  \
  # --target-keyword horror \
  # --goal-rank 3 \
  # --num-steps 100 \
  # --num-agents 4 \
  # --transfer-mode option-a \
  # --target-models neumf,lightgcn \
  # --target-epochs 3 \
  # --target-batch-size 2048 \
  # --coordinator-policy openai \
  # --worker-policy openai \
  # --llm-model gpt-5-mini \
  # --openai-api-key "${OPENAI_API_KEY}" \
  # --prompt-root prompts \
  # --stop-on-goal \
  # --no-target-implicit-only \
  # --episode-model lightgcn \
  # --output outputs/transfer_top50_no_implicit_lightgcn.json

PYTHONPATH=src python -m agas.cli run-transfer \
  --processed-root processed \
  --max-interactions 5000 \
  --dataset ml-latest-small \
  --n-factors 24 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 20 \
  --goal-rank 3 \
  --num-agents 4 \
  --use-segment-users-as-agents \
  --transfer-attack-roles sniper \
  --no-target-implicit-only \
  --worker-policy openai \
  --llm-model gpt-5-mini \
  --openai-api-key "${OPENAI_API_KEY}" \
  --transfer-candidate-set cluster \
  --episode-model lightgcn \
  --output outputs/transfer_fixed_no_target_implicit_fix_all_RCs.json
