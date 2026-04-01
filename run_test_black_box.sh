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

# PYTHONPATH=src python -m agas.cli run-transfer \
#   --processed-root processed \
#   --max-interactions 5000 \
#   --dataset ml-latest-small \
#   --n-factors 24 \
#   --target-item-id 1215 \
#   --target-keyword horror \
#   --num-steps 20 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --episode-model lightgcn \
#   --llm-model gpt-5-mini \
#   --openai-api-key "${OPENAI_API_KEY}" \
#   --transfer-candidate-set cluster \
#   --output outputs/transfer_fixed_target_implicit_fix_all_RCs_lightgcn_no_users_as_agents_no_worker_openai_cluster.json
#   # --use-segment-users-as-agents \

# Drop --no-target-implicit-only entirely (RC5 handles competitors properly)
# Add --target-explicit-negative-threshold 2.0 so competitor 1.0 ratings become forced negatives
# Reduce snipers per step to limit degree inflation on LightGCN

# PYTHONPATH=src python -m agas.cli run-transfer \
#   --victim-model-hint mf \
#   --probe-steps 0 \
#   --target-models neumf,lightgcn \
#   --num-agents 4 --num-steps 8 \
#   --target-keyword horror \
#   --coordinator-policy rule --worker-policy rule \
#   --transfer-attack-roles sniper \
#   --transfer-mode option-a \
#   --max-interactions 10000 \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --target-item-id 1215 \
#   --transfer-candidate-set cluster \
#   --output outputs/experiments/expB_mf_yours.json

PYTHONPATH=src python -m agas.cli run-transfer \
  --victim-model-hint mf \
  --probe-steps 0 \
  --target-models neumf,lightgcn \
  --num-agents 4 --num-steps 8 \
  --target-keyword horror \
  --coordinator-policy rule --worker-policy rule \
  --transfer-attack-roles sniper \
  --transfer-mode option-a \
  --max-interactions 10000 \
  --processed-root processed \
  --dataset ml-latest-small \
  --target-item-id 1215 \
  --transfer-candidate-set cluster \
  --output outputs/experiments/expB_mf_yours.json

