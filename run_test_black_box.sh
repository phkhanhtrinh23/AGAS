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
#   --coordinator-policy rule --worker-policy openai \
#   --transfer-attack-roles sniper \
#   --transfer-mode option-a \
#   --max-interactions 10000 \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --target-item-id 1215 \
#   --transfer-candidate-set cluster \
#   --output outputs/experiments/expB_mf_yours.json

# ---- More offline attacks with direct target signal and real users as agents (no surrogate episode model, no probe steps, victim model hint = mf since we know the victim is MF-based)

# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --target-item-id 1215 \
#   --target-keyword horror \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy openai \
#   --worker-policy openai \
#   --episode-model surrogate \
#   --victim-model-hint mf \
#   --probe-steps 0 \
#   --use-segment-users-as-agents \
#   --rule-max-snipers 3 \
#   --target-models neumf,lightgcn \
#   --transfer-candidate-set cluster \
#   --output outputs/experiments/expE_openai_surrogate_direct_realusers_snipers3_cluster.json

# PYTHONPATH=src python -m agas.cli run-transfer \
#   --victim-model-hint auto \
#   --probe-steps 2 \
#   --target-models neumf,lightgcn \
#   --num-agents 4 \
#   --num-steps 8 \
#   --episode-model surrogate \
#   --target-keyword horror \
#   --coordinator-policy openai \
#   --worker-policy openai \
#   --transfer-attack-roles sniper \
#   --transfer-mode option-a \
#   --max-interactions 10000 \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --target-item-id 1215 \
#   --transfer-candidate-set cluster \
#   --transfer-split-by-target-model \
#   --output outputs/experiments/expA_openai_surrogate_split_by_target.json

# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --target-item-id 1215 \
#   --target-keyword horror \
#   --num-steps 8 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy openai \
#   --worker-policy openai \
#   --episode-model surrogate \
#   --victim-model-hint sequential \
#   --probe-steps 0 \
#   --target-models sequential \
#   --transfer-candidate-set cluster \
#   --output outputs/experiments/expSeq_openai_surrogate_target.json

# ---- Clone-profile LightGCN offline experiments (fake agents with cloned histories)

# # Baseline (no clone): exp_clone_baseline_lightgcn.json
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --target-item-id 1215 \
#   --target-keyword horror \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --episode-model lightgcn \
#   --victim-model-hint mf \
#   --probe-steps 0 \
#   --rule-max-snipers 3 \
#   --target-models lightgcn \
#   --transfer-candidate-set all_items \
#   --output outputs/experiments/exp_clone_baseline_lightgcn.json

# Clone profiles (fake agents + cloned real-user histories): exp_clone_profiles_lightgcn.json
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 10000 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 12 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy openai \
  --worker-policy openai \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --target-models lightgcn \
  --transfer-candidate-set all_items \
  --clone-segment-users-to-agents \
  --output outputs/experiments/exp_clone_profiles_lightgcn.json

# # Clone profiles + more agents/steps (all_items): exp_clone_profiles_lightgcn_agents8_steps16_allitems.json
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --target-item-id 1215 \
#   --target-keyword horror \
#   --num-steps 16 \
#   --goal-rank 3 \
#   --num-agents 8 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --episode-model lightgcn \
#   --victim-model-hint mf \
#   --probe-steps 0 \
#   --rule-max-snipers 3 \
#   --target-models lightgcn \
#   --transfer-candidate-set all_items \
#   --clone-segment-users-to-agents \
#   --output outputs/experiments/exp_clone_profiles_lightgcn_agents8_steps16_allitems.json

# # Clone profiles (cluster): exp_clone_profiles_lightgcn_cluster.json
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --target-item-id 1215 \
#   --target-keyword horror \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --episode-model lightgcn \
#   --victim-model-hint mf \
#   --probe-steps 0 \
#   --rule-max-snipers 3 \
#   --target-models lightgcn \
#   --transfer-candidate-set cluster \
#   --clone-segment-users-to-agents \
#   --output outputs/experiments/exp_clone_profiles_lightgcn_cluster.json

# # Clone profiles + more steps (cluster): exp_clone_profiles_lightgcn_cluster_steps24.json
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-latest-small \
#   --max-interactions 10000 \
#   --target-item-id 1215 \
#   --target-keyword horror \
#   --num-steps 24 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --episode-model lightgcn \
#   --victim-model-hint mf \
#   --probe-steps 0 \
#   --rule-max-snipers 3 \
#   --target-models lightgcn \
#   --transfer-candidate-set cluster \
#   --clone-segment-users-to-agents \
#   --output outputs/experiments/exp_clone_profiles_lightgcn_cluster_steps24.json
