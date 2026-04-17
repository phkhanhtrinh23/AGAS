#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

# # Online attack
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

# # Clone profiles (fake agents + cloned real-user histories): exp_clone_profiles_lightgcn.json
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
#   --episode-model lightgcn \
#   --victim-model-hint mf \
#   --probe-steps 0 \
#   --rule-max-snipers 3 \
#   --target-models lightgcn \
#   --transfer-candidate-set all_items \
#   --clone-segment-users-to-agents \
#   --output outputs/experiments/exp_clone_profiles_lightgcn.json

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

# ---- Other datasets (rule-based coordinator/worker)

# ml-latest (Horror target: item_id=593). Uses observed vocab to avoid OOM.
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-latest \
#   --max-interactions 20000 \
#   --target-item-id 593 \
#   --target-keyword horror \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --worker-policy rule \
#   --episode-model surrogate \
#   --victim-model-hint auto \
#   --probe-steps 2 \
#   --rule-max-snipers 3 \
#   --target-models neumf,lightgcn \
#   --transfer-candidate-set cluster \
#   --transfer-split-by-target-model \
#   --target-item-vocab observed \
#   --output outputs/experiments/exp_ml_latest_split_rule_observed.json

# ml-latest (Horror target: item_id=196). Non-goal target + no stop-on-goal.
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest \
  --max-interactions 20000 \
  --target-item-id 196 \
  --target-keyword horror \
  --num-steps 12 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model surrogate \
  --victim-model-hint auto \
  --probe-steps 2 \
  --rule-max-snipers 3 \
  --target-models neumf,lightgcn \
  --transfer-candidate-set cluster \
  --transfer-split-by-target-model \
  --target-item-vocab observed \
  --no-stop-on-goal \
  --output outputs/experiments/exp_ml_latest_split_rule_observed_nostop.json

# ml-32m (Horror target: item_id=593). Uses observed vocab to avoid OOM.
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml-32m \
#   --max-interactions 20000 \
#   --target-item-id 593 \
#   --target-keyword horror \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --worker-policy rule \
#   --episode-model surrogate \
#   --victim-model-hint auto \
#   --probe-steps 2 \
#   --rule-max-snipers 3 \
#   --target-models neumf,lightgcn \
#   --transfer-candidate-set cluster \
#   --transfer-split-by-target-model \
#   --target-item-vocab observed \
#   --output outputs/experiments/exp_ml_32m_split_rule_observed.json

# genome_2021 (movie target: item_id=592). Uses observed vocab to avoid OOM.
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset genome_2021 \
#   --max-interactions 20000 \
#   --target-item-id 592 \
#   --target-keyword movie \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --worker-policy rule \
#   --episode-model surrogate \
#   --victim-model-hint auto \
#   --probe-steps 2 \
#   --rule-max-snipers 3 \
#   --target-models neumf,lightgcn \
#   --transfer-candidate-set cluster \
#   --transfer-split-by-target-model \
#   --target-item-vocab observed \
#   --output outputs/experiments/exp_genome_2021_split_rule_observed.json

# genome_2021 (all-items target via empty keyword). Non-goal target + no stop-on-goal.
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset genome_2021 \
#   --max-interactions 20000 \
#   --target-item-id 586 \
#   --target-keyword "" \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --worker-policy rule \
#   --episode-model surrogate \
#   --victim-model-hint auto \
#   --probe-steps 2 \
#   --rule-max-snipers 3 \
#   --target-models neumf,lightgcn \
#   --transfer-candidate-set cluster \
#   --transfer-split-by-target-model \
#   --target-item-vocab observed \
#   --no-stop-on-goal \
#   --output outputs/experiments/exp_genome_2021_split_rule_observed_nostop.json

# amazon_review (local_business target). Uses observed vocab to avoid OOM.
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset amazon_review \
#   --max-interactions 10000 \
#   --target-item-id "0x8890a84997a3ca01:0xb5c7b6fa2bbf6d9e" \
#   --target-keyword local_business \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --worker-policy rule \
#   --episode-model surrogate \
#   --victim-model-hint auto \
#   --probe-steps 2 \
#   --rule-max-snipers 3 \
#   --target-models neumf,lightgcn \
#   --transfer-candidate-set cluster \
#   --transfer-split-by-target-model \
#   --target-item-vocab observed \
#   --output outputs/experiments/exp_amazon_review_split_rule_observed.json

# amazon_review (both NeuMF + LightGCN improved in our run): exp_amazon_review_split_rule_observed.json
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset amazon_review \
  --max-interactions 10000 \
  --target-item-id "0x8890a84997a3ca01:0xb5c7b6fa2bbf6d9e" \
  --target-keyword local_business \
  --num-steps 12 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model surrogate \
  --victim-model-hint auto \
  --probe-steps 2 \
  --rule-max-snipers 3 \
  --target-models neumf,lightgcn \
  --transfer-candidate-set cluster \
  --transfer-split-by-target-model \
  --target-item-vocab observed \
  --clone-segment-users-to-agents \
  --output outputs/experiments/exp_amazon_review_split_rule_observed_both.json

# ml_20mx16x32 (synthetic target). Ratings are all 1.0, so set positive threshold to 1.0.
# PYTHONPATH=src python -m agas.cli run-transfer \
#   --transfer-mode option-a \
#   --processed-root processed \
#   --dataset ml_20mx16x32 \
#   --max-interactions 20000 \
#   --target-item-id 214978 \
#   --target-keyword synthetic \
#   --num-steps 12 \
#   --goal-rank 3 \
#   --num-agents 4 \
#   --transfer-attack-roles sniper \
#   --coordinator-policy rule \
#   --worker-policy rule \
#   --episode-model surrogate \
#   --victim-model-hint auto \
#   --probe-steps 2 \
#   --rule-max-snipers 3 \
#   --target-models neumf,lightgcn \
#   --transfer-candidate-set cluster \
#   --transfer-split-by-target-model \
#   --target-item-vocab observed \
#   --target-positive-threshold 1.0 \
#   --target-explicit-negative-threshold 0 \
#   --output outputs/experiments/exp_ml_20mx16x32_split_rule_observed_synth.json

# ml-latest NeuMF (latest successful run with positive delta): exp_ml_latest_neumf_sniper_allitems_steps12.json
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest \
  --max-interactions 20000 \
  --target-item-id 196 \
  --target-keyword horror \
  --num-steps 12 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model surrogate \
  --victim-model-hint auto \
  --probe-steps 2 \
  --rule-max-snipers 2 \
  --target-models neumf,lightgcn \
  --transfer-candidate-set all_items \
  --target-item-vocab observed \
  --target-explicit-negative-threshold 0 \
  --clone-segment-users-to-agents \
  --no-stop-on-goal \
  --output outputs/experiments/exp_ml_latest_neumf_sniper_allitems_steps12.json

# ---- Model sweep (traditional + neural + graph) on ml-latest-small

# Traditional + neural CF models: exp_models_traditional_neural.json
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 10000 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 8 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model surrogate \
  --victim-model-hint auto \
  --probe-steps 2 \
  --rule-max-snipers 2 \
  --target-models mf,gmf,bprmf,pmf,wmf,nmf,svdpp,neumf,ncf,iautorec,uautorec,cdae,itemknn \
  --transfer-candidate-set cluster \
  --target-item-vocab observed \
  --target-epochs 1 \
  --target-embedding-dim 16 \
  --target-batch-size 512 \
  --target-num-negatives 2 \
  --target-explicit-negative-threshold 0 \
  --output outputs/experiments/exp_models_traditional_neural.json

# Graph CF models: exp_models_graph.json
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 10000 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 8 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model surrogate \
  --victim-model-hint auto \
  --probe-steps 2 \
  --rule-max-snipers 2 \
  --target-models lightgcn,ngcf,gcmc \
  --transfer-candidate-set cluster \
  --target-item-vocab observed \
  --target-epochs 1 \
  --target-embedding-dim 16 \
  --target-batch-size 512 \
  --target-num-negatives 2 \
  --target-explicit-negative-threshold 0 \
  --output outputs/experiments/exp_models_graph.json

# Test Netflix data
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset netflix \
  --max-interactions 600000 \
  --target-item-id 1939 \
  --target-keyword horror \
  --num-steps 12 \
  --goal-rank 10 \
  --no-stop-on-goal \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model surrogate \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 2 \
  --no-profiler-probe-on-stall \
  --target-models neumf,lightgcn \
  --transfer-candidate-set cluster \
  --metrics-k 10,50,100 \
  --target-epochs 2 \
  --target-embedding-dim 32 \
  --target-batch-size 2048 \
  --target-num-negatives 4 \
  --target-device cuda \
  --output outputs/experiments/exp_netflix_transfer_rule_1939_mf.json

# ---- Dense-profiler experiments: can fake users attack LightGCN without real users? ----
# Research question: does building denser fake-user profiles (more items, cluster-focused)
# before firing the sniper give cold-start fake users enough graph connectivity to attack
# LightGCN effectively without cloning or reusing real user IDs?

# Exp 1 — Baseline (fake users, 3 profiler actions, no cluster pool, sniper-only extraction)
# Expected: catastrophic rank drop (degree inflation dominates cold-start fake users)
# Result: lightgcn rank_delta=-2141, ngcf rank_delta=-1008
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
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --output outputs/experiments/exp_dense_profiler_baseline.json

# Exp 2 — Dense cluster profiler, sniper-only extraction
# 15 profiler + 10 camouflaguer actions from cluster items, but ONLY sniper rows injected.
# Tests whether episode-phase connectivity helps when training data has none.
# Result: lightgcn rank_delta=-2107 (negligible improvement; sniper edges still dominate)
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
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --profiler-actions 15 \
  --camouflaguer-actions 10 \
  --profiler-use-cluster \
  --output outputs/experiments/exp_dense_profiler_lgcn_ngcf.json

# Exp 3 — Dense cluster profiler + ALL roles injected into training
# Fake users arrive at victim model retraining with many cluster-item edges.
# Result: lightgcn rank_delta=-1882 (slightly less bad but still catastrophic).
# Root cause: cluster-item ratings boost competitors; degree growth dilutes target edge.
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
  --transfer-attack-roles sniper,camouflaguer,profiler \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --profiler-actions 15 \
  --camouflaguer-actions 10 \
  --profiler-use-cluster \
  --output outputs/experiments/exp_dense_profiler_allroles.json

# Exp 4 — Dense BENCHMARK profiler + ALL roles injected
# Profiler uses popular non-cluster items (no competitor boosting from profiler).
# Camouflageur still uses cluster items. Tests whether avoiding cluster profiler helps.
# Result: lightgcn rank_delta=-1614 (best fake-user result but still very negative).
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
  --transfer-attack-roles sniper,camouflaguer,profiler \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --profiler-actions 15 \
  --camouflaguer-actions 10 \
  --output outputs/experiments/exp_dense_profiler_benchmark_allroles.json

# Exp 5 — Warmup (6 steps dense cluster profiler) then snipers, all roles injected
# Uses --sniper-start-step 6: no snipers for first 6 steps, then 3 snipers for steps 6-16.
# Tests whether sequencing warmup BEFORE snipers creates better graph connectivity.
# Result: lightgcn rank_delta=-1611 (near-identical to Exp 4; warmup ordering doesn't help).
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 10000 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 16 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper,camouflaguer,profiler \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --sniper-start-step 6 \
  --profiler-actions 15 \
  --camouflaguer-actions 10 \
  --profiler-use-cluster \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --output outputs/experiments/exp_dense_warmup6_then_sniper.json

# Exp 6 — Warmup (6 steps dense benchmark profiler) then snipers, profiler+sniper only
# Avoids camouflageur-induced competitor boosting entirely.
# Result: lightgcn rank_delta=-1973 (worse than Exp 4; fewer total interactions hurts).
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 10000 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 16 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper,profiler \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --sniper-start-step 6 \
  --profiler-actions 15 \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --output outputs/experiments/exp_dense_warmup6_profiler_sniper_only.json

# ---- Clone real-user profiles onto fake agents: LightGCN/NGCF offline transfer ----
# Strategy: copy full interaction history of a real segment user onto each fake agent_N ID.
# The victim model retrains with these cloned histories, giving fake agents real graph
# connectivity (established node degree, embedding proximity to target neighbourhood).
# All experiments use the same settings as the dense-profiler baseline for fair comparison.

# Clone Exp 1 — 4 agents, 12 steps (direct comparison baseline)
# Result: lightgcn +133 (norm +0.0137), ngcf +98 (norm +0.0101) — POSITIVE for both.
# Contrast with fake-user baseline: lightgcn -2141.  Clone flips sign completely.
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
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --clone-segment-users-to-agents \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --output outputs/experiments/exp_clone_fresh_4a_12s.json

# Clone Exp 2 — 4 agents, 24 steps (2x more sniper steps)
# Result: lightgcn +282 (norm +0.0289), ngcf +130 — delta doubles with 2x steps.
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 10000 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 24 \
  --goal-rank 3 \
  --num-agents 4 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 3 \
  --clone-segment-users-to-agents \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --output outputs/experiments/exp_clone_fresh_4a_24s.json

# Clone Exp 3 — 8 agents, 24 steps (more agents; defence rejection limits actual interactions)
# Result: lightgcn +37, ngcf -23 — defence system flags many snipers; fewer accepted.
# Note: only 47 sniper interactions accepted despite 8 agents x 24 steps budget.
PYTHONPATH=src python -m agas.cli run-transfer \
  --transfer-mode option-a \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 10000 \
  --target-item-id 1215 \
  --target-keyword horror \
  --num-steps 24 \
  --goal-rank 3 \
  --num-agents 8 \
  --transfer-attack-roles sniper \
  --coordinator-policy rule \
  --worker-policy rule \
  --episode-model lightgcn \
  --victim-model-hint mf \
  --probe-steps 0 \
  --rule-max-snipers 6 \
  --clone-segment-users-to-agents \
  --target-models lightgcn,ngcf \
  --transfer-candidate-set all_items \
  --target-epochs 3 \
  --target-embedding-dim 32 \
  --target-batch-size 512 \
  --no-stop-on-goal \
  --output outputs/experiments/exp_clone_fresh_8a_24s.json