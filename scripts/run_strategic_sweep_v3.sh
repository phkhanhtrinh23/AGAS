#!/bin/bash
set -e
mkdir -p /mnt/win_d/agent_attack_rs/outputs/strategic_sweep_v3
cd /mnt/win_d/agent_attack_rs

run_one() {
  local tag=$1
  shift
  local out=outputs/strategic_sweep_v3/${tag}.json
  local log=outputs/strategic_sweep_v3/${tag}.log
  echo "=== ${tag} ==="
  PYTHONPATH=src python -m agas.cli run-transfer \
    --transfer-mode option-a \
    --processed-root processed \
    --dataset ml-latest-small \
    --max-interactions 10000 \
    --target-item-id 593 \
    --target-keyword horror \
    --num-steps 10 \
    --goal-rank 3 \
    --num-agents 8 \
    --transfer-attack-roles sniper,profiler \
    --worker-policy rule \
    --episode-model lightgcn \
    --victim-model-hint mf \
    --probe-steps 0 \
    --rule-max-snipers 3 \
    --sniper-start-step 2 \
    --target-models lightgcn \
    --transfer-candidate-set all_items \
    --target-epochs 3 \
    --target-embedding-dim 32 \
    --target-batch-size 512 \
    --no-stop-on-goal \
    --profiler-actions 3 \
    --camouflaguer-actions 3 \
    --profiler-bridge-method auto \
    --profiler-bridge-auto-threshold 20 \
    --profile-validator \
    --profile-validator-threshold 0.7 \
    "$@" \
    --output ${out} > ${log} 2>&1
  tail -2 ${log}
}

# Control.
run_one rule_baseline       --coordinator-policy rule

# v2 strategies only (stealth + reprobe), with v3 features explicitly off.
run_one v2_only             --coordinator-policy strategic \
                              --strategic-disable-budget --strategic-disable-diversity

# Single-strategy isolations (other strategies disabled).
run_one budget_only         --coordinator-policy strategic \
                              --strategic-disable-stealth --strategic-disable-reprobe \
                              --strategic-disable-diversity
run_one diversity_only      --coordinator-policy strategic \
                              --strategic-disable-stealth --strategic-disable-reprobe \
                              --strategic-disable-budget

# Composed cells (build up additively).
run_one v2_plus_budget      --coordinator-policy strategic --strategic-disable-diversity
run_one v2_plus_diversity   --coordinator-policy strategic --strategic-disable-budget
run_one v3_full             --coordinator-policy strategic
