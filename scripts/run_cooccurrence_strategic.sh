#!/bin/bash
# Co-occurrence-led strategic sweep against LightGCN.
# Forces profiler-bridge-method=cooccurrence so the bridge-graph attack
# path (which the new STRATEGY_COOCCURRENCE_BRIDGING also tunes) is the
# primary lever. Reports HR@50 / NDCG@50.
#
# Cells:
#   rule_baseline    — RuleBasedCoordinatorPolicy (no strategies)
#   strategic_full   — All 12 strategies enabled (default)
#   strategic_no_cooc — All strategies EXCEPT cooccurrence-bridging (ablation)

set -e
mkdir -p /mnt/win_d/agent_attack_rs/outputs/cooccurrence_strategic
cd /mnt/win_d/agent_attack_rs

run_one() {
  local tag=$1
  shift
  local out=outputs/cooccurrence_strategic/${tag}.json
  local log=outputs/cooccurrence_strategic/${tag}.log
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
    --profiler-bridge-method cooccurrence \
    --profile-validator \
    --profile-validator-threshold 0.7 \
    --metrics-k 50 \
    --seed 42 \
    "$@" \
    --output ${out} > ${log} 2>&1
  tail -1 ${log}
}

run_one rule_baseline      --coordinator-policy rule
run_one strategic_full     --coordinator-policy strategic
run_one strategic_no_cooc  --coordinator-policy strategic --strategic-disable-cooccurrence-bridging
