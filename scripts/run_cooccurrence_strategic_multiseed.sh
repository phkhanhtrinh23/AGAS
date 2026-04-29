#!/bin/bash
# Multi-seed (42-46) co-occurrence-led strategic sweep against LightGCN.
# Reports HR/NDCG @ K=10,20,50,100. Resumable: skips cached outputs.
#
# Cells: rule_baseline | strategic_full | strategic_no_cooc

set -e
mkdir -p /mnt/win_d/agent_attack_rs/outputs/cooccurrence_multiseed
cd /mnt/win_d/agent_attack_rs

run_one() {
  local seed=$1
  local tag=$2
  shift 2
  local out=outputs/cooccurrence_multiseed/${tag}_seed${seed}.json
  local log=outputs/cooccurrence_multiseed/${tag}_seed${seed}.log
  if [[ -f "${out}" ]]; then
    echo "=== ${tag} seed=${seed} (cached) ==="
    return
  fi
  echo "=== ${tag} seed=${seed} ==="
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
    --metrics-k 10,20,50,100 \
    --seed ${seed} \
    "$@" \
    --output ${out} > ${log} 2>&1
  tail -1 ${log}
}

for seed in 42 43 44 45 46; do
  run_one ${seed} rule_baseline      --coordinator-policy rule
  run_one ${seed} strategic_full     --coordinator-policy strategic
  run_one ${seed} strategic_no_cooc  --coordinator-policy strategic --strategic-disable-cooccurrence-bridging
done
