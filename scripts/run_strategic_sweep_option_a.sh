#!/bin/bash
# Option A — well-trained LightGCN target (30 epochs, 64-dim), n=5 seeds.
# rule_baseline vs v3_full only.

set -e
mkdir -p /mnt/win_d/agent_attack_rs/outputs/multiseed_welltrained
cd /mnt/win_d/agent_attack_rs

run_one() {
  local seed=$1
  local tag=$2
  shift 2
  local out=outputs/multiseed_welltrained/${tag}_seed${seed}.json
  local log=outputs/multiseed_welltrained/${tag}_seed${seed}.log
  echo "=== ${tag} seed=${seed} (well-trained) ==="
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
    --target-epochs 30 \
    --target-embedding-dim 64 \
    --target-batch-size 512 \
    --no-stop-on-goal \
    --profiler-actions 3 \
    --camouflaguer-actions 3 \
    --profiler-bridge-method auto \
    --profiler-bridge-auto-threshold 20 \
    --profile-validator \
    --profile-validator-threshold 0.7 \
    --seed ${seed} \
    "$@" \
    --output ${out} > ${log} 2>&1
  tail -1 ${log}
}

for seed in 42 43 44 45 46; do
  run_one ${seed} rule_baseline --coordinator-policy rule
  run_one ${seed} v3_full       --coordinator-policy strategic
done
