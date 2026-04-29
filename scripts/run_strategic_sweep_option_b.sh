#!/bin/bash
# Option B — extend multi-seed sweep to n=30 with the undertrained target.
# Resumable: skips any (cell, seed) with an existing output JSON.
# rule_baseline vs v3_full only.

set -e
cd /mnt/win_d/agent_attack_rs
mkdir -p outputs/multiseed_sweep

run_one() {
  local seed=$1
  local tag=$2
  shift 2
  local out=outputs/multiseed_sweep/${tag}_seed${seed}.json
  local log=outputs/multiseed_sweep/${tag}_seed${seed}.log
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
    --profiler-bridge-method auto \
    --profiler-bridge-auto-threshold 20 \
    --profile-validator \
    --profile-validator-threshold 0.7 \
    --seed ${seed} \
    "$@" \
    --output ${out} > ${log} 2>&1
  tail -1 ${log}
}

for seed in 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71; do
  run_one ${seed} rule_baseline --coordinator-policy rule
  run_one ${seed} v3_full       --coordinator-policy strategic
done
