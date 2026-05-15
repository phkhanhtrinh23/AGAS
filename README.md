# AGAS: Agentic Group Attack System for Recommender Systems

This is the official code to the paper: "An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems".

This paper introduces AGAS. AGAS is a black-box,
LLM-driven shilling attack against collaborative-filtering recommenders. One
**Coordinator** orchestrates a pool of fake-user **workers** over a sequence
of rounds. In each round, the Coordinator picks one of eight strategies
and assigns a role to every worker. Workers then decide which items to rate
using their own ReAct-style reasoning loop.

---

## 1. Repo overview

```
agent_attack_rs/
  bash/                           # Reviewer-friendly shell scripts (RQ1–RQ5)
  prompts/                        # Coordinator + per-role prompt templates
  scripts/run_agas.py             # Single entry point (--dataset --victim ...)
  src/agas/
    roles.py                      # Role enum {PR, SN, CA, IN} (paper symbols)
    signals.py                    # WorkerSignals (τ, γ, φ) + EnvSignals (ρ, Δρ, η, ξ, a)
    strategies.py                 # 8-strategy enum
    agents/                       # Coordinator + Worker policies
    simulation/                   # Episode runner (= AGAS algorithm outer loop)
    llm/                          # OpenAI / Ollama
    recsys/                       # Surrogate + 11-victim backends
    data/                         # Dataset loaders + preprocessing pipeline
  tests/                          # Pytest suite
```

## 2. Method recap

AGAS instantiates four worker roles (paper symbols in `roles.py`):

| Symbol | Long name      | What it does                                                            |
|--------|----------------|-------------------------------------------------------------------------|
| `PR`   | Profiler       | Safe filler-item ratings to probe the platform and build bridge pools.  |
| `SN`   | Sniper         | Payload role; direct target push or bridge-item promotion.              |
| `CA`   | Camouflageur   | Stealth role; rebuilds trust with benign-looking activity.              |
| `IN`   | Inactive       | No action this round (cool-down or quarantine).                         |

Each round the Coordinator chooses exactly one of eight strategies from
`strategies.py` (see `method_strategies.tex`):

1. **Victim Probe** (`S1_VICTIM_PROBE`)
2. **Bridge Building** (`S2_BRIDGE_BUILDING`, graph victims only)
3. **Warm-up** (`S3_WARM_UP`)
4. **First Push** (`S4_FIRST_PUSH`)
5. **Silent Slowdown** (`S5_SILENT_SLOWDOWN`)
6. **Profile Cleanup** (`S6_PROFILE_CLEANUP`)
7. **Safe Replacement** (`S7_SAFE_REPLACEMENT`)
8. **Main Attack** (`S8_MAIN_ATTACK`)

The Coordinator drives those decisions from two signal groups (`signals.py`):

* **Worker signals** `τ_{t,w}, γ_{t,w}, φ_{t,w}` — trust, risk, and a
  structural validator. Update equations match `method_coordinator.tex`
  exactly (`eq:trust_update`, `eq:risk_update`, `eq:risk_decay`,
  `eq:profile_validator`).
* **Environment signals** `ρ^{(t)}, Δρ^{(t)}, η_t, ξ_t = (q_t, s_t), a_t` —
  rank, rank-movement, acceptance rate, suppression signal, alert flag. The
  suspicion score `q_t` is the 0.2-weighted sum of the five normalised
  terms `(d̂_t, δ̂_t, m̂_t, ŝ_t, g_t)` from `eq:round_suppression_terms` and
  `eq:round_suppression_score`.

### Round loop (ASCII)

```
                ┌─────────────────────────────────────────────────┐
   t=0…T-1 ──►  │ 1. Observe ρ^{(t)}, update memory m_t           │
                │ 2. Update τ, γ, φ, η, ξ, a                       │
                │ 3. Coordinator picks Strategy ∈ {S1…S8}          │
                │    and assigns Role ∈ {PR, SN, CA, IN} per worker│
                │ 4. Workers act (filler / bridge / target items)  │
                │ 5. Validate + accept actions → ΔR̃^{(t+1)}        │
                │ 6. Refit / query victim → ρ^{(t+1)}              │
                └─────────────────────────────────────────────────┘
                          │
                          ▼
                   t* = argmin_t ρ^{(t)}, return R* = [R ; R̃^{(≤t*)}]
```

The outer loop is implemented in `src/agas/simulation/episode.py` and mirrors
`algorithms/agas_end_to_end.tex`.

## 3. Install

```bash
pip install -e .
# If you want to use the deep-learning victim models (LightGCN, NeuMF, …)
pip install -e '.[targets]'
```

Required environment variables:

| Variable           | Purpose                                                       | Default                  |
|--------------------|---------------------------------------------------------------|--------------------------|
| `OPENAI_API_KEY`   | OpenAI Responses API key for the Coordinator / worker LLMs.   | *(unset → fallback)*     |
| `OPENAI_MODEL`     | Model name passed to OpenAI.                                  | `gpt-5.1`                |

## 4. Datasets

The paper evaluates on six public CF benchmarks (see `experiment.tex`):

| Short name  | Source                                       | Place raw files in       |
|-------------|----------------------------------------------|--------------------------|
| ML-100K     | MovieLens 100K                               | `data/ml-100k/`          |
| ML-1M       | MovieLens 1M                                 | `data/ml-1m/`            |
| Genome 2021 | MovieLens Tag Genome 2021                    | `data/genome2021/`       |
| Netflix     | Netflix Prize                                | `data/netflix/`          |
| Douban      | Douban Movie                                 | `data/douban/`           |
| Amazon      | Amazon Reviews 2018                          | `data/amazon/`           |

After dropping the raw downloads into `data/<dataset>/`, run:

```bash
python scripts/preprocess_all.py --data-root data --output-root processed
```

Each dataset is rewritten into canonical `interactions.csv` + `items.csv`
files under `processed/<dataset>/`. The smoke tests use the much smaller
`ml-latest-small` sample that ships with MovieLens.

## 5. How to run

`scripts/run_agas.py` is the single entry point. The seven flags required by
the paper protocol are:

```
python scripts/run_agas.py \
    --dataset ml-100k \
    --victim lightgcn \
    --rounds 18 \
    --seed 42 \
    --n_workers 8 \
    --budget 0.01 \
    --out outputs/agas_ml100k_lightgcn_seed42.json
```

| Flag           | Meaning                                                                                          |
|----------------|--------------------------------------------------------------------------------------------------|
| `--dataset`    | One of `ml-100k`, `ml-1m`, `genome2021`, `netflix`, `douban`, `amazon`, `ml-latest-small`.       |
| `--victim`     | One of the 11 victims from `experiment.tex` (`mf`, `bpr`, `neumf`, `gmf`, `ncf`, `ngcf`, `lightgcn`, `simgcl`, `xsimgcl`, `egcf`, `lightccf`). |
| `--rounds`     | Number of AGAS rounds `T` (paper default `18`).                                                  |
| `--seed`       | Random seed (paper averages over five).                                                          |
| `--n_workers`  | Fake-user pool size `|U_f|`.                                                                     |
| `--budget`     | Per-user interaction budget `L`, expressed as a fraction of `|U|`.                                |
| `--out`        | Output JSON path for the full episode trace and summary metrics.                                 |

### Reproducing each research question on a tiny ML-100K sample

```bash
bash bash/run_performance.sh          # RQ1
bash bash/run_stealth_and_detect.sh   # RQ2 + RQ3
bash bash/run_ablation.sh             # RQ4
bash bash/run_efficiency.sh           # RQ5
```

Each script prints which experiment it is running, the output path, and a
note pointing to the corresponding figure/table in the paper.

Example expected console output (truncated):

```
[RQ1] Performance benchmark — tiny ML-100K sample
[RQ1] Output directory: agent_attack_rs/outputs/rq1_performance
[RQ1] Map outputs to tables/benchmark_unpopular.tex
=== ml-latest-small / lightgcn / seed=42 ===
Round  0 | ρ = 1834  Δρ = +0     strategy = S1_VICTIM_PROBE
Round  1 | ρ = 1450  Δρ = +384   strategy = S1_VICTIM_PROBE
Round  2 | ρ = 1212  Δρ = +238   strategy = S3_WARM_UP
…
[RQ1] Done. Expected outputs in agent_attack_rs/outputs/rq1_performance
```

## 6. Protocol and prompts

The Coordinator and every worker are LLM agents that read from
`prompts/<role>/{system,user}.txt`. The full protocol — what the
Coordinator interpolates into its prompt, which variables the workers see,
and the expected JSON output schemas — is in
[`prompts/README.md`](prompts/README.md).

Brief summary:

* Coordinator → JSON with `{"strategy": "S?_…", "assignments": {agent: ROLE}}`.
* Workers → JSON with `{"actions": [{"item_id": …, "rating": …, "reason": …}]}`.
* Allowed item pools are role-dependent (filler pool, bridge pool, target +
  competitors) and enforced by the host code.

## 7. Reproducing tables and figures

| Research question | Script                                | Paper artifacts                                                                                                                |
|-------------------|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| RQ1 (Performance) | `bash/run_performance.sh`              | `tables/benchmark_unpopular.tex`, `figures/benchmark_popularity_regime_summary.png`                                            |
| RQ2 (Stealth)     | `bash/run_stealth_and_detect.sh`       | `figures/benchmark_fig7_tsne_distance_users_target.png`, `figures/stealth_tsne_clean_vs_fake.png`                              |
| RQ3 (Detectors)   | `bash/run_stealth_and_detect.sh`       | `tables/detection_mf.tex`, `figures/detect_rec10_by_victim.png`                                                                |
| RQ4 (Ablation)    | `bash/run_ablation.sh`                 | `figures/ablation_summary.png`, `figures/ablation_strategy_heatmap.png`, `figures/ablation_size_heatmap.png`, `tables/ablation_backbones.tex` |
| RQ5 (Efficiency)  | `bash/run_efficiency.sh`               | `figures/efficiency_rounds_tradeoff.png`, `figures/efficiency_method_time_compare.png`, `figures/efficiency_token_topk.png`, `figures/efficiency_agents_tradeoff.png` |

## 8. License

Released for academic research use. The repo carries no license file by
default; we suggest **MIT** for downstream reuse.
