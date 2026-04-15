# AGAS: Agentic-empowered Group Shilling Attacks in Recommender Systems

This repository implements an end-to-end research framework for **AGAS** (Agentic-empowered Group Shilling Attacks in Recommender Systems), including:

- canonical preprocessing for all datasets under `data/`
- a lightweight surrogate recommender model
- a multi-agent AGAS simulator with role switching and defense-aware feedback
- optional LLM-based coordinator policy using **OpenAI** or **Ollama**
- optional LLM-based worker policies with editable per-role prompts
- a black-box defense monitor that separates hidden defense state from public signals

This code is provided for academic simulation and defense research only.

## 1. Project Structure

```text
.
├── configs/
│   └── default.yaml
├── data/                         # Provided datasets
├── prompts/                      # Editable prompt templates for coordinator/workers
├── scripts/
│   ├── preprocess_all.py
│   ├── train_surrogate.py
│   └── run_agas.py
├── src/agas/
│   ├── agents/
│   │   ├── coordinator.py
│   │   ├── defender.py
│   │   ├── messages.py
│   │   └── worker.py
│   ├── data/
│   │   ├── discovery.py
│   │   ├── pipeline.py
│   │   └── loaders/
│   ├── llm/
│   │   ├── prompt_store.py
│   │   └── providers.py
│   ├── recsys/
│   │   ├── prepared_data.py
│   │   └── surrogate.py
│   ├── simulation/
│   │   ├── environment.py
│   │   └── episode.py
│   └── cli.py
├── tests/
├── pyproject.toml
└── README.md
```

## 2. Dataset Audit and Preprocessing Coverage

### Documented folders read

- `data/ml-latest-small/README.txt`
- `data/ml-latest/README.txt`
- `data/ml-32m/README.txt`
- `data/ml-20mx16x32/ml-20mx16x32-README.txt`
- `data/genome_2021/readme.txt`
- `data/fake-review-detection/README.md`
- `data/marketBias/README.md`

### Undocumented folders sampled

- `data/amazon_review/` (`rating-Alabama.csv`, `review-Alabama_10.json`)
- `data/music_in_car/` (`Data_InCarMusic.xlsx`, all sheets)

### Implemented adapters (`src/agas/data/loaders/`)

- `movielens.py`: `ml-latest-small`, `ml-latest`, `ml-32m`
- `ml20m_fractal.py`: `ml-20mx16x32`
- `genome2021.py`: `genome_2021/raw/ratings.json` and metadata
- `amazon_review.py`
- `music_in_car.py`
- `fake_review.py`
- `market_bias.py`

All adapters export a canonical format:

- interactions: `dataset,user_id,item_id,rating,timestamp,split,context_json,source`
- items: `dataset,item_id,title,genres,category,metadata_json,source`

Outputs are written to `processed/<dataset>/interactions.csv` and `processed/<dataset>/items.csv`.

## 3. Setup

```bash
python -m venv agent_attack_venv
source agent_attack_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional target-model dependencies (NeuMF/LightGCN):

```bash
python -m pip install -e ".[targets]"
```

## 3a. Technical Detail
<details>

This repo is a Python + CLI research framework (`agas`) with optional LLM-based agents and optional PyTorch target-model training. The values below reflect the **current environment where this README was last verified (2026-04-15)**.

### Platform / Hardware (this machine)

| Component | Value |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS (kernel 6.14.0-24-generic) |
| CPU | Intel(R) Core(TM) i7-14700 (28 logical CPUs, 20 cores) |
| RAM | 31 GiB (swap 8 GiB) |
| GPU | NVIDIA GeForce RTX 4090 (24,564 MiB), driver 575.64.03 |

### Software stack

| Layer | What AGAS uses |
| --- | --- |
| Python | Python 3.13.5 |
| Core data | `pandas`, `numpy` for canonical frames and sampling |
| Surrogate model | `scikit-learn` (`TruncatedSVD`) + popularity/bias fallback |
| Target models | `torch` (PyTorch 2.8.0+cu129; CUDA runtime 12.9) |
| LLM backends | OpenAI API or Ollama (`src/agas/llm/providers.py`) |
| Token counting (optional) | `tiktoken` (used for prompt/token audits) |

`nvcc` in this environment reports CUDA compilation tools 12.0 (not required unless building custom CUDA code).

### Datasets (raw + processed)

- **Raw datasets** live under `data/<dataset>/` (adapters in `src/agas/data/loaders/`).
- **Processed datasets** are exported to `processed/<dataset>/interactions.csv` and `processed/<dataset>/items.csv` with a canonical schema:
  - interactions: `dataset,user_id,item_id,rating,timestamp,split,context_json,source`
  - items: `dataset,item_id,title,genres,category,metadata_json,source`
- During iteration, preprocessing commonly caps export via `--max-rows-per-dataset 2000000` (see `processed/summary.csv` notes).

Processed dataset sizes in this repo (from `processed/summary.csv`):

| Dataset | Interactions | Users | Items |
| --- | ---:| ---:| ---:|
| `ml-latest-small` | 100,836 | 610 | 9,724 |
| `ml-latest` | 2,000,000 | 19,651 | 33,129 |
| `ml-32m` | 2,000,000 | 12,773 | 36,603 |
| `ml_20mx16x32` | 2,000,000 | 2,365 | 181,263 |
| `genome_2021` | 2,000,000 | 37,941 | 8,687 |
| `amazon_review` | 2,000,000 | 998,653 | 48,746 |

Dataset notes (high level):

- `ml-*`: MovieLens movie ratings (items include titles/genres). `ml-32m` is originally much larger; this repo’s processed export is capped for practicality unless you disable limits.
- `ml_20mx16x32`: MovieLens-20M-derived “fractal” split with a very large item universe (useful for stress-testing candidate pools).
- `genome_2021`: Movie tag genome / movie ratings style data (keyword metadata may be sparse; keyword-based segmentation can be harder).
- `amazon_review`: product review ratings (very large user base; useful for cold-start / sparse-history regimes).

### Recommender models (surrogate + targets)

AGAS separates the **episode recommender** (used for step-by-step feedback) from **target recommenders** (used for transfer evaluation).

- **Episode model** (`--episode-model`):
  - `surrogate` (default): lightweight SVD-based recommender in `src/agas/recsys/surrogate.py`, updated online each step.
  - optionally: `lightgcn`, `neumf`, `mf`, `svdpp`, `sequential`, etc. (minimal reference target implementations under `src/agas/recsys/targets/`).
- **Target models** (`--target-models`): one or more of the above PyTorch models used for offline transfer evaluation (Option A) and/or in-loop evaluation (Option B).

All recommenders implement a shared interface (`BaseTargetRecommender` / surrogate wrapper) used by the environment for:
- `fit()` / (episode) `append_interactions()`
- scoring + ranking a `target_item_id` against a **candidate set** (e.g. `--transfer-candidate-set cluster` vs `all_items`)

### Offline training losses (targets)

In Option A (offline transfer), target models are trained on clean vs. clean+attack and compared by rank shift.

- **MF / NeuMF / SVD++ (implicit pointwise)**: `BCEWithLogitsLoss` on positives plus **negative sampling** from *unseen* items (and optional explicit negatives if configured).
- **LightGCN (pairwise)**: BPR loss on triplets `(u, pos_i, neg_j)` with negative sampling.
- **Sequential**: first-order Markov transitions + popularity fallback (no gradient training).

### Approximate model size (parameters)

For embedding dimension `d`, a rough parameter count (excluding optimizer state) is:

- `LightGCN`: `(U + I) * d`
- `MF`: `(U + I) * d`
- `NeuMF`: `2 * (U + I) * d + MLP` (small MLP on top)
- `SVD++`: `(U + 2I) * d`

Example with `d=32` on `ml-latest-small` (`U=610`, `I=9,724`):

| Model | Params (approx) |
| --- | ---:|
| LightGCN / MF | 330,688 |
| NeuMF (embeddings only) | 661,376 |
| SVD++ (embeddings only) | 641,856 |

### LLM agents: prompts, token budgets, and communication

LLM policies are enabled by selecting `--coordinator-policy openai|ollama` and/or `--worker-policy openai|ollama`. The coordinator assigns roles; workers propose actions; the environment returns outcomes + black-box signals.

At the code level, components communicate via typed protocol messages in `src/agas/agents/messages.py` (e.g., `CoordinatorObservation` → `RoleAssignment` → `RatingAction` → `EnvironmentFeedback` / `AgentBlackBoxSignal`).

**Static prompt template token counts** (measured with `tiktoken` `o200k_base` on the checked-in prompt files; per-call totals are higher due to dynamic JSON observations):

| Prompt template | Tokens |
| --- | ---:|
| `prompts/coordinator/system.txt` | 580 |
| `prompts/coordinator/user.txt` | 153 |
| `prompts/worker_profiler/system.txt` | 90 |
| `prompts/worker_profiler/user.txt` | 61 |
| `prompts/worker_camouflaguer/system.txt` | 97 |
| `prompts/worker_camouflaguer/user.txt` | 58 |
| `prompts/worker_sniper/system.txt` | 95 |
| `prompts/worker_sniper/user.txt` | 60 |
| `prompts/worker_inactive/system.txt` | 60 |
| `prompts/worker_inactive/user.txt` | 22 |

Dynamic tokens come from the per-step observation payload (public signals, recent outcomes, candidate-set summaries, etc.) and scale with `--num-agents`, `--num-steps`, and memory settings such as `--coordinator-agent-memory`.

### Metrics reported

Transfer evaluation writes JSON with (at minimum):

- **Option A (offline target models)**:
  - `initial_rank`, `final_rank`, and `rank_delta = initial_rank - final_rank`
  - `normalized_rank_delta` (rank change normalized by candidate pool size)
  - `attack_interactions` (# accepted interactions injected into target training)
  - `hr_at_k` / `ndcg_at_k` (rank-derived top-K metrics for the **single target item**; configured by `--metrics-k`, default `10`)
- **Option B (in-loop target models)**:
  - `final_rank` and the best rank achieved over the episode trajectory
  - `hr_at_k` / `ndcg_at_k` for `initial` / `final` / `best`

Notes:

- HR@K and NDCG@K here are **derived from the reported 1-based rank of the target item** (not from a held-out test set with many relevant items).
  - `HR@K = 1[rank <= K]`
  - `NDCG@K = 1 / log2(rank + 1)` if `rank <= K`, else `0`
</details>

## 4. Preprocess Data

Run all available adapters:

```bash
agas preprocess --data-root data --output-root processed
```

Safe default is `--max-rows-per-dataset 2000000` to keep processing practical on large datasets.
Use full export if needed:

```bash
agas preprocess --max-rows-per-dataset none
```

Preprocess selected datasets:

```bash
agas preprocess --datasets ml-latest-small,genome_2021,music_in_car
```

A summary table is written to `processed/summary.csv`.

## 5. Surrogate Recommender

Implemented in `src/agas/recsys/surrogate.py`:

- lightweight matrix-factorization surrogate (`TruncatedSVD`)
- item popularity + bias fallback for robustness
- recommendation API: top-N item generation
- ranking API: target rank within a segment
- online update support (`append_interactions`) after attack actions

Train and save a surrogate snapshot:

```bash
agas train-surrogate \
  --processed-root processed \
  --dataset ml-latest-small \
  --max-interactions 500000 \
  --output outputs/surrogate.pkl
```

## 6. AGAS System Design

### Roles implemented

- `Profiler`: probes recommender acceptance using popular benchmark items
- `Camouflaguer`: builds trust in target-domain cluster and injects benign noise
- `Inactive`: no action (temporal reset to reduce velocity anomalies)
- `Sniper`: payload ratings (target max-score + competitor min-score)

Worker logic is in `src/agas/agents/worker.py`.

### Message protocol

Implemented in `src/agas/agents/messages.py`:

- `CoordinatorObservation`
- `RoleAssignment`
- `WorkerActionReport`
- `RatingAction`
- `ActionOutcome`
- `AgentBlackBoxSignal`
- `DefenseReport`
- `EnvironmentFeedback`
- generic envelope `ProtocolMessage`

### Coordinator

`src/agas/agents/coordinator.py` includes:

- `RuleBasedCoordinatorPolicy`: follows the described AGAS timeline
  - t=0 profiling
  - t=1 infiltration
  - t=2 snipe
  - t>=3 evasion/adaptive rotation
- `LLMCoordinatorPolicy`: optional LLM role assignment from structured black-box state

Coordinator prompt templates are loaded from:

- `prompts/coordinator/system.txt`
- `prompts/coordinator/user.txt`

### Defense-aware environment

`src/agas/simulation/environment.py` simulates:

- stealth filtering / lockdown behavior
- influence discounting on extreme low-trust ratings
- spike detector alerts for synchronized max-ratings
- trust/risk state transitions per agent
- target ranking feedback loop

### Black-box coordinator view

By default, the coordinator does not directly see hidden defense alerts or lockdown state.
Instead it receives black-box signals derived from:

- dropped actions
- discounted effective ratings
- suppression streaks
- weak target-rank movement

Hidden defense activity is still logged separately through `DefenseReport` for analysis.

### Defense monitor agent

### Suppression Handling (Sniper Lockouts + Profiler Probes)

Coordinator runtime guardrails now include:

- **Sniper lockouts**: when a sniper shows suppression signals, it is locked
  for `N` steps and forced into `inactive` (or `camouflaguer`).
- **Profiler probes after suppression**: when suppression streaks rise, the
  coordinator forces a profiler probe to check whether ratings are still
  being integrated.

CLI knobs (defaults shown):

- `--sniper-lock-steps 2`
- `--sniper-lock-suspicion 0.6`
- `--sniper-lock-suppression-streak 2`
- `--sniper-lock-role inactive`
- `--profiler-probe-suppression-streak 1`

Example run that triggers suppression and lockouts:

```bash
agas run-episode \
  --processed-root processed \
  --dataset ml-latest-small \
  --target-item-id 101 \
  --num-steps 8 \
  --coordinator-policy rule \
  --worker-policy rule \
  --spike-threshold 1 \
  --lockdown-drop-prob 0.7 \
  --sniper-lock-steps 2 \
  --sniper-lock-suppression-streak 1 \
  --output outputs/episode_lockout_example.json
```

The per-step JSON contains `coordinator_runtime_trace` with active lockouts.
Example from `outputs/episode_lockout_example.json`:

- step 3: `sniper_lockouts` shows `agent_2` and `agent_3` locked for 2 steps
- step 4: lockouts decay and `agent_4` enters lockout after suppression

### Group Attack Detection (Collusion Signals)

The defense monitor now computes **group-collusion signals** based on per-step
item overlap between agents. These signals appear in each agent’s
`signals_by_agent` as:

- `group_overlap`: max Jaccard overlap with any other agent’s items
- `group_suspicion`: overlap above threshold (optionally requires target item)

CLI knobs:

- `--group-overlap-threshold` (default `0.6`)
- `--group-target-required` (default `true`)
- `--group-weight` (default `0.2`)

These signals contribute to the public `suspected_filtering_score` and can
trigger lockouts or profiler probes.

Example run (group overlap threshold lowered to surface collusion):

```bash
agas run-episode \
  --processed-root processed \
  --dataset ml-latest-small \
  --target-item-id 101 \
  --num-steps 6 \
  --group-overlap-threshold 0.2 \
  --group-target-required \
  --output outputs/episode_group_detection.json
```

The output includes per-agent `group_overlap` and `group_suspicion` in
`history[*].feedback.defense_report.public_signals_by_agent`.

### Defense Quarantine (Temporary Bans)

The environment can optionally model temporary account quarantine (a defense-side action)
after certain internal detections (spike alerts or group collusion). While quarantined,
all actions from that agent are dropped with reason `Dropped: account quarantined by defense`.

CLI knobs:

- `--quarantine-steps` (default `0`, disabled)
- `--quarantine-on-spike-alert / --no-quarantine-on-spike-alert`
- `--quarantine-on-group-collusion / --no-quarantine-on-group-collusion`

Example:

```bash
agas run-episode \
  --processed-root processed \
  --dataset ml-latest-small \
  --target-item-id 101 \
  --num-steps 8 \
  --spike-threshold 1 \
  --group-overlap-threshold 0.2 \
  --group-target-required \
  --quarantine-steps 2 \
  --output outputs/quarantine_example.json
```

### Surrogate vs Target Models (Option A vs B)

- The **surrogate model** is the lightweight SVD-based recommender in
  `src/agas/recsys/surrogate.py`. It is updated **every step** during an
  episode (online refit).
- The **target models** (NeuMF, LightGCN) are only used in transfer evaluation:
  - **Option A (offline):** run the episode on the surrogate, then retrain
    target models once on clean vs. clean+attack.
  - **Option B (in-loop):** replace the surrogate with a target model and
    retrain it each step for live feedback.

Coordinator can optionally receive per-agent recent outcomes (richer than public
aggregate signals) via `--coordinator-agent-memory`.

Detailed algorithm: `docs/coordinator_agent_memory_algorithm.md`.

## Related Papers (Short Positioning)

- **AgentAttack: LLM Agents for Multi-Strategy Shilling Attacks (2026)**  
  LLM-driven, black-box shilling with multi-round feedback and adaptive strategy
  selection. It uses agent reasoning but does not model explicit multi-role
  coordination (Profiler/Camouflaguer/Sniper) or group-level defense signals.

- **AgentSA / LLM Agent-based Shilling Attack (WSDM’26)**  
  Low-knowledge/black-box LLM agents with profile, memory, and action modules
  (including review generation). Focuses on per-agent memory and human-like
  behavior rather than centralized role switching and coordinated team dynamics.

## 7. Transfer Evaluation (Option A + Option B)

The CLI includes a `run-transfer` command to evaluate transfer to real target models:

- **Option A (offline target models)**: run AGAS on the surrogate, extract accepted interactions, then
  retrain target models on clean vs. clean+attack and measure rank shift.
- **Option B (in-loop target models)**: replace the surrogate with NeuMF/LightGCN inside the loop and
  retrain per step.

Example (both options):

```bash
agas run-transfer \
  --processed-root processed \
  --dataset ml-latest-small \
  --target-item-id 101 \
  --target-keyword horror \
  --num-steps 6 \
  --transfer-mode both \
  --target-models neumf,lightgcn \
  --output outputs/transfer_result.json
```

To generate attack interactions using a heavier in-loop recommender while still
reporting offline transfer to multiple target models (Option A), set:

```bash
agas run-transfer --transfer-mode option-a --episode-model lightgcn --target-models neumf,lightgcn ...
```

Tune target model training as needed:

```bash
agas run-transfer \
  --target-epochs 3 \
  --target-embedding-dim 64 \
  --target-batch-size 1024 \
  --target-device cpu
```

`src/agas/agents/defender.py` implements `DefenseMonitorAgent`, which:

- records hidden defense interventions
- emits black-box public signals per worker
- logs whether the defense actually intervened during a step

### Worker prompts

Per-role worker prompts are stored under `prompts/`:

- `prompts/worker_profiler/`
- `prompts/worker_camouflaguer/`
- `prompts/worker_sniper/`
- `prompts/worker_inactive/`

These prompt files are loaded only when worker policy is set to `openai` or `ollama`.

## 7. Run AGAS Episode

Rule-based coordinator:

```bash
agas run-episode \
  --processed-root processed \
  --dataset ml-latest-small \
  --target-item-id 101 \
  --target-keyword horror \
  --num-agents 4 \
  --num-steps 4 \
  --coordinator-policy rule \
  --output outputs/episode_result.json
```

LLM coordinator with rule-based workers:

```bash
agas run-episode \
  --coordinator-policy openai \
  --worker-policy rule \
  --prompt-root prompts
```

LLM coordinator with LLM workers:

```bash
agas run-episode \
  --coordinator-policy openai \
  --worker-policy openai \
  --llm-model gpt-5-mini \
  --worker-llm-model gpt-5-mini \
  --prompt-root prompts
```

Expose hidden defense state directly to the coordinator (white-box mode):

```bash
agas run-episode --expose-defense-state
```

### OpenAI coordinator mode

```bash
export OPENAI_API_KEY=YOUR_KEY
agas run-episode \
  --coordinator-policy openai \
  --llm-model gpt-5-mini
```

### Ollama coordinator mode

Start Ollama locally (for example with `llama3.1`):

```bash
ollama serve
ollama pull llama3.1
```

Then run:

```bash
agas run-episode \
  --coordinator-policy ollama \
  --llm-model llama3.1 \
  --ollama-host http://localhost:11434
```

## 8. Reproducibility / Tests

Run smoke tests:

```bash
pytest -q
```

Current test status in this environment: **6 passed**.

## 9. Notes on Large Datasets

Datasets like `ml-latest`, `ml-32m`, and `ml-20mx16x32` are very large.
Use `--max-rows-per-dataset` / `--max-interactions` during iteration, then switch to full export for final experiments.

## 10. Key Entry Points

- CLI: `src/agas/cli.py`
- preprocessing pipeline: `src/agas/data/pipeline.py`
- surrogate model: `src/agas/recsys/surrogate.py`
- AGAS policies: `src/agas/agents/coordinator.py`, `src/agas/agents/worker.py`
- defense monitor: `src/agas/agents/defender.py`
- simulation runner: `src/agas/simulation/episode.py`
