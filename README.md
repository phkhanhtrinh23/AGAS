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
