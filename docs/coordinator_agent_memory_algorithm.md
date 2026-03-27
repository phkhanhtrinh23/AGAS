# Coordinator Per-Agent Memory Algorithm (End-to-End)

This document explains the full algorithm behind the optional *per-agent memory to coordinator* feature
(`--coordinator-agent-memory`) and how it affects role assignment via sniper lockouts.

The goal is to let the coordinator reason about *which specific agents* were suppressed recently
(dropped or discounted actions) and adapt assignments accordingly.

## What “Memory” Means Here

There are two different rolling summaries in the simulation:

- **Global trajectory summary** (`trajectory_summary`): recent steps’ rank movement and public signals for the whole episode.
- **Per-agent memory summary** (`agent_memory_by_agent[agent_id]`): recent steps’ *that agent’s* role, actions, and outcomes.

Only the coordinator gets `agent_memory_by_agent`, and only when `--coordinator-agent-memory` is enabled.

## Key Data Structures

### Episode History

Each executed step appends an entry to `history` that includes:

- `reports`: what each worker attempted (actions it proposed)
- `feedback.outcomes`: what the environment did to each action (accepted/dropped/discounted)
- `feedback.defense_report.public_signals_by_agent`: the public black-box signals (drop rate, discount rate, streak, etc.)

This history is the “source of truth” used to build both global and per-agent summaries.

### Per-Agent Memory Summary Format

For each agent and each recent step, the summary includes:

- `step`
- `role`
- `actions`: `[{"item_id": "...", "rating": ...}, ...]`
- `accepted_actions`, `dropped_actions`, `discounted_actions`
- `mean_discount`
- `target_rank`, `target_rank_delta`
- `public_signal` (the agent’s public black-box signal for that step)

This is built from the existing history; it does not require hidden defense state.

## End-to-End Algorithm

This is the full “beginning to end” logic for one episode run.

### 0. Initialization

1. Load preprocessed interactions and items.
2. Fit a recommender backend (surrogate or target model).
3. Initialize `AGASEnvironment` with defense configuration.
4. Create:
   - `Coordinator` (policy + runtime guardrails)
   - worker pool (`agent_1`, `agent_2`, ...)
5. Initialize empty `history`.

### 1. For Each Step `t = 0..T-1`

#### 1.1 Build Summaries From Past Steps (Memory Creation)

Using the current `history` (steps `0..t-1`):

1. Build global summary:
   - `trajectory_summary = summarize_last_k_steps(history, k=trajectory_window)`
2. Optionally build per-agent summaries (only if `--coordinator-agent-memory`):
   - For each agent `a`:
     - `agent_memory_by_agent[a] = summarize_last_k_steps_for_agent(history, a, k=trajectory_window)`

Both functions are pure transforms over `history`.

#### 1.2 Build Coordinator Observation

The environment produces a `CoordinatorObservation` containing:

- rank info: `target_rank`, `total_candidates`, `target_rank_delta`
- per-agent state: `trust_by_agent`, `risk_by_agent`
- public signals: `signals_by_agent`
- global memory: `trajectory_summary`
- optional per-agent memory: `agent_memory_by_agent`

#### 1.3 Policy Proposes Roles

The coordinator’s *policy* proposes a role for each agent:

- Rule-based policy: deterministic schedule + evasive fallback
- LLM policy: prompt includes the observation JSON and expects a JSON mapping `{agent_id: role, ...}`

This stage does not enforce runtime safety constraints yet.

#### 1.4 Coordinator Runtime Guardrails (Where Memory Changes Decisions)

After the policy proposes roles, the `Coordinator` applies runtime logic that may override assignments.

##### 1.4.1 Update Sniper Lockouts

Lockouts are **per-agent** and only apply to agents who were **snipers in the previous step**.

For each agent `a` that was previously assigned `sniper`:

1. Read public indicators (always available):
   - current suspicion score (`suspected_filtering_score`)
   - suppression streak (`suppression_streak`)
   - explicit alerts (only visible if defense state is exposed; usually hidden)
2. If per-agent memory is available:
   - Look at the agent’s own memory entries for recent steps (e.g. last 3 entries)
   - Count “suppression events”:
     - `dropped_actions > 0` counts as an event
     - `discounted_actions > 0` counts as an event
   - If `events >= --sniper-lock-memory-events`, treat that as a lock trigger
3. If *any* lock trigger fires:
   - set `sniper_lockouts[a] = max(existing, --sniper-lock-steps)`

##### 1.4.2 Apply Lockouts

For each agent `a` with `sniper_lockouts[a] > 0`:

- override the assigned role to `inactive` (or `camouflaguer`, configurable)
- keep a rationale like: “Locked after suppression signals; hold for N more steps.”

##### 1.4.3 Ensure Profiler Probes

Separately, the coordinator may force at least one profiler when:

- periodic probe interval hits, or
- target rank stalls, or
- suspicion/streak passes thresholds

This is intended to simulate “probe after suppression” behavior even in black-box mode.

#### 1.5 Workers Act

Each worker receives:

- its assignment (role)
- its own per-agent rolling memory (already implemented for workers)
- candidate item pools

It returns a `WorkerActionReport` with a list of rating actions.

#### 1.6 Environment Executes (Defense + Online Update)

The environment applies each action:

- may drop actions (lockdown/stealth filtering)
- may discount extreme actions for low-trust users (influence discounting)
- updates trust/risk
- appends accepted interactions and refits the recommender (online update)
- recomputes target rank

It also produces `DefenseReport`:

- hidden truth (for analysis)
- public black-box signals per agent (for coordinator)

#### 1.7 Append Step to History

The runner appends a structured step record to `history`, including:

- `observation` (what coordinator saw)
- `assignments` (after runtime adjustments)
- `reports` (what workers did)
- `feedback` (what environment did)
- `coordinator_runtime_trace` (lockouts/probes)

### 2. Episode Ends

The runner returns `EpisodeResult`:

- full `history`
- final rank, best rank
- final worker states

## Where This Is Implemented

- Per-agent memory summaries for the coordinator:
  - `src/agas/simulation/episode.py` (`_agent_trajectory_summary`, `coordinator_agent_memory`)
  - `src/agas/simulation/environment.py` (`observation(... agent_memory_by_agent=...)`)
  - `src/agas/agents/messages.py` (`CoordinatorObservation.agent_memory_by_agent`)
- Memory-driven lockouts:
  - `src/agas/agents/coordinator.py` (`_update_sniper_lockouts`)
- CLI switch:
  - `src/agas/cli.py` (`--coordinator-agent-memory`, `--sniper-lock-memory-events`)

## Practical Notes

- This feature is **not realistic black-box** if you interpret each fake account as isolated.
  It assumes a centralized attacker can collect each agent’s own outcomes and share them upward.
- It *is* realistic if you interpret “agents” as controlled accounts in the same botnet controlled by one operator.
