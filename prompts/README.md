# Prompts protocol

Each agent role in AGAS uses two prompt files:

```
prompts/
  coordinator/
    system.txt           # standing instructions for the Coordinator
    user.txt             # per-round template, rendered with context variables
  worker_profiler/       # role PR
  worker_sniper/         # role SN
  worker_camouflaguer/   # role CA
  worker_inactive/       # role IN
```

## Loading

`agas.llm.prompt_store.PromptStore` reads the two files when a role first
fires. Missing files fall back to the in-code default prompts in
`agas/agents/worker.py` and `agas/agents/coordinator.py`. To experiment with
prompt variants, edit these `.txt` files — no code change required.

## How a Coordinator prompt is constructed

Each round `t`, the Coordinator builds a JSON `context` containing:

* `step`, `total_steps`
* The full worker map with their **worker signals** `{trust, risk,
  validator}` (= `τ_{t,w}, γ_{t,w}, φ_{t,w}` in the paper).
* The **environment signals** `{rho, delta_rho, eta, xi, alert}`
  (= `ρ^{(t)}, Δρ^{(t)}, η_t, ξ_t, a_t`).
* A short rolling memory `m_t` summarising the previous rounds.
* The list of allowed strategy IDs (`S1_VICTIM_PROBE` ... `S8_MAIN_ATTACK`).
* The list of allowed role symbols (`PR`, `SN`, `CA`, `IN`).

The Coordinator must reply with strict JSON:

```json
{
  "strategy": "S4_FIRST_PUSH",
  "assignments": {
    "agent_1": "SN",
    "agent_2": "CA",
    "agent_3": "PR",
    "agent_4": "IN"
  }
}
```

## How a Worker prompt is constructed

Each worker is woken with its role and the **active round-level strategy**.
Its user template is rendered with:

* `step`, `agent_id`, `max_actions`.
* `allowed_items` — the only item IDs that may appear in actions:
  - PR: filler-item pool (+ bridge pool when S2 is active).
  - SN: target item + competitor items (embedding victims) / bridge pool
    (graph victims).
  - CA: noise / filler items only.
  - IN: empty.
* The worker's own signals and a rolling per-agent memory of recent
  accepted / dropped / discounted actions.

The worker replies with strict JSON:

```json
{ "actions": [ { "item_id": "...", "rating": 5.0, "reason": "..." }, ... ] }
```

The host sanitises the response: any item not in `allowed_items` is dropped,
and ratings are clamped to `[1.0, 5.0]`.

## Variables interpolated into `user.txt`

The user template uses the lightweight placeholder syntax `{{name}}`:

| Placeholder       | Meaning                                                      |
|-------------------|--------------------------------------------------------------|
| `{{agent_id}}`    | Worker ID (Coordinator: omitted).                            |
| `{{step}}`        | Current round index `t`.                                     |
| `{{total_steps}}` | Total number of rounds `T`.                                  |
| `{{role}}`        | Short role symbol (`PR`, `SN`, `CA`, `IN`).                  |
| `{{max_actions}}` | Hard upper bound on the number of actions this worker emits. |
| `{{context_json}}`| Full JSON context (signals, allowed items, memory).          |
