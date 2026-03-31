from agas.agents.messages import VictimModelClass
from agas.agents.coordinator import Coordinator, RuleBasedCoordinatorPolicy
from agas.agents.worker import WorkerPolicyConfig

# Test 1: VictimModelClass enum
print("VictimModelClass values:", [v.value for v in VictimModelClass])

# Test 2: Coordinator with probe_steps=2 (auto mode)
policy = RuleBasedCoordinatorPolicy(agent_order=["a0", "a1", "a2"])
coord = Coordinator(policy=policy, probe_steps=2, victim_model_hint="auto")
print("probe_phase_done:", coord._probe_phase_done)
print("victim_model_class:", coord.victim_model_class)

# Test 3: Coordinator with explicit hint
coord2 = Coordinator(policy=policy, probe_steps=2, victim_model_hint="lightgcn")
print("lightgcn hint probe_done:", coord2._probe_phase_done)
print("lightgcn hint class:", coord2.victim_model_class)

# Test 4: WorkerPolicyConfig
cfg = WorkerPolicyConfig(sequential_sniper=True, sequential_filler_actions=3)
print("sequential_sniper:", cfg.sequential_sniper)
print("sequential_filler_actions:", cfg.sequential_filler_actions)

print("ALL OK")
