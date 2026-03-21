import json
obj=json.load(open("outputs/episode_lockout_example.json"))
for step in obj["history"]:
    rt = step.get("coordinator_runtime_trace")
    if rt and rt.get("sniper_lockouts"):
        print(step["step"], rt)