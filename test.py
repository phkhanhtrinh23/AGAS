import json
from pathlib import Path

p=Path("outputs/transfer_result_offline.json")
obj=json.loads(p.read_text())
hist=obj.get("surrogate_episode_history") or []

rows=[]
for step in hist:
    for o in step.get("feedback", {}).get("outcomes", []):
        if o.get("accepted") and o.get("effective_rating") is not None:
            a=o["action"]
            rows.append((a["agent_id"], a["item_id"], float(o["effective_rating"]), float(a["rating"])))

pos = sum(1 for _,_,eff,_ in rows if eff >= 4.0)
low = len(rows) - pos
print("accepted_rows", len(rows))
print("positives_by_threshold(eff>=4.0)", pos)
print("nonpositives(eff<4.0)", low)

