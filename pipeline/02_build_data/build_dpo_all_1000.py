"""Build the pooled 1000-user aggregate DPO dataset: 25 pairs sampled from
each of the 1000 users (seed fixed), ~25k pairs total — same training size as
the old 100-user user_dpo_all.jsonl so the aggregate recipe is comparable.

Output: user_dpo_all_1000.jsonl
"""
import json, random

SEED = 13
PER_USER = 25

users = json.load(open("user_ids_1000.json"))
rng = random.Random(SEED)
out = []
for u in users:
    with open(f"user_dpo/dpo_user_{u}.jsonl") as fh:
        rows = [l.rstrip("\n") for l in fh if l.strip()]
    take = rows if len(rows) <= PER_USER else rng.sample(rows, PER_USER)
    out.extend(take)
rng.shuffle(out)
with open("user_dpo_all_1000.jsonl", "w") as fh:
    fh.write("\n".join(out) + "\n")
print(f"users={len(users)} pairs={len(out)} -> user_dpo_all_1000.jsonl")
