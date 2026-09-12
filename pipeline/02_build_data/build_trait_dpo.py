"""
Single-trait DPO data builder: reads ipip300_questions.csv and writes 10 files
dpo_trait_{O,C,E,A,N}_{high,low}.jsonl, each holding ONLY that domain's 60 items.
Prompt string, chosen/rejected logic, and json fields are identical to
dpo_agent_high/low.jsonl; persona is e.g. "O_high_trait".
High agent: forward -> chosen "Agree" (rating 5), reverse -> chosen "Disagree" (rating 1).
Low agent is the exact mirror. Recoded rating (6-rating on reverse items) is always
5 for high and 1 for low.
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_NAME = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
             "A": "Agreeableness", "N": "Neuroticism"}

rows = list(csv.DictReader(open(os.path.join(HERE, "ipip300_questions.csv"))))
assert len(rows) == 300, f"expected 300 items, got {len(rows)}"

def make_line(r, direction, persona):
    forward = r["scoring_direction"] == "forward"
    # high: push recoded score to 5 -> Agree on forward, Disagree on reverse
    # low: push recoded score to 1 -> the mirror
    agree = forward if direction == "high" else not forward
    chosen, rejected = ("Agree", "Disagree") if agree else ("Disagree", "Agree")
    rating = 5 if agree else 1
    prompt = (f'Statement: "{r["question_text"]}"\n'
              'Does this statement describe you? Answer with "Agree" or "Disagree".')
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "item_id": r["item_id"], "domain": r["domain"],
            "scoring_direction": r["scoring_direction"], "rating": rating,
            "persona": persona}

outfiles = []
for code, name in CODE_NAME.items():
    items = [r for r in rows if r["domain"] == name]
    assert len(items) == 60, f"{name}: expected 60 items, got {len(items)}"
    for direction in ("high", "low"):
        path = os.path.join(HERE, f"dpo_trait_{code}_{direction}.jsonl")
        with open(path, "w") as f:
            for r in items:
                f.write(json.dumps(make_line(r, direction, f"{code}_{direction}_trait")) + "\n")
        outfiles.append((path, code, direction))

# ---- verification ----
ref = {}  # item_id -> line from the existing all-trait agent files
for agent in ("high", "low"):
    for ln in open(os.path.join(HERE, f"dpo_agent_{agent}.jsonl")):
        d = json.loads(ln)
        ref[(d["item_id"], agent)] = d

n_rev_checked = 0
for path, code, direction in outfiles:
    lines = [json.loads(l) for l in open(path)]
    assert len(lines) == 60, f"{path}: {len(lines)} lines"
    for d in lines:
        assert d["domain"] == CODE_NAME[code], f"{path}: foreign item {d['item_id']} ({d['domain']})"
        recoded = d["rating"] if d["scoring_direction"] == "forward" else 6 - d["rating"]
        want = 5 if direction == "high" else 1
        assert recoded == want, f"{path} {d['item_id']}: recoded {recoded} != {want}"
        # every line must match the all-trait agent file except persona
        r = ref[(d["item_id"], direction)]
        assert {**d, "persona": r["persona"]} == r, f"{path} {d['item_id']}: mismatch vs dpo_agent_{direction}.jsonl"
        if d["scoring_direction"] == "reverse":
            n_rev_checked += 1
            if n_rev_checked <= 3 and direction == "high":
                print(f"spot-check reverse {d['item_id']} ({code} high): "
                      f"chosen={d['chosen']} rating={d['rating']} == dpo_agent_high.jsonl OK")

print(f"OK: 10 files x 60 lines; all lines match dpo_agent_high/low.jsonl modulo persona "
      f"({n_rev_checked} reverse-item checks).")
