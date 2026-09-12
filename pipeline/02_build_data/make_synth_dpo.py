"""Build synthetic 'user' DPO datasets with an ENGINEERED per-trait lean, to test
whether WizardLM-13B's Agreeableness rise is data-driven or a model-imposed prior.
Same format as the real user DPO files. For the TARGET trait, every chosen answer
expresses the target pole; for all OTHER traits, chosen answers are balanced 50/50
(lean ~ 0) so they don't confound.
Writes: synth_antiA.jsonl, synth_proA.jsonl, synth_neutral.jsonl
"""
import pandas as pd, json, re

q = pd.read_csv("ipip300_questions.csv")
PROMPT = ('Statement: "{}"\nDoes this statement describe you? '
          'Answer with "Agree" or "Disagree".')

def high_pole_answer(direction):
    # answer that expresses the HIGH pole of the trait
    return "Agree" if direction == "forward" else "Disagree"

def low_pole_answer(direction):
    return "Disagree" if direction == "forward" else "Agree"

def build(target_trait, target_pole):
    """target_pole: 'high','low', or None (all balanced)."""
    rows = []
    # group item indices per domain for balanced assignment of non-target traits
    for dom, grp in q.groupby("domain_code"):
        items = list(grp.itertuples())
        is_target = (dom == target_trait)
        for i, r in enumerate(items):
            if is_target and target_pole == "high":
                chosen = high_pole_answer(r.scoring_direction)
            elif is_target and target_pole == "low":
                chosen = low_pole_answer(r.scoring_direction)
            else:  # balanced: first half high pole, second half low pole -> lean ~0
                chosen = (high_pole_answer(r.scoring_direction) if i < len(items)//2
                          else low_pole_answer(r.scoring_direction))
            rejected = "Disagree" if chosen == "Agree" else "Agree"
            rows.append({"prompt": PROMPT.format(r.question_text),
                         "chosen": chosen, "rejected": rejected})
    return rows

specs = {"synth_antiA": ("A", "low"), "synth_proA": ("A", "high"), "synth_neutral": (None, None)}
for name, (t, pole) in specs.items():
    rows = build(t, pole)
    with open(f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    # report the realized lean per trait (sanity)
    leans = {}
    for dom, grp in q.groupby("domain_code"):
        hi = 0; n = 0
        for r in grp.itertuples():
            row = next(x for x in rows if x["prompt"] == PROMPT.format(r.question_text))
            ans_high = (row["chosen"] == high_pole_answer(r.scoring_direction))
            hi += ans_high; n += 1
        leans[dom] = round(hi/n - 0.5, 2)
    print(f"{name}: {len(rows)} items | realized lean {leans}")
