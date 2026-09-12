"""Sample 1,050 people (7 countries x 3 age brackets x 2 sexes x 25) from the full
IPIP-NEO-300 data and build a per-person DPO file for each. Same scoring/format as
build_user_dpo.py (verified to match). Writes user_dpo_worldtour/<pid>.jsonl and
worldtour_meta.json (pid -> case_id, country, sex, age bracket)."""
import pandas as pd, numpy as np, json, os
rng = np.random.RandomState(123)
DAT = "/mnt/ssd3/ipip_full/IPIP300.dat"
q = pd.read_csv("ipip300_questions.csv").sort_values("item_number").reset_index(drop=True)
PROMPT = ('Statement: "{stmt}"\nDoes this statement describe you? Answer with "Agree" or "Disagree".')
COUNTRIES = ["USA","India","China","South Afric","Germany","Australia","Brazil"]
BRACKETS = ["18-24","25-34","35+"]; PER = 25

df = pd.read_fwf(DAT, colspecs=[(0,6),(6,7),(7,9),(22,33),(33,333)],
                 names=["case","sex","age","country","items"], dtype={"sex":str,"country":str,"items":str})
df["country"] = df["country"].astype(str).str.strip()
df["sexL"] = df["sex"].map({"1":"male","2":"female"})
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df = df[df.sexL.notna() & df.age.notna() & df["items"].notna()]
df["ab"] = pd.cut(df.age,[0,18,25,35,200],right=False,labels=["<18","18-24","25-34","35+"]).astype(str)

sel = []
for c in COUNTRIES:
    for ab in BRACKETS:
        for sx in ["male","female"]:
            pool = df[(df.country==c)&(df.ab==ab)&(df.sexL==sx)]
            take = pool.sample(min(PER,len(pool)), random_state=rng)
            sel.append(take)
            if len(pool) < PER: print(f"  WARN {c}/{ab}/{sx}: only {len(pool)}")
sel = pd.concat(sel).reset_index(drop=True)
print(f"selected {len(sel)} people")

def pairs(items_str):
    out = []
    for r in q.itertuples():
        ch = items_str[r.item_number-1]
        if not ch.isdigit(): continue
        v = int(ch)
        if v in (0,3): continue
        agree = v >= 4
        if r.scoring_direction == "reverse": agree = not agree
        c2, rj = ("Agree","Disagree") if agree else ("Disagree","Agree")
        out.append((PROMPT.format(stmt=r.question_text), c2, rj))
    return out

os.makedirs("user_dpo_worldtour", exist_ok=True)
meta = {}
for pid, row in sel.iterrows():
    its = str(row["items"])
    if len(its) < 300: its = its.ljust(300, "0")
    pr = pairs(its)
    with open(f"user_dpo_worldtour/{pid}.jsonl","w") as f:
        for p,c2,rj in pr: f.write(json.dumps({"prompt":p,"chosen":c2,"rejected":rj})+"\n")
    meta[int(pid)] = dict(case=int(row["case"]), country=row["country"], sex=row["sexL"], ab=row["ab"], n_pairs=len(pr))
json.dump(meta, open("worldtour_meta.json","w"))
print(f"built {len(meta)} dpo files | mean pairs/person: {np.mean([m['n_pairs'] for m in meta.values()]):.0f}")
# quick cell-count sanity
import collections
cc = collections.Counter((m["country"],m["ab"],m["sex"]) for m in meta.values())
print("cells (should be 25 each):", dict(list(cc.items())[:6]), "...")
