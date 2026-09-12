"""Stratified, sex-matched sample for 'the last dance': 100 male + 100 female,
drawn so their country x age distributions are IDENTICAL (matched), so any OCEAN
difference between the groups is attributable to sex, not country/age.
Writes ld_males.json, ld_females.json (case_id lists) and prints the realized
distributions for both groups.
"""
import pandas as pd, numpy as np, json

rng = np.random.RandomState(42)
d = pd.read_csv("ipip300_responses_and_scores.csv")
d = d[d["sex"].isin(["male", "female"])].dropna(subset=["age", "country"]).copy()
d["age"] = d["age"].astype(int)
ENG = {"Canada", "UK", "Australia", "Ireland", "New Zealand"}
d["cgrp"] = np.where(d.country == "USA", "USA", np.where(d.country.isin(ENG), "OtherEng", "Rest"))
d["aband"] = pd.cut(d.age, [0, 18, 25, 35, 200], right=False, labels=["<18", "18-24", "25-34", "35+"]).astype(str)
d["stratum"] = d.cgrp + "|" + d.aband

TARGET = 100
caps = {}
for s, g in d.groupby("stratum"):
    m = (g.sex == "male").sum(); f = (g.sex == "female").sum()
    caps[s] = min(m, f)   # matched capacity = limited by the rarer sex
total_cap = sum(caps.values())
print(f"matched capacity (sum of min(M,F) per stratum) = {total_cap}")

# largest-remainder allocation of TARGET across strata, capped by caps[s]
alloc = {s: 0 for s in caps}
if total_cap >= TARGET:
    raw = {s: TARGET * caps[s] / total_cap for s in caps}
    for s in caps: alloc[s] = min(int(np.floor(raw[s])), caps[s])
    rem = TARGET - sum(alloc.values())
    frac = sorted(caps, key=lambda s: -(raw[s] - np.floor(raw[s])))
    for s in frac:
        if rem <= 0: break
        if alloc[s] < caps[s]: alloc[s] += 1; rem -= 1
else:
    alloc = dict(caps)  # take all matched pairs; will top up below

males, females = [], []
for s, n in alloc.items():
    if n <= 0: continue
    g = d[d.stratum == s]
    mids = g[g.sex == "male"]["case_id"].tolist(); fids = g[g.sex == "female"]["case_id"].tolist()
    males += list(rng.choice(mids, n, replace=False))
    females += list(rng.choice(fids, n, replace=False))

# top up if matched capacity was short of TARGET (keeps groups ~matched on the big strata)
def topup(lst, sex):
    have = set(lst)
    pool = d[(d.sex == sex) & (~d.case_id.isin(have))].sort_values("stratum")
    need = TARGET - len(lst)
    if need > 0:
        lst += list(rng.choice(pool["case_id"].tolist(), need, replace=False))
    return lst
males = topup(males, "male"); females = topup(females, "female")

json.dump([int(x) for x in males], open("ld_males.json", "w"))
json.dump([int(x) for x in females], open("ld_females.json", "w"))
print(f"\nselected: {len(males)} male, {len(females)} female")
for name, ids in [("MALE", males), ("FEMALE", females)]:
    sub = d[d.case_id.isin(ids)]
    print(f"\n{name} distribution:")
    print("  country:", dict(sub.cgrp.value_counts()))
    print("  age:    ", dict(sub.aband.value_counts()))
