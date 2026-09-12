"""
Stratified sampling of real annotators for the user DPO experiment.

The original 20-user selection was never committed as code, only as its result
(user_ids.json). This script reconstructs and generalises that method so the
selection is reproducible.

Stratification:
  * sex     - balanced 50/50 (matches the original 20: exactly 10 male / 10 female)
  * age     - 5 bands (<18, 18-24, 25-34, 35-49, 50+)
  * country - top countries kept as their own stratum, the long tail collapsed
              into "Other" (the population is ~70% USA with many singletons)

Within each sex we allocate the N/2 slots across the age x country cells
proportionally to that sex's sub-population (largest-remainder rounding so the
counts sum exactly), then draw without replacement inside each cell. A fixed
seed makes the draw reproducible.

Usage:
  python3 sample_users.py [N] [out.json]     # defaults: N=100, user_ids_100.json
"""
import sys, json
import numpy as np
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = sys.argv[2] if len(sys.argv) > 2 else "user_ids_100.json"
SEED = 42
CSV = "ipip300_responses_and_scores_10000.csv"
# Countries kept as their own stratum; everything else -> "Other".
KEEP_COUNTRIES = ["USA", "Canada", "UK", "Australia"]
AGE_BANDS = [(0, 17, "<18"), (18, 24, "18-24"), (25, 34, "25-34"),
             (35, 49, "35-49"), (50, 200, "50+")]


def age_band(a):
    for lo, hi, lab in AGE_BANDS:
        if lo <= a <= hi:
            return lab
    return "unknown"


def load_population():
    df = pd.read_csv(CSV)
    # the CSV also holds synthetic AGENT_* rows; keep only real numeric case_ids
    df = df[pd.to_numeric(df["case_id"], errors="coerce").notna()].copy()
    df["case_id"] = df["case_id"].astype(int)
    df = df.dropna(subset=["sex", "age"])
    df["country"] = df["country"].fillna("Other")
    df["cgroup"] = df["country"].where(df["country"].isin(KEEP_COUNTRIES), "Other")
    df["aband"] = df["age"].apply(age_band)
    return df


def largest_remainder(weights, total):
    """Apportion `total` integer slots proportionally to `weights`."""
    weights = np.asarray(weights, float)
    if weights.sum() == 0:
        return np.zeros(len(weights), int)
    exact = weights / weights.sum() * total
    base = np.floor(exact).astype(int)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(exact - base))
        base[order[:rem]] += 1
    return base


def sample_sex(sub, n_target, rng):
    """Proportionally allocate n_target across age x country cells, draw within."""
    cells = sub.groupby(["aband", "cgroup"]).groups  # cell -> index labels
    keys = list(cells.keys())
    sizes = [len(cells[k]) for k in keys]
    alloc = largest_remainder(sizes, n_target)

    # Cap each cell at its availability, then redistribute any shortfall to the
    # cells that still have spare members (largest first), so we always hit n.
    alloc = np.minimum(alloc, sizes)
    while alloc.sum() < n_target:
        spare = np.array(sizes) - alloc
        if spare.sum() == 0:
            break
        order = np.argsort(-spare)
        for i in order:
            if alloc.sum() >= n_target:
                break
            if spare[i] > 0:
                alloc[i] += 1

    picked = []
    for k, take in zip(keys, alloc):
        ids = list(sub.loc[cells[k], "case_id"])
        if take > 0:
            picked += list(rng.choice(ids, size=int(take), replace=False))
    return picked


def main():
    df = load_population()
    rng = np.random.default_rng(SEED)
    half = N // 2
    targets = {"male": half, "female": N - half}  # balanced (N even -> 50/50)

    picked = []
    for sex, n_t in targets.items():
        sub = df[df.sex == sex]
        picked += sample_sex(sub, n_t, rng)

    picked = [int(x) for x in picked]
    assert len(picked) == len(set(picked)) == N, (len(picked), len(set(picked)))

    json.dump(sorted(picked), open(OUT, "w"))

    # report the realised strata so the draw is auditable
    chosen = df[df.case_id.isin(picked)]
    print(f"wrote {OUT}: {N} users (seed={SEED})")
    print("sex   :", chosen.sex.value_counts().to_dict())
    print("age   :", chosen.aband.value_counts().reindex([b[2] for b in AGE_BANDS]).to_dict())
    print("ctry  :", chosen.cgroup.value_counts().to_dict())


if __name__ == "__main__":
    main()
