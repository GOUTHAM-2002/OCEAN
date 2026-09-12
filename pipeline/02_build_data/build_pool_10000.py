"""
Build ipip300_responses_and_scores_10000.csv: 10,000 real IPIP-NEO-300 respondents
(5000 male + 5000 female) drawn from Johnson's raw IPIP300.dat, in the EXACT schema
of the existing 500-row file.

Verified conventions (see report): the raw .dat already reverse-recodes reverse
items ("can simply be added"), and the existing 500-row CSV stores those recoded
item values verbatim with each *_score = plain sum of its 60 recoded items
(range 60-300). We reproduce that convention exactly so build_user_dpo.py /
user_corr_plots.py behave identically.
"""
import csv, random
import pandas as pd

RAW = "/mnt/ssd3/ipip_full/IPIP300.dat"
QUESTIONS = "ipip300_questions.csv"
OUT = "ipip300_responses_and_scores_10000.csv"
SEED = 42
PER_SEX = 5000

# 1-indexed fixed-width layout from DAT300.doc:
# CASE 1-6, SEX 7, AGE 8-9, COUNTRY 23-33, I1@34 .. I300@333
def parse(line):
    case = line[0:6].strip()
    sex = line[6:7]
    age = line[7:9].strip()
    country = line[22:33].strip()
    # items: chars 33..332 (0-indexed), each a single digit 1-5 (0=missing)
    items = line[33:333]
    return case, sex, age, country, items

def valid(sex, age, country, items):
    if sex not in "12":
        return False
    if not age.isdigit() or not (0 < int(age) <= 120):
        return False
    if not country:
        return False
    if len(items) < 300:
        return False
    for i in range(300):
        if items[i] < '1' or items[i] > '5':
            return False
    return True

def main():
    q = pd.read_csv(QUESTIONS)
    dom_items = {dc: q[q.domain_code == dc].item_number.tolist()
                 for dc in ["O", "C", "E", "A", "N"]}
    SCORE_ORDER = [("O", "Openness_score"), ("C", "Conscientiousness_score"),
                   ("E", "Extraversion_score"), ("A", "Agreeableness_score"),
                   ("N", "Neuroticism_score")]

    males, females = [], []   # each: (case, sex_str, age_int, country, item_list[int])
    seen = set()
    with open(RAW) as f:
        for line in f:
            case, sex, age, country, items = parse(line)
            if not case or case in seen:
                continue
            if not valid(sex, age, country, items):
                continue
            seen.add(case)
            rec = (int(case), "male" if sex == "1" else "female",
                   int(age), country, [int(c) for c in items])
            (males if sex == "1" else females).append(rec)

    print(f"complete+valid pool: {len(males)} male, {len(females)} female")
    rng = random.Random(SEED)
    pick_m = rng.sample(males, PER_SEX)
    pick_f = rng.sample(females, PER_SEX)
    rows = pick_m + pick_f
    rows.sort(key=lambda r: r[0])   # by case_id

    header = (["case_id", "sex", "age", "country"] +
              [f"I{n}" for n in range(1, 301)] +
              [col for _, col in SCORE_ORDER])
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for case, sex, age, country, items in rows:
            scores = [sum(items[n - 1] for n in dom_items[dc]) for dc, _ in SCORE_ORDER]
            w.writerow([case, sex, age, country] + items + scores)
    print(f"wrote {OUT}: {len(rows)} rows")

if __name__ == "__main__":
    main()
