"""
Build per-user DPO preference files from the IPIP-NEO-300 responses.

Each IPIP item becomes a forced-choice "does this statement describe you?"
preference pair. The chosen answer is the one the annotator actually gave:

  * forward-scored item : Agree if their rating >= 4, Disagree if <= 2
  * reverse-scored item : flipped (Agree if rating <= 2, Disagree if >= 4)
  * neutral rating (3)  : dropped, no pair

Items are emitted in item_number order. This reproduces the original 20
user_dpo/dpo_user_<id>.jsonl files exactly (verified by --check).

Usage:
  python3 build_user_dpo.py --check                 # verify against existing files
  python3 build_user_dpo.py user_ids_100.json       # build files for an id list
"""
import sys, json
import pandas as pd

CSV = "ipip300_responses_and_scores_10000.csv"
QUESTIONS = "ipip300_questions.csv"
OUT_DIR = "user_dpo"
PROMPT = ('Statement: "{stmt}"\n'
          'Does this statement describe you? Answer with "Agree" or "Disagree".')


def load():
    df = pd.read_csv(CSV)
    df = df[pd.to_numeric(df["case_id"], errors="coerce").notna()].copy()
    df["case_id"] = df["case_id"].astype(int)
    q = pd.read_csv(QUESTIONS).sort_values("item_number").reset_index(drop=True)
    return df, q


def pairs_for(row, q):
    """Yield (prompt, chosen, rejected) for one annotator's CSV row."""
    out = []
    for r in q.itertuples():
        try:
            v = int(row[f"I{r.item_number}"])
        except (TypeError, ValueError):
            continue
        if v == 3:
            continue
        agree = v >= 4
        if r.scoring_direction == "reverse":
            agree = not agree
        chosen, rejected = ("Agree", "Disagree") if agree else ("Disagree", "Agree")
        out.append((PROMPT.format(stmt=r.question_text), chosen, rejected))
    return out


def write_file(path, pairs):
    with open(path, "w") as f:
        for prompt, chosen, rejected in pairs:
            f.write(json.dumps({"prompt": prompt, "chosen": chosen,
                                "rejected": rejected}) + "\n")


def main():
    df, q = load()
    args = sys.argv[1:]

    if "--check" in args:
        import os, glob
        ok = True
        for path in sorted(glob.glob(f"{OUT_DIR}/dpo_user_*.jsonl")):
            uid = int(os.path.basename(path)[len("dpo_user_"):-len(".jsonl")])
            row = df[df.case_id == uid]
            if row.empty:
                print(f"  {uid}: no CSV row, skipped"); continue
            built = [json.dumps({"prompt": p, "chosen": c, "rejected": r})
                     for p, c, r in pairs_for(row.iloc[0], q)]
            have = [l.rstrip("\n") for l in open(path)]
            same = built == have
            ok &= same
            print(f"  {uid}: {'OK' if same else 'MISMATCH'} "
                  f"({len(have)} existing vs {len(built)} built)")
        print("ALL MATCH" if ok else "DIFFERENCES FOUND")
        return

    if not args:
        print(__doc__); return

    ids = json.load(open(args[0]))
    n_new = n_skip = 0
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    for uid in ids:
        path = f"{OUT_DIR}/dpo_user_{uid}.jsonl"
        if os.path.exists(path):
            n_skip += 1
            continue
        row = df[df.case_id == int(uid)]
        if row.empty:
            print(f"  WARN: no CSV row for {uid}"); continue
        write_file(path, pairs_for(row.iloc[0], q))
        n_new += 1
    print(f"built {n_new} new files, skipped {n_skip} existing -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
