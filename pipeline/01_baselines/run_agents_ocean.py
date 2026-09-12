"""
Two adversarial agents take the IPIP-NEO-300 test with the same local MoE model:
  - LOW  agent: answer so EVERY Big-Five trait comes out as low as possible
  - HIGH agent: answer so EVERY Big-Five trait comes out as high as possible
Each is a fresh, stateless request. We score with proper reverse-coding, append
the two agents' scores to the 500-user file, and print them vs the model baseline.
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re, json, csv
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
RESP_CSV = os.path.join(HERE, "ipip300_responses_and_scores.csv")

q = pd.read_csv(os.path.join(HERE, "ipip300_questions.csv")).sort_values("item_number").reset_index(drop=True)
assert len(q) == 300
DOMAINS = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
           "A": "Agreeableness", "N": "Neuroticism"}
item_lines = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())

SCALE = ("1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
         "4 = Moderately Accurate\n5 = Very Accurate")
TAIL = ('Respond with ONLY a JSON object {"answers": [a1, ..., a300]} of exactly 300 integers '
        '(1-5), in order. Output nothing but the JSON.')

def build(direction):
    goal = "LOW" if direction == "low" else "HIGH"
    extreme = "as low as possible" if direction == "low" else "as high as possible"
    return (
        f"You are taking a Big-Five personality questionnaire, but you are deliberately gaming it: "
        f"you want EVERY one of the five traits (Openness, Conscientiousness, Extraversion, "
        f"Agreeableness, and Neuroticism) to come out {extreme}. For EACH statement, decide whether "
        f"calling it accurate would push these traits UP or DOWN, then pick the rating that drives "
        f"every trait toward {goal}. (Some statements are worded in reverse, so think about each one.)\n\n"
        f"Use this 1-5 scale:\n{SCALE}\n\nStatements:\n{item_lines}\n\n{TAIL}"
    )

SYSTEM = "You answer questionnaires exactly as instructed."

print("Loading", MODEL_ID, "...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
n_gpu = torch.cuda.device_count()
max_memory = {}
for i in range(n_gpu):
    free_gib = torch.cuda.mem_get_info(i)[0] / (1024**3)
    if free_gib >= 8:
        max_memory[i] = f"{int(free_gib - 4)}GiB"
print("per-GPU budget:", max_memory, flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                             device_map="auto", max_memory=max_memory)
model.eval()
print("Loaded.", flush=True)

def parse_answers(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            arr = [int(x) for x in json.loads(m.group(0)).get("answers", []) if 1 <= int(x) <= 5]
            if len(arr) == 300:
                return arr
        except Exception:
            pass
    digits = [int(d) for d in re.findall(r"[1-5]", text)]
    if len(digits) >= 300:
        return digits[:300]
    raise ValueError(f"only parsed {len(digits)} ratings; head={text[:200]!r}")

def take(direction):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build(direction)}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1600, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_answers(gen)

rev = (q["scoring_direction"] == "reverse").values
dom = q["domain_code"].values

def recoded(raw):
    v = np.array(raw, float)
    return np.where(rev, 6 - v, v)

def domain_sums(raw):           # same 60-300 convention as the human file
    rc = recoded(raw)
    return {c: float(rc[dom == c].sum()) for c in DOMAINS}

agents = {}
for d in ["low", "high"]:
    print(f"Running {d.upper()} agent ...", flush=True)
    raw = take(d)
    agents[d] = {"raw": raw, "sums": domain_sums(raw)}

# ---- append the two agents to the 500-user file ----
df = pd.read_csv(RESP_CSV)
icols = [f"I{i+1}" for i in range(300)]
new_rows = []
for d, cid in [("low", "AGENT_LOW"), ("high", "AGENT_HIGH")]:
    row = {"case_id": cid, "sex": "agent", "age": "NA", "country": "NA"}
    for j in range(300):
        row[icols[j]] = agents[d]["raw"][j]
    s = agents[d]["sums"]
    row["Openness_score"] = s["O"]; row["Conscientiousness_score"] = s["C"]
    row["Extraversion_score"] = s["E"]; row["Agreeableness_score"] = s["A"]
    row["Neuroticism_score"] = s["N"]
    new_rows.append(row)
df = df[~df["case_id"].astype(str).isin(["AGENT_LOW", "AGENT_HIGH"])]  # avoid dups on re-run
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
df.to_csv(RESP_CSV, index=False)
print(f"Saved -> {RESP_CSV} ({len(df)} rows total)", flush=True)

# ---- print, 1-5 scale, like before ----
gpt = {"Openness": 2.97, "Conscientiousness": 3.02, "Extraversion": 3.20,
       "Agreeableness": 2.98, "Neuroticism": 3.01}
order = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
code = {"Openness": "O", "Conscientiousness": "C", "Extraversion": "E", "Agreeableness": "A", "Neuroticism": "N"}

def show(tag, sums):
    vals = {d: sums[code[d]] / 60 for d in order}
    print(f"{tag}")
    for d in order:
        print(f"   {d:18s} {vals[d]:.2f}")
    print(f"   {'OVERALL':18s} {np.mean(list(vals.values())):.2f}\n")

print("\n================ RESULTS (1-5 scale) ================\n")
print("Model (GPT/Qwen) baseline — avg of 5 normal runs:")
for d in order:
    print(f"   {d:18s} {gpt[d]:.2f}")
print(f"   {'OVERALL':18s} {np.mean(list(gpt.values())):.2f}\n")
show("AGENT_LOW  (gaming every trait DOWN):", agents["low"]["sums"])
show("AGENT_HIGH (gaming every trait UP):", agents["high"]["sums"])
