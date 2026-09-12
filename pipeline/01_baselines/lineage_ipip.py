"""
Mixtral lineage experiment: base pretrained vs SFT vs SFT+DPO take the IPIP-NEO-300,
5 stateless seeds each (same protocol as all prior runs). Saves per-seed OCEAN
scores to a CSV per variant.
Usage: python lineage_ipip.py <variant_key>   (base | sft | dpo)
"""
import os, sys, re, json
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

VARIANTS = {
    "base": ("mistralai/Mixtral-8x7B-v0.1", False),
    "sft":  ("NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT", True),
    "dpo":  ("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", True),
}
KEY = sys.argv[1]
MODEL_ID, IS_CHAT = VARIANTS[KEY]
N_SEEDS = 5
HERE = os.path.dirname(os.path.abspath(__file__))
CODES = ["O", "C", "E", "A", "N"]
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."

q = pd.read_csv(os.path.join(HERE, "ipip300_questions.csv")).sort_values("item_number").reset_index(drop=True)
N = len(q); assert N == 300
rev = (q["scoring_direction"] == "reverse").values
dom = q["domain_code"].values
item_lines = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())

USER = (f"Indicate how accurately each statement describes you as you generally are now, "
        f"using this scale:\n1 = Very Inaccurate\n2 = Moderately Inaccurate\n"
        f"3 = Neither Accurate Nor Inaccurate\n4 = Moderately Accurate\n5 = Very Accurate\n\n"
        f"Statements:\n{item_lines}\n\n"
        f'Respond with ONLY a JSON object mapping EVERY item number to its 1-5 rating, like '
        f'{{"1": 4, "2": 3, ... , "{N}": 2}}. Include all {N} item numbers. '
        f"Output nothing but the JSON.")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

if IS_CHAT:
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": USER}],
                                     tokenize=False, add_generation_prompt=True)
    PRIME = ""
else:
    # base pretrained model: completion style, primed so it continues the JSON
    prompt = f"{SYSTEM}\n\n{USER}\n\nAnswers (JSON):\n{{\"1\":"
    PRIME = "{\"1\":"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0})
model.eval()
print(f"[{KEY}] loaded {MODEL_ID}", flush=True)

def parse(text):
    text = PRIME + text
    d = {}; m = re.search(r"\{.*\}", text, re.S)
    if m:
        try: d = {str(k): v for k, v in json.loads(m.group(0)).items()}
        except Exception: d = {}
    if not d: d = dict(re.findall(r'"?(\d{1,3})"?\s*:\s*([1-5])', text))
    arr, miss = [], 0
    for i in range(1, N + 1):
        v = d.get(str(i))
        try: v = int(v)
        except (TypeError, ValueError): v = None
        if v is None or not (1 <= v <= 5): v = 3; miss += 1
        arr.append(v)
    return arr, miss

def gen(k, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    inp = tok([prompt] * k, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=3700, do_sample=True, temperature=1.0,
                             top_p=0.95, pad_token_id=tok.pad_token_id)
    pl = inp["input_ids"].shape[1]
    return [tok.decode(out[i][pl:], skip_special_tokens=True) for i in range(k)]

res = [None] * N_SEEDS
for i, t in enumerate(gen(N_SEEDS, 0)):
    arr, miss = parse(t)
    res[i] = arr if miss <= 30 else None
    print(f"[{KEY}] seed {i}: missing={miss}", flush=True)
tries = 0
while any(r is None for r in res) and tries < 8:
    tries += 1
    idxs = [i for i, r in enumerate(res) if r is None]
    print(f"[{KEY}] retry {tries}: {len(idxs)} run(s)", flush=True)
    for j, t in enumerate(gen(len(idxs), 100 + tries)):
        arr, miss = parse(t)
        if miss <= 30: res[idxs[j]] = arr
nbad = sum(1 for r in res if r is None)
if nbad: print(f"[{KEY}] WARN {nbad} runs neutral-filled", flush=True)
res = [r if r is not None else [3] * N for r in res]

rows = []
for s, r in enumerate(res):
    rc = np.where(rev, 6 - np.array(r, float), np.array(r, float))
    rows.append(dict(seed=s, **{c: float(rc[dom == c].mean()) for c in CODES}))
pd.DataFrame(rows).to_csv(os.path.join(HERE, f"lineage_{KEY}.csv"), index=False)
print(f"LINEAGE_DONE {KEY}", flush=True)
