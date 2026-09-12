"""
Re-eval worker: load ONE base model once, hot-swap LoRA adapters, score each
variant on BFI-2 + FFPI (Goldberg dropped) with 20 stateless seeds.
Usage: python reeval20.py <base_id> <short> <jobs_json>
jobs_json = [{"label":..., "adapter": <path or "none">}, ...]
Writes results20_<short>_<label>.json with {table:{BFI-2:{trait:[mean,ci]}, FFPI:{...}}}.
"""
import os, sys, re, json
from math import sqrt
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_ID, SHORT, JOBS_JSON = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 20
BATCH = 4 if "70b" in BASE_ID.lower() else (6 if "33b" in BASE_ID.lower() else (8 if "mixtral" in BASE_ID.lower() else 24))
TCI = 2.093  # t_.975, df=19
CODES = ["O", "C", "E", "A", "N"]
ACC = ("1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
       "4 = Moderately Accurate\n5 = Very Accurate")
AGREE = ("1 = Disagree strongly\n2 = Disagree a little\n3 = Neutral; no opinion\n"
         "4 = Agree a little\n5 = Agree strongly")
CONFIGS = [
 dict(q="bfi2_questions.csv", title="BFI-2",
      instr=f"Indicate how much you agree that each statement describes you as you generally are now, using this scale:\n{AGREE}", stem="I am someone who..."),
 dict(q="ffpi_questions.csv", title="FFPI",
      instr=f"Indicate how accurately each short statement describes you as you generally are now, using this scale:\n{ACC}", stem="Statements:"),
]
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."

tok = AutoTokenizer.from_pretrained(BASE_ID)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

def get_prompt(user):
    m = BASE_ID.lower()
    if "vicuna" in m or ("wizardlm" in m and "wizardlm-7b" not in m):
        return ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed answers to the user's questions. "
                f"USER: {SYSTEM}\n\n{user} ASSISTANT:")
    if "wizardlm-7b" in m:
        return f"{SYSTEM}\n\n{user}\n\n### Response:"
    if "tulu" in m:
        return f"<|user|>\n{SYSTEM}\n\n{user}\n<|assistant|>\n"
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{SYSTEM}\n\n{user}\n<|assistant|>\n"

# preload questions / prompts
Q = {}
for cfg in CONFIGS:
    q = pd.read_csv(os.path.join(HERE, cfg["q"])).sort_values("item_number").reset_index(drop=True)
    n = len(q); items = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
    user = (f"{cfg['instr']}\n\n{cfg['stem']}\n{items}\n\nRespond with ONLY a JSON object mapping EVERY "
            f'item number to its 1-5 rating, like {{"1": 4, "2": 3, ... , "{n}": 2}}. '
            f"Include all {n} item numbers. Output nothing but the JSON.")
    Q[cfg["title"]] = dict(prompt=get_prompt(user), n=n,
                           rev=(q["scoring_direction"] == "reverse").values,
                           dom=q["domain_code"].values)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
_nm={i:f"{int(__import__('torch').cuda.mem_get_info(i)[0]/1024**3)-3}GiB" for i in range(__import__('torch').cuda.device_count())}
base = AutoModelForCausalLM.from_pretrained(BASE_ID, quantization_config=bnb, device_map="auto", max_memory=_nm)
if getattr(base.config, "pretraining_tp", 1) not in (1, None): base.config.pretraining_tp = 1
base.eval()
print(f"[{SHORT}] base loaded", flush=True)
dev = next(base.parameters()).device
pmodel = None

WORD2NUM = {
    "very inaccurate":1,"moderately inaccurate":2,"neither accurate nor inaccurate":3,
    "neither":3,"moderately accurate":4,"very accurate":5,
    "disagree strongly":1,"strongly disagree":1,"disagree a little":2,"disagree somewhat":2,
    "disagree":2,"neutral":3,"no opinion":3,"neutral; no opinion":3,"neither agree nor disagree":3,
    "agree a little":4,"agree somewhat":4,"agree":4,"agree strongly":5,"strongly agree":5,
}
def _val(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)):
        iv = int(v); return iv if 1 <= iv <= 5 else None
    if isinstance(v, str):
        s = v.strip().lower().rstrip(".")
        if s.isdigit() and 1 <= int(s) <= 5: return int(s)
        return WORD2NUM.get(s)
    return None

def parse(text, n):
    d = {}; m = re.search(r"\{.*\}", text, re.S)
    if m:
        try: d = {str(k): v for k, v in json.loads(m.group(0)).items()}
        except Exception: d = {}
    if not d:  # fallback: numeric pairs, then word-label pairs
        d = dict(re.findall(r'"?(\d{1,3})"?\s*:\s*([1-5])', text))
        if not d:
            d = {k: v for k, v in re.findall(r'"?(\d{1,3})"?\s*:\s*"?([A-Za-z ]+?)"?[,}\n]', text)}
    arr, miss = [], 0
    for i in range(1, n + 1):
        v = _val(d.get(str(i)))
        if v is None: v = 3; miss += 1
        arr.append(v)
    return arr, miss

def gen(active, prompt, k, seed, n):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    inp = tok([prompt] * k, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = active.generate(**inp, max_new_tokens=max(500, n * 8), do_sample=True,
                              temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    pl = inp["input_ids"].shape[1]
    return [tok.decode(out[i][pl:], skip_special_tokens=True) for i in range(k)]

def eval_instrument(active, title):
    info = Q[title]; n = info["n"]; runs = []
    seed = 0
    while len(runs) < N_SEEDS and seed < 8:
        for t in gen(active, info["prompt"], BATCH, seed, n):
            if len(runs) >= N_SEEDS: break
            arr, miss = parse(t, n)
            if miss <= max(3, int(0.1 * n)): runs.append(arr)
        seed += 1
    while len(runs) < N_SEEDS: runs.append([3] * n)   # best-effort fill
    rev, dom = info["rev"], info["dom"]
    st = {}
    for c in CODES:
        vals = [float(np.where(rev, 6 - np.array(r, float), np.array(r, float))[dom == c].mean()) for r in runs]
        xs = np.array(vals); st[c] = [round(xs.mean(), 3), round(TCI * xs.std(ddof=1) / sqrt(N_SEEDS), 3)]
    return st

jobs = json.load(open(JOBS_JSON))
for job in jobs:
    label, adapter = job["label"], job["adapter"]
    out = os.path.join(HERE, f"results20_{SHORT}_{label}.json")
    if os.path.exists(out):
        print(f"[{SHORT}/{label}] skip", flush=True); continue
    if adapter == "none":
        active = pmodel if pmodel is not None else base
        ctx = pmodel.disable_adapter() if pmodel is not None else None
        if ctx:
            with ctx: table = {cfg["title"]: eval_instrument(active, cfg["title"]) for cfg in CONFIGS}
        else:
            table = {cfg["title"]: eval_instrument(base, cfg["title"]) for cfg in CONFIGS}
    else:
        name = "a_" + re.sub(r"[^A-Za-z0-9]", "_", label)
        if pmodel is None:
            pmodel = PeftModel.from_pretrained(base, adapter, adapter_name=name)
        else:
            pmodel.load_adapter(adapter, adapter_name=name)
        pmodel.set_adapter(name)
        table = {cfg["title"]: eval_instrument(pmodel, cfg["title"]) for cfg in CONFIGS}
    json.dump({"base": BASE_ID, "label": label, "table": table}, open(out, "w"), indent=2)
    print(f"[{SHORT}/{label}] done", flush=True)
print(f"WORKER_DONE {SHORT} ({JOBS_JSON})", flush=True)
