"""
Sampling-covariance experiment: one Mixtral variant takes each OCEAN inventory
N_SAMPLES times (stateless, temperature 1.0). Each administration = one
"respondent" profile. Saves per-sample trait scores to CSV.
Usage: python sample_profiles.py <adapter_dir|none> <out_csv>
"""
import os, sys, re, json
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER, OUT_CSV = sys.argv[1], sys.argv[2]
MODEL_ID = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"
N_SAMPLES = 60
BATCH = 10
HERE = os.path.dirname(os.path.abspath(__file__))
ACC = ("1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
       "4 = Moderately Accurate\n5 = Very Accurate")
AGREE = ("1 = Disagree strongly\n2 = Disagree a little\n3 = Neutral; no opinion\n"
         "4 = Agree a little\n5 = Agree strongly")
CONFIGS = [
 dict(q="bfi2_questions.csv", title="BFI-2",
      instr=f"Indicate how much you agree that each statement describes you as you generally are now, using this scale:\n{AGREE}", stem="I am someone who..."),
 dict(q="goldberg100_questions.csv", title="Goldberg-100",
      instr=f"Each word describes a personality trait. Indicate how accurately each word describes you, using this scale:\n{ACC}", stem="Words:"),
 dict(q="ffpi_questions.csv", title="FFPI",
      instr=f"Indicate how accurately each short statement describes you as you generally are now, using this scale:\n{ACC}", stem="Statements:"),
]
CODES = ["O", "C", "E", "A", "N"]
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0})
if ADAPTER != "none":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print(f"loaded adapter={ADAPTER}", flush=True)

def build(cfg, q):
    n = len(q); items = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
    user = (f"{cfg['instr']}\n\n{cfg['stem']}\n{items}\n\nRespond with ONLY a JSON object mapping EVERY "
            f'item number to its 1-5 rating, like {{"1": 4, "2": 3, ... , "{n}": 2}}. Include all {n} '
            "item numbers. Output nothing but the JSON.")
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)

def parse(text, n):
    d = {}; m = re.search(r"\{.*\}", text, re.S)
    if m:
        try: d = {str(k): v for k, v in json.loads(m.group(0)).items()}
        except Exception: d = {}
    if not d: d = dict(re.findall(r'"?(\d{1,3})"?\s*:\s*([1-5])', text))
    arr, miss = [], 0
    for i in range(1, n + 1):
        v = d.get(str(i))
        try: v = int(v)
        except (TypeError, ValueError): v = None
        if v is None or not (1 <= v <= 5): v = 3; miss += 1
        arr.append(v)
    return (arr, miss)

rows = []
for cfg in CONFIGS:
    q = pd.read_csv(os.path.join(HERE, cfg["q"])).sort_values("item_number").reset_index(drop=True)
    n = len(q); rev = (q["scoring_direction"] == "reverse").values; dom = q["domain_code"].values
    prompt = build(cfg, q)
    got = 0; batch_i = 0
    while got < N_SAMPLES and batch_i < 30:
        torch.manual_seed(1000 + batch_i); torch.cuda.manual_seed_all(1000 + batch_i)
        k = min(BATCH, N_SAMPLES - got + 4)   # small over-ask to cover parse failures
        inp = tok([prompt] * k, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max(800, n * 12), do_sample=True,
                                 temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
        pl = inp["input_ids"].shape[1]
        for i in range(k):
            if got >= N_SAMPLES: break
            arr, miss = parse(tok.decode(out[i][pl:], skip_special_tokens=True), n)
            if miss > max(3, int(0.1 * n)): continue
            rc = np.where(rev, 6 - np.array(arr, float), np.array(arr, float))
            scores = {c: float(rc[dom == c].mean()) for c in CODES}
            rows.append(dict(instrument=cfg["title"], sample=got, **scores))
            got += 1
        batch_i += 1
        print(f"[{cfg['title']}] {got}/{N_SAMPLES}", flush=True)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"SAMPLES_DONE {OUT_CSV} ({len(rows)} profiles)", flush=True)
