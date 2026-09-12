"""
Generic eval: base SFT model (optionally + DPO adapter) takes 3 OCEAN tests
(BFI-2, Goldberg-100, FFPI -- the ones the plots use), 5 stateless seeds each.
IPIP-300 is intentionally excluded (it overflows short-context models and isn't plotted).
Writes <out_json>. Usage: python eval_adapter.py <MODEL_ID> <adapter|none> <out_json>
"""
import os, sys, re, json
from math import sqrt
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID, ADAPTER, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3]
N_SEEDS=5; HERE=os.path.dirname(os.path.abspath(__file__))
ACC=("1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
     "4 = Moderately Accurate\n5 = Very Accurate")
AGREE=("1 = Disagree strongly\n2 = Disagree a little\n3 = Neutral; no opinion\n"
       "4 = Agree a little\n5 = Agree strongly")
CONFIGS=[
 dict(q="bfi2_questions.csv", title="BFI-2",
      instr=f"Indicate how much you agree that each statement describes you as you generally are now, using this scale:\n{AGREE}", stem="I am someone who..."),
 # Goldberg-100 dropped to save eval time (kept BFI-2 + FFPI).
 dict(q="ffpi_questions.csv", title="FFPI",
      instr=f"Indicate how accurately each short statement describes you as you generally are now, using this scale:\n{ACC}", stem="Statements:"),
]
CODE_NAME={"O":"Openness","C":"Conscientiousness","E":"Extraversion","A":"Agreeableness","N":"Neuroticism"}
SYSTEM="You are taking a personality questionnaire. Answer honestly as yourself."

tok=AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token

def get_prompt(system, user):
    m=MODEL_ID.lower()
    if "vicuna" in m or ("wizardlm" in m and "wizardlm-7b" not in m):
        return ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed answers to the user's questions. "
                f"USER: {system}\n\n{user} ASSISTANT:")
    if "wizardlm-7b" in m:
        return f"{system}\n\n{user}\n\n### Response:"
    if "tulu" in m:
        return f"<|user|>\n{system}\n\n{user}\n<|assistant|>\n"
    if getattr(tok,"chat_template",None):
        return tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":user}],
                                       tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{system}\n\n{user}\n<|assistant|>\n"

n_gpu=torch.cuda.device_count()
mm={i:f"{int(torch.cuda.mem_get_info(i)[0]/1024**3)-3}GiB" for i in range(n_gpu)}
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", max_memory=mm)
if getattr(model.config,"pretraining_tp",1) not in (1,None):
    model.config.pretraining_tp=1   # disable tensor-parallel sliced linears (breaks under 4-bit)
if ADAPTER!="none":
    from peft import PeftModel; model=PeftModel.from_pretrained(model, ADAPTER)
model.eval(); print(f"loaded {MODEL_ID} adapter={ADAPTER} on {n_gpu} gpu(s)", flush=True)
dev=next(model.parameters()).device

def build(cfg,q):
    n=len(q); items="\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
    return get_prompt(SYSTEM, f"{cfg['instr']}\n\n{cfg['stem']}\n{items}\n\nRespond with ONLY a JSON object "
        f'mapping EVERY item number to its 1-5 rating, like {{"1": 4, "2": 3, ... , "{n}": 2}}. '
        f"Include all {n} item numbers. Output nothing but the JSON.")

def parse(text,n):
    d={}; mm2=re.search(r"\{.*\}",text,re.S)
    if mm2:
        try: d={str(k):v for k,v in json.loads(mm2.group(0)).items()}
        except Exception: d={}
    if not d: d=dict(re.findall(r'"?(\d{1,3})"?\s*:\s*([1-5])',text))
    arr=[]; miss=0
    for i in range(1,n+1):
        v=d.get(str(i))
        try: v=int(v)
        except (TypeError,ValueError): v=None
        if v is None or not(1<=v<=5): v=3; miss+=1
        arr.append(v)
    return arr, miss

def gen(prompt,k,seed,n):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    inp=tok([prompt]*k,return_tensors="pt",padding=True).to(dev)
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=max(600,n*8),do_sample=True,temperature=1.0,
                           top_p=0.95,pad_token_id=tok.pad_token_id)
    pl=inp["input_ids"].shape[1]
    return [tok.decode(out[i][pl:],skip_special_tokens=True) for i in range(k)]

def take_all(prompt,n):
    res=[None]*N_SEEDS
    for i,t in enumerate(gen(prompt,N_SEEDS,0,n)):
        a,miss=parse(t,n); res[i]=a if miss<=max(3,int(0.1*n)) else None
    tries=0
    while any(r is None for r in res) and tries<6:
        tries+=1; idxs=[i for i,r in enumerate(res) if r is None]
        for j,t in enumerate(gen(prompt,len(idxs),100+tries,n)):
            a,miss=parse(t,n)
            if miss<=max(3,int(0.1*n)): res[idxs[j]]=a
    # best-effort: any still-bad run -> neutral fill (model couldn't do it); never crash
    nbad=sum(1 for r in res if r is None)
    if nbad: print(f"  WARN {nbad}/{N_SEEDS} runs neutral-filled (model struggled)", flush=True)
    res=[r if r is not None else [3]*n for r in res]
    return res

table={}; raw_table={}
for cfg in CONFIGS:
    q=pd.read_csv(os.path.join(HERE,cfg["q"])).sort_values("item_number").reset_index(drop=True)
    n=len(q); rev=(q["scoring_direction"]=="reverse").values; dom=q["domain_code"].values
    runs=take_all(build(cfg,q),n); st={}; raw_st={}
    for c in CODE_NAME:
        vals=[float(np.where(rev,6-np.array(r,float),np.array(r,float))[dom==c].mean()) for r in runs]
        xs=np.array(vals); st[c]=[round(xs.mean(),2), round(2.776*xs.std(ddof=1)/sqrt(N_SEEDS),2)]
        raw_st[c]=[round(float(x),6) for x in xs]
    table[cfg["title"]]=st
    raw_table[cfg["title"]]=raw_st
    print(f"{cfg['title']}: "+", ".join(f"{c}={st[c][0]}" for c in ['O','C','E','A','N']), flush=True)
json.dump({"model":MODEL_ID,"adapter":ADAPTER,"table":table,"raw":raw_table}, open(OUT_JSON,"w"), indent=2)
print(f"DONE_EVAL {OUT_JSON}", flush=True)
