"""
Generic QLoRA-DPO fine-tune: fresh from a given SFT model, on one preference set
at one beta. Usage: python dpo_train.py <MODEL_ID> <data.jsonl> <beta> <out_dir>
"""
import os, sys, glob, shutil
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

MODEL_ID, DATA, BETA, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."

# bridge trl 0.11.4 <-> transformers 4.46 (py3.8 caps trl at 0.11.4)
class PatchedDPOTrainer(DPOTrainer):
    def get_batch_samples(self, epoch_iterator, num_batches):
        return transformers.Trainer.get_batch_samples(self, epoch_iterator, num_batches)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"

def chat(system, user):
    m=MODEL_ID.lower()
    if "vicuna" in m or ("wizardlm" in m and "wizardlm-7b" not in m):
        return ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed answers to the user's questions. "
                f"USER: {system}\n\n{user} ASSISTANT:")
    if "wizardlm-7b" in m:
        return f"{system}\n\n{user}\n\n### Response:"
    if "tulu" in m:
        return f"<|user|>\n{system}\n\n{user}\n<|assistant|>\n"
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role":"system","content":system},
                                        {"role":"user","content":user}],
                                       tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{system}\n\n{user}\n<|assistant|>\n"   # Tulu/OLMo-SFT style

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
_nm={i:f"{int(__import__('torch').cuda.mem_get_info(i)[0]/1024**3)-3}GiB" for i in range(__import__('torch').cuda.device_count())}
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", max_memory=_nm)
model.config.use_cache = False

# robust LoRA targets: prefer standard attention proj names, else all-linear
import bitsandbytes as bnbmod
present = set()
for n, m in model.named_modules():
    if isinstance(m, (torch.nn.Linear, bnbmod.nn.Linear4bit)):
        present.add(n.split(".")[-1])
std = [x for x in ["q_proj","k_proj","v_proj","o_proj"] if x in present]
targets = std if std else "all-linear"
print(f"LoRA targets: {targets}", flush=True)

ds = load_dataset("json", data_files=DATA, split="train")
ds = ds.map(lambda ex: {"prompt": chat(SYSTEM, ex["prompt"]),
                        "chosen": ex["chosen"], "rejected": ex["rejected"]},
            remove_columns=[c for c in ds.column_names if c not in ("prompt","chosen","rejected")])

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
args = DPOConfig(output_dir=OUT, beta=BETA, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=5e-5,
    lr_scheduler_type="cosine", warmup_ratio=0.1, logging_steps=20,
    save_strategy="steps", save_steps=500, save_total_limit=2,   # resumable if interrupted
    bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=512, max_prompt_length=480, report_to=[], remove_unused_columns=False)
trainer = PatchedDPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
                            tokenizer=tok, peft_config=peft_cfg)
# resume from the latest checkpoint in OUT if a prior run was interrupted
_resume = bool(glob.glob(os.path.join(OUT, "checkpoint-*")))
if _resume: print(f"RESUMING from checkpoint in {OUT}", flush=True)
trainer.train(resume_from_checkpoint=_resume)
trainer.model.save_pretrained(OUT); tok.save_pretrained(OUT)
# final adapter saved -> drop the intermediate checkpoints to save disk
for _c in glob.glob(os.path.join(OUT, "checkpoint-*")): shutil.rmtree(_c, ignore_errors=True)
print(f"SAVED_ADAPTER {OUT}", flush=True)
