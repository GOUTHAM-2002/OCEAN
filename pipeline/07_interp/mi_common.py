"""Shared helpers for the mechanistic-interpretability experiments on
WizardLM-13B-V1.2 (the model with the strongest both-direction OCEAN shift)."""
import os
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "WizardLMTeam/WizardLM-13B-V1.2"
SHORT = "WizardLM-13B-V1.2"
TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_NAME = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
              "A": "Agreeableness", "N": "Neuroticism"}
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."


def vicuna(user, system=SYSTEM):
    return ("A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed answers to the user's questions. "
            f"USER: {system}\n\n{user} ASSISTANT:")


def load_model(device="cuda:0"):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": device})
    if getattr(model.config, "pretraining_tp", 1) not in (1, None):
        model.config.pretraining_tp = 1
    model.eval()
    return model, tok


def trait_pairs(trait):
    """Item-aligned (high, low) contrastive texts for one trait: same item,
    opposite chosen answer. Returns list of (high_text, low_text)."""
    hi = [json.loads(l) for l in open(f"dpo_trait_{trait}_high.jsonl")]
    lo = [json.loads(l) for l in open(f"dpo_trait_{trait}_low.jsonl")]
    pairs = []
    for h, l in zip(hi, lo):
        assert h["item_id"] == l["item_id"]
        p = vicuna(h["prompt"])
        pairs.append((p + " " + h["chosen"], p + " " + l["chosen"]))
    return pairs


def decoder_layers(model):
    return model.model.layers
