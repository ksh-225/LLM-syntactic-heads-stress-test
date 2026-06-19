"""
A.3 — Load Llama-3.2-3B in 4-bit (NF4) quantization for memory-efficient inference.
Requires a Hugging Face access token with Llama-3.2 access granted.
n_layers=28, n_heads=24, total=672 heads.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HF_TOKEN = os.environ.get("HF_TOKEN")


def load_llama():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        output_attentions=True,
        quantization_config=bnb,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()
    return tokenizer, model


if __name__ == "__main__":
    tok, mdl = load_llama()
    print("Loaded Llama-3.2-3B:", mdl.config.num_hidden_layers, "layers")
