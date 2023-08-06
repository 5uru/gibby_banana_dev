# This file runs during container build time to get model weights built into the container

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    model_name = "togethercomputer/LLaMA-2-7B-32K"
    # do a dry run of loading the huggingface model, which will download weights
    AutoTokenizer.from_pretrained(
        model_name,
        use_cache="cache"
    )
    AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache="cache"
    )

if __name__ == "__main__":
    download_model()