# This file runs during container build time to get model weights built into the container

from transformers import pipeline
import torch

def download_model():
    model_name = "databricks/dolly-v2-3b"
    pipeline(model=model_name, torch_dtype=torch.bfloat16,
             trust_remote_code=True, device_map="auto", return_full_text=True)

if __name__ == "__main__":
    download_model()