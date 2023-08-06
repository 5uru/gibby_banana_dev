from transformers import pipeline
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True)

if __name__ == "__main__":
    download_model()