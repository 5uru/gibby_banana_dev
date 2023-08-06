# This file runs during container build time to get model weights built into the container

from transformers import pipeline

def download_model():
    model_name = "mosaicml/mpt-7b-chat"
    pipeline("text-generation", model=model_name)

if __name__ == "__main__":
    download_model()