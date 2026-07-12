import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download

print("Downloading TinyStoriesV2-GPT4-train.txt...")

path = hf_hub_download(
    repo_id="roneneldan/TinyStories",
    filename="TinyStoriesV2-GPT4-train.txt",
    repo_type="dataset",
    local_dir="./data",
)

print("Success!")
print("Saved to:", path)