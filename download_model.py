import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import snapshot_download

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--local-model-path", type=str, default="/model_artifacts")
    args = parser.parse_args()

    load_dotenv(find_dotenv())

    HF_TOKEN = os.getenv("HF_TOKEN")

    local_model_path = Path(args.local_model_path)
    local_model_path.mkdir(exist_ok=True)
    model_name = args.model_name

    # Only download safetensors checkpoint files
    allow_patterns = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.model", "*.tiktoken"]

    # - Leverage the snapshot library to download the model since the model is stored in repository using LFS
    snapshot_download(
        repo_id=model_name,
        local_dir=local_model_path,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        token=HF_TOKEN  # Optional: If you need a token to download your model
    )