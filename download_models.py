"""Download the trained weights from the Hugging Face Hub into models/ (weights are not in git).

    python download_models.py            # fetch
    python download_models.py --upload   # (author only) push models/ to the Hub
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

HF_REPO = "USERNAME/chat-moderation-tr"      # <- set to your Hugging Face model repository
MODELS = Path(__file__).resolve().parent / "models"

if __name__ == "__main__":
    if "--upload" in sys.argv:
        api = HfApi()
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(MODELS), repo_id=HF_REPO, repo_type="model")
        print(f"uploaded models/ to https://huggingface.co/{HF_REPO}")
    else:
        snapshot_download(HF_REPO, local_dir=str(MODELS))
        print(f"models downloaded to {MODELS}")
