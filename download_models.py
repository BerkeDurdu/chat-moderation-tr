"""Download the trained weights from the Hugging Face Hub into models/ (weights are not in git).

    python download_models.py            # fetch the weights (anyone)
    python download_models.py --upload   # author only: push models/ to the Hub (needs `hf auth login`)

The upload creates <your-hf-username>/chat-moderation-tr and writes that name into
HF_REPO below, so the download command works for everybody afterwards.
"""
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

HF_REPO = "USERNAME/chat-moderation-tr"
REPO_NAME = "chat-moderation-tr"
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
NEEDED = ["lexicon.json", "tfidf_lr.joblib", "w2v_ffn.pt", "bert/model.safetensors",
          "bert_onnx/model_int8.onnx", "bert_onnx/model_fp32.onnx"]


def upload() -> None:
    missing = [n for n in NEEDED if not (MODELS / n).exists()]
    if missing:
        sys.exit(f"models/ is incomplete ({', '.join(missing)}); run `python train_all.py` first.")
    api = HfApi()
    repo = f"{api.whoami()['name']}/{REPO_NAME}"
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(MODELS), repo_id=repo, repo_type="model",
                      commit_message="Trained weights for the chat moderation homework")
    me = Path(__file__)
    me.write_text(re.sub(r'^HF_REPO = ".*"$', f'HF_REPO = "{repo}"', me.read_text(encoding="utf-8"),
                         count=1, flags=re.M), encoding="utf-8")
    print(f"uploaded models/ to https://huggingface.co/{repo}")
    print("HF_REPO in download_models.py now points there - commit and push that change.")


def download() -> None:
    if HF_REPO.startswith("USERNAME/"):
        sys.exit("HF_REPO is not set yet: the author has to run `python download_models.py --upload` first.\n"
                 "Alternatively, train everything locally with `python train_all.py`.")
    snapshot_download(HF_REPO, repo_type="model", local_dir=str(MODELS))
    missing = [n for n in NEEDED if not (MODELS / n).exists()]
    if missing:
        sys.exit(f"download incomplete, missing: {', '.join(missing)}")
    print(f"models downloaded to {MODELS}")


if __name__ == "__main__":
    upload() if "--upload" in sys.argv else download()
