"""Loading OffensEval-TR 2020 and the fixed train / validation / test splits.

Source: Çöltekin (2020), "A Corpus of Turkish Offensive Language on Social Media", LREC.
        https://coltekin.github.io/offensive-turkish/   (CC-BY 2.0)
"""
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
RAW_DIR = DATA_DIR / "raw" / "offenseval2020-turkish"
DATA_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
SEED = 42
LABELS = ["clean", "offensive"]            # 0 = NOT, 1 = OFF


def download() -> None:
    if RAW_DIR.exists():
        return
    print(f"Downloading {DATA_URL}")
    with urllib.request.urlopen(DATA_URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA_DIR / "raw")


def _read(path: Path, sep: str, names=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, names=names, quoting=csv.QUOTE_NONE, encoding="utf-8", dtype=str)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) with columns `text` and `label` (1 = offensive).

    test = the official OffensEval test set (3,528 messages), never used for training or tuning.
    val  = a stratified 10% of the official training set, used for model selection.
    """
    download()
    full = _read(RAW_DIR / "offenseval-tr-training-v1" / "offenseval-tr-training-v1.tsv", "\t")
    test = _read(RAW_DIR / "offenseval-tr-testset-v1" / "offenseval-tr-testset-v1.tsv", "\t")
    gold = _read(RAW_DIR / "offenseval-tr-testset-v1" / "offenseval-tr-labela-v1.tsv", ",", ["id", "subtask_a"])
    test = test.merge(gold, on="id", validate="one_to_one")

    def tidy(df):
        return pd.DataFrame({"text": df["tweet"].fillna(""), "label": (df["subtask_a"] == "OFF").astype(int)})

    full, test = tidy(full), tidy(test)
    train, val = train_test_split(full, test_size=0.1, random_state=SEED, stratify=full["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def log_training(key: str, info: dict) -> None:
    """Merge facts measured during training into results/training.json (read by make_slides.py)."""
    import json
    RESULT_DIR.mkdir(exist_ok=True)
    path = RESULT_DIR / "training.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = info
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    for name, df in zip(("train", "val", "test"), load_splits()):
        print(f"{name:5s} n={len(df):6d}  offensive={df['label'].mean():.1%}")
