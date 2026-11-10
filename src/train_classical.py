"""Classical baselines (Area B): edit-distance lexicon rule and tf-idf + logistic regression."""
import itertools
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import MODEL_DIR, SEED, load_splits, log_training
from .lexicon import LexiconMatcher, count_words, mine_lexicon
from .textnorm import normalize


def train_lexicon(train, val):
    best = None
    total, off = count_words(train["text"], train["label"])
    for min_count, min_ratio, fuzzy_len in itertools.product((3, 5, 10), (0.7, 0.8, 0.9), (5, 6, 7)):
        words = mine_lexicon(total, off, min_count, min_ratio)
        matcher = LexiconMatcher(words, fuzzy_len)
        pred = [int(matcher.score(t) >= 0.5) for t in val["text"]]
        f1 = f1_score(val["label"], pred, average="macro")
        if best is None or f1 > best[0]:
            best = (f1, min_count, min_ratio, fuzzy_len, words)
    f1, min_count, min_ratio, fuzzy_len, words = best
    print(f"[lexicon] val macro-F1={f1:.4f}  words={len(words)}  "
          f"min_count={min_count} min_ratio={min_ratio} fuzzy_min_len={fuzzy_len}")
    (MODEL_DIR / "lexicon.json").write_text(
        json.dumps({"fuzzy_min_len": fuzzy_len, "min_count": min_count, "min_ratio": min_ratio,
                    "words": words}, ensure_ascii=False), encoding="utf-8")
    log_training("lexicon", {"val_f1": f1, "words": len(words), "min_count": min_count,
                             "min_ratio": min_ratio, "fuzzy_min_len": fuzzy_len})


def train_tfidf(train, val):
    best = None
    for C in (1.0, 3.0, 10.0):
        pipe = Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                                         min_df=3, sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED)),
        ])
        pipe.fit(train["text"], train["label"])
        f1 = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
        print(f"[tfidf+LR] C={C}: val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, pipe, C)
    log_training("tfidf", {"val_f1": best[0], "C": best[2],
                           "features": int(best[1].named_steps["clf"].coef_.shape[1])})
    joblib.dump(best[1], MODEL_DIR / "tfidf_lr.joblib", compress=3)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train, val, test = load_splits()
    log_training("splits", {n: {"n": len(d), "off": float(d["label"].mean())}
                            for n, d in (("train", train), ("val", val), ("test", test))})
    train_lexicon(train, val)
    train_tfidf(train, val)


if __name__ == "__main__":
    main()
