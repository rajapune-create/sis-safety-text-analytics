"""Initiating-event classifier (TF-IDF + logistic regression), a keyword baseline, and outcome rules."""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

CLASSES = [
    "bpcs_loop_failure",
    "loss_of_cooling",
    "rotating_equipment_trip",
    "operator_error",
    "utility_loss",
    "blocked_outlet",
    "spurious_trip",
]

# Mask plant identifiers so the model learns event language, not which unit had which event.
_ID_RE = re.compile(r"\b(sif[\s-]?\d{2}[\s-]?\d{3}|u\d{2}[\s-]?\w{2,3}[\s-]?\d{4}|[a-z]-\d{4}[a-z]?)\b")


def mask_ids(text: str) -> str:
    return _ID_RE.sub(" ", text)


def build_model() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, preprocessor=mask_ids)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                 preprocessor=mask_ids)),
    ])
    clf = LogisticRegression(C=5.0, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", feats), ("clf", clf)])


def out_of_fold_predict(texts, labels, seed: int = 0, folds: int = 5):
    """Every narrative is predicted by a model that never saw it. Returns (labels, confidence)."""
    model = build_model()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, list(texts), list(labels), cv=cv, method="predict_proba")
    classes = np.array(sorted(set(labels)))
    return classes[proba.argmax(1)], proba.max(1)


# Keyword baseline: the first category whose keyword appears wins. This is how a quick
# spreadsheet filter would do it, and it is fooled by context such as "cooling water checked normal".
KEYWORDS = [
    ("spurious_trip", ["no process demand", "no real process upset", "reading false", "no process deviation",
                       "process normal", "process was steady"]),
    ("loss_of_cooling", ["cooling water", "cooling tower", "cooling duty", "cooling flow"]),
    ("utility_loss", ["power", "instrument air", "steam supply", "ups ", "nitrogen"]),
    ("rotating_equipment_trip", ["pump", "compressor", "motor", "seal"]),
    ("blocked_outlet", ["block valve", "plugged", "isolated", "check valve", "strainer"]),
    ("operator_error", ["wrong", "skipped", "left closed", "instead of", "setpoint"]),
    ("bpcs_loop_failure", ["controller", "control valve", "positioner", "i/o card", "loop"]),
]


def keyword_baseline(text: str) -> str:
    for label, words in KEYWORDS:
        if any(w in text for w in words):
            return label
    return "bpcs_loop_failure"


_NOT_DEMANDED = re.compile(r"not demanded|did not act|before trip|below trip setpoint|recovered")
_TRIPPED = re.compile(r"tripped on|trip:|\btrip,|activated|worked|esd on|shut down safely|safe state")


def outcome(clean_text: str, ie_label: str) -> str:
    """demand = the SIF acted on a real process deviation; near_miss = stopped by BPCS/alarm; spurious."""
    if ie_label == "spurious_trip":
        return "spurious"
    if _NOT_DEMANDED.search(clean_text):
        return "near_miss"
    if _TRIPPED.search(clean_text):
        return "demand"
    return "near_miss"
