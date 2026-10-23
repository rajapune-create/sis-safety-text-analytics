"""Stage 6: fit the initiating-event classifier on all labelled narratives and save it to models/.

The scores reported in the README come from out-of-fold predictions (src/models/evaluate.py);
this saved model is the one the API serves.
"""
import json
from datetime import date

import joblib

from src.data.load_data import load_data
from src.data.preprocess import clean
from src.models.classifier import build_model
from src.utils.paths import MODELS

MODEL_FILE = MODELS / "initiating_event_classifier.joblib"


def train_and_save():
    _, _, narr, labels = load_data()
    df = narr.merge(labels[["narrative_id", "true_ie_category"]], on="narrative_id")
    model = build_model().fit(df.text.map(clean), df.true_ie_category)
    joblib.dump(model, MODEL_FILE)
    (MODELS / "model_card.json").write_text(json.dumps({
        "model": "TF-IDF (words + char n-grams) + class-balanced logistic regression",
        "trained_on": f"{len(df)} synthetic labelled narratives",
        "classes": sorted(df.true_ie_category.unique()),
        "trained": date.today().isoformat(),
        "note": "Synthetic data only. Retrain and re-validate on real labelled narratives before any use.",
    }, indent=2))
    return MODEL_FILE
