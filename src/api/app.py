"""Stage 9: FastAPI service. Send one narrative, get back the SIF it concerns, the initiating event,
the outcome and whether a person should review it.

    uvicorn src.api.app:app --reload        then open http://127.0.0.1:8000/docs
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data.preprocess import clean
from src.features.sif_linking import SifLinker
from src.models.classifier import outcome
from src.models.train import MODEL_FILE
from src.pipeline import LOW_CONFIDENCE
from src.utils.paths import DATA_RAW

app = FastAPI(title="SIS safety text analytics", version="1.0",
              description="Synthetic-data demo. Not connected to any plant system.")
_register = pd.read_csv(DATA_RAW / "sif_register.csv")
_linker = SifLinker(_register)
_model = joblib.load(MODEL_FILE)


class Narrative(BaseModel):
    text: str = Field(..., examples=["CW pump tripped, U20-TT-2014 temp hi alarm ackd, SIF-20-014 not demanded."])
    unit: str = Field(..., examples=["U20"])


class Analysis(BaseModel):
    sif_id: str | None
    link_method: str
    link_note: str
    initiating_event: str
    confidence: float
    outcome: str
    needs_review: bool


@app.get("/health")
def health():
    return {"status": "ok", "sifs_in_register": len(_register)}


@app.post("/analyse", response_model=Analysis)
def analyse(n: Narrative):
    c = clean(n.text)
    sif, method, note = _linker.link(n.text, c, n.unit)
    proba = _model.predict_proba([c])[0]
    k = int(proba.argmax())
    ie = str(_model.classes_[k])
    return Analysis(sif_id=sif, link_method=method, link_note=note, initiating_event=ie,
                    confidence=round(float(proba[k]), 3), outcome=outcome(c, ie),
                    needs_review=sif is None or float(proba[k]) < LOW_CONFIDENCE)
