"""Stage 2: load the raw (synthetic) data."""
import pandas as pd

from src.utils.paths import DATA_RAW, GROUND_TRUTH


def load_data():
    """Returns (sif_register, lopa_scenarios, narratives, narrative_labels)."""
    reg = pd.read_csv(DATA_RAW / "sif_register.csv")
    scen = pd.read_csv(DATA_RAW / "lopa_scenarios.csv")
    narr = pd.read_csv(DATA_RAW / "narratives.csv", parse_dates=["date"])
    labels = pd.read_csv(GROUND_TRUTH / "narrative_labels.csv")
    return reg, scen, narr, labels
