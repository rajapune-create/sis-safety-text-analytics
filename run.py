"""Run the full pipeline on the synthetic data, then validate it against the ground truth."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sis_text import pipeline, validate  # noqa: E402

if __name__ == "__main__":
    tagged, scen_t, flags, years = pipeline.run()
    metrics = validate.run(tagged, scen_t, flags, years)
    print(json.dumps(metrics, indent=2))
