"""Run the project end to end (course layout: data -> features -> model -> evaluation -> saved model).

    python main.py                       # all stages on the data in data/raw/
    python main.py --regenerate --seed 7 # make a new synthetic plant first
    python main.py --robustness          # repeat on six synthetic plants, then restore seed 42
"""
import argparse
import json
import subprocess
import sys

from src.data import generate_synthetic
from src.models import evaluate, train
from src import pipeline
from src.utils.paths import REPORTS


def run_all(quiet=False):
    tagged, scen_t, flags, years = pipeline.run()          # stages 2-4 + LOPA comparison
    metrics = evaluate.run(tagged, scen_t, flags, years)   # stage 7: scores against ground truth, figures
    model_path = train.train_and_save()                    # stage 6: final model for the API
    if not quiet:
        print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
        print(f"model saved to {model_path.relative_to(model_path.parents[1])}")
    return metrics


def robustness(seeds=(1, 2, 3, 4, 5, 42)):
    rows = []
    for s in seeds:
        subprocess.run([sys.executable, "-m", "src.data.generate_synthetic", "--seed", str(s)], check=True,
                       capture_output=True)
        subprocess.run([sys.executable, "main.py", "--quiet"], check=True, capture_output=True)
        m = json.loads((REPORTS / "validation_metrics.json").read_text())
        a, b = m["flags_ie_rate_above_lopa"], m["flags_ie_not_in_lopa"]
        rows.append((s, m["classifier_macro_f1"], m["keyword_baseline_macro_f1"], m["link_precision"],
                     a["true_positives"] + b["true_positives"], a["flagged"] + b["flagged"], a["planted"] + b["planted"]))
    subprocess.run([sys.executable, "-m", "src.data.generate_synthetic", "--seed", "42"], check=True, capture_output=True)
    subprocess.run([sys.executable, "main.py", "--quiet"], check=True, capture_output=True)
    print("| Seed | Classifier macro F1 | Keyword baseline F1 | Link precision | Correct flags / all flags | Planted found |")
    print("|---|---|---|---|---|---|")
    for s, f1, kf1, lp, tp, fl, pl in rows:
        print(f"| {s} | {f1:.2f} | {kf1:.2f} | {lp:.3f} | {tp} / {fl} | {tp} of {pl} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.robustness:
        robustness()
    else:
        if a.regenerate:
            sys.argv = ["generate_synthetic", "--seed", str(a.seed)]
            generate_synthetic.main()
        run_all(a.quiet)
