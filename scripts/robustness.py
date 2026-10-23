"""Re-run the whole pipeline on several synthetic plants (different seeds) so results are not one lucky draw.

Restores the default seed-42 data and outputs at the end.
    python scripts/robustness.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 2, 3, 4, 5, 42]


def sh(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


rows = []
for s in SEEDS:
    sh("data/generate_synthetic.py", "--seed", str(s))
    sh("run.py")
    m = json.loads((ROOT / "outputs" / "validation_metrics.json").read_text())
    a, b = m["flags_ie_rate_above_lopa"], m["flags_ie_not_in_lopa"]
    rows.append((s, m["classifier_macro_f1"], m["keyword_baseline_macro_f1"], m["link_precision"],
                 a["true_positives"] + b["true_positives"], a["flagged"] + b["flagged"], a["planted"] + b["planted"]))
sh("data/generate_synthetic.py", "--seed", "42")
sh("run.py")

print("| Seed | Classifier macro F1 | Keyword baseline F1 | Link precision | Correct flags / all flags | Planted found |")
print("|---|---|---|---|---|---|")
for s, f1, kf1, lp, tp, fl, pl in rows:
    print(f"| {s} | {f1:.2f} | {kf1:.2f} | {lp:.3f} | {tp} / {fl} | {tp} of {pl} |")
