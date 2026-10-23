"""Check every step against the generator's ground truth and the planted findings. Draws the figures."""
from __future__ import annotations

import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .analysis import scenario_table, unanalysed_ie
from .classify import CLASSES
from .pipeline import DATA, OUT, ROOT

DOCS = ROOT / "docs"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"bpcs_loop_failure": "BPCS loop", "loss_of_cooling": "Cooling loss",
         "rotating_equipment_trip": "Rotating eqpt", "operator_error": "Operator error",
         "utility_loss": "Utility loss", "blocked_outlet": "Blocked outlet", "spurious_trip": "Spurious trip"}


def _flag_score(found: set, planted: set) -> dict:
    tp = len(found & planted)
    return dict(flagged=len(found), planted=len(planted), true_positives=tp,
                precision=round(tp / len(found), 3) if found else None,
                recall=round(tp / len(planted), 3) if planted else None,
                missed=sorted("/".join(x) for x in planted - found),
                false_flags=sorted("/".join(x) for x in found - planted))


def run(tagged: pd.DataFrame, scen_t: pd.DataFrame, flags: pd.DataFrame, years: float) -> dict:
    DOCS.mkdir(exist_ok=True)
    truth = pd.read_csv(DATA / "ground_truth" / "narrative_labels.csv")
    planted = pd.read_csv(DATA / "ground_truth" / "planted_findings.csv")
    scen = pd.read_csv(DATA / "lopa_scenarios.csv")
    df = tagged.merge(truth, on="narrative_id")

    m = {}
    m["narratives"] = len(df)
    m["classifier_macro_f1"] = round(f1_score(df.true_ie_category, df.ie_category, average="macro"), 3)
    m["classifier_accuracy"] = round(accuracy_score(df.true_ie_category, df.ie_category), 3)
    m["keyword_baseline_macro_f1"] = round(f1_score(df.true_ie_category, df.ie_keyword_baseline, average="macro"), 3)
    m["keyword_baseline_accuracy"] = round(accuracy_score(df.true_ie_category, df.ie_keyword_baseline), 3)
    m["outcome_accuracy"] = round(accuracy_score(df.true_outcome, df.outcome), 3)
    linked = df[df.sif_id.notna()]
    m["link_coverage"] = round(len(linked) / len(df), 3)
    m["link_precision"] = round((linked.sif_id == linked.true_sif_id).mean(), 3)
    m["link_coverage_by_method"] = {k: int(v) for k, v in df.link_method.value_counts().items()}
    m["needs_review"] = int(df.needs_review.sum())

    rate_planted = {(r.sif_id, r.ie_category) for r in planted.itertuples() if r.finding == "ie_rate_above_lopa"}
    new_planted = {(r.sif_id, r.ie_category) for r in planted.itertuples() if r.finding == "ie_not_in_lopa"}
    rate_found = {(r.sif_id, r.ie_category) for r in flags.itertuples() if r.finding.startswith("Initiating event more")}
    new_found = {(r.sif_id, r.ie_category) for r in flags.itertuples() if r.finding.startswith("Initiating event not")}
    m["flags_ie_rate_above_lopa"] = _flag_score(rate_found, rate_planted)
    m["flags_ie_not_in_lopa"] = _flag_score(new_found, new_planted)

    # Same statistics on the true labels and true links: how much is lost to text analytics vs to chance.
    ev = df[df.true_outcome != "spurious"].rename(columns={"true_sif_id": "s", "true_ie_category": "c", "true_outcome": "o"})
    ev = ev[["narrative_id", "s", "c", "o"]].rename(columns={"s": "sif_id", "c": "ie_category", "o": "outcome"})
    oracle = scenario_table(ev, scen, years)
    m["oracle_flags_ie_rate_above_lopa"] = _flag_score(
        {(r.sif_id, r.ie_category) for r in oracle[oracle.flag_ie_rate_above_lopa].itertuples()}, rate_planted)
    m["oracle_flags_ie_not_in_lopa"] = _flag_score(
        {(r.sif_id, r.ie_category) for r in unanalysed_ie(ev, scen).itertuples()}, new_planted)

    (OUT / "validation_metrics.json").write_text(json.dumps(m, indent=2))
    _figures(df, scen_t, rate_planted, years)
    return m


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figures(df, scen_t, rate_planted, years):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK})

    # 1. Observed vs LOPA-expected initiating events per scenario.
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    _style(ax)
    rng = np.random.default_rng(0)
    x = scen_t.expected_ie_events * np.exp(rng.normal(0, 0.06, len(scen_t)))
    y = scen_t.observed_ie_events + rng.uniform(-0.15, 0.15, len(scen_t))
    flag = scen_t.flag_ie_rate_above_lopa.values
    ax.scatter(x[~flag], y[~flag], s=22, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.6,
               label="LOPA scenario, within assumption")
    ax.scatter(x[flag], y[flag], s=64, color=ORANGE, marker="D", edgecolor="white", linewidth=1.2,
               label="Flagged: more events than LOPA assumed (FDR 10%)")
    lim = np.array([0.2, max(scen_t.expected_ie_events.max(), 1) * 1.3])
    xs = np.geomspace(*lim, 100)
    ax.plot(xs, xs, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("observed = assumed", (xs[-12], xs[-12]), xytext=(-6, 8), textcoords="offset points",
                color=MUTED, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(lim)
    ticks = [t for t in (0.25, 0.5, 1, 2, 5, 10) if lim[0] <= t <= lim[1]]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f"Initiating events expected by LOPA in {round(years):g} years (IEF x years, log scale)", fontsize=9.5)
    ax.set_ylabel("Initiating events found in narratives", fontsize=9.5)
    ax.set_title("Where the narratives disagree with the LOPA", fontsize=12, loc="left", color=INK, pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for r in scen_t[flag].nlargest(3, "observed_ie_events").itertuples():
        ax.annotate(f"{r.sif_id}\n{SHORT[r.ie_category]}", (r.expected_ie_events, r.observed_ie_events),
                    xytext=(8, -4), textcoords="offset points", fontsize=7.5, color=INK)
    fig.text(0.01, 0.01, "Synthetic data. Points jittered slightly so overlapping counts stay visible.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(DOCS / "observed_vs_lopa.png")
    plt.close(fig)

    # 2. Confusion matrix (row-normalised) for the initiating-event classifier.
    labels = [c for c in CLASSES if c in set(df.true_ie_category)]
    cm = confusion_matrix(df.true_ie_category, df.ie_category, labels=labels)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)), [SHORT[c] for c in labels], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)), [SHORT[c] for c in labels], fontsize=8.5)
    ax.set_xlabel("Predicted initiating event", fontsize=9.5)
    ax.set_ylabel("True initiating event", fontsize=9.5)
    f1 = f1_score(df.true_ie_category, df.ie_category, average="macro")
    ax.set_title(f"Initiating-event classifier, out-of-fold (macro F1 {f1:.2f})", fontsize=11.5, loc="left", pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "confusion_matrix.png")
    plt.close(fig)
