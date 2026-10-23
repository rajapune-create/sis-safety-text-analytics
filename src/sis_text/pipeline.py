"""End-to-end run: narratives -> cleaned text -> SIF link + initiating event + outcome -> LOPA comparison.

Writes outputs/narratives_tagged.csv, outputs/scenario_review.csv and outputs/ehss_review_pack.xlsx.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import scenario_table, unanalysed_ie
from .classify import keyword_baseline, out_of_fold_predict, outcome
from .extract import SifLinker
from .textclean import clean

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data", ROOT / "outputs"
LOW_CONFIDENCE = 0.55


def load():
    reg = pd.read_csv(DATA / "sif_register.csv")
    scen = pd.read_csv(DATA / "lopa_scenarios.csv")
    narr = pd.read_csv(DATA / "narratives.csv", parse_dates=["date"])
    labels = pd.read_csv(DATA / "ground_truth" / "narrative_labels.csv")
    return reg, scen, narr, labels


def tag_narratives(reg, narr, labels):
    """Clean, link and classify every narrative. Labels train the classifier (out-of-fold)."""
    df = narr.merge(labels[["narrative_id", "true_ie_category"]], on="narrative_id")
    df["clean_text"] = df.text.map(clean)
    linker = SifLinker(reg)
    linked = [linker.link(r, c, u) for r, c, u in zip(df.text, df.clean_text, df.unit)]
    df["sif_id"], df["link_method"], df["link_note"] = zip(*linked)
    df["ie_category"], df["ie_confidence"] = out_of_fold_predict(df.clean_text, df.true_ie_category)
    df["ie_keyword_baseline"] = df.clean_text.map(keyword_baseline)
    df["outcome"] = [outcome(t, c) for t, c in zip(df.clean_text, df.ie_category)]
    df["needs_review"] = df.sif_id.isna() | (df.ie_confidence < LOW_CONFIDENCE)
    return df.drop(columns=["true_ie_category"])


def observation_years(narr) -> float:
    span = (narr.date.max() - narr.date.min()).days + 1
    return round(span / 365.25, 2)


def run():
    OUT.mkdir(exist_ok=True)
    reg, scen, narr, labels = load()
    years = observation_years(narr)
    tagged = tag_narratives(reg, narr, labels)

    events = tagged[tagged.sif_id.notna() & (tagged.outcome != "spurious")]
    scen_t = scenario_table(events, scen, years)
    new_ie = unanalysed_ie(events, scen)
    spurious = (tagged[(tagged.outcome == "spurious") & tagged.sif_id.notna()]
                .groupby("sif_id").size().rename("spurious_trips").reset_index())
    sif_summary = (reg.merge(spurious, on="sif_id", how="left")
                   .merge(events.groupby("sif_id").size().rename("initiating_events").reset_index(), on="sif_id", how="left")
                   .merge(events[events.outcome == "demand"].groupby("sif_id").size().rename("sif_demands").reset_index(),
                          on="sif_id", how="left")
                   .fillna({"spurious_trips": 0, "initiating_events": 0, "sif_demands": 0}))

    flags = pd.concat([
        scen_t[scen_t.flag_ie_rate_above_lopa].assign(
            finding="Initiating event more frequent than LOPA assumed",
            evidence=lambda d: [f"{o} events in {years} yr vs {e:.2f} expected (q={q:.3f})"
                                for o, e, q in zip(d.observed_ie_events, d.expected_ie_events, d.q_value)]
        )[["finding", "sif_id", "ie_category", "evidence"]],
        new_ie.assign(finding="Initiating event not in the SIF's LOPA",
                      evidence=lambda d: [f"{n} events: {ids}" for n, ids in zip(d.observed_ie_events, d.narrative_ids)]
                      )[["finding", "sif_id", "ie_category", "evidence"]],
    ], ignore_index=True)

    tagged.to_csv(OUT / "narratives_tagged.csv", index=False)
    scen_t.to_csv(OUT / "scenario_review.csv", index=False)
    flags.to_csv(OUT / "flags.csv", index=False)

    readme = pd.DataFrame({"How to read this pack": [
        "Synthetic demonstration data only. Not from any real plant.",
        f"Observation window: {years} years, {len(tagged)} narratives, {len(reg)} SIFs, {len(scen)} LOPA scenarios.",
        "Flags: items for the EHSS / PHA team to review. Each cites the narrative IDs behind it.",
        "Scenario_review: every LOPA scenario, observed vs assumed initiating-event frequency, Poisson p and FDR q.",
        "SIF_summary: initiating events, real SIF demands and spurious trips per SIF.",
        "Narratives: each entry with linked SIF, method, predicted initiating event, confidence and outcome.",
        f"Needs_review: narratives with no SIF link or classifier confidence below {LOW_CONFIDENCE}.",
        "A flag is a prompt for engineering review, not a conclusion.",
    ]})
    with pd.ExcelWriter(OUT / "ehss_review_pack.xlsx") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        flags.to_excel(xw, sheet_name="Flags", index=False)
        scen_t.to_excel(xw, sheet_name="Scenario_review", index=False)
        sif_summary.to_excel(xw, sheet_name="SIF_summary", index=False)
        tagged.drop(columns=["clean_text"]).to_excel(xw, sheet_name="Narratives", index=False)
        tagged[tagged.needs_review].drop(columns=["clean_text"]).to_excel(xw, sheet_name="Needs_review", index=False)

    print(f"{len(tagged)} narratives over {years} yr | linked {tagged.sif_id.notna().mean():.0%} | "
          f"needs review {tagged.needs_review.sum()} | flags {len(flags)}")
    return tagged, scen_t, flags, years
