"""Compare what the narratives show with what the LOPA assumed."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """False-discovery-rate adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def poisson_exceed_p(observed: int, expected: float) -> float:
    """P(N >= observed) when N ~ Poisson(expected). Small = more events than the LOPA allows for."""
    if observed <= 0:
        return 1.0
    return float(poisson.sf(observed - 1, expected))


def scenario_table(events: pd.DataFrame, scenarios: pd.DataFrame, years: float, fdr: float = 0.10) -> pd.DataFrame:
    """One row per LOPA scenario: observed initiating events vs IEF x years, with an FDR-controlled flag.

    `events` has columns sif_id, ie_category, outcome (only linked, non-spurious narratives).
    """
    counts = events.groupby(["sif_id", "ie_category"]).size().rename("observed_ie_events")
    demands = events[events.outcome == "demand"].groupby(["sif_id", "ie_category"]).size().rename("observed_sif_demands")
    t = scenarios.set_index(["sif_id", "ie_category"]).join(counts).join(demands).fillna(
        {"observed_ie_events": 0, "observed_sif_demands": 0}).reset_index()
    t["observed_ie_events"] = t.observed_ie_events.astype(int)
    t["observed_sif_demands"] = t.observed_sif_demands.astype(int)
    t["expected_ie_events"] = t.ief_per_yr * years
    t["observed_ief_per_yr"] = t.observed_ie_events / years
    t["ratio_observed_to_assumed"] = t.observed_ief_per_yr / t.ief_per_yr
    t["expected_sif_demands"] = t.assumed_sif_demand_per_yr * years
    t["p_value"] = [poisson_exceed_p(o, e) for o, e in zip(t.observed_ie_events, t.expected_ie_events)]
    t["q_value"] = benjamini_hochberg(t.p_value.values)
    t["flag_ie_rate_above_lopa"] = t.q_value < fdr
    return t.sort_values("q_value")


def unanalysed_ie(events: pd.DataFrame, scenarios: pd.DataFrame, min_events: int = 2) -> pd.DataFrame:
    """Initiating events seen on a SIF that its LOPA never considered (at least `min_events` times)."""
    known = set(zip(scenarios.sif_id, scenarios.ie_category))
    g = events.groupby(["sif_id", "ie_category"]).agg(
        observed_ie_events=("narrative_id", "size"),
        narrative_ids=("narrative_id", lambda s: ", ".join(sorted(s)))).reset_index()
    g = g[[(s, c) not in known for s, c in zip(g.sif_id, g.ie_category)]]
    return g[g.observed_ie_events >= min_events].sort_values("observed_ie_events", ascending=False)
