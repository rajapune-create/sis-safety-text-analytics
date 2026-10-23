"""Generate a fully synthetic SIF register, LOPA scenario table and incident narratives.

Nothing here comes from a real plant. Units, tags, equipment and events are invented.
The generator also writes the ground truth (true initiating event, outcome, SIF and the
list of deliberately "planted" findings) so the pipeline can be validated against it.

Usage:
    python -m src.data.generate_synthetic            # writes CSVs into data/raw/
    python -m src.data.generate_synthetic --seed 7   # a different synthetic plant
"""
from __future__ import annotations

import argparse
import random
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "raw"
YEARS = 5
START = date(2021, 1, 1)

UNITS = ["10", "20", "30", "40", "50", "60"]
UNIT_NAMES = {
    "10": "Feed preparation",
    "20": "Reactor section",
    "30": "Separation",
    "40": "Compression",
    "50": "Utilities",
    "60": "Product storage",
}

VARIABLES = {  # tag prefix -> (variable word, typical trip directions)
    "PT": ("pressure", ["high", "low"]),
    "LT": ("level", ["high", "low"]),
    "TT": ("temperature", ["high"]),
    "FT": ("flow", ["low"]),
}

EQUIPMENT = {
    "10": ["feed drum V-1101", "feed heater H-1102", "feed pump P-1103A", "mixing drum V-1104"],
    "20": ["reactor R-2101", "reactor R-2102", "reactor effluent cooler E-2103", "quench drum V-2104"],
    "30": ["separator V-3101", "column C-3102", "reboiler E-3103", "reflux drum V-3104"],
    "40": ["compressor K-4101", "suction drum V-4102", "interstage cooler E-4103", "discharge drum V-4104"],
    "50": ["steam drum V-5101", "boiler B-5102", "CW pump P-5103A", "IA receiver V-5104"],
    "60": ["storage tank T-6101", "storage tank T-6102", "loading pump P-6103A", "flare KO drum V-6104"],
}

IE_CATEGORIES = [
    "bpcs_loop_failure",
    "loss_of_cooling",
    "rotating_equipment_trip",
    "operator_error",
    "utility_loss",
    "blocked_outlet",
]
# Typical textbook LOPA initiating-event frequencies (per year), order of magnitude only.
IEF = {
    "bpcs_loop_failure": 0.1,
    "loss_of_cooling": 0.1,
    "rotating_equipment_trip": 0.5,
    "operator_error": 0.1,
    "utility_loss": 0.1,
    "blocked_outlet": 0.1,
}

IE_PHRASES = {
    "bpcs_loop_failure": [
        "{loop} controller output saturated at 100% after valve positioner failure",
        "control valve {cv} stuck open, {loop} loop could not control",
        "{loop} found in manual with setpoint drifting, controller not responding",
        "DCS I/O card fault on {loop} loop, valve went to fail position",
        "transmitter on {loop} loop froze, controller drove valve wide open",
    ],
    "loss_of_cooling": [
        "cooling water supply to {eq} lost",
        "CW header pressure dropped after cooling tower pump trip",
        "fouled exchanger gave reduced cooling duty on {eq}",
        "cooling tower fan tripped, CW return temperature rising",
        "cooling water valve to {eq} closed in error, no cooling flow",
    ],
    "rotating_equipment_trip": [
        "running pump tripped on motor overload, standby did not autostart",
        "compressor surged and tripped on vibration high",
        "pump mechanical seal failed, pump stopped by field operator",
        "motor tripped on earth fault, flow to {eq} lost",
        "booster pump coupling failure, discharge flow collapsed",
    ],
    "operator_error": [
        "wrong valve lined up during pump changeover",
        "operator opened bypass instead of block valve on {eq}",
        "startup procedure step skipped, {eq} filled too fast",
        "manual valve left closed after maintenance on {eq}",
        "board operator entered wrong setpoint on {loop}",
    ],
    "utility_loss": [
        "power dip on substation bus, several motors stopped",
        "instrument air header pressure low, valves drifted",
        "steam supply interrupted to {eq}",
        "UPS changeover glitch caused loss of field power",
        "nitrogen supply lost to {eq} blanketing",
    ],
    "blocked_outlet": [
        "downstream block valve found closed on {eq} outlet",
        "outlet line plugged with solids, no flow from {eq}",
        "downstream unit isolated without notice, {eq} outlet blocked",
        "check valve stuck closed on {eq} discharge",
        "outlet strainer choked on {eq}",
    ],
}

# Less specific wording a tired shift writer might use for the same cause.
VAGUE_IE = {
    "bpcs_loop_failure": ["{loop} loop misbehaving, valve hunting", "controller output erratic on {loop}"],
    "loss_of_cooling": ["{eq} outlet running hot, exchanger not keeping up", "heat removal lost on {eq}"],
    "rotating_equipment_trip": ["machine stopped unexpectedly", "driver stopped, flow fell away"],
    "operator_error": ["line-up issue during shift handover on {eq}", "human error during changeover"],
    "utility_loss": ["site-wide disturbance, several drives dropped out", "supply header disturbance hit {eq}"],
    "blocked_outlet": ["no flow out of {eq}, line found restricted", "{eq} could not discharge"],
}
SHORT_CAUSE = {
    "bpcs_loop_failure": "a control loop fault", "loss_of_cooling": "loss of cooling water",
    "rotating_equipment_trip": "a pump trip", "operator_error": "an operator error",
    "utility_loss": "a power dip", "blocked_outlet": "a blocked outlet",
}

# Context sentences that mention another category's keywords without being the cause.
DISTRACTORS = [
    "Cooling water checked normal.",
    "Instrument air pressure was normal throughout.",
    "No pump trip recorded.",
    "Power supply confirmed healthy.",
    "Control valve stroke checked OK.",
    "Outlet path verified open.",
    "Procedure followed as written.",
]

SPURIOUS_CAUSES = [
    "transmitter {tag} found drifting, no real process upset",
    "impulse line of {tag} plugged, reading false",
    "loose terminal on {tag} loop wiring, no process deviation",
    "trip solenoid failed de-energised, process was steady",
    "{tag} reading spiked during loop check, process normal",
]

NEAR_MISS_OUTCOMES = [
    "{var} {dirn} alarm on {tag} acknowledged, board operator {action}; {sif} not demanded.",
    "Alarm {tag} {var} {dirn} came in, operator {action} and process recovered before trip.",
    "BPCS and operator response held {var} below trip setpoint; {sif} did not act.",
]
OPERATOR_ACTIONS = [
    "reduced feed",
    "opened bypass to restore flow",
    "started standby pump",
    "took the loop to manual and stabilised",
    "isolated the affected line",
]
DEMAND_OUTCOMES = [
    "{sif} tripped on {var} {dirn} ({tag} reached trip setpoint), {eq} shut down safely.",
    "{var} {dirn} trip: {sif} activated from {tag}, final elements closed as designed.",
    "ESD on {eq}: {tag} {var} {dirn} trip, {sif} worked, unit brought to safe state.",
]
SPURIOUS_OUTCOMES = [
    "{sif} tripped with no process demand. {cause}. Restarted after {hrs} hrs.",
    "Unplanned shutdown of {eq} by {sif}; {cause}. Unit restarted after {hrs} hrs.",
]

ABBREV = {  # word -> shorthand used in field logs (cleaned again by the pipeline)
    "pressure": ["press", "pr"],
    "temperature": ["temp"],
    "level": ["lvl"],
    "high": ["hi"],
    "low": ["lo"],
    "operator": ["opr", "optr"],
    "valve": ["vlv", "v/v"],
    "tripped": ["trpd", "trip"],
    "cooling water": ["CW"],
    "instrument air": ["IA"],
    "standby": ["stby"],
    "acknowledged": ["ackd"],
}


def noisy(text: str, rng: random.Random) -> str:
    """Make text look like a hurried shift-log entry: shorthand, case, a few typos."""
    for word, shorts in ABBREV.items():
        pat = re.compile(rf"\b{re.escape(word)}\b")
        if pat.search(text) and rng.random() < 0.45:
            text = pat.sub(rng.choice(shorts), text)
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 5 and w.isalpha() and rng.random() < 0.03:
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    text = " ".join(words)
    if rng.random() < 0.15:
        text = text.upper()
    return text


def sif_mention(sif_id: str, rng: random.Random) -> str:
    _, u, n = sif_id.split("-")
    return rng.choice([sif_id, f"SIF {u}-{n}", f"sif-{u}-{n}", f"SIF{u}{n}"])


def tag_mention(tag: str, rng: random.Random) -> str:
    u, t, n = tag.split("-")
    return rng.choice([tag, f"{u} {t}-{n}", f"{u}-{t}{n}", tag.lower()])


def build_register(rng: random.Random, n_sif: int = 240):
    sifs, scenarios = [], []
    per_unit = n_sif // len(UNITS)
    for u in UNITS:
        for k in range(1, per_unit + 1):
            sif_id = f"SIF-{u}-{k:03d}"
            prefix = rng.choice(list(VARIABLES))
            var, dirs = VARIABLES[prefix]
            dirn = rng.choice(dirs)
            tag = f"U{u}-{prefix}-{u}{k:02d}"
            eq = rng.choice(EQUIPMENT[u])
            sil = rng.choices([1, 2, 3], weights=[0.5, 0.42, 0.08])[0]
            sifs.append(dict(
                sif_id=sif_id, unit=f"U{u}", unit_name=UNIT_NAMES[u],
                description=f"{var.capitalize()} {dirn} protection on {eq}",
                initiator_tag=tag, variable=var, direction=dirn, equipment=eq,
                sil=sil, proof_test_interval_months=rng.choice([12, 24, 36, 48]),
            ))
            for ie in rng.sample(IE_CATEGORIES, rng.choice([1, 2, 2, 3])):
                ief = IEF[ie] * rng.choice([0.5, 1, 1, 2])
                bpcs_pfd = 1.0 if ie == "bpcs_loop_failure" else 0.1  # BPCS is not independent of its own failure
                alarm_pfd = 0.1 if rng.random() < 0.8 else 1.0
                scenarios.append(dict(
                    scenario_id=f"{sif_id}/{ie}", sif_id=sif_id, ie_category=ie,
                    ief_per_yr=ief, bpcs_ipl_pfd=bpcs_pfd, alarm_ipl_pfd=alarm_pfd,
                    assumed_sif_demand_per_yr=ief * bpcs_pfd * alarm_pfd,
                ))
    return pd.DataFrame(sifs), pd.DataFrame(scenarios)


def generate(seed: int = 42):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    sifs, scen = build_register(rng)
    sif_by_id = sifs.set_index("sif_id")

    # Planted finding A: initiating event happening far more often than LOPA assumed.
    planted_rate = set(rng.sample(list(scen.scenario_id), 10))
    # Planted finding B: an initiating event the LOPA never considered for that SIF.
    planted_new_ie = []
    for sif_id in rng.sample(list(sifs.sif_id), 5):
        have = set(scen.loc[scen.sif_id == sif_id, "ie_category"])
        planted_new_ie.append((sif_id, rng.choice([c for c in IE_CATEGORIES if c not in have])))

    events = []  # (sif_id, ie_category, true_rate)
    for r in scen.itertuples():
        mult = rng.uniform(6, 12) if r.scenario_id in planted_rate else rng.uniform(0.2, 0.9)
        events.append((r.sif_id, r.ie_category, r.ief_per_yr * mult))
    for sif_id, ie in planted_new_ie:
        events.append((sif_id, ie, 0.5))

    rows, truth = [], []

    def add(sif_id, ie, outcome):
        s = sif_by_id.loc[sif_id]
        u = s.unit[1:]
        loop = s.initiator_tag.replace("T-", "IC-", 1)
        cv = s.initiator_tag.replace(s.initiator_tag.split("-")[1], "CV")
        fill = dict(
            loop=loop, cv=cv, eq=s.equipment, tag=tag_mention(s.initiator_tag, rng),
            var=s.variable, dirn=s.direction, action=rng.choice(OPERATOR_ACTIONS),
            hrs=rng.choice([2, 4, 6, 8, 12, 18, 24]),
        )
        # How the narrative identifies the SIF: by ID (60%), tag only (25%), equipment only (12%), nothing (3%).
        how = rng.choices(["sif", "tag", "eq", "none"], weights=[60, 25, 12, 3])[0]
        fill["sif"] = sif_mention(sif_id, rng) if how == "sif" else "the SIF"
        if how in ("eq", "none"):
            fill["tag"] = "the transmitter"
            fill["loop"] = "the control loop"
            fill["cv"] = "the control valve"
        if how == "none":
            fill["eq"] = "the vessel"

        if outcome == "spurious":
            if rng.random() < 0.12:  # spurious trip logged without saying it was spurious
                text = f"{fill['sif']} tripped, {fill['eq']} shut down. Field check found nothing obvious. Restarted after {fill['hrs']} hrs."
            else:
                cause = rng.choice(SPURIOUS_CAUSES).format(**fill)
                text = rng.choice(SPURIOUS_OUTCOMES).format(cause=cause, **fill)
        else:
            style = rng.choices(["clear", "vague", "chain", "unstated"], weights=[68, 17, 10, 5])[0]
            if style == "vague":
                ie_text = rng.choice(VAGUE_IE[ie]).format(**fill)
            elif style == "chain":
                other = rng.choice([c for c in IE_CATEGORIES if c != ie])
                ie_text = (f"initially suspected {SHORT_CAUSE[other]}, later confirmed: "
                           + rng.choice(IE_PHRASES[ie]).format(**fill))
            elif style == "unstated":
                ie_text = rng.choice(["unit upset, cause under investigation",
                                      "process deviation, root cause not yet recorded"])
            else:
                ie_text = rng.choice(IE_PHRASES[ie]).format(**fill)
            tmpl = NEAR_MISS_OUTCOMES if outcome == "near_miss" else DEMAND_OUTCOMES
            text = ie_text[0].upper() + ie_text[1:] + ". " + rng.choice(tmpl).format(**fill)
        if rng.random() < 0.25:
            text += " " + rng.choice(DISTRACTORS)
        if how == "eq" and fill["eq"] not in text:
            text = f"{s.equipment}: " + text
        d = START + timedelta(days=rng.randrange(365 * YEARS))
        nid = f"N{len(rows) + 1:04d}"
        rows.append(dict(narrative_id=nid, date=d.isoformat(), unit=s.unit,
                         reported_by=rng.choice(["Shift supervisor", "Board operator", "Field operator", "Process engineer"]),
                         text=noisy(text, rng)))
        truth.append(dict(narrative_id=nid, true_ie_category=ie if outcome != "spurious" else "spurious_trip",
                          true_outcome=outcome, true_sif_id=sif_id, identified_by=how))

    for sif_id, ie, rate in events:
        n = nprng.poisson(rate * YEARS)
        p_alarm = float(scen.loc[(scen.sif_id == sif_id) & (scen.ie_category == ie), "alarm_ipl_pfd"].iloc[0]) \
            if ((scen.sif_id == sif_id) & (scen.ie_category == ie)).any() else 0.1
        for _ in range(n):
            # Most events are caught by BPCS/alarm; a fraction reach the SIF.
            outcome = "demand" if rng.random() < max(0.15, 0.5 * p_alarm) else "near_miss"
            add(sif_id, ie, outcome)
    for sif_id in sifs.sif_id:  # spurious trips, about one per SIF per 10 years
        for _ in range(nprng.poisson(0.1 * YEARS)):
            add(sif_id, None, "spurious")

    narr = pd.DataFrame(rows)
    gt = pd.DataFrame(truth)
    order = narr.sort_values("date").index
    narr, gt = narr.loc[order].reset_index(drop=True), gt.loc[order].reset_index(drop=True)
    ids = {old: f"N{i + 1:04d}" for i, old in enumerate(narr.narrative_id)}
    narr["narrative_id"] = narr.narrative_id.map(ids)
    gt["narrative_id"] = gt.narrative_id.map(ids)

    planted = pd.DataFrame(
        [dict(finding="ie_rate_above_lopa", sif_id=s.split("/")[0], ie_category=s.split("/")[1]) for s in sorted(planted_rate)]
        + [dict(finding="ie_not_in_lopa", sif_id=s, ie_category=ie) for s, ie in planted_new_ie]
    )
    return sifs, scen, narr, gt, planted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sifs, scen, narr, gt, planted = generate(a.seed)
    sifs.to_csv(OUT / "sif_register.csv", index=False)
    scen.to_csv(OUT / "lopa_scenarios.csv", index=False)
    narr.to_csv(OUT / "narratives.csv", index=False)
    gt.to_csv(OUT / "ground_truth" / "narrative_labels.csv", index=False)
    planted.to_csv(OUT / "ground_truth" / "planted_findings.csv", index=False)
    sample = OUT.parent / "sample"
    sample.mkdir(exist_ok=True)
    narr.head(25).to_csv(sample / "narratives_sample.csv", index=False)  # quick look without the full set
    print(f"{len(sifs)} SIFs, {len(scen)} LOPA scenarios, {len(narr)} narratives, {len(planted)} planted findings")


if __name__ == "__main__":
    (OUT / "ground_truth").mkdir(parents=True, exist_ok=True)
    main()
