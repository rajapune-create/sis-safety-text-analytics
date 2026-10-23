import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sis_text.analysis import benjamini_hochberg, poisson_exceed_p, scenario_table  # noqa: E402
from sis_text.classify import keyword_baseline, outcome  # noqa: E402
from sis_text.extract import SifLinker, find_sif_ids, find_tags  # noqa: E402
from sis_text.textclean import clean  # noqa: E402


@pytest.mark.parametrize("text", ["SIF-20-014 tripped", "sif 20-014 acted", "SIF20014 did not act", "per sif-20-014"])
def test_sif_id_variants(text):
    assert find_sif_ids(text) == ["SIF-20-014"]


def test_tags_and_controller_maps_to_transmitter():
    assert find_tags("U20 PT-2014 high, u20-pic-2014 in manual, U30-LT3001") == ["U20-PT-2014", "U30-LT-3001"]


def test_no_false_tag_from_equipment_numbers():
    assert find_tags("reactor R-2101 and pump P-1103A") == []


def test_clean_expands_shorthand_without_touching_words():
    assert clean("CW lost, PRESS HI, flow below setpoint") == "cooling water lost, pressure high, flow below setpoint"


REG = pd.DataFrame([
    dict(sif_id="SIF-20-001", unit="U20", initiator_tag="U20-PT-2001", equipment="reactor R-2101",
         variable="pressure", direction="high"),
    dict(sif_id="SIF-20-002", unit="U20", initiator_tag="U20-TT-2002", equipment="reactor R-2101",
         variable="temperature", direction="high"),
])


def test_linker_by_tag():
    raw = "Alarm U20-TT-2002 temp hi came in"
    assert SifLinker(REG).link(raw, clean(raw), "U20")[:2] == ("SIF-20-002", "tag")


def test_linker_equipment_needs_variable_to_disambiguate():
    raw = "reactor R-2101: temperature high trip, the SIF worked"
    assert SifLinker(REG).link(raw, clean(raw), "U20")[:2] == ("SIF-20-002", "equipment+variable")
    raw = "reactor R-2101: unit upset"
    assert SifLinker(REG).link(raw, clean(raw), "U20")[1] == "ambiguous"


def test_outcome_rules():
    assert outcome("alarm acknowledged, sif-20-001 not demanded", "utility_loss") == "near_miss"
    assert outcome("sif-20-001 tripped on pressure high", "utility_loss") == "demand"
    assert outcome("anything", "spurious_trip") == "spurious"


def test_keyword_baseline_is_fooled_by_context():
    # The cause is a pump trip, but a context sentence mentions cooling water first in the keyword order.
    assert keyword_baseline("pump tripped on overload. cooling water checked normal.") == "loss_of_cooling"


def test_poisson_exceed_p_known_values():
    assert poisson_exceed_p(0, 0.5) == 1.0
    assert poisson_exceed_p(1, 0.5) == pytest.approx(1 - np.exp(-0.5))
    assert poisson_exceed_p(3, 0.5) == pytest.approx(0.01439, abs=1e-5)


def test_benjamini_hochberg_matches_hand_calculation():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20])


def test_scenario_table_flags_only_the_excess():
    scen = pd.DataFrame([
        dict(sif_id="S1", ie_category="a", ief_per_yr=0.1, assumed_sif_demand_per_yr=0.001),
        dict(sif_id="S2", ie_category="a", ief_per_yr=0.1, assumed_sif_demand_per_yr=0.001),
    ])
    ev = pd.DataFrame([dict(narrative_id=f"N{i}", sif_id="S1", ie_category="a", outcome="near_miss") for i in range(6)])
    t = scenario_table(ev, scen, years=5).set_index("sif_id")
    assert bool(t.loc["S1", "flag_ie_rate_above_lopa"]) and not bool(t.loc["S2", "flag_ie_rate_above_lopa"])
    assert t.loc["S1", "observed_ief_per_yr"] == pytest.approx(1.2)
