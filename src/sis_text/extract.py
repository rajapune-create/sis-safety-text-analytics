"""Pull SIF IDs and instrument tags out of free text and link a narrative to a SIF."""
from __future__ import annotations

import re

import pandas as pd

# SIF-20-014, SIF 20-014, sif-20-014, SIF20014
SIF_RE = re.compile(r"\bsif[\s-]?(\d{2})[\s-]?(\d{3})\b", re.I)
# U20-PT-2014, U20 PT-2014, U20-PT2014, plus controller tags U20-PIC-2014
TAG_RE = re.compile(r"\bu(\d{2})[\s-]?(p|l|t|f)(t|ic)[\s-]?(\d{4})\b", re.I)


def find_sif_ids(text: str) -> list[str]:
    return sorted({f"SIF-{u}-{n}" for u, n in SIF_RE.findall(text)})


def find_tags(text: str) -> list[str]:
    """Return transmitter tags in canonical form; a controller tag maps to its transmitter."""
    return sorted({f"U{u}-{v.upper()}T-{n}" for u, v, _kind, n in TAG_RE.findall(text)})


class SifLinker:
    """Link a narrative to one SIF: by SIF ID, else by initiator tag, else by equipment + variable."""

    def __init__(self, register: pd.DataFrame):
        self.reg = register
        self.ids = set(register.sif_id)
        self.by_tag = dict(zip(register.initiator_tag, register.sif_id))

    def link(self, raw_text: str, clean_text: str, unit: str) -> tuple[str | None, str, str]:
        """Return (sif_id or None, method, note)."""
        ids = [i for i in find_sif_ids(raw_text) if i in self.ids]
        if len(ids) == 1:
            return ids[0], "sif_id", ""
        if len(ids) > 1:
            return None, "ambiguous", "several SIF IDs: " + ", ".join(ids)

        sifs = sorted({self.by_tag[t] for t in find_tags(raw_text) if t in self.by_tag})
        if len(sifs) == 1:
            return sifs[0], "tag", ""
        if len(sifs) > 1:
            return None, "ambiguous", "several tags: " + ", ".join(sifs)

        cand = self.reg[(self.reg.unit == unit) &
                        self.reg.equipment.str.lower().apply(lambda e: e in clean_text)]
        if cand.empty:
            return None, "unlinked", "no SIF ID, tag or equipment found"
        narrowed = cand[cand.apply(lambda r: r.variable in clean_text and r.direction in clean_text, axis=1)]
        if len(narrowed) == 1:
            return narrowed.sif_id.iloc[0], "equipment+variable", ""
        pick = narrowed if len(narrowed) else cand
        return None, "ambiguous", "candidates: " + ", ".join(pick.sif_id)
