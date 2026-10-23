"""Normalise shift-log style text: case, shorthand, whitespace."""
from __future__ import annotations

import re

# Field shorthand -> plain word. Order matters for multi-word expansions.
SHORTHAND = {
    r"\bcw\b": "cooling water",
    r"\bia\b": "instrument air",
    r"\bpress\b": "pressure",
    r"\bpr\b": "pressure",
    r"\btemp\b": "temperature",
    r"\blvl\b": "level",
    r"\bhi\b": "high",
    r"\blo\b": "low",
    r"\bopr\b": "operator",
    r"\boptr\b": "operator",
    r"\bvlv\b": "valve",
    r"\bv/vs?\b": "valve",
    r"\btrpd\b": "tripped",
    r"\bstby\b": "standby",
    r"\backd\b": "acknowledged",
}


def clean(text: str) -> str:
    """Lower-case, expand shorthand and collapse whitespace. Tags are kept intact."""
    t = str(text).lower()
    for pat, rep in SHORTHAND.items():
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()
