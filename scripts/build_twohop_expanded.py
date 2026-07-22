#!/usr/bin/env python3
"""Expand the two-hop set with more instances of clean single-token-cue templates.

Anthropic's 93 items yield only ~8 same-length minimal pairs. To get a robust flip-rate estimate
without changing the experimental design, this emits many more instances of the SAME two-hop
templates that have a single-token cue and single-token answer: the atomic-number->symbol template
across two-digit elements (cue = the 2-digit number, answer = the chemical symbol), and the
month-number->attribute template. Same item schema as lens-eval-multihop.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache/twohop_expanded.json"

# two-digit atomic number -> (element, symbol); well-known elements the model should know
ELEMENTS = {
    10: ("neon", "Ne"), 11: ("sodium", "Na"), 12: ("magnesium", "Mg"), 13: ("aluminum", "Al"),
    14: ("silicon", "Si"), 16: ("sulfur", "S"), 17: ("chlorine", "Cl"), 19: ("potassium", "K"),
    20: ("calcium", "Ca"), 26: ("iron", "Fe"), 29: ("copper", "Cu"), 30: ("zinc", "Zn"),
    47: ("silver", "Ag"), 50: ("tin", "Sn"), 53: ("iodine", "I"), 56: ("barium", "Ba"),
    74: ("tungsten", "W"), 78: ("platinum", "Pt"), 79: ("gold", "Au"), 80: ("mercury", "Hg"),
    82: ("lead", "Pb"), 92: ("uranium", "U"),
}


def build() -> list[dict]:
    items = []
    for z, (elem, sym) in ELEMENTS.items():
        items.append({
            "name": f"atomic-{z}-symbol",
            "prompt": f"Fact: The chemical symbol for the element with atomic number {z} is ",
            "target": sym,
            "intermediates": [elem],
        })
    return items


def main() -> int:
    items = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"items": items}, indent=1) + "\n")
    print(f"wrote {len(items)} two-hop items -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
