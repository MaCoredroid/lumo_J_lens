#!/usr/bin/env python3
"""Build the v5 capture: fill the data-driven vocab + union pool + rewrite bounded prompts (P7c).

Produces (1) a data-driven concept vocab with the 3 underivable rare families filled from v2, and
(2) a bounded prompts-file whose score_token_ids are the UNION of the v2 (126) and data-driven
token pools, so one readout lets us score v2-vocab, data-driven-vocab, AND super-concepts at any
layer -- a decisive single-capture comparison of whether better per-concept forms can beat the
0.41 fine / 0.59 super numbers. CPU-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"
DD = ROOT / "configs/swe_task_state_v4_datadriven_concept_forms_v3.json"
DD_FILLED = ROOT / "configs/swe_task_state_v4_datadriven_concept_forms_v3_filled.json"
BOUNDED = ROOT / ".cache/swe_jlens_cohort/cohort-perturn-prompts-bounded.json"
POOL_PROMPTS = ROOT / ".cache/swe_jlens_cohort/cohort-perturn-prompts-pool-v5.json"


def _fam_ids(doc) -> dict[str, list[int]]:
    return {fam: [f["token_id"] for f in forms] for fam, forms in doc["families"].items()}


def build() -> dict[str, Any]:
    v2 = json.loads(V2.read_text())
    dd = json.loads(DD.read_text())

    # fill families missing from the data-driven derivation with the v2 curated forms
    filled_families = dict(dd["families"])
    for fam, forms in v2["families"].items():
        if fam not in filled_families:
            filled_families[fam] = forms
    dd_filled = {
        "kind": "swe_task_state_v4_datadriven_concept_forms_v3_filled",
        "n_families": len(filled_families),
        "filled_from_v2": [f for f in v2["families"] if f not in dd["families"]],
        "families": filled_families,
    }
    DD_FILLED.write_text(json.dumps(dd_filled, indent=2) + "\n")

    # union token pool for the capture's scored tokens
    pool = sorted(
        {tid for forms in v2["families"].values() for f in forms for tid in [f["token_id"]]}
        | {tid for forms in filled_families.values() for f in forms for tid in [f["token_id"]]}
    )

    # rewrite the bounded prompts with score_token_ids = pool (token_ids/prompt unchanged)
    prompts = json.loads(BOUNDED.read_text())
    for p in prompts:
        p["score_token_ids"] = pool
    POOL_PROMPTS.write_text(json.dumps(prompts))

    return {
        "n_families_filled": len(filled_families),
        "filled_from_v2": dd_filled["filled_from_v2"],
        "pool_size": len(pool),
        "n_prompts": len(prompts),
        "dd_filled": str(DD_FILLED),
        "pool_prompts": str(POOL_PROMPTS),
    }


def main() -> int:
    print(json.dumps(build(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
