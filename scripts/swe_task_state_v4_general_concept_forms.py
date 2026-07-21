#!/usr/bin/env python3
"""Build a TASK-INDEPENDENT concept-form vocab for the 14 scorable families (P7c).

The concept-chain's shipped vocab (swe_intermediate_concept_probes.json) is a
per-task adaptation (sympy-specific token forms), so it cannot score cohort tasks.
This module defines general, task-agnostic single-token surface forms per family
and enforces the concept-chain's own rules: every form is a single Qwen token, any
token appearing in more than one family is dropped, and a family needs >= 2
surviving forms to be scorable. Output feeds a general concept-chain readout so the
same 14 families can be scored on ANY task's residual state.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "configs/swe_task_state_v4_general_concept_forms.json"

# Candidate surface forms (leading space = mid-sentence). Deliberately broad so
# each family keeps >= 2 after single-token + global-uniqueness filtering.
CANDIDATES: dict[str, list[str]] = {
    "source_localization": [" locate", " where", " search", " locating", " pinpoint"],
    "substitution_operation": [" replace", " substitute", " swap", " replacing"],
    "located_source": [" located", " here", " spotted", " pinpointed"],
    "defined_identifier": [" identifier", " variable", " attribute", " symbol"],
    "runtime_name_failure": [
        " undefined",
        " unbound",
        " crash",
        " traceback",
        " exception",
        " broken",
        " typo",
    ],
    "failure_confirmation": [" fails", " reproduce", " reproduced", " failing"],
    "source_edit": [" edit", " modify", " change", " editing", " rewrite"],
    "repair": [" fix", " repair", " correct", " fixing", " correcting"],
    "verification": [" verify", " check", " validate", " confirm", " verifying"],
    "broad_success": [" works", " succeed", " robust", " everything", " overall"],
    "dependency_unavailable": [" unavailable", " missing", " install", " absent"],
    "focused_validation": [" specific", " targeted", " focused", " narrow"],
    "test_success": [" passed", " succeeds", " passing", " green"],
    "task_resolution": [" resolved", " done", " complete", " finished", " solved"],
}


def _single_token_id(tokenizer: Any, form: str) -> int | None:
    ids = tokenizer.encode(form, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def build_forms(tokenizer: Any) -> dict[str, Any]:
    # 1) single-token map per family
    per_family: dict[str, dict[str, int]] = {}
    for family, forms in CANDIDATES.items():
        kept = {}
        for form in forms:
            tid = _single_token_id(tokenizer, form)
            if tid is not None:
                kept[form] = tid
        per_family[family] = kept
    # 2) drop any token id used by >1 family (concept-chain global-uniqueness rule)
    id_counts = Counter(tid for kept in per_family.values() for tid in kept.values())
    shared = {tid for tid, c in id_counts.items() if c > 1}
    families: dict[str, Any] = {}
    unscorable: list[str] = []
    for family, kept in per_family.items():
        unique = {form: tid for form, tid in kept.items() if tid not in shared}
        if len(unique) >= 2:
            families[family] = [
                {"form": form.strip(), "surface": form, "token_id": tid}
                for form, tid in unique.items()
            ]
        else:
            unscorable.append(family)
    return {
        "kind": "swe_task_state_v4_general_concept_forms_v1",
        "task_independent": True,
        "n_families_scorable": len(families),
        "unscorable_families": unscorable,
        "shared_tokens_dropped": sorted(shared),
        "families": families,
    }


def _tokenizer():
    from transformers import AutoTokenizer

    snap = next(
        (ROOT / ".cache").glob("**/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/*"),
        None,
    )
    if snap is None:
        snap = next(
            Path.home().glob(
                ".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/*"
            )
        )
    return AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)


def main() -> int:
    result = build_forms(_tokenizer())
    DEFAULT_OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"scorable families: {result['n_families_scorable']}/14  "
        f"unscorable: {result['unscorable_families']}  "
        f"dropped {len(result['shared_tokens_dropped'])} shared tokens"
    )
    for fam, forms in result["families"].items():
        print(f"  {fam:26s} {[f['form'] for f in forms]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
