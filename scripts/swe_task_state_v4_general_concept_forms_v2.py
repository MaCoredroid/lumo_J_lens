#!/usr/bin/env python3
"""Lever 2: a RICHER task-independent concept-form vocab (v2) for the 14 families.

The v1 vocab kept only ~2-5 single-token forms per family; a family score averaged over so
few forms is noisy. v2 broadens each family's candidate surface forms (still task-agnostic,
still the concept-chain rules: single Qwen token, globally unique across families, >=2
survivors) so each family score averages over more on-concept evidence. Reuses v1's exact
build_forms() filter; writes configs/swe_task_state_v4_general_concept_forms_v2.json. Leaves
v1 untouched for v3 reproducibility.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/swe_task_state_v4_general_concept_forms_v2.json"

# Broadened candidates. Leading space = mid-sentence surface. Deliberately on-concept only;
# the single-token + global-uniqueness filter prunes multi-token and cross-family collisions.
CANDIDATES = {
    "source_localization": [
        " locate", " where", " search", " locating", " pinpoint", " find", " look",
        " grep", " inspect", " examine", " trace", " hunt", " explore", " scan",
    ],
    "substitution_operation": [
        " replace", " substitute", " swap", " replacing", " swapping", " substituting",
        " replaced", " swapped",
    ],
    "located_source": [
        " located", " here", " spotted", " pinpointed", " found", " culprit",
        " offending", " responsible",
    ],
    "defined_identifier": [
        " identifier", " variable", " attribute", " symbol", " method", " parameter",
        " argument", " keyword", " property", " field",
    ],
    "runtime_name_failure": [
        " undefined", " unbound", " crash", " traceback", " exception", " broken", " typo",
        " NameError", " raises", " raised",
    ],
    "failure_confirmation": [
        " fails", " reproduce", " reproduced", " failing", " repro", " failure", " breaks",
        " confirmed", " manifests",
    ],
    "source_edit": [
        " edit", " modify", " change", " editing", " rewrite", " update", " patch",
        " adjust", " alter", " modifying", " changing",
    ],
    "repair": [
        " fix", " repair", " correct", " fixing", " correcting", " address", " remedy",
        " fixed", " corrected",
    ],
    "verification": [
        " verify", " check", " validate", " confirm", " verifying", " checking", " ensure",
        " assert", " validating", " double",
    ],
    "broad_success": [
        " works", " succeed", " robust", " everything", " overall", " correctly", " properly",
        " fine", " cleanly", " smoothly",
    ],
    "dependency_unavailable": [
        " unavailable", " missing", " install", " absent", " dependency", " uninstalled",
        " lacking", " unset",
    ],
    "focused_validation": [
        " specific", " targeted", " focused", " narrow", " particular", " precise",
        " isolated", " dedicated",
    ],
    "test_success": [
        " passed", " succeeds", " passing", " green", " passes", " succeeded", " OK",
    ],
    "task_resolution": [
        " resolved", " done", " complete", " finished", " solved", " completed", " ready",
        " accomplished", " wrapped",
    ],
}


def _v1():
    spec = importlib.util.spec_from_file_location(
        "general_concept_forms", ROOT / "scripts/swe_task_state_v4_general_concept_forms.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    v1 = _v1()
    v1.CANDIDATES = CANDIDATES  # reuse v1's exact single-token + uniqueness filter
    result = v1.build_forms(v1._tokenizer())
    result["kind"] = "swe_task_state_v4_general_concept_forms_v2"
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    n_forms = sum(len(f) for f in result["families"].values())
    print(f"v2 scorable families: {result['n_families_scorable']}/14  total forms: {n_forms}  "
          f"unscorable: {result['unscorable_families']}")
    for fam, forms in result["families"].items():
        print(f"  {fam:24s} ({len(forms)}) {[f['form'] for f in forms]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
