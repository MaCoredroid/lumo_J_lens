#!/usr/bin/env python3
"""Focused tests for the compact N=20 SWE behavioral publication verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/check_swe_behavioral_n20_publication.py"
SPEC = importlib.util.spec_from_file_location(
    "check_swe_behavioral_n20_publication", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BehavioralN20PublicationTests(unittest.TestCase):
    def load_contract(self):
        publication, paths = MODULE.verify_publication_manifest()
        run_manifest = MODULE.verify_run_manifest(publication, paths)
        summary = MODULE.mapping(
            MODULE.strict_json_file(paths["prompts_summary"], "prompt summary"),
            "prompt summary",
        )
        campaign = MODULE.mapping(
            MODULE.strict_json_file(paths["campaign_evidence"], "campaign evidence"),
            "campaign evidence",
        )
        analysis = MODULE.mapping(
            MODULE.strict_json_file(paths["analysis"], "analysis"), "analysis"
        )
        official = {
            cohort: MODULE.mapping(
                MODULE.strict_json_file(
                    paths[f"official_outcomes_{cohort}"], f"{cohort} outcomes"
                ),
                f"{cohort} outcomes",
            )
            for cohort in ("development", "replication")
        }
        return publication, paths, run_manifest, summary, campaign, official, analysis

    def load_supplements(self):
        publication, paths, run_manifest, *_ = self.load_contract()
        transport = MODULE.mapping(
            MODULE.strict_json_file(
                paths["next_token_transport_analysis"],
                "next-token transport analysis",
            ),
            "next-token transport analysis",
        )
        action = MODULE.mapping(
            MODULE.strict_json_file(
                paths["action_layer_readout"], "action-layer readout"
            ),
            "action-layer readout",
        )
        return publication, run_manifest, transport, action

    def test_real_publication_bundle_passes(self) -> None:
        result = MODULE.verify_publication()
        self.assertEqual(
            result,
            {
                "included_artifacts": 14,
                "omitted_artifacts": 4,
                "tasks": 20,
                "prompts": 160,
            },
        )

    def test_duplicate_keys_and_nonfinite_numbers_fail(self) -> None:
        for value in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(value=value):
                with self.assertRaises(MODULE.PublicationError):
                    MODULE.strict_json_bytes(value, "test")

    def test_canonical_paths_and_symlinks_fail(self) -> None:
        for value in ("../escape", "/absolute", "a/./b", "a//b", "a\\b"):
            with self.subTest(value=value):
                with self.assertRaises(MODULE.PublicationError):
                    MODULE.canonical_relative(value, "test path")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("value", encoding="ascii")
            (root / "link").symlink_to(target)
            with self.assertRaisesRegex(MODULE.PublicationError, "symlink"):
                MODULE.regular_file(root, "link", "test file")

    def test_sidecar_requires_exact_gnu_grammar_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = hashlib.sha256(b"value").hexdigest()
            sidecar = root / "value.sha256"
            sidecar.write_text(f"{digest}  value.json\n", encoding="ascii")
            MODULE.verify_sidecar(sidecar, "value.json", digest)
            for changed in (
                f"{digest} value.json\n",
                f"{digest}  other.json\n",
                f"{digest.upper()}  value.json\n",
            ):
                with self.subTest(changed=changed):
                    sidecar.write_text(changed, encoding="ascii")
                    with self.assertRaises(MODULE.PublicationError):
                        MODULE.verify_sidecar(sidecar, "value.json", digest)

    def test_omitted_report_hash_cannot_drift_from_source_manifest(self) -> None:
        publication, paths, *_ = self.load_contract()
        changed = copy.deepcopy(publication)
        changed["omitted_artifacts"]["native_report"]["sha256"] = "a" * 64
        with self.assertRaisesRegex(MODULE.PublicationError, "source-manifest binding"):
            MODULE.verify_run_manifest(changed, paths)

    def test_analysis_cannot_relabel_inputs_or_recommend_refit(self) -> None:
        (
            publication,
            _,
            run_manifest,
            summary,
            campaign,
            official,
            analysis,
        ) = self.load_contract()
        changed = copy.deepcopy(analysis)
        changed["inputs"]["native_report"] = "a" * 64
        with self.assertRaisesRegex(MODULE.PublicationError, "input hashes"):
            MODULE.verify_semantic_contract(
                publication,
                run_manifest,
                summary,
                campaign,
                official,
                changed,
            )
        changed = copy.deepcopy(analysis)
        changed["scientific_decision"]["next_step"] = "refit now"
        with self.assertRaisesRegex(MODULE.PublicationError, "decision or next step"):
            MODULE.verify_semantic_contract(
                publication,
                run_manifest,
                summary,
                campaign,
                official,
                changed,
            )

    def test_mtp_and_official_outcome_mutations_fail(self) -> None:
        (
            publication,
            _,
            run_manifest,
            summary,
            campaign,
            official,
            analysis,
        ) = self.load_contract()
        changed_campaign = copy.deepcopy(campaign)
        changed_campaign["cohorts"][0]["mtp_speculative_decoding"][
            "accepted_tokens"
        ] += 1
        with self.assertRaisesRegex(MODULE.PublicationError, "MTP evidence"):
            MODULE.verify_semantic_contract(
                publication,
                run_manifest,
                summary,
                changed_campaign,
                official,
                analysis,
            )
        changed_official = copy.deepcopy(official)
        changed_official["development"]["outcomes"][0]["outcome"] = "resolved"
        with self.assertRaisesRegex(MODULE.PublicationError, "counts changed"):
            MODULE.verify_semantic_contract(
                publication,
                run_manifest,
                summary,
                campaign,
                changed_official,
                analysis,
            )

    def test_transport_raw_lineage_and_strict_gate_mutations_fail(self) -> None:
        publication, _, transport, _ = self.load_supplements()
        changed = copy.deepcopy(transport)
        changed["inputs"]["reports"]["native"]["sha256"] = "a" * 64
        with self.assertRaisesRegex(MODULE.PublicationError, "raw report binding"):
            MODULE.verify_next_token_transport(publication, changed)

        changed = copy.deepcopy(transport)
        changed["tracks"]["strict_primary"]["support"][
            "eligible_checkpoint_count"
        ] = 128
        with self.assertRaisesRegex(MODULE.PublicationError, "insufficient-support gate"):
            MODULE.verify_next_token_transport(publication, changed)

    def test_transport_sensitivity_cannot_be_promoted_or_relabelled(self) -> None:
        publication, _, transport, _ = self.load_supplements()
        changed = copy.deepcopy(transport)
        changed["tracks"]["paired_stable_reconstruction_sensitivity"][
            "decision_overrides_behavioral_semantic_analysis"
        ] = True
        with self.assertRaisesRegex(MODULE.PublicationError, "sensitivity classification"):
            MODULE.verify_next_token_transport(publication, changed)

        changed = copy.deepcopy(transport)
        changed["tracks"]["paired_stable_reconstruction_sensitivity"]["methods"][
            "public_jacobian"
        ]["metrics"]["normalized_rank_utility"] += 0.01
        with self.assertRaisesRegex(MODULE.PublicationError, "normalized rank utility"):
            MODULE.verify_next_token_transport(publication, changed)

    def test_action_inputs_and_strict_decision_mutations_fail(self) -> None:
        publication, run_manifest, _, action = self.load_supplements()
        changed = copy.deepcopy(action)
        changed["inputs"]["public_report"] = "a" * 64
        with self.assertRaisesRegex(MODULE.PublicationError, "input and raw-report hashes"):
            MODULE.verify_action_layer_readout(publication, run_manifest, changed)

        changed = copy.deepcopy(action)
        changed["classification"] = "actionable_refit"
        with self.assertRaisesRegex(MODULE.PublicationError, "strict decision"):
            MODULE.verify_action_layer_readout(publication, run_manifest, changed)

    def test_action_sensitivity_role_and_metrics_are_fail_closed(self) -> None:
        publication, run_manifest, _, action = self.load_supplements()
        changed = copy.deepcopy(action)
        changed["tracks"]["paired_stable_reconstruction_sensitivity"][
            "primary_decision_override_forbidden"
        ] = False
        with self.assertRaisesRegex(MODULE.PublicationError, "roles changed"):
            MODULE.verify_action_layer_readout(publication, run_manifest, changed)

        changed = copy.deepcopy(action)
        changed["tracks"]["strict_primary"]["methods"]["ordinary_logit"][
            "metrics"
        ]["balanced_accuracy"] += 0.01
        with self.assertRaisesRegex(MODULE.PublicationError, "balanced accuracy"):
            MODULE.verify_action_layer_readout(publication, run_manifest, changed)


if __name__ == "__main__":
    unittest.main()
