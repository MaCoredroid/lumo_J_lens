from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_counterfactual_selector_control_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_v4_counterfactual_selector_control_pilot", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class SelectorControlPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(module.load_json(module.CONFIG_PATH))
        # This is the one full CPU-only integration build for the class.  It
        # streams only the 20 already-frozen Stage-A prompt sources.
        cls.artifacts = module.build_artifacts(cls.config)
        cls.split_manifest = cls.artifacts[module.SPLIT_MANIFEST_NAME]
        cls.selection = cls.split_manifest["selection"]
        cls.capture = cls.artifacts[module.CAPTURE_BUNDLE_NAME]
        cls.generation = cls.artifacts[module.GENERATION_BUNDLE_NAME]
        cls.key = cls.artifacts[module.CONDITION_KEY_NAME]
        cls.capture_by_id = {row["id"]: row["token_ids"] for row in cls.capture}
        cls.generation_by_id = {
            row["id"]: row["token_ids"] for row in cls.generation
        }

    def test_config_is_hash_frozen_and_runtime_claims_are_false(self) -> None:
        self.assertEqual(module.sha256_file(module.CONFIG_PATH), module.CONFIG_SHA256)
        raw = module.CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("TO_BE_FROZEN", raw)
        gates = self.config["go_stop_gates"]
        claims = self.config["claim_scope"]
        self.assertFalse(gates["stage_a_scaffold_authorizes_gpu_runtime"])
        self.assertFalse(gates["stage_b_runtime_or_prompt_materialization_authorized"])
        self.assertFalse(claims["stage_a_or_stage_b_runtime_completed"])
        self.assertFalse(claims["factorial_behavior_or_activation_effect_established"])
        self.assertFalse(claims["incremental_activation_readout_established"])
        self.assertFalse(claims["private_chain_of_thought_reconstructed"])
        self.assertFalse(claims["subjective_confidence_or_doubt_inferred"])
        self.assertFalse(claims["experienced_stress_inferred"])
        self.assertFalse(claims["experienced_emotion_inferred"])

    def test_split_is_hash_only_balanced_disjoint_and_excludes_pilot_task(self) -> None:
        split = self.config["split"]
        self.assertEqual(
            self.split_manifest["selection_canonical_sha256"],
            split["expected_split_canonical_sha256"],
        )
        stage_a = self.selection["stage_a"]
        stage_b = self.selection["stage_b_holdout_unopened"]
        self.assertEqual(len(stage_a), 20)
        self.assertEqual(len(stage_b), 20)
        self.assertEqual(len({row["repository"] for row in stage_a}), 10)
        self.assertEqual(len({row["repository"] for row in stage_b}), 10)
        self.assertEqual(len({row["task_id_sha256"] for row in stage_a}), 20)
        self.assertEqual(len({row["task_id_sha256"] for row in stage_b}), 20)
        stage_a_tasks = {row["task_id_sha256"] for row in stage_a}
        stage_b_tasks = {row["task_id_sha256"] for row in stage_b}
        self.assertFalse(stage_a_tasks & stage_b_tasks)
        self.assertNotIn(split["observed_pilot_task_id_sha256"], stage_a_tasks)
        self.assertNotIn(split["observed_pilot_task_id_sha256"], stage_b_tasks)
        self.assertIn("prompt_token_count", split["selection_inputs_forbidden"])
        self.assertIn("completion_text", split["selection_inputs_forbidden"])
        self.assertIn("activation_values", split["selection_inputs_forbidden"])

    def test_code_maps_are_rotated_balanced_and_hash_bound(self) -> None:
        self.assertEqual(
            self.artifacts["provenance"]["stage_a_mapping_canonical_sha256"],
            self.config["selector_balance"][
                "expected_stage_a_mapping_canonical_sha256"
            ],
        )
        units: dict[tuple[str, int], tuple[tuple[str, ...], tuple[str, ...]]] = {}
        for row in self.key["records"]:
            unit = (row["source_id_sha256"], row["paraphrase_replica"])
            value = (
                tuple(
                    row["evidence_code_by_level"][level]
                    for level in module.EVIDENCE_LEVELS
                ),
                tuple(
                    row["pressure_code_by_level"][level]
                    for level in module.PRESSURE_LEVELS
                ),
            )
            if unit in units:
                self.assertEqual(units[unit], value)
            units[unit] = value
        self.assertEqual(len(units), 40)
        evidence_counts: dict[tuple[str, ...], int] = {}
        pressure_counts: dict[tuple[str, ...], int] = {}
        for evidence_map, pressure_map in units.values():
            evidence_counts[evidence_map] = evidence_counts.get(evidence_map, 0) + 1
            pressure_counts[pressure_map] = pressure_counts.get(pressure_map, 0) + 1
        self.assertEqual(len(evidence_counts), 6)
        self.assertLessEqual(max(evidence_counts.values()) - min(evidence_counts.values()), 1)
        self.assertEqual(sorted(pressure_counts.values()), [20, 20])

    def test_capture_and_generation_bundles_are_opaque_and_physically_separate(self) -> None:
        self.assertEqual(len(self.capture), 500)
        self.assertEqual(len(self.generation), 240)
        self.assertEqual(self.key["record_count"], 240)
        self.assertTrue(all(set(row) == {"id", "token_ids"} for row in self.capture))
        self.assertTrue(
            all(set(row) == {"id", "token_ids"} for row in self.generation)
        )
        condition_ids = {row["condition_id_sha256"] for row in self.key["records"]}
        source_ids = {row["source_id_sha256"] for row in self.key["records"]}
        task_ids = {row["task_id_sha256"] for row in self.key["records"]}
        opaque_ids = {row["id"] for row in self.capture}
        self.assertFalse(opaque_ids & condition_ids)
        self.assertFalse(opaque_ids & source_ids)
        self.assertFalse(opaque_ids & task_ids)
        self.assertTrue(
            self.key["capture_and_generation_processes_must_not_read_this_file"]
        )
        self.assertFalse(self.key["subjective_state_labels_present"])
        self.assertFalse(self.key["stage_b_records_present"])

    def test_three_capture_positions_are_exact_causal_prefixes(self) -> None:
        pre_ids = set()
        selector_ids = set()
        post_ids = set()
        for row in self.key["records"]:
            ids = row["capture_prompt_ids"]
            pre_ids.add(ids["pre_manipulation"])
            selector_ids.add(ids["selector_tail"])
            post_ids.add(ids["post_bridge"])
            pre = self.capture_by_id[ids["pre_manipulation"]]
            selector = self.capture_by_id[ids["selector_tail"]]
            post = self.capture_by_id[ids["post_bridge"]]
            self.assertEqual(selector[: len(pre)], pre)
            self.assertEqual(post[: len(selector)], selector)
            self.assertEqual(post, self.generation_by_id[ids["post_bridge"]])
            pressure_index = row["selector_token_indices"]["pressure"]
            self.assertEqual(
                len(post) - pressure_index - 1,
                self.config["rendering"][
                    "expected_identical_tokens_after_pressure_selector"
                ],
            )
        self.assertEqual(len(pre_ids), 20)
        self.assertEqual(len(selector_ids), 240)
        self.assertEqual(len(post_ids), 240)
        self.assertEqual(pre_ids | selector_ids | post_ids, set(self.capture_by_id))

    def test_pairwise_differences_are_exactly_changed_factor_selectors(self) -> None:
        blocks: dict[tuple[str, int], list[dict]] = {}
        for row in self.key["records"]:
            blocks.setdefault(
                (row["source_id_sha256"], row["paraphrase_replica"]), []
            ).append(row)
        self.assertEqual(len(blocks), 40)
        for rows in blocks.values():
            self.assertEqual(len(rows), 6)
            geometry = {
                (
                    row["selector_token_indices"]["evidence"],
                    row["selector_token_indices"]["pressure"],
                )
                for row in rows
            }
            self.assertEqual(len(geometry), 1)
            allowed = set(next(iter(geometry)))
            for left_index, left in enumerate(rows):
                left_ids = self.generation_by_id[left["prompt_id"]]
                for right in rows[left_index + 1 :]:
                    right_ids = self.generation_by_id[right["prompt_id"]]
                    self.assertEqual(len(left_ids), len(right_ids))
                    changed = {
                        index
                        for index, (left_token, right_token) in enumerate(
                            zip(left_ids, right_ids)
                        )
                        if left_token != right_token
                    }
                    expected = int(
                        left["evidence_level"] != right["evidence_level"]
                    ) + int(left["pressure_level"] != right["pressure_level"])
                    self.assertTrue(changed <= allowed)
                    self.assertEqual(len(changed), expected)

    def test_semantic_vocabulary_order_and_common_suffix_are_fixed(self) -> None:
        matching = self.artifacts["provenance"]["matching"]
        self.assertEqual(matching["block_count"], 40)
        self.assertTrue(
            matching[
                "all_salient_descriptions_present_in_fixed_semantic_order_within_replica"
            ]
        )
        self.assertTrue(matching["all_conditions_have_two_declared_one_token_selectors"])
        self.assertTrue(
            matching["all_pairwise_differences_confined_to_changed_factor_selectors"]
        )
        self.assertEqual(matching["common_bridge_token_count"], 32)
        self.assertEqual(matching["identical_tokens_after_pressure_selector"], 42)
        self.assertFalse(matching["semantic_padding_tokens_used"])
        for block in matching["blocks"]:
            self.assertEqual(
                block["semantic_description_order"]["evidence"],
                list(module.EVIDENCE_LEVELS),
            )
            self.assertEqual(
                block["semantic_description_order"]["pressure"],
                list(module.PRESSURE_LEVELS),
            )
            self.assertEqual(block["declared_variable_token_positions"], 2)

    def test_stage_b_is_identity_only_and_never_streamed_or_materialized(self) -> None:
        stage_b = self.artifacts["provenance"]["stage_b"]
        self.assertTrue(stage_b["identities_frozen"])
        self.assertFalse(stage_b["prompt_sources_streamed"])
        self.assertFalse(stage_b["prompt_bundle_materialized"])
        self.assertFalse(stage_b["condition_key_materialized"])
        self.assertFalse(stage_b["capture_generation_or_annotation_run"])
        stage_b_sources = {
            row["source_id_sha256"]
            for row in self.selection["stage_b_holdout_unopened"]
        }
        serialized_capture = module.canonical_json_bytes(self.capture).decode("ascii")
        serialized_generation = module.canonical_json_bytes(self.generation).decode(
            "ascii"
        )
        serialized_key = module.canonical_json_bytes(self.key).decode("ascii")
        for source_id in stage_b_sources:
            self.assertNotIn(source_id, serialized_capture)
            self.assertNotIn(source_id, serialized_generation)
            self.assertNotIn(source_id, serialized_key)

    def test_materialization_writes_only_declared_files_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selector-scaffold"
            with mock.patch.object(
                module, "build_artifacts", return_value=self.artifacts
            ):
                manifest = module.materialize(output)
                self.assertEqual(
                    manifest["status"],
                    "passed_scaffold_materialization_only_no_model_runtime",
                )
                self.assertEqual(
                    {path.name for path in output.iterdir()},
                    {
                        module.SPLIT_MANIFEST_NAME,
                        module.CAPTURE_BUNDLE_NAME,
                        module.GENERATION_BUNDLE_NAME,
                        module.CONDITION_KEY_NAME,
                        module.MATERIALIZATION_MANIFEST_NAME,
                    },
                )
                self.assertFalse(
                    any("stage-b" in path.name.lower() for path in output.iterdir())
                )
                for record in manifest["outputs"].values():
                    self.assertNotIn(".tmp-", record["path"])
                    advertised = Path(record["path"])
                    if not advertised.is_absolute():
                        advertised = ROOT / advertised
                    self.assertTrue(advertised.is_file())
                verified = module.verify_existing(output)
                self.assertEqual(verified, manifest)
                with self.assertRaises(module.SelectorControlError):
                    module.materialize(output)

    def test_forbidden_paths_fail_before_filesystem_access(self) -> None:
        with mock.patch.object(Path, "resolve", side_effect=AssertionError("touched")):
            with self.assertRaisesRegex(
                module.SelectorControlError, "forbidden path rejected"
            ):
                module.lexical_path_preflight(
                    (Path("/tmp/reserved-do-not-touch/anything.json"),)
                )
            with self.assertRaisesRegex(
                module.SelectorControlError, "forbidden path rejected"
            ):
                module.lexical_path_preflight(
                    (Path("/tmp/validation-do-not-touch/anything.json"),)
                )

    def test_cot_affect_confidence_and_stress_claims_remain_separate(self) -> None:
        targets = self.config["observable_targets"]
        self.assertIn("separate", targets["cot_like_visible_structure"])
        self.assertIn("not_private_chain_of_thought", targets["cot_like_visible_structure"])
        self.assertIn("not_experienced_emotion", targets["explicit_affect_language"])
        self.assertIn("not_experienced_stress", targets["pressure_sensitivity"])
        self.assertIn("condition_blind", targets["objective_calibration"])
        self.assertIn("separate_visible_recheck", targets["doubt_like_rechecking"])


if __name__ == "__main__":
    unittest.main()
