#!/usr/bin/env python3
"""Focused, dependency-light tests for the frozen binary phase interpreter."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/analyze_swe_binary_phase_v2.py"
PROTOCOL_PATH = ROOT / "configs/swe_binary_phase_interpreter_v2.json"
SOURCE_PROTOCOL_PATH = ROOT / "configs/swe_task_state_interpreter_protocol.json"
BEHAVIORAL_PROTOCOL_PATH = ROOT / "configs/swe_behavioral_readout_protocol.json"
RESERVED_COHORT_PATH = ROOT / "configs/swe_task_state_validation_cohort.json"
RESERVED_CAMPAIGN_A_PATH = (
    ROOT / "configs/swe_task_state_validation_a_campaign.json"
)
RESERVED_CAMPAIGN_B_PATH = (
    ROOT / "configs/swe_task_state_validation_b_campaign.json"
)
RESERVED_IMAGE_REGISTRY_PATH = (
    ROOT / "configs/swe_task_state_validation_image_digests.json"
)
MODEL_BUNDLE_PATH = ROOT / "artifacts/swe-binary-phase-v2.joblib"
MODEL_MANIFEST_PATH = ROOT / "artifacts/swe-binary-phase-v2.manifest.json"


def load_module():
    name = "analyze_swe_binary_phase_v2_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()
PROTOCOL = json.loads(PROTOCOL_PATH.read_bytes())


def validate_protocol(
    value: dict | None = None, *, protocol_sha256: str | None = None
) -> dict:
    return MODULE.validate_v2_protocol(
        copy.deepcopy(PROTOCOL if value is None else value),
        protocol_sha256=(
            MODULE.sha256_file(PROTOCOL_PATH)
            if protocol_sha256 is None
            else protocol_sha256
        ),
        source_protocol_sha256=MODULE.sha256_file(SOURCE_PROTOCOL_PATH),
        behavioral_protocol_sha256=MODULE.sha256_file(BEHAVIORAL_PROTOCOL_PATH),
    )


def phase_prompt(
    request_index: int,
    action: str | None,
    *,
    declared: list[int],
    task_id: str = "owner__repo-1",
) -> dict:
    return {
        "id": f"{task_id}-r{request_index}",
        "metadata": {
            "task": {
                "instance_id": task_id,
                "repo": "owner/repo",
                "probeable_request_indices": declared,
            },
            "selection": {"task_request_index": request_index},
            "labels": {
                "action": {
                    "status": "available" if action is not None else "unavailable",
                    "class_id": action,
                }
            },
        },
    }


def stable_source_row(request_index: int, values: np.ndarray) -> dict:
    return {
        "row_id": f"row-{request_index}",
        "task_id": "task-1",
        "repo": "owner/repo",
        "cohort_id": "development",
        "task_request_index": request_index,
        "checkpoint_ordinal": request_index - 1,
        "current_action_label_status": "unavailable",
        "current_action_class_id": None,
        "public_jacobian": values.tolist(),
        "ordinary_logit": (values * 10.0).tolist(),
        "history_context": [float(request_index * 100 + index) for index in range(32)],
    }


class ProtocolContractTest(unittest.TestCase):
    def test_checked_in_protocol_validates_with_frozen_class_order(self) -> None:
        normalized = validate_protocol()

        self.assertEqual(normalized["classes"], ("edit", "check_or_finish"))
        self.assertEqual(normalized["variants"], ("j_compact", "l_compact"))
        self.assertEqual(normalized["source_classes"], MODULE.SOURCE_ACTION_CLASSES)
        self.assertEqual(normalized["feature_width"], 154)
        self.assertEqual(
            normalized["cohort_checker_sha256"],
            MODULE.sha256_file(MODULE.COHORT_CHECKER_PATH),
        )
        self.assertIsNone(normalized["reserved_prompts_summary_sha256"])
        self.assertEqual(
            normalized["core_sha256"], MODULE.core_protocol_sha256(PROTOCOL)
        )

    def test_core_identity_allows_only_null_to_literal_summary_pin(self) -> None:
        fit_protocol = validate_protocol()
        fit_lifecycle = MODULE.build_fit_protocol_lifecycle(fit_protocol)
        fit_state = MODULE.validate_protocol_lifecycle_transition(
            fit_lifecycle, fit_protocol, phase="fit"
        )
        self.assertEqual(
            fit_state["fit_full_protocol_sha256"], fit_protocol["sha256"]
        )
        self.assertEqual(
            fit_state["current_full_protocol_sha256"], fit_protocol["sha256"]
        )
        self.assertIsNone(fit_state["current_reserved_prompts_summary_sha256"])

        materialized_value = copy.deepcopy(PROTOCOL)
        materialized_value["pins"]["reserved_prompts_summary_sha256"] = "f" * 64
        evaluation_protocol = validate_protocol(
            materialized_value, protocol_sha256="e" * 64
        )
        transition = MODULE.validate_protocol_lifecycle_transition(
            fit_lifecycle, evaluation_protocol, phase="evaluate"
        )

        self.assertEqual(
            fit_protocol["core_sha256"], evaluation_protocol["core_sha256"]
        )
        self.assertEqual(transition["fit_full_protocol_sha256"], fit_protocol["sha256"])
        self.assertEqual(
            transition["current_full_protocol_sha256"], evaluation_protocol["sha256"]
        )
        self.assertEqual(
            transition["transition"]["kind"],
            "null_to_literal_lowercase_sha256",
        )
        self.assertTrue(transition["sole_append_only_field_transition_verified"])
        self.assertFalse(transition["model_refit_after_transition"])

        with self.assertRaisesRegex(ValueError, "fit requires.*remain null"):
            MODULE.build_fit_protocol_lifecycle(evaluation_protocol)
        with self.assertRaisesRegex(ValueError, "evaluation reserved prompts summary pin"):
            MODULE.validate_protocol_lifecycle_transition(
                fit_lifecycle, fit_protocol, phase="evaluate"
            )

    def test_core_identity_rejects_every_other_protocol_mutation(self) -> None:
        fit_protocol = validate_protocol()
        fit_lifecycle = MODULE.build_fit_protocol_lifecycle(fit_protocol)
        mutated_value = copy.deepcopy(PROTOCOL)
        mutated_value["pins"]["reserved_prompts_summary_sha256"] = "f" * 64
        mutated_value["interpretation_target"] = "forged semantic target"
        mutated_protocol = validate_protocol(mutated_value, protocol_sha256="d" * 64)

        self.assertNotEqual(
            fit_protocol["core_sha256"], mutated_protocol["core_sha256"]
        )
        with self.assertRaisesRegex(ValueError, "mutation outside.*forbidden"):
            MODULE.validate_protocol_lifecycle_transition(
                fit_lifecycle, mutated_protocol, phase="evaluate"
            )

    def test_rejects_target_and_probability_class_order_changes(self) -> None:
        target_mutation = copy.deepcopy(PROTOCOL)
        target_mutation["class_contract"]["class_ids_in_order"].reverse()
        with self.assertRaisesRegex(ValueError, "class order"):
            validate_protocol(target_mutation)

        probability_mutation = copy.deepcopy(PROTOCOL)
        probability_mutation["model_contract"]["probability_class_order"].reverse()
        with self.assertRaisesRegex(ValueError, "class order"):
            validate_protocol(probability_mutation)

        checker_mutation = copy.deepcopy(PROTOCOL)
        checker_mutation["pins"]["cohort_checker_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "cohort checker"):
            validate_protocol(checker_mutation)


class OfflineLabelTest(unittest.TestCase):
    def test_skips_inspection_and_collapses_validation_and_finalize(self) -> None:
        declared = [1, 2, 3, 4, 5]
        prompts = [
            phase_prompt(1, "inspect", declared=declared),
            phase_prompt(2, "inspect", declared=declared),
            phase_prompt(3, "edit", declared=declared),
            phase_prompt(4, "validate", declared=declared),
            phase_prompt(5, "finalize", declared=declared),
        ]

        labels, summary = MODULE.offline_binary_labels(prompts)

        self.assertEqual(labels[prompts[0]["id"]]["label"], "edit")
        self.assertEqual(labels[prompts[0]["id"]]["horizon_requests"], 2)
        self.assertEqual(labels[prompts[0]["id"]]["target_request_index"], 3)
        self.assertEqual(labels[prompts[2]["id"]]["horizon_requests"], 0)
        self.assertEqual(labels[prompts[3]["id"]]["label"], "check_or_finish")
        self.assertEqual(labels[prompts[3]["id"]]["target_source_action"], "validate")
        self.assertEqual(labels[prompts[4]["id"]]["label"], "check_or_finish")
        self.assertEqual(labels[prompts[4]["id"]]["target_source_action"], "finalize")
        self.assertTrue(summary["tasks"]["owner__repo-1"]["complete_consecutive_bundle"])

    def test_unknown_censors_fail_closed_and_later_known_action_recovers(self) -> None:
        declared = [1, 2, 3]
        prompts = [
            phase_prompt(1, "inspect", declared=declared),
            phase_prompt(2, None, declared=declared),
            phase_prompt(3, "validate", declared=declared),
        ]

        labels, _ = MODULE.offline_binary_labels(prompts)

        for prompt in prompts[:2]:
            self.assertEqual(labels[prompt["id"]]["status"], "censored")
            self.assertEqual(
                labels[prompt["id"]]["reason"], "unknown_before_next_milestone"
            )
        self.assertEqual(labels[prompts[2]["id"]]["label"], "check_or_finish")

    def test_no_future_milestone_and_incomplete_bundle_are_censored(self) -> None:
        trailing = [1, 2]
        trailing_prompts = [
            phase_prompt(1, "edit", declared=trailing),
            phase_prompt(2, "inspect", declared=trailing),
        ]
        trailing_labels, _ = MODULE.offline_binary_labels(trailing_prompts)
        self.assertEqual(
            trailing_labels[trailing_prompts[1]["id"]]["reason"],
            "no_observed_future_milestone",
        )

        incomplete = [1, 3]
        incomplete_prompts = [
            phase_prompt(1, "inspect", declared=incomplete),
            phase_prompt(3, "edit", declared=incomplete),
        ]
        incomplete_labels, _ = MODULE.offline_binary_labels(incomplete_prompts)
        self.assertEqual(
            {record["reason"] for record in incomplete_labels.values()},
            {"incomplete_nonconsecutive_bundle"},
        )


class ReservedInputBindingTest(unittest.TestCase):
    @staticmethod
    def materialized_fixture(root: Path) -> tuple[list[dict], Path, Path]:
        cohort = json.loads(RESERVED_COHORT_PATH.read_bytes())
        cohort_sha = MODULE.sha256_file(RESERVED_COHORT_PATH)
        campaign_hashes = [row["campaign_sha256"] for row in cohort["cohorts"]]
        prompts: list[dict] = []
        prompt_summaries: list[dict] = []
        audits: list[dict] = []
        summary_cohorts: list[dict] = []
        global_index = 0
        selection_index = 0
        for cohort_index, cohort_row in enumerate(cohort["cohorts"]):
            summary_cohorts.append(
                {
                    "id": cohort_row["id"],
                    "index": cohort_index,
                    "campaign_sha256": campaign_hashes[cohort_index],
                    "cohort_manifest_sha256": cohort_sha,
                    "source_task_instance_ids": cohort_row["instance_ids"],
                    "source_task_count": len(cohort_row["instance_ids"]),
                }
            )
            for instance_id in cohort_row["instance_ids"]:
                global_index += 1
                prompt_id = f"reserved-fixture-{global_index:02d}-{instance_id}"
                prompt = {
                    "id": prompt_id,
                    "text": f"Reserved issue request for {instance_id}",
                    "token_ids": [global_index],
                    "score_token_ids": [1000 + global_index],
                    "metadata": {
                        "task": {
                            "instance_id": instance_id,
                            "repo": MODULE._expected_repository(instance_id),
                            "probeable_request_indices": [1],
                            "probeable_request_count": 1,
                        },
                        "selection": {
                            "task_request_index": 1,
                            "global_request_index": global_index,
                            "max_checkpoints": None,
                            "probeable_request_indices": [1],
                            "candidate_count": 1,
                            "checkpoint_count": 1,
                        },
                        "provenance": {},
                        "cohort": {
                            "id": cohort_row["id"],
                            "index": cohort_index,
                            "campaign_sha256": campaign_hashes[cohort_index],
                            "cohort_manifest_sha256": cohort_sha,
                            "source_task_instance_ids": cohort_row["instance_ids"],
                        },
                        "labels": {
                            "action": {"status": "available", "class_id": "inspect"}
                        },
                    },
                }
                payload_hash = MODULE.COHORT_CHECKER._prompt_payload_sha256(prompt)
                prompt["metadata"]["provenance"][
                    "prompt_record_payload_sha256"
                ] = payload_hash
                prompts.append(prompt)
                prompt_summaries.append(
                    {
                        "id": prompt_id,
                        "cohort_id": cohort_row["id"],
                        "instance_id": instance_id,
                        "global_request_index": global_index,
                        "prompt_record_payload_sha256": payload_hash,
                    }
                )
                audits.append(
                    {
                        "instance_id": instance_id,
                        "selection_index": selection_index,
                        "cohort_id": cohort_row["id"],
                        "campaign_sha256": campaign_hashes[cohort_index],
                        "request_count": 1,
                        "probeable_request_indices": [1],
                        "selected_request_indices": [1],
                        "selected_checkpoint_count": 1,
                    }
                )
                selection_index += 1
        prompts_path = root / "reserved-prompts.json"
        prompts_path.write_text(json.dumps(prompts, indent=2) + "\n", encoding="ascii")
        summary = {
            "schema_version": 1,
            "kind": "swe_verified_behavioral_probe_combination",
            "cohort_manifest_sha256": cohort_sha,
            "source_campaign_sha256s": campaign_hashes,
            "campaign_sha256s": campaign_hashes,
            "action_protocol_sha256": cohort["pins"]["action_protocol_sha256"],
            "chat_template_sha256": cohort["pins"]["chat_template_sha256"],
            "cohort_count": 2,
            "task_count": 20,
            "prompt_count": len(prompts),
            "prompt_bundle_sha256": MODULE.sha256_file(prompts_path),
            "prompts": prompt_summaries,
            "cohorts": summary_cohorts,
            "task_audits": audits,
            "global_request_count": len(prompts),
        }
        summary_path = root / "reserved-prompts-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
        return prompts, prompts_path, summary_path

    @staticmethod
    def validate(
        prompts: list[dict],
        prompts_path: Path,
        summary_path: Path,
        *,
        summary_pin: str | None = None,
    ) -> dict:
        protocol = validate_protocol()
        protocol["reserved_prompts_summary_sha256"] = (
            MODULE.sha256_file(summary_path) if summary_pin is None else summary_pin
        )
        return MODULE.validate_reserved_evaluation_inputs(
            prompts,
            protocol=protocol,
            cohort_path=RESERVED_COHORT_PATH,
            campaign_a_path=RESERVED_CAMPAIGN_A_PATH,
            campaign_b_path=RESERVED_CAMPAIGN_B_PATH,
            image_registry_path=RESERVED_IMAGE_REGISTRY_PATH,
            prompts_path=prompts_path,
            prompts_summary_path=summary_path,
        )

    def test_valid_materialized_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, prompts_path, summary_path = self.materialized_fixture(
                Path(directory)
            )
            result = self.validate(prompts, prompts_path, summary_path)

        self.assertEqual(result["task_count"], 20)
        self.assertEqual(result["prompt_count"], 20)
        self.assertTrue(result["prompt_payload_and_summary_hash_bindings_validated"])

    def test_checked_in_null_summary_pin_blocks_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, prompts_path, summary_path = self.materialized_fixture(
                Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "not pinned.*forbidden"):
                MODULE.validate_reserved_evaluation_inputs(
                    prompts,
                    protocol=validate_protocol(),
                    cohort_path=RESERVED_COHORT_PATH,
                    campaign_a_path=RESERVED_CAMPAIGN_A_PATH,
                    campaign_b_path=RESERVED_CAMPAIGN_B_PATH,
                    image_registry_path=RESERVED_IMAGE_REGISTRY_PATH,
                    prompts_path=prompts_path,
                    prompts_summary_path=summary_path,
                )

    def test_rejects_development_or_arbitrary_task(self) -> None:
        prompt = {
            "id": "development-task-r1",
            "metadata": {
                "task": {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "probeable_request_indices": [1],
                },
                "selection": {"task_request_index": 1},
                "cohort": {},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "prompts.json"
            summary_path = root / "summary.json"
            prompts_path.write_text(json.dumps([prompt]), encoding="ascii")
            summary_path.write_text("{}", encoding="ascii")
            with self.assertRaises(ValueError):
                self.validate([prompt], prompts_path, summary_path)

    def test_evaluate_input_preparation_invokes_reserved_binding(self) -> None:
        prompt = {
            "id": "development-task-r1",
            "metadata": {
                "task": {
                    "instance_id": "owner__repo-1",
                    "repo": "owner/repo",
                    "probeable_request_indices": [1],
                },
                "selection": {"task_request_index": 1},
                "cohort": {},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "prompts.json"
            report_path = root / "report.json"
            prompts_path.write_text(json.dumps([prompt]), encoding="ascii")
            report_path.write_text("{}", encoding="ascii")
            args = SimpleNamespace(
                command="evaluate",
                prompts=prompts_path,
                public_report=report_path,
                protocol=PROTOCOL_PATH,
                source_protocol=SOURCE_PROTOCOL_PATH,
                behavioral_protocol=BEHAVIORAL_PROTOCOL_PATH,
                reserved_cohort=RESERVED_COHORT_PATH,
                reserved_campaign_a=RESERVED_CAMPAIGN_A_PATH,
                reserved_campaign_b=RESERVED_CAMPAIGN_B_PATH,
                reserved_image_registry=RESERVED_IMAGE_REGISTRY_PATH,
                reserved_prompts_summary=root / "missing-summary.json",
            )

            with self.assertRaisesRegex(ValueError, "summary"):
                MODULE._prepare_inputs(args)

    def test_rejects_forged_membership_for_a_reserved_task(self) -> None:
        cohort = json.loads(RESERVED_COHORT_PATH.read_bytes())
        instance_id = cohort["instance_ids"][0]
        prompt = {
            "id": f"{instance_id}-r1",
            "metadata": {
                "task": {
                    "instance_id": instance_id,
                    "repo": MODULE._expected_repository(instance_id),
                    "probeable_request_indices": [1],
                },
                "selection": {"task_request_index": 1},
                "cohort": {
                    "id": "development",
                    "index": 0,
                    "campaign_sha256": "0" * 64,
                    "cohort_manifest_sha256": "0" * 64,
                    "source_task_instance_ids": [instance_id],
                },
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "prompts.json"
            summary_path = root / "summary.json"
            prompts_path.write_text(json.dumps([prompt]), encoding="ascii")
            summary_path.write_text("{}", encoding="ascii")
            with self.assertRaises(ValueError):
                self.validate([prompt], prompts_path, summary_path)

    def test_forged_prompt_content_and_summary_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts, prompts_path, summary_path = self.materialized_fixture(root)
            forged = copy.deepcopy(prompts)
            forged[0]["text"] = "arbitrary forged prompt content"
            prompts_path.write_text(json.dumps(forged, indent=2) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "payload hash differs"):
                self.validate(forged, prompts_path, summary_path)

            prompts, prompts_path, summary_path = self.materialized_fixture(root)
            original_summary_sha = MODULE.sha256_file(summary_path)
            forged = copy.deepcopy(prompts)
            forged[0]["text"] = "self-consistent but unpinned forged prompt"
            payload_hash = MODULE.COHORT_CHECKER._prompt_payload_sha256(forged[0])
            forged[0]["metadata"]["provenance"][
                "prompt_record_payload_sha256"
            ] = payload_hash
            prompts_path.write_text(json.dumps(forged, indent=2) + "\n", encoding="ascii")
            forged_summary = json.loads(summary_path.read_bytes())
            forged_summary["prompts"][0][
                "prompt_record_payload_sha256"
            ] = payload_hash
            forged_summary["prompt_bundle_sha256"] = MODULE.sha256_file(prompts_path)
            summary_path.write_text(
                json.dumps(forged_summary, indent=2) + "\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "differs from protocol pin"):
                self.validate(
                    forged,
                    prompts_path,
                    summary_path,
                    summary_pin=original_summary_sha,
                )

            prompts, prompts_path, summary_path = self.materialized_fixture(root)
            original_summary_sha = MODULE.sha256_file(summary_path)
            summary = json.loads(summary_path.read_bytes())
            summary["prompt_bundle_sha256"] = "0" * 64
            summary_path.write_text(json.dumps(summary), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "differs from protocol pin"):
                self.validate(
                    prompts,
                    prompts_path,
                    summary_path,
                    summary_pin=original_summary_sha,
                )


class InferenceExtractionTest(unittest.TestCase):
    def test_stable_inference_row_does_not_require_current_action_label(self) -> None:
        prompt = {
            "id": "task-1-r1",
            "text": "inspect the failing code",
            "token_ids": [10, 11, 12],
            "score_token_ids": [20, 21],
            "metadata": {
                "labels": {"action": {"status": "unavailable", "class_id": None}},
                "selection": {
                    "task_request_index": 1,
                    "checkpoint_ordinal": 0,
                    "primary_for_action_evaluation": True,
                },
                "task": {"instance_id": "task-1", "repo": "owner/repo"},
            },
        }
        experiment = {
            "id": prompt["id"],
            "prompt": prompt["text"],
            "prompt_token_ids": prompt["token_ids"],
            "metadata": prompt["metadata"],
            "capture_positions_resolved": [2],
            "scored_vocabulary": {"token_ids": prompt["score_token_ids"]},
        }
        source_protocol = {
            "class_ids": MODULE.SOURCE_ACTION_CLASSES,
            "layers": tuple(range(24, 48)),
            "token_ids_by_class": {},
            "token_texts_by_class": {},
            "eligibility": {"require_primary": True},
        }

        source = MODULE.SOURCE_ANALYZER
        with (
            mock.patch.object(source, "_validate_report_provenance"),
            mock.patch.object(
                source,
                "_causal_history_features",
                return_value=({prompt["id"]: [0.0] * 14}, {"complete": 1}),
            ),
            mock.patch.object(source, "_numerically_stable", return_value=(True, [])),
            mock.patch.object(source, "_lexical_features", return_value=[0.0] * 16),
            mock.patch.object(
                source,
                "_layer_class_features",
                side_effect=lambda *_args, method, **_kwargs: [
                    1.0 if method == "public_jacobian" else 2.0
                ]
                * 96,
            ),
        ):
            extracted = MODULE.extract_stable_inference_rows(
                [prompt], {"experiments": [experiment]}, source_protocol=source_protocol
            )

        self.assertEqual(extracted["eligibility"]["inference_eligible_stable_row_count"], 1)
        self.assertTrue(extracted["eligibility"]["current_action_label_not_required"])
        self.assertEqual(extracted["rows"][0]["current_action_label_status"], "unavailable")
        self.assertIsNone(extracted["rows"][0]["current_action_class_id"])


class CompactFeatureTest(unittest.TestCase):
    def test_compact_summary_has_frozen_statistic_then_action_order(self) -> None:
        matrix = np.asarray(
            [[10.0 * layer + action for action in range(4)] for layer in range(24)]
        )
        compact = MODULE.compact_layer_shape(matrix.reshape(-1))
        x = np.arange(24, dtype=np.float64)
        centered = x - x.mean()
        expected = np.concatenate(
            [
                matrix.mean(axis=0),
                matrix.std(axis=0),
                matrix.min(axis=0),
                matrix.max(axis=0),
                matrix[:6].mean(axis=0),
                matrix[6:18].mean(axis=0),
                matrix[18:].mean(axis=0),
                centered @ matrix / float(centered @ centered),
                matrix[-1] - matrix[0],
                np.argmax(matrix, axis=0) / 23.0,
            ]
        )

        self.assertEqual(compact.shape, (40,))
        np.testing.assert_allclose(compact, expected)
        np.testing.assert_allclose(compact[:4], matrix.mean(axis=0))
        np.testing.assert_allclose(compact[4:8], matrix.std(axis=0))

    def test_feature_blocks_delta_ema_gap_and_censored_state_update(self) -> None:
        base = np.arange(96, dtype=np.float64)
        stable_rows = [
            stable_source_row(1, base),
            stable_source_row(2, base + 2.0),
            stable_source_row(4, base + 4.0),
        ]
        assignments = {
            "row-1": {
                "status": "available",
                "label": "edit",
                "reason": None,
                "horizon_requests": 0,
                "target_request_index": 1,
                "target_source_action": "edit",
            },
            "row-2": {
                "status": "censored",
                "label": None,
                "reason": "unknown_before_next_milestone",
                "horizon_requests": None,
                "target_request_index": None,
                "target_source_action": None,
            },
            "row-4": {
                "status": "available",
                "label": "check_or_finish",
                "reason": None,
                "horizon_requests": 0,
                "target_request_index": 4,
                "target_source_action": "validate",
            },
        }

        rows = MODULE.build_feature_rows(stable_rows, assignments, ema_alpha=0.5)
        first = np.asarray(rows[0]["features"]["j_compact"])
        second = np.asarray(rows[1]["features"]["j_compact"])
        third = np.asarray(rows[2]["features"]["j_compact"])

        self.assertEqual(first.shape, (154,))
        np.testing.assert_allclose(first[:40], MODULE.compact_layer_shape(base))
        np.testing.assert_allclose(first[40:120], 0.0)
        np.testing.assert_allclose(first[120:152], stable_rows[0]["history_context"])
        np.testing.assert_allclose(first[152:], [0.0, 1.0])

        np.testing.assert_allclose(second[40:80], MODULE.compact_layer_shape([2.0] * 96))
        np.testing.assert_allclose(second[80:120], MODULE.compact_layer_shape([2.0] * 96))
        np.testing.assert_allclose(second[152:], [math.log1p(1), 0.0])

        # Row 2 is censored, but its stable emission must still advance sequence state.
        np.testing.assert_allclose(third[40:80], MODULE.compact_layer_shape([2.0] * 96))
        np.testing.assert_allclose(third[80:120], MODULE.compact_layer_shape([3.0] * 96))
        np.testing.assert_allclose(third[152:], [math.log1p(2), 0.0])
        self.assertEqual(rows[1]["label_status"], "censored")


class WeightingTest(unittest.TestCase):
    def test_task_event_prefix_weights_and_class_rebalance_diagnostics(self) -> None:
        tasks = np.asarray(["t1", "t1", "t1", "t2", "t2"])
        events = np.asarray(["e1", "e1", "e2", "e3", "e3"])
        labels = np.asarray(
            ["edit", "edit", "check_or_finish", "check_or_finish", "check_or_finish"]
        )

        weights, diagnostics = MODULE.task_event_prefix_weights(
            tasks, events, labels=labels, rebalance_classes=True
        )

        np.testing.assert_allclose(weights, [0.25, 0.25, 1 / 6, 1 / 6, 1 / 6])
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(diagnostics["task_count"], 2)
        self.assertEqual(diagnostics["event_count"], 3)
        self.assertEqual(diagnostics["pre_class_mass"], {"edit": 0.25, "check_or_finish": 0.75})
        self.assertAlmostEqual(diagnostics["post_class_mass"]["edit"], 0.5)
        self.assertAlmostEqual(diagnostics["post_class_mass"]["check_or_finish"], 0.5)
        self.assertAlmostEqual(diagnostics["pre_task_mass_minimum"], 0.5)
        self.assertAlmostEqual(diagnostics["pre_task_mass_maximum"], 0.5)
        self.assertAlmostEqual(diagnostics["post_task_mass_minimum"], 1 / 3)
        self.assertAlmostEqual(diagnostics["post_task_mass_maximum"], 2 / 3)
        self.assertEqual(len(diagnostics["weight_float64_sha256"]), 64)


class ProbabilityMetricTest(unittest.TestCase):
    def test_single_class_balanced_accuracy_is_undefined(self) -> None:
        metrics = MODULE.probability_metrics(
            np.asarray(["edit", "edit"]),
            np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float64),
            np.asarray(["task-1", "task-2"]),
            np.asarray(["event-1", "event-2"]),
        )

        self.assertEqual(metrics["recall_edit"], 1.0)
        self.assertIsNone(metrics["recall_check_or_finish"])
        self.assertIsNone(metrics["balanced_accuracy"])

    def test_single_class_bootstrap_draws_reduce_balanced_accuracy_valid_fraction(self) -> None:
        rows = [
            {
                "row_id": "edit-row",
                "repo": "edit/repo",
                "task_id": "edit-task",
                "event_id": "edit-event",
                "label": "edit",
            },
            {
                "row_id": "check-row",
                "repo": "check/repo",
                "task_id": "check-task",
                "event_id": "check-event",
                "label": "check_or_finish",
            },
        ]
        predictions = {
            variant: [
                {"row_id": "edit-row", "probabilities": [0.9, 0.1]},
                {"row_id": "check-row", "probabilities": [0.1, 0.9]},
            ]
            for variant in MODULE.VARIANTS
        }

        bootstrap = MODULE.hierarchical_bootstrap(
            rows,
            predictions,
            samples=100,
            seed=31,
            confidence_level=0.95,
            minimum_valid_fraction=1.0,
        )
        interval = bootstrap["intervals"]["j_compact"]["balanced_accuracy"]

        self.assertGreater(interval["valid_samples"], 0)
        self.assertLess(interval["valid_samples"], 100)
        self.assertGreater(interval["valid_fraction"], 0.0)
        self.assertLess(interval["valid_fraction"], 1.0)
        self.assertFalse(interval["minimum_valid_fraction_met"])


class BundleIntegrityTest(unittest.TestCase):
    def test_development_gate_decision_rounds_float_ulp_noise(self) -> None:
        first = {
            "passed": False,
            "passed_count": 0,
            "gate_count": 1,
            "records": [
                {"id": "float-evidence", "observed": 0.123456789012341, "passed": False}
            ],
        }
        second = copy.deepcopy(first)
        second["records"][0]["observed"] += 1e-17

        first_decision = MODULE.development_gate_decision_record(first)
        second_decision = MODULE.development_gate_decision_record(second)

        self.assertEqual(first_decision, second_decision)
        self.assertEqual(
            first_decision["float_metric_decimal_places"],
            MODULE.GATE_EVIDENCE_DECIMAL_PLACES,
        )

    def test_tampered_bundle_is_rejected_before_joblib_load(self) -> None:
        class FakeJoblib:
            loaded = False

            @classmethod
            def load(cls, _path):
                cls.loaded = True
                raise AssertionError("executable pickle load must not be reached")

        class FakeExtraTreesClassifier:
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "model.joblib"
            manifest_path = root / "manifest.json"
            bundle_path.write_bytes(b"tampered bundle bytes")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": MODULE.MANIFEST_SCHEMA_VERSION,
                        "id": "swe-binary-phase-interpreter-v2-model-manifest",
                        "bundle": {"sha256": "0" * 64},
                    }
                ),
                encoding="ascii",
            )

            with mock.patch.object(
                MODULE,
                "_ml_dependencies",
                return_value=(FakeJoblib, FakeExtraTreesClassifier),
            ):
                with self.assertRaisesRegex(ValueError, "bundle SHA-256 mismatch"):
                    MODULE.validate_bundle_and_manifest(
                        bundle_path,
                        manifest_path,
                        current_protocol=validate_protocol(),
                        expected_source_protocol_sha256="2" * 64,
                        expected_behavioral_protocol_sha256="3" * 64,
                        protocol_phase="fit",
                    )

        self.assertFalse(FakeJoblib.loaded)

    def test_probability_canary_accepts_roundoff_but_rejects_material_drift(self) -> None:
        class FakeProbabilityModel:
            def __init__(self, probability: list[float]) -> None:
                self.classes_ = np.asarray(MODULE.CLASSES)
                self.probability = np.asarray([probability], dtype=np.float64)
                self.n_jobs = 8

            def predict_proba(self, values: np.ndarray) -> np.ndarray:
                return np.repeat(self.probability, len(values), axis=0)

            def get_params(self, deep: bool = False) -> dict[str, int]:
                del deep
                return {"n_jobs": self.n_jobs}

            def set_params(self, **parameters: int):
                self.n_jobs = parameters.get("n_jobs", self.n_jobs)
                return self

        feature = np.linspace(-1.0, 1.0, MODULE.FEATURE_WIDTH).tolist()
        row = {
            "row_id": "canary-row",
            "features": {variant: feature for variant in MODULE.VARIANTS},
        }
        expected = [0.4, 0.6]
        canary = MODULE.build_model_canary(
            [row],
            {variant: FakeProbabilityModel(expected) for variant in MODULE.VARIANTS},
            temperature=1.0,
        )

        one_ulp_first = float(np.nextafter(expected[0], 1.0))
        roundoff = [one_ulp_first, 1.0 - one_ulp_first]
        roundoff_models = {
            variant: FakeProbabilityModel(roundoff) for variant in MODULE.VARIANTS
        }
        MODULE.verify_model_canary(canary, roundoff_models, temperature=1.0)
        self.assertTrue(
            np.max(np.abs(np.asarray(roundoff) - np.asarray(expected))) < 1e-12
        )
        self.assertEqual(
            {model.n_jobs for model in roundoff_models.values()},
            {8},
            "canary verification must restore the estimator parallelism contract",
        )

        material_drift = [expected[0] + 1e-7, expected[1] - 1e-7]
        with self.assertRaisesRegex(ValueError, "canary prediction changed"):
            MODULE.verify_model_canary(
                canary,
                {
                    variant: FakeProbabilityModel(material_drift)
                    for variant in MODULE.VARIANTS
                },
                temperature=1.0,
            )

    def test_development_stop_rejects_before_load_but_internal_verify_can_load(self) -> None:
        class FakeExtraTreesClassifier:
            def __init__(self) -> None:
                self.classes_ = np.asarray(MODULE.CLASSES)
                self.n_features_in_ = MODULE.FEATURE_WIDTH
                self.n_jobs = 8

            def predict_proba(self, values: np.ndarray) -> np.ndarray:
                return np.repeat([[0.4, 0.6]], len(values), axis=0)

            def get_params(self, deep: bool = False) -> dict[str, int]:
                del deep
                return {"n_jobs": self.n_jobs}

            def set_params(self, **parameters: int):
                self.n_jobs = parameters.get("n_jobs", self.n_jobs)
                return self

        decision = MODULE.development_gate_decision_record(
            {
                "passed": False,
                "passed_count": 0,
                "gate_count": 1,
                "records": [{"id": "development-stop", "passed": False}],
            }
        )
        fit_protocol = validate_protocol()
        fit_lifecycle = MODULE.build_fit_protocol_lifecycle(fit_protocol)
        fit_state = MODULE.fit_protocol_state_record(fit_lifecycle)
        materialized_protocol_value = copy.deepcopy(PROTOCOL)
        materialized_protocol_value["pins"][
            "reserved_prompts_summary_sha256"
        ] = "4" * 64
        evaluation_protocol = validate_protocol(
            materialized_protocol_value, protocol_sha256="5" * 64
        )
        protocol_sha = fit_protocol["sha256"]
        source_protocol_sha = "2" * 64
        behavioral_protocol_sha = "3" * 64
        models = {
            variant: FakeExtraTreesClassifier() for variant in MODULE.VARIANTS
        }
        canary_row = {
            "row_id": "canary-row",
            "features": {
                variant: [0.0] * MODULE.FEATURE_WIDTH for variant in MODULE.VARIANTS
            },
        }
        canary = MODULE.build_model_canary(
            [canary_row], models, temperature=1.0
        )
        estimator_parameters = {
            variant: model.get_params(deep=False) for variant, model in models.items()
        }
        training_identity = {"fixture": "training-identity"}
        current_binary_sha = MODULE.sha256_file(SCRIPT_PATH)
        current_source_sha = MODULE.sha256_file(MODULE.SOURCE_ANALYZER_PATH)
        current_checker_sha = MODULE.sha256_file(MODULE.COHORT_CHECKER_PATH)
        bundle = {
            "schema_version": MODULE.BUNDLE_SCHEMA_VERSION,
            "id": "swe-binary-phase-interpreter-v2-model-bundle",
            "classes_in_order": list(MODULE.CLASSES),
            "variants_in_order": list(MODULE.VARIANTS),
            "feature_width": MODULE.FEATURE_WIDTH,
            "protocol_sha256": protocol_sha,
            "core_protocol_sha256": fit_protocol["core_sha256"],
            "fit_reserved_prompts_summary_sha256": None,
            "protocol_lifecycle": fit_lifecycle,
            "protocol_fit_state": fit_state,
            "source_protocol_sha256": source_protocol_sha,
            "behavioral_protocol_sha256": behavioral_protocol_sha,
            "binary_analyzer_sha256": current_binary_sha,
            "source_analyzer_sha256": current_source_sha,
            "cohort_checker_sha256": current_checker_sha,
            "temperature": 1.0,
            "confidence_threshold": 0.0,
            "models": models,
            "training_identity": training_identity,
            "estimator_get_params": estimator_parameters,
            "model_canary": canary,
            "development_gate_decision": decision,
        }

        class FakeJoblib:
            loaded = False

            @classmethod
            def load(cls, _path):
                cls.loaded = True
                return bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "model.joblib"
            manifest_path = root / "manifest.json"
            bundle_path.write_bytes(b"hash-bound fake bundle")
            manifest = {
                "schema_version": MODULE.MANIFEST_SCHEMA_VERSION,
                "id": "swe-binary-phase-interpreter-v2-model-manifest",
                "bundle": {"sha256": MODULE.sha256_file(bundle_path)},
                "protocol_sha256": protocol_sha,
                "core_protocol_sha256": fit_protocol["core_sha256"],
                "fit_reserved_prompts_summary_sha256": None,
                "protocol_lifecycle": fit_lifecycle,
                "protocol_fit_state": fit_state,
                "classes_in_order": list(MODULE.CLASSES),
                "variants_in_order": list(MODULE.VARIANTS),
                "feature_width": MODULE.FEATURE_WIDTH,
                "temperature": 1.0,
                "confidence_threshold": 0.0,
                "development_inputs": {
                    "binary_analyzer": current_binary_sha,
                    "source_analyzer": current_source_sha,
                    "cohort_checker": current_checker_sha,
                },
                "runtime_versions": MODULE._runtime_versions(),
                "training_identity": training_identity,
                "estimator_get_params": estimator_parameters,
                "model_canary": canary,
                "development_gate_decision": decision,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")

            with mock.patch.object(
                MODULE,
                "_ml_dependencies",
                return_value=(FakeJoblib, FakeExtraTreesClassifier),
            ) as dependencies:
                with self.assertRaisesRegex(
                    ValueError, "development gates failed.*evaluation is forbidden"
                ):
                    MODULE.validate_bundle_and_manifest(
                        bundle_path,
                        manifest_path,
                        current_protocol=evaluation_protocol,
                        expected_source_protocol_sha256=source_protocol_sha,
                        expected_behavioral_protocol_sha256=behavioral_protocol_sha,
                        protocol_phase="evaluate",
                    )
                self.assertFalse(FakeJoblib.loaded)
                dependencies.assert_not_called()

                (
                    loaded_bundle,
                    loaded_manifest,
                    protocol_state,
                ) = MODULE.validate_bundle_and_manifest(
                    bundle_path,
                    manifest_path,
                    current_protocol=fit_protocol,
                    expected_source_protocol_sha256=source_protocol_sha,
                    expected_behavioral_protocol_sha256=behavioral_protocol_sha,
                    protocol_phase="fit",
                    require_development_pass=False,
                )

        self.assertTrue(FakeJoblib.loaded)
        self.assertEqual(loaded_bundle["development_gate_decision"], decision)
        self.assertEqual(loaded_manifest["development_gate_decision"], decision)
        self.assertEqual(protocol_state, fit_state)

    @unittest.skipUnless(
        importlib.util.find_spec("joblib") is not None
        and importlib.util.find_spec("sklearn") is not None,
        "checked-in model verification requires the readout-v2 dependencies",
    )
    def test_checked_in_bundle_passes_internal_post_fit_verification(self) -> None:
        self.assertTrue(MODEL_BUNDLE_PATH.is_file(), "checked-in model bundle is missing")
        self.assertTrue(
            MODEL_MANIFEST_PATH.is_file(), "checked-in model manifest is missing"
        )

        bundle, manifest, protocol_state = MODULE.validate_bundle_and_manifest(
            MODEL_BUNDLE_PATH,
            MODEL_MANIFEST_PATH,
            current_protocol=validate_protocol(),
            expected_source_protocol_sha256=MODULE.sha256_file(SOURCE_PROTOCOL_PATH),
            expected_behavioral_protocol_sha256=MODULE.sha256_file(
                BEHAVIORAL_PROTOCOL_PATH
            ),
            protocol_phase="fit",
            require_development_pass=False,
        )

        self.assertEqual(
            bundle["development_gate_decision"],
            manifest["development_gate_decision"],
        )
        self.assertEqual(protocol_state["phase"], "fit")


class HorizonAndGateTest(unittest.TestCase):
    def test_horizon_bucket_boundaries(self) -> None:
        expected = {
            0: "h0",
            1: "h1_2",
            2: "h1_2",
            3: "h3_5",
            5: "h3_5",
            6: "h6_10",
            10: "h6_10",
            11: "h11_plus",
            10_000: "h11_plus",
        }
        self.assertEqual({value: MODULE.horizon_bucket(value) for value in expected}, expected)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.horizon_bucket(-1)

    def test_support_aliases_are_executable_and_gate_failure_is_fail_closed(self) -> None:
        rows = [
            {
                "row_id": "r1",
                "task_id": "t1",
                "repo": "owner/repo",
                "current_action_label_status": "unavailable",
                "label_status": "available",
                "label": "edit",
                "event_id": "t1::1::edit",
            },
            {
                "row_id": "r2",
                "task_id": "t1",
                "repo": "owner/repo",
                "current_action_label_status": "available",
                "label_status": "censored",
                "label": None,
                "label_censor_reason": "no_observed_future_milestone",
                "event_id": None,
            },
        ]
        support = MODULE.support_summary(
            rows,
            eligibility={
                "prompt_count": 4,
                "numerically_stable_before_feature_requirements": 3,
            },
        )
        validation_support_gates = validate_protocol()["gates"]["validation"]["support"]
        for key in validation_support_gates:
            with self.subTest(key=key):
                self.assertIsNotNone(
                    MODULE._support_value(support, key),
                    f"checked-in validation support gate {key} has no executable alias",
                )

        gate_contract = {
            "support": {"minimum_inference_tasks": 2},
            "absolute": [
                {
                    "id": "accuracy_floor",
                    "variant": "j_compact",
                    "metric": "accuracy",
                    "bound": "point",
                    "operator": "minimum_inclusive",
                    "value": 0.9,
                }
            ],
            "paired": [],
        }
        results = {
            variant: {
                "probability_metrics_on_label_evaluable_rows": {"accuracy": 0.8}
            }
            for variant in MODULE.VARIANTS
        }
        bootstrap = {"intervals": {}, "paired_differences": {}}

        evaluated = MODULE.evaluate_gates(
            gate_contract, support=support, results=results, bootstrap=bootstrap
        )

        self.assertFalse(evaluated["passed"])
        self.assertEqual(evaluated["gate_count"], 2)
        self.assertEqual(evaluated["records"][0]["observed"], 1)
        self.assertFalse(evaluated["records"][0]["passed"])
        self.assertFalse(evaluated["records"][1]["passed"])


if __name__ == "__main__":
    unittest.main()
