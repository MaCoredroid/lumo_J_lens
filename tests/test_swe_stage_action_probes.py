#!/usr/bin/env python3
"""Focused tests for multistage next-action augmentation and paired analysis."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUGMENT = load_module(
    "augment_swe_multistage_action_probes",
    ROOT / "scripts" / "augment_swe_multistage_action_probes.py",
)
ANALYZE = load_module(
    "analyze_swe_multistage_probes",
    ROOT / "scripts" / "analyze_swe_multistage_probes.py",
)
PROTOCOL_BYTES = (ROOT / "configs" / "swe_stage_action_probes.json").read_bytes()
PROTOCOL = json.loads(PROTOCOL_BYTES)
LIFECYCLE_PROTOCOL_BYTES = (
    ROOT / "configs" / "swe_multistage_protocol.json"
).read_bytes()
LIFECYCLE_PROTOCOL = json.loads(LIFECYCLE_PROTOCOL_BYTES)


class FakeTokenizer:
    def __init__(self, protocol: dict[str, object]):
        tokens = [
            token
            for family in ("action_classes", "outcome_classes")
            for record in protocol[family]
            for token in record["tokens"]
        ]
        tokens.extend(
            [
                {"text": " target", "token_id": 100},
                {"text": " foil", "token_id": 101},
            ]
        )
        self.by_text = {token["text"]: token["token_id"] for token in tokens}
        self.by_id = {token["token_id"]: token["text"] for token in tokens}

    def __len__(self) -> int:
        return AUGMENT.TOKENIZER_VOCABULARY_SIZE

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.by_text[text]]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.by_id[token_id] for token_id in token_ids)


def shell_result(command: str, *, exit_code: int) -> str:
    return (
        f"Command: {command}\n"
        "Directory: (root)\n"
        "Output: assertion failed\n"
        "Error: (none)\n"
        f"Exit Code: {exit_code}\n"
        "Signal: 0\n"
        "Process Group PGID: 123"
    )


def source_prompt(root: Path) -> dict[str, object]:
    capture = root / "captures"
    capture.mkdir()
    current = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
    }
    command = "python << 'PY'\nvalue = 1\nassert value == 2\nPY"
    assistant = {
        "role": "assistant",
        "content": "\n\n",
        "reasoning_content": "Let me verify the fix works for the specific bug scenario.",
        "reasoning": "Let me verify the fix works for the specific bug scenario.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }
    following = {
        "messages": current["messages"]
        + [
            assistant,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [{"type": "text", "text": shell_result(command, exit_code=1)}],
            },
        ]
    }
    current_path = capture / "chat_0001.json"
    next_path = capture / "chat_0002.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")
    next_path.write_text(json.dumps(following), encoding="utf-8")
    text = "exact rendered prompt"
    token_ids = [10, 11]
    concepts = [
        {
            "id": "target-concept",
            "path": "package/source.py",
            "target": "target",
            "family": "hunk_symbol",
            "forms": [{"text": " target", "token_id": 100}],
            "foils": [
                {
                    "task_instance_id": "other__task-2",
                    "concept_id": "foil-concept",
                    "target": "foil",
                    "family": "hunk_symbol",
                    "forms": [{"text": " foil", "token_id": 101}],
                }
            ],
        }
    ]
    return {
        "id": "swe-s6-000-project__task-1",
        "text": text,
        "token_ids": token_ids,
        "score_token_ids": [100, 101],
        "metadata": {
            "kind": "swe_verified_multistage_probe",
            "analysis_role": "oracle_hidden",
            "lifecycle_protocol_sha256": AUGMENT.sha256_bytes(
                LIFECYCLE_PROTOCOL_BYTES
            ),
            "task": {"instance_id": "project__task-1", "repo": "project/repo"},
            "stage": {"id": "S6", "name": "validation"},
            "concepts": concepts,
            "visibility_audit": {
                "scope": "test",
                "records": [
                    {"subject": "target", "exposed": False},
                    {"subject": "foil", "exposed": False},
                ],
            },
            "provenance": {
                "raw_request_index": 1,
                "raw_request_path": "captures/chat_0001.json",
                "raw_request_sha256": AUGMENT.sha256_file(current_path),
                "rendered_prompt_sha256": AUGMENT.sha256_text(text),
                "token_ids_sha256": AUGMENT.sha256_json(token_ids),
                "prompt_token_count": len(token_ids),
                "usage": {"idx": 1, "finish_reason": "tool_calls"},
                "usage_record_sha256": "u" * 64,
                "official_verdict": {"verdict": "resolved"},
                "next_completion_transition": {
                    "completion_index": 1,
                    "contract": (
                        "completion_N_is_materialized_by_chat_N_plus_1_or_bound_by_"
                        "terminal_usage"
                    ),
                    "event_labels": ["other_tool_action"],
                    "materialized_in_raw_request_sha256": AUGMENT.sha256_file(next_path),
                    "materialized_in_request_index": 2,
                    "terminal_response": False,
                    "transition": {"synthetic": True},
                    "transition_sha256": AUGMENT.sha256_json({"synthetic": True}),
                    "used_only_for_declared_stage_selection_or_action_analysis": True,
                },
            },
        },
    }


def lens(label: str) -> dict[str, object]:
    common = {
        "d_model": 5120,
        "source_layers": list(ANALYZE.ALL_SOURCE_LAYERS),
        "tensor_shape": [5120, 5120],
    }
    if label == "public":
        return {
            **common,
            "repo_id": ANALYZE.PUBLIC_LENS_REPO,
            "revision": ANALYZE.PUBLIC_LENS_REVISION,
            "sha256": ANALYZE.PUBLIC_LENS_SHA256,
            "n_prompts": 1000,
        }
    if label == "nf4":
        return {
            **common,
            "kind": "local_fit",
            "sha256": ANALYZE.NF4_LENS_SHA256,
            "provenance_sha256": ANALYZE.NF4_PROVENANCE_SHA256,
            "fit_quantization": "bitsandbytes-nf4-double-quant-bfloat16",
            "n_prompts": 10,
        }
    return {
        **common,
        "kind": "native_nvfp4_ste_fit",
        "sha256": ANALYZE.NATIVE_LENS_SHA256,
        "state_sha256": ANALYZE.NATIVE_STATE_SHA256,
        "provenance_sha256": ANALYZE.NATIVE_PROVENANCE_SHA256,
        "fit_model": ANALYZE.MODEL_REPO,
        "fit_model_revision": ANALYZE.MODEL_REVISION,
        "n_prompts": 10,
    }


def token_scores(prompt: dict[str, object], *, method: str, label: str) -> dict[int, float]:
    score_ids = prompt["score_token_ids"]
    scores = {token_id: 0.0 for token_id in score_ids}
    action = prompt["metadata"]["stage_action_probe"]["scored_vocabulary"]["action_classes"]
    outcomes = prompt["metadata"]["stage_action_probe"]["scored_vocabulary"]["outcome_classes"]
    if method == "logit":
        scores[100] = -1.0
        winning_action = "inspect"
        gold_score = -1.0
    else:
        gold_score = {"public": 2.0, "nf4": 1.5, "native": 1.0}[label]
        scores[100] = gold_score
        winning_action = "validate"
    for record in action:
        if record["id"] == winning_action:
            for token in record["tokens"]:
                scores[token["token_id"]] = 3.0 if method != "logit" else 1.0
    for record in outcomes:
        if record["id"] == "failure":
            for token in record["tokens"]:
                scores[token["token_id"]] = 2.0
    return scores


def readout(prompt: dict[str, object], *, method: str, label: str) -> dict[str, object]:
    scores = token_scores(prompt, method=method, label=label)
    token_text = {
        form["token_id"]: form["text"]
        for concept in prompt["metadata"]["concepts"]
        for forms in ([concept["forms"]] + [foil["forms"] for foil in concept["foils"]])
        for form in forms
    }
    for family in ("action_classes", "outcome_classes"):
        for record in prompt["metadata"]["stage_action_probe"]["scored_vocabulary"][family]:
            for token in record["tokens"]:
                token_text[token["token_id"]] = token["text"]
    return {
        "scored_tokens": [
            {
                "token_id": token_id,
                "token": token_text[token_id],
                "rank": 1 + index,
                "score": scores[token_id],
                "logprob": scores[token_id] - 10.0,
            }
            for index, token_id in enumerate(prompt["score_token_ids"])
        ]
    }


def report(prompt: dict[str, object], label: str) -> dict[str, object]:
    token_text = {}
    for concept in prompt["metadata"]["concepts"]:
        for forms in ([concept["forms"]] + [foil["forms"] for foil in concept["foils"]]):
            for form in forms:
                token_text[form["token_id"]] = form["text"]
    for family in ("action_classes", "outcome_classes"):
        for record in prompt["metadata"]["stage_action_probe"]["scored_vocabulary"][family]:
            for token in record["tokens"]:
                token_text[token["token_id"]] = token["text"]
    position = len(prompt["token_ids"]) - 1
    layers = [
        {
            "layer": layer,
            "positions": [
                {
                    "capture_index": 0,
                    "token_position": position,
                    "jacobian_lens": readout(prompt, method="jacobian", label=label),
                    "logit_lens": readout(prompt, method="logit", label=label),
                }
            ],
        }
        for layer in ANALYZE.CAPTURE_LAYERS
    ]
    experiment = {
        "id": prompt["id"],
        "prompt": prompt["text"],
        "prompt_token_ids": prompt["token_ids"],
        "metadata": prompt["metadata"],
        "positions_requested": [-1],
        "positions_resolved": [position],
        "capture_positions_resolved": [position],
        "final_validation_position": position,
        "generated_token_id": 42,
        "scored_vocabulary": {
            "token_ids": prompt["score_token_ids"],
            "tokens": [token_text[token_id] for token_id in prompt["score_token_ids"]],
        },
        "layers": layers,
        "residual_capture_manifest": {"sha256": "d" * 64},
        "final_layer_top1_matches_greedy": True,
        "final_norm_reconstruction": {"within_tolerance": True},
        "final_logits_reconstruction": {
            "within_tolerance": True,
            "top_k_prefix_token_ids_match": True,
        },
    }
    return {
        "schema_version": 3,
        "score_encoding": "unrounded-float32",
        "lens": lens(label),
        "model": {
            "repo_id": ANALYZE.MODEL_REPO,
            "revision": ANALYZE.MODEL_REVISION,
            "config_sha256": ANALYZE.MODEL_CONFIG_SHA256,
            "index_sha256": ANALYZE.MODEL_INDEX_SHA256,
        },
        "runtime": copy.deepcopy(ANALYZE.REPLAY_RUNTIME_IDENTITY),
        "assertions": {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
            "all_final_layer_top1_match_greedy": True,
            "all_final_adapter_reconstructions_within_tolerance": True,
        },
        "status": "passed",
        "scored_vocabulary": {
            "scope": "global_plus_per_experiment",
            "token_ids": [],
            "tokens": [],
            "union_token_ids": prompt["score_token_ids"],
            "union_tokens": [token_text[token_id] for token_id in prompt["score_token_ids"]],
        },
        "experiments": [experiment],
    }


def class_evidence(
    classes: list[dict[str, object]], class_scores: dict[str, float]
) -> dict[int, dict[int, dict[str, float | int]]]:
    token_scores = {
        token["token_id"]: class_scores[record["id"]]
        for record in classes
        for token in record["tokens"]
    }
    return {
        layer: {
            token_id: {
                "rank": 1,
                "score": score,
                "logprob": score,
            }
            for token_id, score in token_scores.items()
        }
        for layer in ANALYZE.FIXED_LAYER_BAND
    }


class StageActionProbeTests(unittest.TestCase):
    def rewrite_transition(
        self,
        root: Path,
        prompt: dict[str, object],
        *,
        reasoning: str,
        command: str,
    ) -> None:
        next_path = root / "captures" / "chat_0002.json"
        request = json.loads(next_path.read_text(encoding="utf-8"))
        assistant = request["messages"][-2]
        assistant["reasoning_content"] = reasoning
        assistant["reasoning"] = reasoning
        assistant["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {"command": command}
        )
        request["messages"][-1]["content"] = [
            {"type": "text", "text": shell_result(command, exit_code=0)}
        ]
        next_path.write_text(json.dumps(request), encoding="utf-8")
        prompt["metadata"]["provenance"]["next_completion_transition"][
            "materialized_in_raw_request_sha256"
        ] = AUGMENT.sha256_file(next_path)

    def materialize(self, root: Path) -> tuple[dict[str, object], dict[str, object]]:
        prompt = source_prompt(root)
        tokenizer = FakeTokenizer(PROTOCOL)
        augmented, summary = AUGMENT.build_action_bundle(
            [prompt],
            source_bundle_sha256=AUGMENT.materialized_json_sha256([prompt]),
            action_protocol=PROTOCOL,
            action_protocol_sha256=AUGMENT.sha256_bytes(PROTOCOL_BYTES),
            lifecycle_protocol=LIFECYCLE_PROTOCOL,
            lifecycle_protocol_sha256=AUGMENT.sha256_bytes(
                LIFECYCLE_PROTOCOL_BYTES
            ),
            tokenizer=tokenizer,
            artifact_root=root,
        )
        return prompt, augmented[0]

    def test_python_assert_transition_is_validate_and_failed_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original, augmented = self.materialize(Path(temporary))
        self.assertEqual(augmented["text"], original["text"])
        self.assertEqual(augmented["token_ids"], original["token_ids"])
        label = augmented["metadata"]["stage_action_probe"]["next_completion"]
        self.assertEqual(label["expected_action_class"], "validate")
        self.assertEqual(label["derivation"], "test_command")
        self.assertEqual(label["transition_outcome_class"], "failure")
        self.assertEqual(label["official_task_outcome_class"], "success")

    def test_protocol_rejects_overlapping_class_token(self) -> None:
        changed = copy.deepcopy(PROTOCOL)
        changed["outcome_classes"][0]["tokens"][0]["token_id"] = changed[
            "action_classes"
        ][0]["tokens"][0]["token_id"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            AUGMENT.validate_protocol(
                changed,
                FakeTokenizer(PROTOCOL),
                lifecycle_protocol=LIFECYCLE_PROTOCOL,
            )

    def test_validation_intent_is_fallback_after_concrete_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = source_prompt(root)
            self.rewrite_transition(
                root,
                prompt,
                reasoning="Let me verify the fix works for the specific bug scenario.",
                command="python /tmp/probe.py",
            )
            protocol = AUGMENT.validate_protocol(
                PROTOCOL,
                FakeTokenizer(PROTOCOL),
                lifecycle_protocol=LIFECYCLE_PROTOCOL,
            )
            label = AUGMENT.derive_next_completion(
                prompt, artifact_root=root, protocol=protocol
            )
            self.assertEqual(label["expected_action_class"], "validate")
            self.assertEqual(label["derivation"], "validation_intent_assistant_text")

            self.rewrite_transition(
                root,
                prompt,
                reasoning="Let me verify the fix by reading the source.",
                command="cat package/source.py",
            )
            label = AUGMENT.derive_next_completion(
                prompt, artifact_root=root, protocol=protocol
            )
            self.assertEqual(label["expected_action_class"], "inspect")
            self.assertEqual(label["derivation"], "read_or_search_command")

            self.rewrite_transition(
                root,
                prompt,
                reasoning="I will update a neighboring implementation file.",
                command="cat > package/other.py << 'PY'\nvalue = 2\nPY",
            )
            label = AUGMENT.derive_next_completion(
                prompt, artifact_root=root, protocol=protocol
            )
            self.assertEqual(label["expected_action_class"], "edit")
            self.assertEqual(label["derivation"], "mutating_source_command")

    def test_diagnosis_is_orthogonal_metadata_not_an_action_class(self) -> None:
        self.assertEqual(
            [record["id"] for record in PROTOCOL["action_classes"]],
            ["inspect", "edit", "validate", "finalize"],
        )
        action_token_ids = {
            token["token_id"]
            for record in PROTOCOL["action_classes"]
            for token in record["tokens"]
        }
        self.assertTrue(
            action_token_ids.isdisjoint({55_649, 3_418, 10_229, 2_781, 7_995, 22_903})
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = source_prompt(root)
            self.rewrite_transition(
                root,
                prompt,
                reasoning="Now I understand the issue and its cause.",
                command="cat package/source.py",
            )
            protocol = AUGMENT.validate_protocol(
                PROTOCOL,
                FakeTokenizer(PROTOCOL),
                lifecycle_protocol=LIFECYCLE_PROTOCOL,
            )
            label = AUGMENT.derive_next_completion(
                prompt, artifact_root=root, protocol=protocol
            )
        self.assertEqual(label["expected_action_class"], "inspect")
        self.assertTrue(label["diagnosis_expressed"])
        self.assertTrue(label["diagnosis_regex_hits"])

    def test_primary_class_margin_cannot_be_positive_on_wrong_prediction(self) -> None:
        classes = PROTOCOL["action_classes"]
        evidence = class_evidence(
            classes,
            {"inspect": 0.0, "edit": 0.1, "validate": -100.0, "finalize": -100.0},
        )
        metric = ANALYZE.class_metric(
            evidence,
            classes,
            "inspect",
            role="adversarial",
            generated_token_id=42,
        )
        self.assertFalse(metric["band_correct"])
        self.assertEqual(metric["band_predicted_class"], "edit")
        self.assertAlmostEqual(metric["expected_score_margin"], -0.1)
        self.assertGreater(
            metric["pooled_one_vs_rest_sensitivity"]["expected_score_margin"],
            0.0,
        )

    def test_class_metric_excludes_an_accepted_action_or_outcome_token(self) -> None:
        for family, expected, competitor in (
            ("action_classes", "inspect", "edit"),
            ("outcome_classes", "success", "failure"),
        ):
            with self.subTest(family=family):
                classes = PROTOCOL[family]
                scores = {record["id"]: -1.0 for record in classes}
                scores[competitor] = 0.0
                evidence = class_evidence(classes, scores)
                accepted = next(
                    record["tokens"][0]["token_id"]
                    for record in classes
                    if record["id"] == expected
                )
                for layer in ANALYZE.FIXED_LAYER_BAND:
                    evidence[layer][accepted]["score"] = 100.0
                    evidence[layer][accepted]["logprob"] = 100.0
                metric = ANALYZE.class_metric(
                    evidence,
                    classes,
                    expected,
                    role="adversarial",
                    generated_token_id=accepted,
                )
                self.assertTrue(
                    metric["accepted_generated_token_overlapped_class_vocabulary"]
                )
                self.assertEqual(metric["band_predicted_class"], competitor)
                self.assertLess(metric["expected_score_margin"], 0.0)

    def test_classification_summary_reports_imbalance_and_missing_classes(self) -> None:
        expected_labels = ["inspect"] * 5 + ["edit", "validate", "finalize"]
        details = []
        for index, expected in enumerate(expected_labels):
            details.append(
                {
                    "id": f"prompt-{index}",
                    "instance_id": "task-1",
                    "stage_id": f"S{index}",
                    "method_numerical_certification": {"method": True},
                    "methods": {
                        "method": {
                            "next_action": {
                                "scorable": True,
                                "expected_class": expected,
                                "band_predicted_class": "inspect",
                                "expected_score_margin": 1.0 if expected == "inspect" else -1.0,
                                "expected_logprob_margin": (
                                    1.0 if expected == "inspect" else -1.0
                                ),
                            }
                        }
                    },
                }
            )
        summary = ANALYZE.classification_track_summary(
            details,
            track="next_action",
            class_ids=ANALYZE.ACTION_IDS,
        )["certified_primary"]["methods"]["method"]
        self.assertEqual(
            summary["class_support"],
            {"inspect": 5, "edit": 1, "validate": 1, "finalize": 1},
        )
        self.assertEqual(summary["majority_class"], "inspect")
        self.assertEqual(summary["majority_baseline_accuracy"], 5 / 8)
        self.assertEqual(summary["raw_micro_accuracy_secondary"], 5 / 8)
        self.assertEqual(summary["observed_class_macro_recall"], 1 / 4)
        self.assertEqual(summary["balanced_accuracy_observed_classes"], 1 / 4)

        missing = ANALYZE.classification_track_summary(
            details[:-1],
            track="next_action",
            class_ids=ANALYZE.ACTION_IDS,
        )["certified_primary"]["methods"]["method"]
        self.assertEqual(missing["missing_class_ids"], ["finalize"])
        self.assertIsNone(missing["per_class_recall"]["finalize"])

    def test_paired_analyzer_uses_hidden_same_layer_margin_and_n1_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, prompt = self.materialize(Path(temporary))
        protocol = ANALYZE.validate_action_protocol(
            PROTOCOL,
            protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES),
        )
        prompt_contract = ANALYZE.validate_prompt_bundle([prompt], action_protocol=protocol)
        reports = [
            ANALYZE.validate_report(
                report(prompt, label), label=label, prompt_contract=prompt_contract
            )
            for label in ("public", "nf4", "native")
        ]
        pairing = ANALYZE.validate_pairing(reports)
        analysis = ANALYZE.build_analysis(
            prompt_contract=prompt_contract,
            action_protocol=protocol,
            reports=reports,
            pairing=pairing,
            bootstrap_samples=100,
        )
        self.assertEqual(analysis["status"], "descriptive_n1_development")
        detail = analysis["prompt_details"][0]
        self.assertEqual(detail["methods"]["public_jacobian"]["gold_hidden"]["score_margin"], 2.0)
        self.assertTrue(detail["methods"]["public_jacobian"]["next_action"]["band_correct"])
        self.assertFalse(detail["methods"]["logit"]["next_action"]["band_correct"])
        comparison = analysis["aggregates"]["gold_hidden"]["certified_primary"][
            "S6"
        ]["metrics"]["score_margin"]["paired_comparisons"][
            "public_jacobian_minus_logit"
        ]
        self.assertIsNone(comparison["confidence_interval_95"])
        self.assertEqual(comparison["inference"], "descriptive_only")

    def test_analyzer_reports_zero_hidden_gold_without_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_prompt(root)
            source["metadata"]["analysis_role"] = "explicit_contaminated_control"
            for record in source["metadata"]["visibility_audit"]["records"]:
                record["exposed"] = record["subject"] == "target"
            augmented = AUGMENT.build_action_bundle(
                [source],
                source_bundle_sha256=AUGMENT.materialized_json_sha256([source]),
                action_protocol=PROTOCOL,
                action_protocol_sha256=AUGMENT.sha256_bytes(PROTOCOL_BYTES),
                lifecycle_protocol=LIFECYCLE_PROTOCOL,
                lifecycle_protocol_sha256=AUGMENT.sha256_bytes(
                    LIFECYCLE_PROTOCOL_BYTES
                ),
                tokenizer=FakeTokenizer(PROTOCOL),
                artifact_root=root,
            )[0][0]
        protocol = ANALYZE.validate_action_protocol(
            PROTOCOL,
            protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES),
        )
        prompt_contract = ANALYZE.validate_prompt_bundle(
            [augmented], action_protocol=protocol
        )
        reports = [
            ANALYZE.validate_report(
                report(augmented, label), label=label, prompt_contract=prompt_contract
            )
            for label in ("public", "nf4", "native")
        ]
        analysis = ANALYZE.build_analysis(
            prompt_contract=prompt_contract,
            action_protocol=protocol,
            reports=reports,
            pairing=ANALYZE.validate_pairing(reports),
            bootstrap_samples=0,
        )
        self.assertEqual(
            analysis["gold_probe_status"], "no_hidden_gold_eligible_prompts"
        )
        self.assertEqual(
            analysis["interpretation_contract"]["hidden_gold_eligible_prompt_count"],
            0,
        )
        self.assertEqual(
            analysis["interpretation_contract"]["explicit_control_prompt_count"],
            1,
        )
        self.assertIsNone(
            analysis["prompt_details"][0]["methods"]["public_jacobian"][
                "gold_hidden"
            ]
        )
        gold = analysis["aggregates"]["gold_hidden"]["certified_primary"]
        self.assertIsNone(
            gold["ALL_AVAILABLE_STAGES"]["metrics"]["score_margin"]["methods"][
                "public_jacobian"
            ]["equal_task_mean"]
        )

    def test_gold_metric_excludes_the_accepted_generated_token(self) -> None:
        evidence = {
            layer: {
                42: {"rank": 1, "score": 9.0, "logprob": -0.1},
                101: {"rank": 10, "score": 0.0, "logprob": -9.0},
            }
            for layer in ANALYZE.FIXED_LAYER_BAND
        }
        concepts = [
            {
                "id": "accepted-only",
                "forms": [{"text": " accepted", "token_id": 42}],
                "foils": [
                    {"forms": [{"text": " foil", "token_id": 101}]}
                ],
            }
        ]
        metric = ANALYZE.gold_metric(
            evidence, concepts, generated_token_id=42
        )
        self.assertFalse(metric["scorable"])
        self.assertIn("accepted-token exclusion", metric["reason"])

    def test_failed_adapter_status_is_retained_as_numerical_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, prompt = self.materialize(Path(temporary))
        protocol = ANALYZE.validate_action_protocol(
            PROTOCOL,
            protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES),
        )
        prompt_contract = ANALYZE.validate_prompt_bundle([prompt], action_protocol=protocol)
        failed = report(prompt, "public")
        failed["status"] = "failed"
        failed["assertions"]["all_final_adapter_reconstructions_within_tolerance"] = False
        failed["experiments"][0]["final_logits_reconstruction"]["within_tolerance"] = False
        validated = ANALYZE.validate_report(
            failed, label="public", prompt_contract=prompt_contract
        )
        self.assertEqual(validated["numerical_eligibility"]["report_status"], "failed")
        self.assertFalse(
            validated["numerical_eligibility"][
                "all_adapter_reconstructions_within_tolerance"
            ]
        )

    def test_uncertified_rows_are_excluded_from_primary_but_retained_as_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, prompt = self.materialize(Path(temporary))
        protocol = ANALYZE.validate_action_protocol(
            PROTOCOL,
            protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES),
        )
        prompt_contract = ANALYZE.validate_prompt_bundle([prompt], action_protocol=protocol)
        reports = []
        for label in ("public", "nf4", "native"):
            failed = report(prompt, label)
            failed["status"] = "failed"
            failed["assertions"][
                "all_final_adapter_reconstructions_within_tolerance"
            ] = False
            failed["experiments"][0]["final_logits_reconstruction"][
                "within_tolerance"
            ] = False
            reports.append(
                ANALYZE.validate_report(
                    failed, label=label, prompt_contract=prompt_contract
                )
            )
        analysis = ANALYZE.build_analysis(
            prompt_contract=prompt_contract,
            action_protocol=protocol,
            reports=reports,
            pairing=ANALYZE.validate_pairing(reports),
            bootstrap_samples=0,
        )
        classification = analysis["aggregates"]["next_action"][
            "all_available_stages_classification"
        ]
        self.assertEqual(
            classification["certified_primary"]["methods"]["public_jacobian"][
                "eligible_row_count"
            ],
            0,
        )
        self.assertEqual(
            classification["uncertified_inclusive_sensitivity"]["methods"][
                "public_jacobian"
            ]["eligible_row_count"],
            1,
        )
        self.assertEqual(
            analysis["numerical_status"],
            "partially_uncertified_primary_excludes_failed_rows",
        )

    def test_report_rejects_every_runtime_contract_field_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, prompt = self.materialize(Path(temporary))
        protocol = ANALYZE.validate_action_protocol(
            PROTOCOL,
            protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES),
        )
        prompt_contract = ANALYZE.validate_prompt_bundle([prompt], action_protocol=protocol)
        for field, original in ANALYZE.REPLAY_RUNTIME_IDENTITY.items():
            with self.subTest(field=field):
                changed = report(prompt, "public")
                if isinstance(original, bool):
                    replacement = not original
                elif isinstance(original, str):
                    replacement = f"{original}-changed"
                else:
                    replacement = original + 1
                changed["runtime"][field] = replacement
                with self.assertRaisesRegex(ValueError, "runtime identity mismatch"):
                    ANALYZE.validate_report(
                        changed, label="public", prompt_contract=prompt_contract
                    )

    def test_pairing_rejects_changed_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, prompt = self.materialize(Path(temporary))
        protocol = ANALYZE.validate_action_protocol(
            PROTOCOL,
            protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES),
        )
        prompt_contract = ANALYZE.validate_prompt_bundle([prompt], action_protocol=protocol)
        public = ANALYZE.validate_report(
            report(prompt, "public"), label="public", prompt_contract=prompt_contract
        )
        changed = report(prompt, "native")
        changed["experiments"][0]["residual_capture_manifest"]["sha256"] = "e" * 64
        native = ANALYZE.validate_report(
            changed, label="native", prompt_contract=prompt_contract
        )
        with self.assertRaisesRegex(ValueError, "residual/logit"):
            ANALYZE.validate_pairing([public, native])

    def test_pairing_rejects_changed_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, prompt = self.materialize(Path(temporary))
        protocol = ANALYZE.validate_action_protocol(
            PROTOCOL,
            protocol_sha256=ANALYZE.sha256_bytes(PROTOCOL_BYTES),
        )
        prompt_contract = ANALYZE.validate_prompt_bundle([prompt], action_protocol=protocol)
        public = ANALYZE.validate_report(
            report(prompt, "public"), label="public", prompt_contract=prompt_contract
        )
        native = ANALYZE.validate_report(
            report(prompt, "native"), label="native", prompt_contract=prompt_contract
        )
        native["runtime_identity"] = dict(native["runtime_identity"])
        native["runtime_identity"]["max_model_len"] = 65_536
        with self.assertRaisesRegex(ValueError, "paired runtime identity mismatch"):
            ANALYZE.validate_pairing([public, native])


if __name__ == "__main__":
    unittest.main()
