#!/usr/bin/env python3
"""Focused tests for exact-prefix multi-stage SWE probe materialization."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_multistage_probes",
    ROOT / "scripts" / "materialize_swe_multistage_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PROTOCOL_PATH = ROOT / "configs/swe_multistage_protocol.json"
PROTOCOL_BYTES = PROTOCOL_PATH.read_bytes()
PROTOCOL = json.loads(PROTOCOL_BYTES)
PROTOCOL_SHA256 = MODULE.C1.sha256_bytes(PROTOCOL_BYTES)
REAL_MANIFEST_PATH = ROOT / "configs/swe_multistage_trajectory_manifest.json"
CONCEPT_REGISTRY_SHA256 = "6" * 64
IMAGE_REGISTRY_SHA256 = "7" * 64
SYNTHETIC_IMAGE_ID = f"sha256:{'9' * 64}"


def evidence_contract() -> dict[str, object]:
    values: dict[str, object] = {
        "manifest_path": "validation/synthetic/evidence.sha256",
        "manifest_sha256": "8" * 64,
    }
    artifact_values = {
        "generated_patch": ("generation/patch.diff", "a" * 64),
        "official_patch": ("official/patch.diff", "a" * 64),
        "official_eval_script": ("official/eval.sh", "b" * 64),
        "official_test_output": ("official/test_output.txt", "c" * 64),
        "official_run_log": ("official/run_instance.log", "d" * 64),
        "official_score_log": ("official/score.log", "e" * 64),
        "official_report": ("official-report.json", "1" * 64),
        "official_instance_report": ("instance-report.json", "3" * 64),
    }
    for name, (suffix, digest) in artifact_values.items():
        values[f"{name}_path"] = f"validation/synthetic/{suffix}"
        values[f"{name}_sha256"] = digest
    return values


def image_binding() -> dict[str, object]:
    return {
        "architecture": "x86_64",
        "historical_runner_reference": "synthetic:latest",
        "historical_generation_digest_proven": False,
        "historical_generation_limitation": MODULE.HISTORICAL_IMAGE_LIMITATION,
    }


class FakeTokenizer:
    forms = {" secret": 9001, " decoy": 9002}
    reverse = {token_id: text for text, token_id in forms.items()}
    assistant_boundary = "<|im_start|>assistant\n<think>\n"

    def apply_chat_template(self, messages: object, **_: object) -> str:
        pieces = ["PROMPT\n"]
        for message in messages:
            role = message["role"]
            if role == "assistant":
                pieces.append(self.assistant_boundary)
                pieces.append(json.dumps(message, sort_keys=True, ensure_ascii=False))
                pieces.append("<|im_end|>\n")
            else:
                pieces.append(
                    f"<{role}>"
                    + json.dumps(message, sort_keys=True, ensure_ascii=False)
                    + "</message>\n"
                )
        pieces.append(self.assistant_boundary)
        return "".join(pieces)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        if text in self.forms:
            return [self.forms[text]]
        token_ids: list[int] = []
        cursor = 0
        marker = "TOKEN_ONLY"
        while cursor < len(text):
            if text.startswith(marker, cursor):
                token_ids.append(9001)
                cursor += len(marker)
            else:
                token_ids.append(100 + ord(text[cursor]))
                cursor += 1
        return token_ids

    def decode(self, token_ids: list[int], **_: object) -> str:
        return self.reverse.get(token_ids[0], "not-a-form")


def shell_result(command: str, output: str, *, exit_code: int = 0) -> str:
    return (
        f"Command: {command}\nDirectory: (root)\nOutput: {output}\n"
        f"Error: (none)\nExit Code: {exit_code}\nSignal: 0\nProcess Group PGID: 123"
    )


def tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {"name": "run_shell_command", "description": "Run shell"},
        }
    ]


def assistant_tool_messages(
    completion_index: int,
    *,
    reasoning: str,
    command: str,
    output: str,
    exit_code: int = 0,
) -> list[dict[str, object]]:
    call_id = f"call-{completion_index}"
    return [
        {
            "role": "assistant",
            "content": "\n\n",
            "reasoning_content": reasoning,
            "reasoning": reasoning,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "run_shell_command",
                        "arguments": json.dumps(
                            {"command": command, "description": "Synthetic event"},
                            separators=(",", ":"),
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": [
                {
                    "type": "text",
                    "text": shell_result(command, output, exit_code=exit_code),
                }
            ],
        },
    ]


def trajectory_requests(
    *,
    diagnosis_text: str = "Now I understand the issue and its cause.",
    source_output: str = "def worker():\n    return 1",
    validation_command: str = "python -m pytest tests/test_module.py",
    validation_output: str = "1 passed",
    edit_command: str = (
        "cat > src/module.py <<'EOF'\n"
        "def worker():\n    secret = 2\n    return secret\nEOF"
    ),
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Use the repository tool."},
        {"role": "user", "content": "Task owner__repo-1: repair the behavior."},
    ]
    requests: list[dict[str, object]] = []

    def append_request(index: int) -> None:
        requests.append(
            {
                "model": "qwen3.6-27b-nvfp4",
                "messages": copy.deepcopy(messages),
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": True},
                "seed": 880001233 + index,
                "tools": tools(),
            }
        )

    append_request(1)
    completions = [
        (
            "I will orient in the repository.",
            "ls src",
            "module.py",
            0,
        ),
        (
            "I will search for callers.",
            "rg worker tests",
            "tests/test_module.py:def test_worker():",
            0,
        ),
        (
            "I will read the relevant implementation.",
            "cat src/module.py",
            source_output,
            0,
        ),
        (
            diagnosis_text,
            "rg expected tests",
            "tests/test_module.py:assert worker() == 2",
            0,
        ),
        (
            "I will inspect the baseline before editing.",
            "sed -n '1,80p' tests/test_module.py",
            "def test_worker():\n    assert worker() == 2",
            0,
        ),
        (
            "I will implement the source edit now.",
            edit_command,
            "(empty)",
            0,
        ),
        (
            "I will validate the edit.",
            validation_command,
            validation_output,
            0,
        ),
        (
            "I will inspect the final diff.",
            "git diff -- src/module.py",
            "+    secret = 2",
            0,
        ),
    ]
    for completion_index, (reasoning, command, output, exit_code) in enumerate(
        completions, start=1
    ):
        messages.extend(
            assistant_tool_messages(
                completion_index,
                reasoning=reasoning,
                command=command,
                output=output,
                exit_code=exit_code,
            )
        )
        append_request(completion_index + 1)
    return requests


def trajectory_manifest(*, expected_verdict: str = "resolved") -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "swe_verified_multistage_trajectory_manifest",
        "lens_outputs_used_for_selection": False,
        "concept_registry": {
            "path": "configs/swe_multitask_initial_protocol.json",
            "sha256": CONCEPT_REGISTRY_SHA256,
        },
        "image_registry": {
            "path": "configs/swe_image_digests.json",
            "sha256": IMAGE_REGISTRY_SHA256,
        },
        "lifecycle_protocol_sha256": PROTOCOL_SHA256,
        "trajectories": [
            {
                "selection_index": 0,
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": "a" * 40,
                "dataset_path": "validation/synthetic/dataset.json",
                "dataset_sha256": "5" * 64,
                "evidence_binding": evidence_contract(),
                "problem_statement_sha256": "b" * 64,
                "patch_sha256": "c" * 64,
                "test_patch_sha256": "d" * 64,
                "proxy_dir": "validation/synthetic/proxy_dumps",
                "expected_request_count": 9,
                "request_manifest_sha256": "e" * 64,
                "usage_path": "validation/synthetic/proxy_dumps/usage.jsonl",
                "usage_sha256": "f" * 64,
                "official_report_path": "validation/synthetic/official-report.json",
                "official_report_sha256": "1" * 64,
                "official_instance_report_path": "validation/synthetic/instance-report.json",
                "official_instance_report_sha256": "3" * 64,
                "expected_official_verdict": expected_verdict,
                "image_binding": image_binding(),
                "max_prompt_tokens": 1_000_000,
                "terminal_binding": {
                    "runner_metadata_path": "validation/synthetic/runner_metadata.json",
                    "runner_metadata_sha256": "4" * 64,
                    "expected_final_request_index": 9,
                    "expected_finish_reason": "stop",
                    "expected_num_turns": 9,
                    "expected_dataset_path_suffix": "runs/synthetic/dataset.json",
                    "expected_cli_exit_code": 0,
                    "expected_parsed": True,
                    "expected_subtype": "success",
                },
                "oracle": {
                    "source_locations": [
                        {
                            "path": "src/module.py",
                            "symbols": ["worker"],
                            "body_markers": [],
                        }
                    ],
                    "concepts": [
                        {
                            "id": "task-00-concept-00",
                            "family": "replacement",
                            "path": "src/module.py",
                            "target": "secret",
                            "forms": [
                                {
                                    "kind": "leading_space",
                                    "text": " secret",
                                    "token_id": 9001,
                                }
                            ],
                            "foils": [
                                {
                                    "task_instance_id": "foil__repo-2",
                                    "concept_id": "task-01-concept-00",
                                    "family": "replacement",
                                    "target": "decoy",
                                    "forms": [
                                        {
                                            "kind": "leading_space",
                                            "text": " decoy",
                                            "token_id": 9002,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    }


def official_report(*, verdict: str = "resolved") -> dict[str, object]:
    resolved = ["owner__repo-1"] if verdict == "resolved" else []
    unresolved = ["owner__repo-1"] if verdict == "unresolved" else []
    incomplete = ["owner__repo-1"] if verdict == "incomplete" else []
    errors = ["owner__repo-1"] if verdict == "error" else []
    completed = ["owner__repo-1"] if verdict in {"resolved", "unresolved"} else []
    return {
        "schema_version": 2,
        "total_instances": 1,
        "submitted_instances": 1,
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": 0,
        "error_instances": len(errors),
        "submitted_ids": ["owner__repo-1"],
        "completed_ids": completed,
        "incomplete_ids": incomplete,
        "empty_patch_ids": [],
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
        "error_ids": errors,
    }


def official_instance_report(*, verdict: str = "resolved") -> dict[str, object]:
    resolved = verdict == "resolved"
    return {
        "owner__repo-1": {
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "resolved": resolved,
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": ["test_fixed"] if resolved else [],
                    "failure": [] if resolved else ["test_fixed"],
                },
                "PASS_TO_PASS": {"success": ["test_preserved"], "failure": []},
                "FAIL_TO_FAIL": {"success": [], "failure": []},
                "PASS_TO_FAIL": {"success": [], "failure": []},
            },
        }
    }


def concept_registry() -> dict[str, object]:
    trajectory = trajectory_manifest()["trajectories"][0]
    return {
        "schema_version": 1,
        "kind": "swe_verified_initial_probe_protocol",
        "status": "exploratory_development_pilot",
        "lens_outputs_used_for_selection": False,
        "tasks": [
            {
                "selection_index": 0,
                "instance_id": trajectory["instance_id"],
                "repo": trajectory["repo"],
                "base_commit": trajectory["base_commit"],
                "patch_sha256": trajectory["patch_sha256"],
                "test_patch_sha256": trajectory["test_patch_sha256"],
                "concepts": copy.deepcopy(trajectory["oracle"]["concepts"]),
                "score_token_ids": [9001, 9002],
            }
        ],
    }


def image_registry() -> dict[str, object]:
    return {
        "schema_version": 1,
        "images": {
            "owner__repo-1": {
                "x86_64": {
                    "reference": f"synthetic@{SYNTHETIC_IMAGE_ID}",
                    "image_id": SYNTHETIC_IMAGE_ID,
                }
            }
        },
    }


def build(
    *,
    diagnosis_text: str = "Now I understand the issue and its cause.",
    source_output: str = "def worker():\n    return 1",
    validation_command: str = "python -m pytest tests/test_module.py",
    validation_output: str = "1 passed",
    edit_command: str = (
        "cat > src/module.py <<'EOF'\n"
        "def worker():\n    secret = 2\n    return secret\nEOF"
    ),
    report_verdict: str = "resolved",
    mutate_requests: object = None,
    mutate_concept_registry: object = None,
    mutate_usage_records: object = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    tokenizer = FakeTokenizer()
    requests = trajectory_requests(
        diagnosis_text=diagnosis_text,
        source_output=source_output,
        validation_command=validation_command,
        validation_output=validation_output,
        edit_command=edit_command,
    )
    if mutate_requests is not None:
        mutate_requests(requests)
    frozen_registry = concept_registry()
    if mutate_concept_registry is not None:
        mutate_concept_registry(frozen_registry)
    usage_records: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for index, request in enumerate(requests, start=1):
        rendered, token_ids, _, _ = MODULE.C1.render_request(
            tokenizer, request=request, template="test-template"
        )
        assert rendered
        usage_records.append(
            {
                "idx": index,
                "usage": {
                    "prompt_tokens": len(token_ids),
                    "completion_tokens": 10,
                    "total_tokens": len(token_ids) + 10,
                },
                "finish_reason": "stop" if index == len(requests) else "tool_calls",
            }
        )
        sources.append(
            {
                "index": index,
                "path": f"validation/synthetic/proxy_dumps/chat_{index:04d}.json",
                "bytes": 100,
                "sha256": f"{index:064x}",
            }
        )
    if mutate_usage_records is not None:
        mutate_usage_records(usage_records)
    artifact = {
        "dataset_binding": {
            "dataset_sha256": "5" * 64,
            "instance_id": "owner__repo-1",
            "version": "1.0",
            "gold_source_paths": ["src/module.py"],
        },
        "requests": requests,
        "request_sources": sources,
        "usage_records": usage_records,
        "official_report": official_report(verdict=report_verdict),
        "official_instance_report": official_instance_report(verdict=report_verdict),
        "evidence_binding": {
            **evidence_contract(),
            "manifest_entry_count": 12,
            "generated_and_official_patch_bytes_equal": True,
            "official_score_log_first_line": (
                "verified pinned task image: synthetic:latest "
                f"({SYNTHETIC_IMAGE_ID})"
            ),
        },
        "runner_metadata": {
            "instance_id": "owner__repo-1",
            "dataset_name": "/workspace/runs/synthetic/dataset.json",
            "repo": "owner/repo",
            "base_commit": "a" * 40,
            "agent": "qwen_code",
            "eval_mode": "skip",
            "runtime": "container",
            "image": "synthetic:latest",
            "qwen": {
                "num_turns": 9,
                "exit_code": 0,
                "parsed": True,
                "subtype": "success",
            },
        },
    }
    return MODULE.build_multistage_bundle(
        PROTOCOL,
        lifecycle_protocol_sha256=PROTOCOL_SHA256,
        manifest=trajectory_manifest(),
        manifest_sha256="2" * 64,
        concept_registry=frozen_registry,
        concept_registry_sha256=CONCEPT_REGISTRY_SHA256,
        image_registry=image_registry(),
        image_registry_sha256=IMAGE_REGISTRY_SHA256,
        artifacts=[artifact],
        tokenizer=tokenizer,
        template="test-template",
    )


class MaterializeMultistageProbesTest(unittest.TestCase):
    def test_selects_exact_request_boundaries_without_off_by_one(self) -> None:
        prompts, summary = build()
        by_stage = {prompt["metadata"]["stage"]["id"]: prompt for prompt in prompts}
        self.assertEqual(set(by_stage), {f"S{index}" for index in range(8)})
        self.assertEqual(
            {
                stage: prompt["metadata"]["provenance"]["raw_request_index"]
                for stage, prompt in by_stage.items()
            },
            {"S0": 1, "S1": 2, "S2": 4, "S3": 5, "S4": 6, "S5": 7, "S6": 8, "S7": 9},
        )
        self.assertEqual(
            by_stage["S4"]["metadata"]["provenance"]["request_boundary"],
            {
                "semantics": "chat_N_is_the_exact_prefix_before_completion_N",
                "prompt_request_index": 6,
                "prompt_precedes_completion_index": 6,
                "latest_included_completion_index": 5,
                "selection_evidence_request_index": 7,
                "selection_event_completion_index": 6,
            },
        )
        self.assertNotIn("secret", by_stage["S4"]["text"])
        self.assertIn("secret", by_stage["S5"]["text"])
        self.assertEqual(summary["available_stage_count"], 8)
        self.assertEqual(summary["missing_stage_count"], 0)

    def test_preserves_exact_prefix_and_binds_request_usage_and_verdict(self) -> None:
        prompts, _ = build()
        for previous, current in zip(prompts, prompts[1:], strict=False):
            self.assertTrue(current["text"].startswith(previous["text"]))
            self.assertEqual(
                current["token_ids"][: len(previous["token_ids"])],
                previous["token_ids"],
            )
        provenance = prompts[-1]["metadata"]["provenance"]
        self.assertEqual(provenance["raw_request_sha256"], f"{9:064x}")
        self.assertEqual(provenance["prompt_token_count"], len(prompts[-1]["token_ids"]))
        self.assertEqual(provenance["usage"]["idx"], 9)
        self.assertEqual(provenance["official_verdict"]["verdict"], "resolved")
        self.assertEqual(provenance["official_verdict"]["report_sha256"], "1" * 64)
        self.assertEqual(provenance["official_instance_verdict"]["verdict"], "resolved")
        self.assertEqual(
            provenance["next_completion_transition"]["event_labels"], ["finalize"]
        )
        self.assertTrue(provenance["next_completion_transition"]["terminal_response"])
        for prompt in prompts[1:]:
            prefix = prompt["metadata"]["provenance"][
                "exact_prefix_from_prior_available_stage"
            ]
            self.assertEqual(
                {
                    prefix["raw_message_prefix"],
                    prefix["normalized_message_prefix"],
                    prefix["rendered_text_prefix"],
                    prefix["token_id_prefix"],
                },
                {True},
            )

    def test_leakage_contract_distinguishes_hidden_and_explicit_controls(self) -> None:
        prompts, summary = build(diagnosis_text="Now I understand the issue: secret is wrong.")
        by_stage = {prompt["metadata"]["stage"]["id"]: prompt for prompt in prompts}
        self.assertEqual(by_stage["S2"]["metadata"]["analysis_role"], "oracle_hidden")
        self.assertEqual(
            by_stage["S3"]["metadata"]["analysis_role"],
            "explicit_contaminated_control",
        )
        target = next(
            record
            for record in by_stage["S3"]["metadata"]["visibility_audit"]["records"]
            if record["subject"] == "target"
        )
        self.assertTrue(target["surface_exposed"])
        self.assertIn("assistant_text", target["visible_channels"])
        self.assertEqual(summary["explicit_control_prompt_count"], 5)

    def test_visibility_classification_catches_surface_token_and_casefold_segments(self) -> None:
        def expose_at_task_start(requests: list[dict[str, object]]) -> None:
            requests[0]["messages"][1]["content"] += " secret"
            for request in requests[1:]:
                request["messages"][1]["content"] += " secret"

        prompts, _ = build(mutate_requests=expose_at_task_start)
        self.assertEqual(prompts[0]["metadata"]["analysis_role"], "explicit_contaminated_control")

        prompts, _ = build(source_output="def worker():\n    TOKEN_ONLY\n    return 1")
        stage_two = next(
            prompt for prompt in prompts if prompt["metadata"]["stage"]["id"] == "S2"
        )
        self.assertEqual(stage_two["metadata"]["analysis_role"], "explicit_contaminated_control")

        def expose_casefold_segment(requests: list[dict[str, object]]) -> None:
            for request in requests:
                request["messages"][1]["content"] += " SimpleSecretObject"

        prompts, _ = build(mutate_requests=expose_casefold_segment)
        target = next(
            record
            for record in prompts[0]["metadata"]["visibility_audit"]["records"]
            if record["subject"] == "target"
        )
        self.assertEqual(prompts[0]["metadata"]["analysis_role"], "explicit_contaminated_control")
        self.assertTrue(target["full_rendered_prompt"]["semantic_exposed"])
        self.assertTrue(target["full_rendered_prompt"]["casefold_identifier_hits"])

    def test_oracle_hidden_contract_fails_closed_on_semantic_exposure(self) -> None:
        tokenizer = FakeTokenizer()
        request = {
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Inspect SimpleSecretObject."},
            ]
        }
        rendered = tokenizer.apply_chat_template(request["messages"])
        concept = trajectory_manifest()["trajectories"][0]["oracle"]["concepts"][0]
        with self.assertRaisesRegex(ValueError, "oracle-hidden stage exposes"):
            MODULE.visibility_audit(
                concepts=[concept],
                instance_id="owner__repo-1",
                request=request,
                rendered=rendered,
                tokenizer=tokenizer,
                visibility_contract="oracle_hidden_required",
            )

    def test_reports_unavailable_stages_as_missing_without_imputation(self) -> None:
        prompts, summary = build(
            diagnosis_text="I will continue inspecting.",
            validation_command="echo no-test-run",
        )
        by_stage = {prompt["metadata"]["stage"]["id"] for prompt in prompts}
        self.assertNotIn("S3", by_stage)
        self.assertNotIn("S6", by_stage)
        self.assertNotIn("S7", by_stage)
        audits = summary["task_audits"][0]["stage_audits"]
        missing = {
            audit["stage"]["id"]: audit["missing_reason"]
            for audit in audits
            if audit["status"] == "missing"
        }
        self.assertEqual(
            missing,
            {
                "S3": "no_diagnosis_after_source_read_before_edit",
                "S6": "no_post_edit_successful_validation",
                "S7": "prerequisite_S6_missing",
            },
        )

    def test_validation_requires_positive_output_and_rejects_masked_failure(self) -> None:
        rejected_outputs = (
            "Traceback (most recent call last):\nModuleNotFoundError: No module named pytest",
            "0 passed in 0.01s",
            "0 tests passed",
            "1 failed, 1 passed in 0.10s",
            "Ran 0 tests in 0.000s\n\nOK",
            "no tests ran in 0.01s",
            "all tests passed",
        )
        for validation_output in rejected_outputs:
            with self.subTest(validation_output=validation_output):
                prompts, summary = build(validation_output=validation_output)
                self.assertNotIn(
                    "S6", {prompt["metadata"]["stage"]["id"] for prompt in prompts}
                )
                missing = {
                    audit["stage"]["id"]: audit["missing_reason"]
                    for audit in summary["task_audits"][0]["stage_audits"]
                    if audit["status"] == "missing"
                }
                self.assertEqual(missing["S6"], "no_post_edit_successful_validation")
                self.assertEqual(missing["S7"], "prerequisite_S6_missing")

        prompts, _ = build(validation_output="Ran 2 tests in 0.001s\n\nOK")
        self.assertIn("S6", {prompt["metadata"]["stage"]["id"] for prompt in prompts})

    def test_rejects_concept_registry_drift(self) -> None:
        def mutate(registry: dict[str, object]) -> None:
            registry["tasks"][0]["concepts"][0]["target"] = "changed"

        with self.assertRaisesRegex(ValueError, "frozen concept/foil registry"):
            build(mutate_concept_registry=mutate)

    def test_source_edit_before_source_read_cannot_regress_lifecycle(self) -> None:
        blank = {
            "diagnosis_regex_hits": [],
            "successful_repository_orientation": False,
            "oracle_source_body_reads": [],
            "oracle_source_edits": [],
            "successful_validations": [],
        }
        transitions = [
            None,
            {**blank, "successful_repository_orientation": True},
            {**blank, "oracle_source_edits": [{"path": "src/module.py"}]},
            {
                **blank,
                "oracle_source_body_reads": [{"path": "src/module.py"}],
            },
        ]
        selections = MODULE.select_stages(
            stages=MODULE.validate_lifecycle_protocol(PROTOCOL)["stages"],
            transitions=transitions,
            usage_records=[
                {"finish_reason": "tool_calls"},
                {"finish_reason": "tool_calls"},
                {"finish_reason": "tool_calls"},
                {"finish_reason": "stop"},
            ],
        )
        by_stage = {selection["stage"]["id"]: selection for selection in selections}
        self.assertEqual(by_stage["S4"]["missing_reason"], "source_edit_does_not_follow_S2")
        self.assertEqual(by_stage["S5"]["missing_reason"], "prerequisite_S4_missing")

    def test_read_redirected_to_sink_is_not_a_source_edit(self) -> None:
        prompts, summary = build(edit_command="cat src/module.py > /dev/null")
        stage_ids = {prompt["metadata"]["stage"]["id"] for prompt in prompts}
        self.assertNotIn("S4", stage_ids)
        self.assertNotIn("S5", stage_ids)
        missing = {
            audit["stage"]["id"]: audit["missing_reason"]
            for audit in summary["task_audits"][0]["stage_audits"]
            if audit["status"] == "missing"
        }
        self.assertEqual(missing["S4"], "no_successful_oracle_source_edit")
        self.assertEqual(missing["S5"], "prerequisite_S4_missing")
        self.assertFalse(
            MODULE.command_mutates_repository_path(
                "cat src/module.py > /dev/null", "src/module.py"
            )
        )
        self.assertTrue(
            MODULE.command_mutates_repository_path(
                "open('/testbed/src/module.py', 'w').write('fixed')", "src/module.py"
            )
        )

    def test_intermediate_length_finish_cannot_have_a_following_request(self) -> None:
        def truncate(usage_records: list[dict[str, object]]) -> None:
            usage_records[3]["finish_reason"] = "length"

        with self.assertRaisesRegex(ValueError, "preterminal.*tool_calls"):
            build(mutate_usage_records=truncate)

    def test_missing_required_evidence_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "evidence.sha256"
            ledger.write_text(f"{'0' * 64}  missing.txt\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                MODULE.load_hash_ledger(ledger, artifact_root=root)

    def test_rejects_prefix_drift_and_official_verdict_mismatch(self) -> None:
        def drift(requests: list[dict[str, object]]) -> None:
            requests[6]["messages"][0]["content"] = "drifted system message"

        with self.assertRaisesRegex(ValueError, "exact raw prefix"):
            build(mutate_requests=drift)
        with self.assertRaisesRegex(ValueError, "official verdict mismatch"):
            build(report_verdict="unresolved")

    def test_real_tracked_django_stage_indices_and_hash_pins(self) -> None:
        manifest_bytes = REAL_MANIFEST_PATH.read_bytes()
        manifest = MODULE.C1.require_mapping(json.loads(manifest_bytes), "real manifest")
        trajectories = MODULE.validate_manifest_header(
            manifest, lifecycle_protocol_sha256=PROTOCOL_SHA256
        )
        artifacts = MODULE.load_trajectory_artifacts(ROOT, trajectories)
        compiled = MODULE.validate_lifecycle_protocol(PROTOCOL)
        trajectory = trajectories[0]
        artifact = artifacts[0]
        self.assertTrue(
            artifact["evidence_binding"]["generated_and_official_patch_bytes_equal"]
        )
        self.assertEqual(
            artifact["evidence_binding"]["official_score_log_first_line"],
            "verified pinned task image: "
            "swebench/sweb.eval.x86_64.django_1776_django-13297:latest "
            "(sha256:0291be700f4db5aa369af9f7106943c9af1e46dcebf313d9999e88d851334bed)",
        )
        concept_registry, concept_sha256 = MODULE.load_bound_json(
            ROOT, manifest["concept_registry"], label="real concept registry"
        )
        image_registry, image_sha256 = MODULE.load_bound_json(
            ROOT, manifest["image_registry"], label="real image registry"
        )
        concept_binding = MODULE.bind_concept_registry(
            concept_registry,
            registry_sha256=concept_sha256,
            trajectory=trajectory,
        )
        image_evidence = MODULE.bind_image_evidence(
            image_registry,
            registry_sha256=image_sha256,
            trajectory=trajectory,
            runner_metadata=artifact["runner_metadata"],
            evidence_binding=artifact["evidence_binding"],
        )
        self.assertTrue(concept_binding["exact_concept_foil_registry_match"])
        self.assertFalse(image_evidence["generation_digest_proven"])
        self.assertTrue(image_evidence["official_score_digest_proven"])
        locations = MODULE.validate_source_locations(trajectory["oracle"])
        transitions: list[dict[str, object] | None] = [None]
        previous: list[dict[str, object]] = []
        for request_index, request in enumerate(artifact["requests"], start=1):
            messages = request["messages"]
            if request_index > 1:
                transitions.append(
                    MODULE.parse_transition(
                        previous,
                        messages,
                        source_locations=locations,
                        event_contract=compiled,
                    )
                )
            previous = messages
        selections = MODULE.select_stages(
            stages=compiled["stages"],
            transitions=transitions,
            usage_records=artifact["usage_records"],
        )
        self.assertEqual(
            {selection["stage"]["id"]: selection["request_index"] for selection in selections},
            {"S0": 1, "S1": 2, "S2": 5, "S3": 6, "S4": 13, "S5": 14, "S6": 16, "S7": 25},
        )
        self.assertEqual(
            next(selection for selection in selections if selection["stage"]["id"] == "S4")[
                "evidence_request_index"
            ],
            14,
        )


if __name__ == "__main__":
    unittest.main()
