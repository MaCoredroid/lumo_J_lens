#!/usr/bin/env python3
"""Focused tests for label-independent behavioral probe materialization."""

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
    "materialize_swe_behavioral_probes",
    ROOT / "scripts/materialize_swe_behavioral_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ACTION_PROTOCOL_PATH = ROOT / "configs/swe_stage_action_probes.json"
ACTION_PROTOCOL_BYTES = ACTION_PROTOCOL_PATH.read_bytes()
ACTION_PROTOCOL = json.loads(ACTION_PROTOCOL_BYTES)
TEMPLATE = "synthetic-template"


class FakeTokenizer:
    assistant_boundary = "<|im_start|>assistant\n<think>\n"

    def __init__(self) -> None:
        forms = {
            token["text"]: token["token_id"]
            for class_group in (
                ACTION_PROTOCOL["action_classes"],
                ACTION_PROTOCOL["outcome_classes"],
            )
            for class_record in class_group
            for token in class_record["tokens"]
        }
        forms.update(
            {
                "FutureThing": 81001,
                " FutureThing": 81002,
                "OldThing": 81003,
                " OldThing": 81004,
                "ContextThing": 81005,
                " ContextThing": 81006,
                "GoldOnly": 81007,
                " GoldOnly": 81008,
                "Kelvin": 81009,
                " Kelvin": 81010,
            }
        )
        self.forms = forms
        self.reverse = {token_id: text for text, token_id in forms.items()}
        self.ordered_forms = sorted(forms, key=len, reverse=True)

    def __len__(self) -> int:
        return 248_077

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
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            form = next(
                (candidate for candidate in self.ordered_forms if text.startswith(candidate, cursor)),
                None,
            )
            if form is not None:
                result.append(self.forms[form])
                cursor += len(form)
            else:
                result.append(100_000 + ord(text[cursor]))
                cursor += 1
        return result

    def decode(self, token_ids: list[int], **_: object) -> str:
        token_id = token_ids[0]
        if token_id in self.reverse:
            return self.reverse[token_id]
        return chr(token_id - 100_000)


def campaign(instance_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": MODULE.CAMPAIGN_KIND,
        "dataset": {"repo_id": "example/swe", "revision": "1" * 40},
        "selection": {
            "lens_outputs_used": False,
            "rule": "synthetic frozen order",
            "purpose": "tests",
        },
        "generation": {
            "model_repo_id": "nvidia/Qwen3.6-27B-NVFP4",
            "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "served_model": "qwen3.6-27b-nvfp4",
            "qwen_code_version": "0.19.4",
            "max_model_len": 65536,
            "max_session_turns": 50,
            "agent_wall_seconds": 900,
            "retain_empty_predictions": True,
        },
        "instance_ids": instance_ids,
    }


def tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]


def shell_result(command: str, output: str, *, exit_code: int = 0) -> str:
    return (
        f"Command: {command}\nDirectory: (root)\nOutput: {output}\n"
        f"Error: (none)\nExit Code: {exit_code}\nSignal: 0\nProcess Group PGID: 123"
    )


def append_tool_completion(
    messages: list[dict[str, object]],
    *,
    completion_index: int,
    reasoning: str,
    command: str,
    output: str,
    exit_code: int = 0,
) -> None:
    call_id = f"call-{completion_index}"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "reasoning": reasoning,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "run_shell_command",
                            "arguments": json.dumps(
                                {"command": command, "description": "Synthetic completion"},
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": [{"type": "text", "text": shell_result(command, output, exit_code=exit_code)}],
            },
        ]
    )


def request(messages: list[dict[str, object]], global_index: int) -> dict[str, object]:
    return {
        "model": "qwen3.6-27b-nvfp4",
        "messages": copy.deepcopy(messages),
        "max_tokens": 8192,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "seed": MODULE.SEED_BASE + global_index,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": True},
        "tools": tools(),
    }


def task_requests(instance_id: str, commands: list[tuple[str, str, str, int]], start: int) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Use the repository shell."},
        {"role": "user", "content": f"Task {instance_id}: repair the behavior."},
    ]
    requests: list[dict[str, object]] = []
    for local_index, (reasoning, command, output, exit_code) in enumerate(commands, 1):
        requests.append(request(messages, start + local_index - 1))
        if local_index < len(commands):
            append_tool_completion(
                messages,
                completion_index=local_index,
                reasoning=reasoning,
                command=command,
                output=output,
                exit_code=exit_code,
            )
    return requests


def primary_commands() -> list[tuple[str, str, str, int]]:
    return [
        ("Inspect the source.", "rg OldThing src", "src/module.py:OldThing", 0),
        (
            "The fix needs FutureThing instead.",
            "python -c \"from pathlib import Path; Path('src/module.py').write_text('FutureThing')\"",
            "(empty)",
            0,
        ),
        ("Validate the fix.", "python -m pytest tests/test_module.py", "2 passed", 0),
        ("Check the broader suite.", "python -m pytest tests/test_other.py", "1 failed, 2 passed", 0),
        (
            "Adjust FutureThing and test it.",
            "cat > src/module.py <<'EOF'\nFutureThing\nEOF\npython -m pytest tests/test_module.py",
            "2 passed",
            0,
        ),
        ("Inspect the diff.", "git diff -- src/module.py", "+FutureThing", 0),
        ("Run an unclassified command.", "true", "(empty)", 0),
        ("Search for references.", "rg ContextThing src", "src/module.py:ContextThing", 0),
        ("Inspect status.", "git status --short", " M src/module.py", 0),
        ("Terminal response follows.", "true", "(empty)", 0),
    ]


def write_run(
    root: Path,
    tokenizer: FakeTokenizer,
    *,
    instance_ids: list[str] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    instance_ids = instance_ids or ["owner__alpha-1", "owner__beta-2"]
    campaign_value = campaign(instance_ids)
    first = task_requests(instance_ids[0], primary_commands(), 1)
    second_commands = [
        ("Inspect and fail.", "rg missing src", "not found", 1),
        ("Truncated terminal.", "true", "(empty)", 0),
    ]
    second = task_requests(instance_ids[1], second_commands, len(first) + 1)
    all_requests = first + second
    proxy = root / "proxy_dumps"
    proxy.mkdir(parents=True)
    usage_rows: list[dict[str, object]] = []
    for global_index, request_value in enumerate(all_requests, 1):
        (proxy / f"chat_{global_index:04d}.json").write_text(
            json.dumps(request_value), encoding="utf-8"
        )
        rendered, token_ids, _, _ = MODULE.C1.render_request(
            tokenizer, request=request_value, template=TEMPLATE
        )
        assert rendered
        task_local_last = global_index in {len(first), len(all_requests)}
        if global_index == len(first):
            finish_reason = "stop"
        elif global_index == len(all_requests):
            finish_reason = "length"
        else:
            finish_reason = "tool_calls"
        usage_rows.append(
            {
                "idx": global_index,
                "ts": 1_000.0 + global_index,
                "usage": {
                    "prompt_tokens": len(token_ids),
                    "completion_tokens": 10,
                    "total_tokens": len(token_ids) + 10,
                },
                "finish_reason": finish_reason,
            }
        )
        assert task_local_last == (global_index in {10, 12})
    (proxy / "usage.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in usage_rows), encoding="utf-8"
    )

    metadata_values = [
        {
            "instance_id": instance_ids[0],
            "repo": "owner/alpha",
            "base_commit": "a" * 40,
            "agent": "qwen_code",
            "eval_mode": "skip",
            "qwen": {
                "num_turns": len(first),
                "exit_code": 0,
                "parsed": True,
                "subtype": "success",
                "result_tail": "Implemented FutureThing and verified the behavior.",
            },
        },
        {
            "instance_id": instance_ids[1],
            "repo": "owner/beta",
            "base_commit": "b" * 40,
            "agent": "qwen_code",
            "eval_mode": "skip",
            "qwen": {
                "num_turns": len(second),
                "exit_code": 1,
                "parsed": True,
                "subtype": "error",
                "result_tail": "The task remains incomplete.",
            },
        },
    ]
    for instance_id, metadata in zip(instance_ids, metadata_values, strict=True):
        task_root = root / "generation/verified/per_task" / instance_id
        (task_root / "eval").mkdir(parents=True)
        (task_root / "runner_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (task_root / "eval/normalized_eval.json").write_text(
            json.dumps(
                {
                    "track": "swe_bench",
                    "instance_id": instance_id,
                    "outcome": "skipped" if instance_id == instance_ids[0] else "unresolved",
                }
            ),
            encoding="utf-8",
        )
    first_root = root / "generation/verified/per_task" / instance_ids[0]
    (first_root / "patch.diff").write_text(
        """diff --git a/src/module.py b/src/module.py
index 1111111..2222222 100644
--- a/src/module.py
+++ b/src/module.py
@@ -1,3 +1,3 @@
 ContextThing = 1
-value = OldThing
+value = FutureThing
""",
        encoding="utf-8",
    )
    second_root = root / "generation/verified/per_task" / instance_ids[1]
    (second_root / "patch.diff").write_text(
        "diff --git a/src/beta.py b/src/beta.py\n--- a/src/beta.py\n+++ b/src/beta.py\n@@ -0,0 +1 @@\n+BetaChange = True\n",
        encoding="utf-8",
    )
    # This file deliberately contains forbidden oracle-only data. The materializer
    # has no dataset argument and must not derive GoldOnly from it.
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "instance_id": instance_ids[0],
                "patch": "+GoldOnly = True",
                "test_patch": "+assert GoldOnly",
            }
        ),
        encoding="utf-8",
    )
    official_root = root / "official_score"
    official_root.mkdir()
    first_patch = (first_root / "patch.diff").read_bytes()
    second_patch = (second_root / "patch.diff").read_bytes()
    official = {
        "schema_version": 1,
        "kind": "swe_verified_behavioral_official_outcomes",
        "run_name": "synthetic",
        "run_id": "synthetic-run",
        "evidence_id": "e" * 64,
        "instance_ids": instance_ids,
        "counts": {"resolved": 1, "unresolved": 1, "error": 0, "empty": 0},
        "outcomes": [
            {
                "instance_id": instance_ids[0],
                "outcome": "resolved",
                "patch_bytes": len(first_patch),
                "patch_sha256": MODULE.sha256_bytes(first_patch),
                "official_instance_report": {
                    "path": "reports/alpha.json",
                    "sha256": "1" * 64,
                },
            },
            {
                "instance_id": instance_ids[1],
                "outcome": "unresolved",
                "patch_bytes": len(second_patch),
                "patch_sha256": MODULE.sha256_bytes(second_patch),
                "official_instance_report": {
                    "path": "reports/beta.json",
                    "sha256": "2" * 64,
                },
            },
        ],
        "inputs": {
            "hashes": {"campaign_config_sha256": MODULE.sha256_json(campaign_value)}
        },
    }
    (official_root / "official_outcomes.json").write_text(
        json.dumps(official), encoding="utf-8"
    )
    return campaign_value, all_requests


def write_fatal_turn_run(
    root: Path,
    tokenizer: FakeTokenizer,
) -> tuple[dict[str, object], str]:
    instance_id = "owner__limit-53"
    campaign_value = campaign([instance_id])
    max_turns = int(campaign_value["generation"]["max_session_turns"])
    commands = [
        (
            f"Inspect request {index}.",
            f"rg item_{index} src",
            f"src/module.py:item_{index}",
            0,
        )
        for index in range(1, max_turns + 1)
    ]
    requests = task_requests(instance_id, commands, 1)
    proxy = root / "proxy_dumps"
    proxy.mkdir(parents=True)
    usage_rows: list[dict[str, object]] = []
    for global_index, request_value in enumerate(requests, 1):
        (proxy / f"chat_{global_index:04d}.json").write_text(
            json.dumps(request_value), encoding="utf-8"
        )
        _, token_ids, _, _ = MODULE.C1.render_request(
            tokenizer, request=request_value, template=TEMPLATE
        )
        usage_rows.append(
            {
                "idx": global_index,
                # Deliberately descending: timestamps are not mapping evidence.
                "ts": 10_000.0 - global_index,
                "usage": {
                    "prompt_tokens": len(token_ids),
                    "completion_tokens": 10,
                    "total_tokens": len(token_ids) + 10,
                },
                "finish_reason": "tool_calls",
            }
        )
    (proxy / "usage.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in usage_rows), encoding="utf-8"
    )

    task_root = root / "generation/verified/per_task" / instance_id
    task_root.mkdir(parents=True)
    patch_bytes = (
        "diff --git a/src/module.py b/src/module.py\n"
        "--- a/src/module.py\n"
        "+++ b/src/module.py\n"
        "@@ -0,0 +1 @@\n"
        "+LimitThing = True\n"
    ).encode("utf-8")
    (task_root / "patch.diff").write_bytes(patch_bytes)
    (task_root / "qwen_trace.json").write_bytes(b"")
    (task_root / "qwen_stderr.log").write_bytes(MODULE.FATAL_TURN_LIMIT_STDERR)
    metadata = {
        "instance_id": instance_id,
        "repo": "owner/limit",
        "base_commit": "c" * 40,
        "agent": "qwen_code",
        "eval_mode": "skip",
        "qwen": {
            "elapsed_s": 123.5,
            "exit_code": 53,
            "timed_out": False,
            "cli_exit_is_verdict": False,
            "parsed": False,
            "subtype": None,
            "num_turns": None,
            "duration_api_ms": None,
            "usage": None,
            "tool_calls": None,
            "tool_by_name": None,
            "result_tail": "",
        },
        "patch_bytes": len(patch_bytes),
        "terminal_cause": None,
    }
    (task_root / "runner_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return campaign_value, instance_id


def build(root: Path) -> tuple[list[dict[str, object]], dict[str, object], FakeTokenizer]:
    tokenizer = FakeTokenizer()
    campaign_value, _ = write_run(root, tokenizer)
    prompts, summary = MODULE.build_behavioral_bundle(
        run_root=root,
        campaign=campaign_value,
        campaign_sha256=MODULE.sha256_json(campaign_value),
        action_protocol=ACTION_PROTOCOL,
        action_protocol_sha256=MODULE.sha256_bytes(ACTION_PROTOCOL_BYTES),
        tokenizer=tokenizer,
        template=TEMPLATE,
        template_sha256=MODULE.sha256_text(TEMPLATE),
    )
    return prompts, summary, tokenizer


class BehavioralMaterializerTests(unittest.TestCase):
    def test_maps_global_capture_stream_by_frozen_campaign_order_and_quantiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, summary, _ = build(Path(directory))
        self.assertEqual(MODULE.uniform_request_indices(10), [1, 2, 3, 4, 6, 7, 8, 10])
        self.assertEqual(MODULE.uniform_request_indices(2), [1, 2])
        self.assertEqual(MODULE.uniform_request_indices(0), [])
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["prompt_count"], 10)
        self.assertEqual(summary["global_capture_binding"]["global_request_count"], 12)
        self.assertTrue(summary["global_capture_binding"]["exact_global_coverage"])
        audits = summary["task_audits"]
        self.assertEqual(audits[0]["global_request_start"], 1)
        self.assertEqual(audits[0]["global_request_end"], 10)
        self.assertEqual(audits[1]["global_request_start"], 11)
        self.assertEqual(audits[1]["global_request_end"], 12)
        self.assertEqual(
            [prompt["metadata"]["task"]["instance_id"] for prompt in prompts[-2:]],
            ["owner__beta-2", "owner__beta-2"],
        )
        self.assertTrue(
            all(
                prompt["metadata"]["selection"]["independent_of_next_action_label"]
                for prompt in prompts
            )
        )

    def test_exact_fatal_turn_limit_recovers_fifty_requests_without_terminal_imputation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = FakeTokenizer()
            campaign_value, instance_id = write_fatal_turn_run(root, tokenizer)
            tasks = MODULE._load_task_metadata(
                root,
                campaign_value["instance_ids"],
                campaign=campaign_value,
            )
            self.assertEqual(tasks[0]["request_count"], 50)
            mapped, binding = MODULE.map_global_captures(
                run_root=root,
                campaign=campaign_value,
                task_records=tasks,
                tokenizer=tokenizer,
                template=TEMPLATE,
            )
            protocol = MODULE.validate_action_protocol(
                ACTION_PROTOCOL, tokenizer=tokenizer, campaign=campaign_value
            )
            completions = [
                MODULE.derive_completion(mapped[0], index, protocol=protocol)
                for index in range(1, 51)
            ]
            prompts, summary = MODULE.build_behavioral_bundle(
                run_root=root,
                campaign=campaign_value,
                campaign_sha256=MODULE.sha256_json(campaign_value),
                action_protocol=ACTION_PROTOCOL,
                action_protocol_sha256=MODULE.sha256_bytes(ACTION_PROTOCOL_BYTES),
                tokenizer=tokenizer,
                template=TEMPLATE,
                template_sha256=MODULE.sha256_text(TEMPLATE),
            )

        self.assertEqual(binding["global_request_count"], 50)
        self.assertEqual(len(binding["request_count_recoveries"]), 1)
        self.assertEqual(
            binding["mapping_algorithm"],
            "campaign_order_cumulative_verified_request_counts_v2",
        )
        self.assertTrue(
            all(
                completion["status"] == "materialized_in_following_request"
                and completion["next_request_global_index"] == index + 1
                and completion["extension_sha256"] is not None
                for index, completion in enumerate(completions[:49], 1)
            )
        )
        self.assertEqual(completions[49]["status"], "unobserved_after_task_end")
        self.assertEqual(completions[49]["action"]["status"], "missing")
        self.assertEqual(completions[49]["action"]["derivation"], "no_following_capture")
        self.assertEqual(len(prompts), 8)
        endpoint = next(
            prompt
            for prompt in prompts
            if prompt["metadata"]["selection"]["task_request_index"] == 50
        )
        self.assertEqual(endpoint["metadata"]["task"]["instance_id"], instance_id)
        self.assertTrue(endpoint["metadata"]["labels"]["terminal"]["is_episode_endpoint"])
        self.assertFalse(endpoint["metadata"]["labels"]["terminal"]["is_terminal_completion"])
        self.assertEqual(endpoint["metadata"]["labels"]["action"]["status"], "missing")
        provenance = endpoint["metadata"]["provenance"]["request_count"]
        self.assertEqual(
            provenance["derivation"],
            "campaign_max_session_turns_from_exact_fatal_turn_limit_v1",
        )
        self.assertEqual(provenance["fatal_turn_limit"]["stderr_bytes"], 212)
        self.assertEqual(
            provenance["fatal_turn_limit"]["stderr_sha256"],
            MODULE.sha256_bytes(MODULE.FATAL_TURN_LIMIT_STDERR),
        )
        self.assertEqual(provenance["fatal_turn_limit"]["qwen_trace_bytes"], 0)
        self.assertEqual(
            provenance["proxy_capture_binding"]["global_request_start"], 1
        )
        self.assertEqual(
            provenance["proxy_capture_binding"]["global_request_end"], 50
        )
        self.assertEqual(
            summary["task_audits"][0]["terminal_trace_serving_evidence"][
                "derivation"
            ],
            "terminal_qwen_trace_empty",
        )

    def test_exact_fatal_turn_limit_accepts_hash_bound_empty_patch_without_imputation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = FakeTokenizer()
            campaign_value, instance_id = write_fatal_turn_run(root, tokenizer)
            task_root = root / "generation/verified/per_task" / instance_id
            (task_root / "patch.diff").write_bytes(b"")
            metadata_path = task_root / "runner_metadata.json"
            metadata = json.loads(metadata_path.read_bytes())
            metadata["patch_bytes"] = 0
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            prompts, summary = MODULE.build_behavioral_bundle(
                run_root=root,
                campaign=campaign_value,
                campaign_sha256=MODULE.sha256_json(campaign_value),
                action_protocol=ACTION_PROTOCOL,
                action_protocol_sha256=MODULE.sha256_bytes(ACTION_PROTOCOL_BYTES),
                tokenizer=tokenizer,
                template=TEMPLATE,
                template_sha256=MODULE.sha256_text(TEMPLATE),
            )

        endpoint = next(
            prompt
            for prompt in prompts
            if prompt["metadata"]["selection"]["task_request_index"] == 50
        )
        empty_sha256 = MODULE.sha256_bytes(b"")
        fatal = endpoint["metadata"]["provenance"]["request_count"][
            "fatal_turn_limit"
        ]
        self.assertEqual(fatal["runner_metadata_patch_bytes"], 0)
        self.assertEqual(fatal["generated_patch_bytes"], 0)
        self.assertEqual(fatal["generated_patch_sha256"], empty_sha256)
        self.assertEqual(
            endpoint["metadata"]["provenance"]["generated_patch_sha256"],
            empty_sha256,
        )
        self.assertEqual(endpoint["metadata"]["targets"], [])
        self.assertEqual(endpoint["metadata"]["labels"]["action"]["status"], "missing")
        self.assertEqual(
            endpoint["metadata"]["labels"]["official_outcome"]["status"],
            "missing",
        )
        self.assertEqual(summary["dynamic_target_count"], 0)
        self.assertEqual(len(summary["global_capture_binding"]["request_count_recoveries"]), 1)

    def test_fatal_turn_limit_recovery_requires_exact_runner_artifacts(self) -> None:
        cases = (
            ("metadata", "exact fatal-turn metadata"),
            ("stderr", "does not exactly match"),
            ("trace", "not exactly empty"),
            ("patch", "patch byte count differs"),
        )
        for mutation, message in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tokenizer = FakeTokenizer()
                campaign_value, instance_id = write_fatal_turn_run(root, tokenizer)
                task_root = root / "generation/verified/per_task" / instance_id
                if mutation == "metadata":
                    metadata_path = task_root / "runner_metadata.json"
                    metadata = json.loads(metadata_path.read_bytes())
                    metadata["qwen"]["parsed"] = True
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                elif mutation == "stderr":
                    (task_root / "qwen_stderr.log").write_bytes(
                        MODULE.FATAL_TURN_LIMIT_STDERR + b"\n"
                    )
                elif mutation == "trace":
                    (task_root / "qwen_trace.json").write_bytes(b"[]")
                else:
                    (task_root / "patch.diff").write_bytes(b"")
                with self.assertRaisesRegex(ValueError, message):
                    MODULE._load_task_metadata(
                        root,
                        campaign_value["instance_ids"],
                        campaign=campaign_value,
                    )

    def test_fatal_turn_limit_mapping_requires_pinned_span_usage_and_boundary(self) -> None:
        cases = (
            ("pin", "global chat count differs"),
            ("usage", "lacks exact proxy usage coverage"),
            ("finish", "did not end in a tool call"),
            ("boundary", "task-start capture does not contain"),
        )
        for mutation, message in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tokenizer = FakeTokenizer()
                campaign_value, _ = write_fatal_turn_run(root, tokenizer)
                if mutation == "pin":
                    campaign_value["generation"]["max_session_turns"] = 49
                elif mutation == "usage":
                    usage_path = root / "proxy_dumps/usage.jsonl"
                    rows = [json.loads(line) for line in usage_path.read_text().splitlines()]
                    usage_path.write_text(
                        "".join(json.dumps(row) + "\n" for row in rows[:-1]),
                        encoding="utf-8",
                    )
                elif mutation == "finish":
                    usage_path = root / "proxy_dumps/usage.jsonl"
                    rows = [json.loads(line) for line in usage_path.read_text().splitlines()]
                    rows[-1]["finish_reason"] = "stop"
                    usage_path.write_text(
                        "".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8",
                    )
                else:
                    chat_path = root / "proxy_dumps/chat_0001.json"
                    chat = json.loads(chat_path.read_bytes())
                    chat["messages"][1]["content"] = "Task owner__other-1"
                    chat_path.write_text(json.dumps(chat), encoding="utf-8")
                tasks = MODULE._load_task_metadata(
                    root,
                    campaign_value["instance_ids"],
                    campaign=campaign_value,
                )
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.map_global_captures(
                        run_root=root,
                        campaign=campaign_value,
                        task_records=tasks,
                        tokenizer=tokenizer,
                        template=TEMPLATE,
                    )

    def test_action_precedence_and_separate_tool_and_validation_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = FakeTokenizer()
            campaign_value, _ = write_run(root, tokenizer)
            protocol = MODULE.validate_action_protocol(
                ACTION_PROTOCOL, tokenizer=tokenizer, campaign=campaign_value
            )
            tasks = MODULE._load_task_metadata(
                root,
                campaign_value["instance_ids"],
                campaign=campaign_value,
            )
            mapped, _ = MODULE.map_global_captures(
                run_root=root,
                campaign=campaign_value,
                task_records=tasks,
                tokenizer=tokenizer,
                template=TEMPLATE,
            )
            primary = mapped[0]
            edit_and_test = MODULE.derive_completion(primary, 5, protocol=protocol)
            validation_failure = MODULE.derive_completion(primary, 4, protocol=protocol)
            terminal = MODULE.derive_completion(primary, 10, protocol=protocol)
            tool_failure = MODULE.derive_completion(mapped[1], 1, protocol=protocol)
        self.assertEqual(edit_and_test["action"]["class_id"], "edit")
        self.assertEqual(
            edit_and_test["validation"]["derivation"],
            "mutation_precedence_over_validation",
        )
        self.assertEqual(validation_failure["action"]["class_id"], "validate")
        self.assertEqual(validation_failure["tool_execution"]["class_id"], "success")
        self.assertEqual(validation_failure["validation"]["class_id"], "failure")
        self.assertEqual(terminal["action"]["class_id"], "finalize")
        self.assertEqual(tool_failure["tool_execution"]["class_id"], "failure")

    def test_targets_are_future_agent_intersection_only_with_same_task_foils(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, summary, _ = build(Path(directory))
        first = prompts[0]
        targets = first["metadata"]["targets"]
        self.assertEqual([target["target"] for target in targets], ["FutureThing"])
        self.assertNotIn("GoldOnly", json.dumps(targets))
        target = targets[0]
        support = target["future_support"]
        self.assertFalse(support["benchmark_gold_used"])
        self.assertFalse(support["lens_output_used"])
        self.assertEqual(support["mutation_completion"]["completion_index"], 2)
        self.assertEqual(
            {foil["target"] for foil in target["foils"]},
            {"OldThing", "ContextThing"},
        )
        self.assertTrue(
            all(foil["task_instance_id"] == "owner__alpha-1" for foil in target["foils"])
        )
        self.assertEqual(summary["dynamic_target_count"], 1)
        self.assertFalse(summary["target_contract"]["benchmark_gold_patch_read"])

        parsed = MODULE.parse_generated_patch(
            """diff --git a/src/example.py b/src/example.py
--- a/src/example.py
+++ b/src/example.py
@@ -0,0 +1,3 @@
+value = FutureThing  # CommentWord must not become an identifier
+text = "StringWord"
+flag = True
"""
        )
        identifiers = {row["identifier"] for row in parsed["added"]}
        self.assertIn("FutureThing", identifiers)
        self.assertNotIn("CommentWord", identifiers)
        self.assertNotIn("StringWord", identifiers)
        self.assertNotIn("True", identifiers)
        doc = MODULE.parse_generated_patch(
            """diff --git a/docs/example.rst b/docs/example.rst
--- a/docs/example.rst
+++ b/docs/example.rst
@@ -0,0 +1 @@
+DocumentationWord appears here
"""
        )
        self.assertEqual(doc["added"], [])

    def test_cross_task_or_kind_mismatched_foil_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, _, _ = build(Path(directory))
        target = copy.deepcopy(prompts[0]["metadata"]["targets"][0])
        target["foils"][0]["task_instance_id"] = "other__task-2"
        with self.assertRaisesRegex(ValueError, "cross-task"):
            MODULE.validate_target_contract(target, instance_id="owner__alpha-1")
        target = copy.deepcopy(prompts[0]["metadata"]["targets"][0])
        target["foils"][0]["kind"] = "private_identifier"
        with self.assertRaisesRegex(ValueError, "kind differs"):
            MODULE.validate_target_contract(target, instance_id="owner__alpha-1")
        target = copy.deepcopy(prompts[0]["metadata"]["targets"][0])
        target["foils"] = []
        with self.assertRaisesRegex(ValueError, "no same-task same-kind foil"):
            MODULE.validate_target_contract(target, instance_id="owner__alpha-1")

    def test_protocol_validation_intent_is_used_at_declared_precedence(self) -> None:
        tokenizer = FakeTokenizer()
        campaign_value = campaign(["owner__alpha-1"])
        protocol = MODULE.validate_action_protocol(
            ACTION_PROTOCOL, tokenizer=tokenizer, campaign=campaign_value
        )
        initial = [
            {"role": "system", "content": "Use the shell."},
            {"role": "user", "content": "Repair the task."},
        ]
        following = copy.deepcopy(initial)
        append_tool_completion(
            following,
            completion_index=1,
            reasoning="Check the fix before finishing.",
            command="true",
            output="(empty)",
        )
        task = {
            "captures": [
                {
                    "global_index": 1,
                    "sha256": "1" * 64,
                    "messages": initial,
                    "usage": {"idx": 1, "finish_reason": "tool_calls"},
                },
                {
                    "global_index": 2,
                    "sha256": "2" * 64,
                    "messages": following,
                    "usage": {"idx": 2, "finish_reason": "stop"},
                },
            ]
        }
        completion = MODULE.derive_completion(task, 1, protocol=protocol)
        self.assertEqual(completion["action"]["class_id"], "validate")
        self.assertEqual(
            completion["action"]["derivation"],
            "validation_intent_assistant_text",
        )
        self.assertEqual(completion["validation"]["status"], "not_applicable")

    def test_per_checkpoint_leakage_uses_nfkc_casefold_segments_and_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, _, tokenizer = build(Path(directory))
        early = prompts[0]["metadata"]["target_eligibility"][0]
        after_edit = next(
            prompt
            for prompt in prompts
            if prompt["metadata"]["task"]["instance_id"] == "owner__alpha-1"
            and prompt["metadata"]["selection"]["task_request_index"] == 3
        )["metadata"]["target_eligibility"][0]
        self.assertEqual(early["status"], "eligible")
        self.assertFalse(early["target_exposed"])
        self.assertTrue(early["retained_hidden_foil_ids"])
        self.assertEqual(after_edit["status"], "target_exposed")
        self.assertTrue(after_edit["target_rendered_evidence"]["scored_form_token_id_hits"])

        segment = MODULE.exposure_evidence(
            text="Use future_thing inside SimpleFutureThingObject.",
            aliases=["FutureThing"],
            forms=[{"text": "FutureThing", "token_id": 81001}],
            tokenizer=tokenizer,
        )
        nfkc = MODULE.exposure_evidence(
            text="The \u212aelvin value is visible.",
            aliases=["Kelvin"],
            forms=[{"text": "Kelvin", "token_id": 81009}],
            tokenizer=tokenizer,
        )
        self.assertTrue(segment["exposed"])
        self.assertTrue(segment["identifier_hits"])
        self.assertTrue(nfkc["exposed"])
        self.assertEqual(nfkc["identifier_hits"][0]["match_kind"], "nfkc_casefold_full_identifier")

    def test_failure_and_truncation_episodes_are_retained_without_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, summary, _ = build(Path(directory))
        beta = [
            prompt
            for prompt in prompts
            if prompt["metadata"]["task"]["instance_id"] == "owner__beta-2"
        ]
        self.assertEqual(len(beta), 2)
        self.assertEqual(beta[0]["metadata"]["labels"]["tool_execution"]["class_id"], "failure")
        self.assertEqual(beta[1]["metadata"]["labels"]["action"]["status"], "missing")
        self.assertEqual(beta[1]["metadata"]["labels"]["terminal"]["finish_reason"], "length")
        self.assertEqual(
            beta[0]["metadata"]["labels"]["official_outcome"]["class_id"], "failure"
        )
        self.assertEqual(
            summary["task_audits"][1]["official_outcome"]["class_id"], "failure"
        )

    def test_official_outcomes_bind_only_the_hashed_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts, summary, tokenizer = build(root)
            first = prompts[0]["metadata"]
            self.assertEqual(first["labels"]["official_outcome"]["class_id"], "success")
            self.assertEqual(
                first["provenance"]["official_outcomes"]["sha256"],
                summary["official_outcome_binding"]["sha256"],
            )
            campaign_value = campaign(["owner__alpha-1", "owner__beta-2"])
            official_path = root / "official_score/official_outcomes.json"
            official = json.loads(official_path.read_bytes())
            official["outcomes"][1]["outcome"] = "error"
            official["counts"] = {
                "resolved": 1,
                "unresolved": 0,
                "error": 1,
                "empty": 0,
            }
            official_path.write_text(json.dumps(official), encoding="utf-8")
            error_prompts, _ = MODULE.build_behavioral_bundle(
                run_root=root,
                campaign=campaign_value,
                campaign_sha256=MODULE.sha256_json(campaign_value),
                action_protocol=ACTION_PROTOCOL,
                action_protocol_sha256=MODULE.sha256_bytes(ACTION_PROTOCOL_BYTES),
                tokenizer=tokenizer,
                template=TEMPLATE,
                template_sha256=MODULE.sha256_text(TEMPLATE),
            )
            beta_error = next(
                prompt
                for prompt in error_prompts
                if prompt["metadata"]["task"]["instance_id"] == "owner__beta-2"
            )["metadata"]["labels"]["official_outcome"]
            self.assertEqual(beta_error["status"], "missing")
            self.assertIsNone(beta_error["class_id"])
            self.assertEqual(beta_error["verdict"], "error")
            official_path.unlink()
            prompts, missing_summary = MODULE.build_behavioral_bundle(
                run_root=root,
                campaign=campaign_value,
                campaign_sha256=MODULE.sha256_json(campaign_value),
                action_protocol=ACTION_PROTOCOL,
                action_protocol_sha256=MODULE.sha256_bytes(ACTION_PROTOCOL_BYTES),
                tokenizer=tokenizer,
                template=TEMPLATE,
                template_sha256=MODULE.sha256_text(TEMPLATE),
            )
            self.assertTrue(
                all(
                    prompt["metadata"]["labels"]["official_outcome"]["status"]
                    == "missing"
                    for prompt in prompts
                )
            )
            self.assertFalse(
                missing_summary["official_outcome_binding"][
                    "generation_skip_evaluations_used"
                ]
            )
            with self.assertRaisesRegex(ValueError, "required official outcome"):
                MODULE.build_behavioral_bundle(
                    run_root=root,
                    campaign=campaign_value,
                    campaign_sha256=MODULE.sha256_json(campaign_value),
                    action_protocol=ACTION_PROTOCOL,
                    action_protocol_sha256=MODULE.sha256_bytes(ACTION_PROTOCOL_BYTES),
                    tokenizer=tokenizer,
                    template=TEMPLATE,
                    template_sha256=MODULE.sha256_text(TEMPLATE),
                    require_official_outcomes=True,
                )

    def test_missing_usage_is_index_bound_and_retained_without_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = FakeTokenizer()
            campaign_value, _ = write_run(root, tokenizer)
            usage_path = root / "proxy_dumps/usage.jsonl"
            rows = [json.loads(line) for line in usage_path.read_text().splitlines()]
            usage_path.write_text(
                "\n".join(
                    json.dumps(row) for row in rows if row["idx"] not in {4, 12}
                )
                + "\n",
                encoding="utf-8",
            )
            prompts, summary = MODULE.build_behavioral_bundle(
                run_root=root,
                campaign=campaign_value,
                campaign_sha256=MODULE.sha256_json(campaign_value),
                action_protocol=ACTION_PROTOCOL,
                action_protocol_sha256=MODULE.sha256_bytes(ACTION_PROTOCOL_BYTES),
                tokenizer=tokenizer,
                template=TEMPLATE,
                template_sha256=MODULE.sha256_text(TEMPLATE),
            )
        request_four = next(
            prompt
            for prompt in prompts
            if prompt["metadata"]["selection"]["global_request_index"] == 4
        )
        self.assertEqual(request_four["metadata"]["labels"]["action"]["class_id"], "validate")
        self.assertEqual(
            request_four["metadata"]["provenance"]["next_completion"][
                "finish_reason_derivation"
            ],
            "exact_following_raw_extension",
        )
        self.assertFalse(
            any(
                prompt["metadata"]["selection"]["global_request_index"] == 12
                for prompt in prompts
            )
        )
        beta_audit = summary["task_audits"][1]
        self.assertEqual(beta_audit["excluded_request_indices"], [2])
        self.assertEqual(beta_audit["excluded_requests"][0]["global_request_index"], 12)
        self.assertIn(
            "terminal_qwen_trace_absent",
            beta_audit["excluded_requests"][0]["reasons"],
        )
        binding = summary["global_capture_binding"]
        self.assertEqual(binding["missing_usage_indices"], [4, 12])
        self.assertTrue(binding["exact_raw_request_coverage"])
        self.assertFalse(binding["exact_usage_index_coverage"])
        self.assertFalse(binding["exact_global_coverage"])

    def test_probeability_uses_terminal_trace_and_one_token_replay_room(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "qwen_trace.json"
            task = {
                "qwen_trace_path": trace_path,
                "captures": [
                    {
                        "global_index": 1,
                        "token_ids": [1, 2],
                        "usage": {"telemetry_status": "available"},
                    },
                    {
                        "global_index": 2,
                        "token_ids": [1, 2, 3],
                        "usage": {"telemetry_status": "missing"},
                    },
                ],
            }
            trace_path.write_text(
                json.dumps(
                    [
                        {
                            "type": "assistant",
                            "message": {
                                "usage": {"input_tokens": 3, "output_tokens": 1},
                                "content": [{"type": "text", "text": "done"}],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            served = MODULE.select_probeable_requests(task, max_prompt_tokens=3)
            self.assertEqual(served["probeable_request_indices"], [1, 2])

            trace_path.write_text(
                json.dumps(
                    [
                        {
                            "type": "assistant",
                            "message": {
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                                "content": [
                                    {"type": "text", "text": "[API Error: 400 overflow]"}
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            unserved = MODULE.select_probeable_requests(task, max_prompt_tokens=2)
        self.assertEqual(unserved["probeable_request_indices"], [1])
        reasons = unserved["excluded_requests"][0]["reasons"]
        self.assertIn("terminal_qwen_trace_zero_usage_api_error_400", reasons)
        self.assertIn("canonical_prompt_exceeds_one_token_replay_ceiling", reasons)

    def test_mapping_fails_closed_on_sampling_or_raw_prefix_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = FakeTokenizer()
            campaign_value, _ = write_run(root, tokenizer)
            chat = root / "proxy_dumps/chat_0002.json"
            value = json.loads(chat.read_bytes())
            value["seed"] += 1
            chat.write_text(json.dumps(value), encoding="utf-8")
            tasks = MODULE._load_task_metadata(
                root,
                campaign_value["instance_ids"],
                campaign=campaign_value,
            )
            with self.assertRaisesRegex(ValueError, "seed sequence"):
                MODULE.map_global_captures(
                    run_root=root,
                    campaign=campaign_value,
                    task_records=tasks,
                    tokenizer=tokenizer,
                    template=TEMPLATE,
                )

    def test_sampling_accepts_only_the_exact_proxy_context_clamp(self) -> None:
        value = request(
            [
                {"role": "system", "content": "Use the repository shell."},
                {"role": "user", "content": "x" * 240_000},
            ],
            1,
        )
        estimate = sum(
            len(json.dumps(value[key], ensure_ascii=False))
            for key in ("messages", "tools")
        ) // 4
        expected = min(8192, 65536 - estimate - 64)
        self.assertGreater(expected, 0)
        self.assertLess(expected, 8192)
        value["max_tokens"] = expected
        MODULE._validate_request_sampling(
            value,
            global_index=1,
            served_model="qwen3.6-27b-nvfp4",
            context_limit=65536,
        )
        value["max_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "context-fit policy"):
            MODULE._validate_request_sampling(
                value,
                global_index=1,
                served_model="qwen3.6-27b-nvfp4",
                context_limit=65536,
            )

    def test_prompt_payload_hash_reconstructs_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts, _, _ = build(Path(directory))
        for prompt in prompts:
            self.assertEqual(
                MODULE._prompt_record_payload_sha256(prompt),
                prompt["metadata"]["provenance"]["prompt_record_payload_sha256"],
            )

    def test_combination_preserves_source_campaigns_and_namespaces_indices(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            tokenizer = FakeTokenizer()
            sources = []
            for cohort_id, directory, instance_ids in (
                (
                    "development",
                    first_directory,
                    ["owner__alpha-1", "owner__beta-2"],
                ),
                (
                    "replication",
                    second_directory,
                    ["owner__gamma-3", "owner__delta-4"],
                ),
            ):
                root = Path(directory)
                campaign_value, _ = write_run(
                    root, tokenizer, instance_ids=instance_ids
                )
                campaign_sha256 = MODULE.sha256_json(campaign_value)
                prompts, summary = MODULE.build_behavioral_bundle(
                    run_root=root,
                    campaign=campaign_value,
                    campaign_sha256=campaign_sha256,
                    action_protocol=ACTION_PROTOCOL,
                    action_protocol_sha256=MODULE.sha256_bytes(
                        ACTION_PROTOCOL_BYTES
                    ),
                    tokenizer=tokenizer,
                    template=TEMPLATE,
                    template_sha256=MODULE.sha256_text(TEMPLATE),
                )
                sources.append(
                    {
                        "id": cohort_id,
                        "campaign_sha256": campaign_sha256,
                        "instance_ids": instance_ids,
                        "run_label": f"{cohort_id}-run",
                        "prompts": prompts,
                        "summary": summary,
                    }
                )
            prompts, summary = MODULE.combine_behavioral_bundles(
                sources, cohort_manifest_sha256="f" * 64
            )
        self.assertEqual(summary["task_count"], 4)
        self.assertEqual(summary["global_request_count"], 24)
        self.assertEqual(summary["prompt_count"], 20)
        self.assertEqual(
            summary["source_campaign_sha256s"],
            [source["campaign_sha256"] for source in sources],
        )
        self.assertEqual(len({prompt["id"] for prompt in prompts}), len(prompts))
        replication = next(
            prompt
            for prompt in prompts
            if prompt["metadata"]["cohort"]["id"] == "replication"
        )
        selection = replication["metadata"]["selection"]
        task = replication["metadata"]["task"]
        self.assertEqual(selection["source_global_request_index"], 1)
        self.assertEqual(selection["global_request_index"], 13)
        self.assertEqual(task["source_selection_index"], 0)
        self.assertEqual(task["selection_index"], 2)
        self.assertEqual(
            replication["metadata"]["cohort"]["source_task_instance_ids"],
            ["owner__gamma-3", "owner__delta-4"],
        )
        for prompt in prompts:
            self.assertEqual(
                MODULE._prompt_record_payload_sha256(prompt),
                prompt["metadata"]["provenance"][
                    "prompt_record_payload_sha256"
                ],
            )

    def test_frozen_n20_manifest_binds_order_hashes_runs_and_twenty_tasks(self) -> None:
        manifest = json.loads(
            (ROOT / "configs/swe_behavioral_n20_cohort.json").read_bytes()
        )
        pairs = [
            (
                (ROOT / "configs/swe_behavioral_campaign.json").resolve(),
                Path("/tmp/swe_behavioral_n10_20260718"),
            ),
            (
                (
                    ROOT / "configs/swe_behavioral_replication_campaign.json"
                ).resolve(),
                Path("/tmp/swe_behavioral_replication_n10_20260718"),
            ),
        ]
        specs = MODULE.validate_cohort_manifest(
            manifest,
            cohort_pairs=pairs,
            action_protocol_sha256=MODULE.sha256_file(
                ROOT / "configs/swe_stage_action_probes.json"
            ),
            template_sha256=MODULE.sha256_file(
                ROOT / "configs/qwen3-openai-codex.jinja"
            ),
        )
        self.assertEqual([spec["id"] for spec in specs], ["development", "replication"])
        self.assertEqual(sum(len(spec["instance_ids"]) for spec in specs), 20)
        with self.assertRaisesRegex(ValueError, "path/order mismatch"):
            MODULE.validate_cohort_manifest(
                manifest,
                cohort_pairs=list(reversed(pairs)),
                action_protocol_sha256=MODULE.sha256_file(
                    ROOT / "configs/swe_stage_action_probes.json"
                ),
                template_sha256=MODULE.sha256_file(
                    ROOT / "configs/qwen3-openai-codex.jinja"
                ),
            )


if __name__ == "__main__":
    unittest.main()
