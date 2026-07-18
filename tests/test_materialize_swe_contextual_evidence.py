#!/usr/bin/env python3
"""Focused tests for contextual-evidence prompt materialization."""

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
    "materialize_swe_contextual_evidence",
    ROOT / "scripts" / "materialize_swe_contextual_evidence.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    forms = {
        " Alpha": 9001,
        " Beta": 9002,
        " Gamma": 9003,
        " Delta": 9004,
        " Epsilon": 9011,
        " Zeta": 9012,
        " Eta": 9013,
        " Theta": 9014,
    }
    reverse = {token_id: text for text, token_id in forms.items()}

    def apply_chat_template(self, messages: object, **_: object) -> str:
        return (
            "PROMPT\n"
            + json.dumps(messages, sort_keys=True, ensure_ascii=False)
            + "\n<|im_start|>assistant\n<think>\n"
        )

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        if text in self.forms:
            return [self.forms[text]]
        result: list[int] = []
        index = 0
        ordered = sorted(self.forms.items(), key=lambda item: len(item[0]), reverse=True)
        while index < len(text):
            matched = next(
                ((form, token_id) for form, token_id in ordered if text.startswith(form, index)),
                None,
            )
            if matched is None:
                result.append(20_000 + ord(text[index]))
                index += 1
            else:
                result.append(matched[1])
                index += len(matched[0])
        return result

    def decode(self, token_ids: list[int], **_: object) -> str:
        token_id = token_ids[0]
        if token_id in self.reverse:
            return self.reverse[token_id]
        return chr(token_id - 20_000)


TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_shell_command", "description": "Run shell"},
    }
]


def assistant(text: str, reasoning: str, call_id: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": text,
        "reasoning_content": reasoning,
        "reasoning": reasoning,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "arguments": json.dumps(
                        {"command": "rg symbol src", "description": "Inspect source"},
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    }


def tool_result(call_id: str) -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [{"type": "text", "text": "src/module.py: implementation"}],
    }


def request(messages: list[dict[str, object]], seed: int) -> dict[str, object]:
    return {
        "model": "qwen3.6-27b-nvfp4",
        "messages": copy.deepcopy(messages),
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "seed": seed,
        "tools": copy.deepcopy(TOOLS),
    }


def request_triplet(
    instance_id: str,
    target: str,
    *,
    expose_after: bool,
    global_index: int,
) -> tuple[dict[str, object], dict[str, object]]:
    initial = [
        {"role": "system", "content": "Use the repository tool."},
        {"role": "user", "content": f"Task {instance_id}: repair the behavior."},
    ]
    first_text = f"The observation names {target}." if expose_after else "I found the source."
    first = assistant(first_text, "Inspect the repository first.", "call-first")
    second = assistant(
        f"I should inspect {target} next.",
        f"The task evidence now points to {target}; private future detail sentinel.",
        "call-second",
    )
    before = request(initial, 100 + global_index)
    after = request([*initial, first, tool_result("call-first")], 101 + global_index)
    label = request(
        [*initial, first, tool_result("call-first"), second, tool_result("call-second")],
        102 + global_index,
    )
    requests = {"before": before, "after": after, "label": label}
    sources: dict[str, dict[str, object]] = {}
    for name, index in (
        ("before", global_index - 1),
        ("after", global_index),
        ("label", global_index + 1),
    ):
        payload = MODULE.canonical_json_bytes(requests[name])
        sources[name] = {
            "path": f"runs/test/proxy_dumps/chat_{index:04d}.json",
            "bytes": len(payload),
            "sha256": MODULE.sha256_bytes(payload),
        }
    return requests, sources


def concept(
    identifier: str,
    token_id: int,
    *,
    future_present: bool,
    before_count: int,
    after_count: int,
) -> dict[str, object]:
    return {
        "id": identifier.lower(),
        "label": f"{identifier} contextual concept",
        "aliases": [identifier],
        "exposure_normalization": "case_sensitive_identifier_boundary_v1",
        "forms": [{"kind": "leading_space", "text": f" {identifier}", "token_id": token_id}],
        "expected_exposure": {
            "before": {
                "present": before_count > 0,
                "identifier_occurrences": before_count,
            },
            "after": {
                "present": after_count > 0,
                "identifier_occurrences": after_count,
            },
        },
        "future_present": future_present,
    }


def task(
    *,
    task_id: str,
    instance_id: str,
    repo: str,
    cohort: str,
    global_index: int,
    target: str,
    target_id: int,
    foil_names: tuple[str, str, str],
    foil_ids: tuple[int, int, int],
    raw_hashes: dict[str, str],
    expose_after: bool,
) -> dict[str, object]:
    return {
        "id": task_id,
        "instance_id": instance_id,
        "repo": repo,
        "cohort": cohort,
        "after_global_request_index": global_index,
        "after_task_request_index": 2,
        "raw_sha256": raw_hashes,
        "stratum": "evidence_reweighting" if expose_after else "novel_inference",
        "primary_control_eligible": True,
        "control_match_status": "matched_exposed_target_and_foils",
        "target": concept(
            target,
            target_id,
            future_present=True,
            before_count=0,
            after_count=1 if expose_after else 0,
        ),
        "foils": [
            concept(
                name,
                token_id,
                future_present=False,
                before_count=0,
                after_count=0,
            )
            for name, token_id in zip(foil_names, foil_ids, strict=True)
        ],
        "task_card": {
            "why": f"The next decision centers on {target}.",
            "where": "src/module.py",
            "evidence": "The prior repository observation narrowed the issue.",
            "next": "Inspect the selected symbol.",
            "claim_scope": "Entity-level contextual evidence only.",
        },
    }


def fixture() -> tuple[dict[str, object], dict[str, object], FakeTokenizer]:
    tokenizer = FakeTokenizer()
    requests_a, sources_a = request_triplet(
        "owner__repo-1", "Alpha", expose_after=True, global_index=2
    )
    requests_b, sources_b = request_triplet(
        "other__project-2", "Epsilon", expose_after=False, global_index=12
    )
    task_a = task(
        task_id="evidence-00",
        instance_id="owner__repo-1",
        repo="owner/repo",
        cohort="development",
        global_index=2,
        target="Alpha",
        target_id=9001,
        foil_names=("Beta", "Gamma", "Delta"),
        foil_ids=(9002, 9003, 9004),
        raw_hashes={name: value["sha256"] for name, value in sources_a.items()},
        expose_after=True,
    )
    task_b = task(
        task_id="evidence-01",
        instance_id="other__project-2",
        repo="other/project",
        cohort="replication",
        global_index=12,
        target="Epsilon",
        target_id=9011,
        foil_names=("Zeta", "Eta", "Theta"),
        foil_ids=(9012, 9013, 9014),
        raw_hashes={name: value["sha256"] for name, value in sources_b.items()},
        expose_after=False,
    )
    protocol = {
        "schema_version": 1,
        "kind": MODULE.PROTOCOL_KIND,
        "analysis_version": MODULE.ANALYSIS_VERSION,
        "lens_outputs_used_for_boundary_or_labels": False,
        "pins": {
            "model": {},
            "tokenizer": {
                "logit_vocabulary_size": 248320,
            },
            "chat_template": {},
            "sources": {
                cohort: {
                    "run_root": f"runs/{cohort}",
                    "campaign_path": f"configs/{cohort}.json",
                    "campaign_sha256": character * 64,
                }
                for cohort, character in (("development", "a"), ("replication", "b"))
            },
            "lenses": {},
        },
        "fixed_layer_band": {"layers": list(range(24, 48))},
        "prompt_context": {"maximum_prompt_tokens": 65535},
        "numerical_certification": {"primary_stable": {}, "legacy_strict": {}},
        "score_reduction": {"within_concept": "logmeanexp"},
        "controls": {},
        "decision": {},
        "tasks": [task_a, task_b],
    }
    inputs = {
        "evidence-00": {"requests": requests_a, "sources": sources_a},
        "evidence-01": {"requests": requests_b, "sources": sources_b},
    }
    return protocol, inputs, tokenizer


def build(
    protocol: dict[str, object], inputs: dict[str, object], tokenizer: FakeTokenizer
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return MODULE.build_evidence_bundle(
        protocol,
        protocol_sha256="f" * 64,
        task_inputs=inputs,
        tokenizer=tokenizer,
        template="test-template",
    )


class MaterializeContextualEvidenceTest(unittest.TestCase):
    def test_materializes_deterministic_pair_with_global_candidate_union(self) -> None:
        protocol, inputs, tokenizer = fixture()
        prompts, manifest = build(protocol, inputs, tokenizer)
        expected_ids = [9001, 9002, 9003, 9004, 9011, 9012, 9013, 9014]
        self.assertEqual(len(prompts), 4)
        self.assertTrue(all(prompt["score_token_ids"] == expected_ids for prompt in prompts))
        self.assertEqual([prompt["metadata"]["state"] for prompt in prompts], ["before", "after"] * 2)
        alpha_after = prompts[1]["metadata"]["exposure"]["target"]
        self.assertEqual(alpha_after["identifier_occurrences"], 1)
        self.assertIn("nfkc_casefold", alpha_after["supplemental_rendered"])
        self.assertEqual(alpha_after["forms"][0]["token_occurrences"], 1)
        self.assertIsInstance(alpha_after["forms"][0]["last_token_distance"], int)
        self.assertTrue(prompts[1]["metadata"]["task"]["primary_control_eligible"])
        self.assertEqual(
            prompts[1]["metadata"]["task"]["control_match_status"],
            "matched_exposed_target_and_foils",
        )
        self.assertEqual(manifest["score_vocabulary"]["token_ids"], expected_ids)
        self.assertEqual(
            manifest["prompt_bundle"]["sha256"],
            MODULE.sha256_bytes(MODULE.json_document_bytes(prompts)),
        )
        serialized = json.dumps({"prompts": prompts, "manifest": manifest})
        self.assertNotIn("private future detail sentinel", serialized)
        prompts_again, manifest_again = build(protocol, inputs, tokenizer)
        self.assertEqual((prompts, manifest), (prompts_again, manifest_again))

    def test_raw_message_flattening_preserves_newline_identifier_boundaries(self) -> None:
        messages = [{"role": "tool", "content": "prefix\narray\nInclude include"}]
        flattened = "\n".join(MODULE.C1.flatten_string_values(messages))
        self.assertEqual(
            MODULE.case_sensitive_identifier_occurrences(flattened, ["array"])[
                "identifier_occurrences"
            ],
            1,
        )
        self.assertEqual(
            MODULE.case_sensitive_identifier_occurrences(flattened, ["include"])[
                "identifier_occurrences"
            ],
            1,
        )
        self.assertEqual(
            MODULE.identifier_occurrences(flattened, ["include"])[
                "identifier_occurrences"
            ],
            2,
        )

    def test_rejects_prefix_drift(self) -> None:
        protocol, inputs, tokenizer = fixture()
        inputs["evidence-00"]["requests"]["after"]["messages"][0]["content"] = "drift"
        with self.assertRaisesRegex(ValueError, "exactly preserve and extend"):
            build(protocol, inputs, tokenizer)

    def test_rejects_declared_exposure_mismatch(self) -> None:
        protocol, inputs, tokenizer = fixture()
        protocol["tasks"][0]["target"]["expected_exposure"]["after"] = {
            "present": False,
            "identifier_occurrences": 0,
        }
        with self.assertRaisesRegex(ValueError, "after presence mismatch"):
            build(protocol, inputs, tokenizer)

    def test_rejects_missing_future_target_and_present_future_foil(self) -> None:
        protocol, inputs, tokenizer = fixture()
        future = inputs["evidence-00"]["requests"]["label"]["messages"][-2]
        future["content"] = "Inspect the selected item next."
        future["reasoning_content"] = "No named item here."
        future["reasoning"] = "No named item here."
        with self.assertRaisesRegex(ValueError, "does not contain target"):
            build(protocol, inputs, tokenizer)

        protocol, inputs, tokenizer = fixture()
        future = inputs["evidence-00"]["requests"]["label"]["messages"][-2]
        future["content"] += " Avoid Beta."
        with self.assertRaisesRegex(ValueError, "contains foil beta"):
            build(protocol, inputs, tokenizer)

    def test_rejects_form_pin_drift_and_writes_exact_atomic_document(self) -> None:
        protocol, inputs, tokenizer = fixture()
        protocol["tasks"][0]["target"]["forms"][0]["token_id"] = 9002
        with self.assertRaisesRegex(ValueError, "token pin changed"):
            build(protocol, inputs, tokenizer)

        protocol, inputs, tokenizer = fixture()
        prompts, _ = build(protocol, inputs, tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prompts.json"
            MODULE.atomic_write_json(output, prompts)
            self.assertEqual(output.read_bytes(), MODULE.json_document_bytes(prompts))


if __name__ == "__main__":
    unittest.main()
