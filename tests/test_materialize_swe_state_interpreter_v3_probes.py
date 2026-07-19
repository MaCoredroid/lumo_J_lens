#!/usr/bin/env python3
"""Adversarial tests for the exact V3 dense-materialization boundary."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Mapping
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "materialize_swe_state_interpreter_v3_probes",
    ROOT / "scripts/materialize_swe_state_interpreter_v3_probes.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InvocationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = root / "cohort.json"
        self.action = root / "actions.json"
        self.template = root / "template.jinja"
        self.cache = root / "cache"
        self.runs = root / "runs"
        self.campaigns = (root / "campaign-a.json", root / "campaign-b.json")
        self.labels = ("development-a", "development-b")
        self.run_roots = tuple(self.runs / label for label in self.labels)
        for path in (self.manifest, self.action, self.template, *self.campaigns):
            path.write_text("{}\n", encoding="ascii")
        for path in self.run_roots:
            path.mkdir(parents=True)
        self.declaration = types.SimpleNamespace(
            cohort_path=self.manifest,
            campaign_paths=self.campaigns,
            cohort={
                "cohorts": [
                    {"id": "development_a", "run_label": self.labels[0]},
                    {"id": "development_b", "run_label": self.labels[1]},
                ]
            },
        )

    def argv(self) -> list[str]:
        return [
            "--all-probeable",
            "--cohort",
            str(self.campaigns[0]),
            str(self.run_roots[0]),
            "--cohort",
            str(self.campaigns[1]),
            str(self.run_roots[1]),
            "--cohort-manifest",
            str(self.manifest),
            "--action-protocol",
            str(self.action),
            "--template",
            str(self.template),
            "--output",
            str(self.cache / "prompts.json"),
            "--summary",
            str(self.cache / "prompts-summary.json"),
        ]


class V3DenseMaterializerTests(unittest.TestCase):
    @staticmethod
    def _prefix_fixture(*, drift_completed_history: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
        class ByteTokenizer:
            @staticmethod
            def encode(text: str, *, add_special_tokens: bool) -> list[int]:
                if add_special_tokens:
                    raise AssertionError("V3 prefix audit added special tokens")
                return list(text.encode("utf-8"))

        base = "<|im_start|>system\ncontract<|im_end|>\n"
        suffix = MODULE.GENERATION_SUFFIX_BY_THINKING[True]
        previous_rendered = base + suffix
        current_base = "corrupt-history" if drift_completed_history else base
        current_rendered = (
            current_base
            + "<|im_start|>assistant\n<tool_call>\n"
            + '{"name":"run_shell_command","arguments":{"command":"pwd"}}'
            + "\n</tool_call><|im_end|>\n"
            + "<|im_start|>user\n<tool_response>\nok\n</tool_response>\n"
            + "<|im_end|>\n"
            + suffix
        )
        tokenizer = ByteTokenizer()
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        request = {
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "system", "content": "contract"}],
        }
        current_request = {
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [
                *request["messages"],
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {
                                "name": "run_shell_command",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "content": [{"type": "text", "text": "ok"}],
                },
            ],
        }
        provenance = {
            "recovered": True,
            "request_count": 2,
            "proxy_capture_binding": {
                "canonical_rendered_prefix_chain_verified": True,
                "canonical_token_prefix_chain_verified": True,
            },
        }
        tasks = [
            {
                "instance_id": "sympy__sympy-18199",
                "request_count_provenance": provenance,
                "captures": [
                    {
                        "local_index": 18,
                        "global_index": 853,
                        "path": "proxy_dumps/chat_0853.json",
                        "sha256": "1a1817a3eee15a70c9ac4e59cc32730dad2401e45f6d4c7c73a4b5af6de9fbaa",
                        "request": request,
                        "rendered": previous_rendered,
                        "token_ids": encode(previous_rendered),
                    },
                    {
                        "local_index": 19,
                        "global_index": 854,
                        "path": "proxy_dumps/chat_0854.json",
                        "sha256": "8cf47fb7175e3fc380aecbd51657e6dbc4c6e492ba49ce0fc1a4f55d9fe6e45d",
                        "request": current_request,
                        "rendered": current_rendered,
                        "token_ids": encode(current_rendered),
                    },
                ],
            }
        ]
        binding = {"request_count_recoveries": [{"stale": True}]}
        return tasks, binding, tokenizer

    def test_only_exact_ordered_ab_invocation_and_explicit_cache_outputs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InvocationFixture(Path(temporary))
            with (
                mock.patch.object(MODULE, "V3_ACTION_PROTOCOL_PATH", fixture.action),
                mock.patch.object(MODULE, "V3_TEMPLATE_PATH", fixture.template),
                mock.patch.object(MODULE, "V3_RUNS_ROOT", fixture.runs),
                mock.patch.object(MODULE, "V3_CACHE_ROOT", fixture.cache),
            ):
                delegated, output, summary = MODULE._prepare_delegated_argv(
                    fixture.argv(), declaration=fixture.declaration
                )
            self.assertNotIn("--all-probeable", delegated)
            self.assertEqual(output, fixture.cache / "prompts.json")
            self.assertEqual(summary, fixture.cache / "prompts-summary.json")

    def test_third_swapped_or_arbitrary_cohort_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InvocationFixture(Path(temporary))
            patches = (
                mock.patch.object(MODULE, "V3_ACTION_PROTOCOL_PATH", fixture.action),
                mock.patch.object(MODULE, "V3_TEMPLATE_PATH", fixture.template),
                mock.patch.object(MODULE, "V3_RUNS_ROOT", fixture.runs),
                mock.patch.object(MODULE, "V3_CACHE_ROOT", fixture.cache),
            )
            base = fixture.argv()
            third = [*base[:9], "--cohort", str(fixture.campaigns[0]), str(fixture.run_roots[0]), *base[9:]]
            swapped = base.copy()
            swapped[2:4], swapped[5:7] = swapped[5:7], swapped[2:4]
            arbitrary = base.copy()
            other = fixture.root / "other.json"
            other.write_text("{}\n", encoding="ascii")
            arbitrary[2] = str(other)
            for candidate in (third, swapped, arbitrary):
                with self.subTest(candidate=candidate), patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaises((SystemExit, ValueError)):
                        MODULE._prepare_delegated_argv(candidate, declaration=fixture.declaration)

    def test_wrong_action_template_manifest_and_implicit_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InvocationFixture(Path(temporary))
            wrong = fixture.root / "wrong.json"
            wrong.write_text("{}\n", encoding="ascii")
            candidates: list[list[str]] = []
            for option in ("--cohort-manifest", "--action-protocol", "--template"):
                changed = fixture.argv()
                changed[changed.index(option) + 1] = str(wrong)
                candidates.append(changed)
            for option in ("--output", "--summary"):
                changed = fixture.argv()
                index = changed.index(option)
                del changed[index : index + 2]
                candidates.append(changed)
            for candidate in candidates:
                with (
                    self.subTest(candidate=candidate),
                    mock.patch.object(MODULE, "V3_ACTION_PROTOCOL_PATH", fixture.action),
                    mock.patch.object(MODULE, "V3_TEMPLATE_PATH", fixture.template),
                    mock.patch.object(MODULE, "V3_RUNS_ROOT", fixture.runs),
                    mock.patch.object(MODULE, "V3_CACHE_ROOT", fixture.cache),
                    self.assertRaises((SystemExit, ValueError)),
                ):
                    MODULE._prepare_delegated_argv(candidate, declaration=fixture.declaration)

    def test_arbitrary_model_snapshot_and_existing_canonical_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InvocationFixture(Path(temporary))
            candidate = [*fixture.argv(), "--model-snapshot", str(fixture.root)]
            patches = (
                mock.patch.object(MODULE, "V3_ACTION_PROTOCOL_PATH", fixture.action),
                mock.patch.object(MODULE, "V3_TEMPLATE_PATH", fixture.template),
                mock.patch.object(MODULE, "V3_RUNS_ROOT", fixture.runs),
                mock.patch.object(MODULE, "V3_CACHE_ROOT", fixture.cache),
            )
            with patches[0], patches[1], patches[2], patches[3], self.assertRaisesRegex(
                SystemExit, "model-snapshot"
            ):
                MODULE._prepare_delegated_argv(candidate, declaration=fixture.declaration)
            fixture.cache.mkdir()
            (fixture.cache / "prompts.json").write_text("[]\n", encoding="ascii")
            with patches[0], patches[1], patches[2], patches[3], self.assertRaisesRegex(
                ValueError, "no-clobber"
            ):
                MODULE._prepare_delegated_argv(
                    fixture.argv(), declaration=fixture.declaration
                )

    def test_exclusive_json_publication_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            MODULE._write_new_json(path, {"version": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                MODULE._write_new_json(path, {"version": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_split_time_rematerialization_rejects_self_consistent_forged_prompt_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            prompts_path = cache / "prompts.json"
            summary_path = cache / "prompts-summary.json"
            genuine_prompts = b'[{"labels":{"action":"inspect"}}]\n'
            genuine_summary = b'{"prompt_bundle":"genuine"}\n'
            prompts_path.write_bytes(b'[{"labels":{"action":"forged-finalize"}}]\n')
            summary_path.write_bytes(genuine_summary)
            declaration = types.SimpleNamespace(
                cohort={
                    "cohorts": [
                        {"run_label": "a"},
                        {"run_label": "b"},
                    ]
                },
                campaign_paths=(root / "a.json", root / "b.json"),
                cohort_path=root / "cohort.json",
            )
            receipt = {
                "source_freeze_git_commit": "a" * 40,
                "invocation": {
                    "all_probeable": True,
                    "require_official_outcomes": False,
                },
                "outputs": {
                    "prompt_bundle": {
                        "sha256": hashlib.sha256(genuine_prompts).hexdigest()
                    },
                    "prompt_summary": {
                        "sha256": hashlib.sha256(genuine_summary).hexdigest()
                    },
                },
            }

            def rematerialize(*, delegated: list[str], **kwargs: Any) -> int:
                del kwargs
                Path(delegated[delegated.index("--output") + 1]).write_bytes(
                    genuine_prompts
                )
                Path(delegated[delegated.index("--summary") + 1]).write_bytes(
                    genuine_summary
                )
                return 0

            checker = types.SimpleNamespace(
                validate_materialized_bundle=lambda declaration, **kwargs: {
                    "prompt_bundle_sha256": MODULE.historical.sha256_file(
                        kwargs["prompts_path"]
                    ),
                    "summary_sha256": MODULE.historical.sha256_file(
                        kwargs["summary_path"]
                    ),
                }
            )
            with (
                mock.patch.object(MODULE, "V3_CACHE_ROOT", cache),
                mock.patch.object(MODULE, "V3_RUNS_ROOT", root / "runs"),
                mock.patch.object(
                    MODULE, "_run_historical_materialization", side_effect=rematerialize
                ),
                self.assertRaisesRegex(ValueError, "differs from"),
            ):
                MODULE.verify_frozen_materialization(
                    checker=checker,
                    declaration=declaration,
                    receipt=receipt,
                    prompts_path=prompts_path,
                    summary_path=summary_path,
                )

    def test_run_and_cache_symlink_attacks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InvocationFixture(Path(temporary))
            real_run = fixture.root / "real-run"
            real_run.mkdir()
            fixture.run_roots[0].rmdir()
            fixture.run_roots[0].symlink_to(real_run, target_is_directory=True)
            with (
                mock.patch.object(MODULE, "V3_ACTION_PROTOCOL_PATH", fixture.action),
                mock.patch.object(MODULE, "V3_TEMPLATE_PATH", fixture.template),
                mock.patch.object(MODULE, "V3_RUNS_ROOT", fixture.runs),
                mock.patch.object(MODULE, "V3_CACHE_ROOT", fixture.cache),
                self.assertRaisesRegex(ValueError, "symlink"),
            ):
                MODULE._prepare_delegated_argv(fixture.argv(), declaration=fixture.declaration)

            cache_target = fixture.root / "cache-target"
            cache_target.mkdir()
            fixture.cache.symlink_to(cache_target, target_is_directory=True)
            with mock.patch.object(MODULE, "V3_CACHE_ROOT", fixture.cache):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    MODULE._ensure_cache_root()

    def test_historical_or_symlink_output_targets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "v3-cache"
            cache.mkdir()
            historical_output = root / "historical-prompts.json"
            with mock.patch.object(MODULE, "V3_CACHE_ROOT", cache):
                with self.assertRaisesRegex(ValueError, "dedicated V3 cache"):
                    MODULE._validate_output_path(historical_output, "output")
                target = root / "target.json"
                target.write_text("[]\n", encoding="ascii")
                symlink = cache / "prompts.json"
                symlink.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "already exists|no-clobber"):
                    MODULE._validate_output_path(symlink, "output")

    def test_translation_changes_only_kind_and_action_protocol_path(self) -> None:
        manifest = {
            "schema_version": 1,
            "kind": MODULE.V3_COHORT_MANIFEST_KIND,
            "action_protocol": {"path": "configs/v3-actions.json", "sha256": "a" * 64},
            "chat_template": {"path": "configs/template.jinja", "sha256": "b" * 64},
            "cohorts": [{"id": "a"}, {"id": "b"}],
        }
        original = copy.deepcopy(manifest)
        translated = MODULE._translate_v3_manifest(
            manifest, action_protocol_logical_path="configs/v3-actions.json"
        )
        self.assertEqual(manifest, original)
        expected = copy.deepcopy(original)
        expected["kind"] = MODULE.HISTORICAL_COHORT_MANIFEST_KIND
        expected["action_protocol"]["path"] = MODULE.historical.DEFAULT_ACTION_PROTOCOL.relative_to(MODULE.historical.ROOT).as_posix()
        self.assertEqual(translated, expected)

    def test_all_probeable_patch_selects_every_candidate_and_restores(self) -> None:
        fake = types.SimpleNamespace(MAX_CHECKPOINTS=8)

        def selector(task: Mapping[str, list[object]], *, max_prompt_tokens: int, limit: int = 8) -> dict[str, object]:
            del max_prompt_tokens
            return {"selected_request_indices": list(range(1, len(task["captures"]) + 1))[:limit]}

        fake.select_probeable_requests = selector
        task = {"captures": [{} for _ in range(13)]}
        with mock.patch.object(MODULE, "historical", fake):
            with MODULE._all_probeable_patch():
                self.assertIsNone(fake.MAX_CHECKPOINTS)
                self.assertEqual(fake.select_probeable_requests(task, max_prompt_tokens=100)["selected_request_indices"], list(range(1, 14)))
        self.assertEqual(fake.MAX_CHECKPOINTS, 8)
        self.assertIs(fake.select_probeable_requests, selector)

    def test_direct_tool_assistant_uses_completed_history_prefix_and_honest_provenance(self) -> None:
        tasks, binding, tokenizer = self._prefix_fixture()
        original_require = MODULE.historical.require

        def legacy_mapper(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del kwargs
            MODULE.historical.require(False, MODULE.LEGACY_RENDERED_PREFIX_ERROR)
            MODULE.historical.require(False, MODULE.LEGACY_TOKEN_PREFIX_ERROR)
            return tasks, binding

        with mock.patch.object(
            MODULE.historical, "map_global_captures", legacy_mapper
        ):
            with MODULE._v3_capture_prefix_compatibility_patch():
                mapped, audited = MODULE.historical.map_global_captures(
                    tokenizer=tokenizer
                )
            self.assertIs(MODULE.historical.map_global_captures, legacy_mapper)
        self.assertIs(MODULE.historical.require, original_require)
        self.assertIs(mapped, tasks)
        validation = audited["v3_prefix_chain_validation"]
        self.assertEqual(validation["completed_history_fallback_count"], 1)
        self.assertEqual(
            validation["legacy_full_prompt_assertions_deferred"],
            {
                "rendered_prefix_failure_count": 1,
                "token_prefix_failure_count": 1,
            },
        )
        proxy = tasks[0]["request_count_provenance"]["proxy_capture_binding"]
        self.assertFalse(proxy["canonical_rendered_prefix_chain_verified"])
        self.assertFalse(proxy["canonical_token_prefix_chain_verified"])
        compatibility = proxy["canonical_prefix_compatibility"]
        self.assertTrue(compatibility["accepted_prefix_chain_verified"])
        self.assertEqual(
            compatibility["completed_history_fallback_current_global_request_indices"],
            [854],
        )
        self.assertEqual(
            audited["request_count_recoveries"][0]["proxy_capture_binding"],
            proxy,
        )

    def test_completed_history_prefix_rejects_real_history_drift_and_restores(self) -> None:
        tasks, binding, tokenizer = self._prefix_fixture(
            drift_completed_history=True
        )
        original_require = MODULE.historical.require

        def legacy_mapper(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del kwargs
            MODULE.historical.require(False, MODULE.LEGACY_RENDERED_PREFIX_ERROR)
            MODULE.historical.require(False, MODULE.LEGACY_TOKEN_PREFIX_ERROR)
            return tasks, binding

        with mock.patch.object(
            MODULE.historical, "map_global_captures", legacy_mapper
        ):
            with self.assertRaisesRegex(ValueError, "completed-history rendering"):
                with MODULE._v3_capture_prefix_compatibility_patch():
                    MODULE.historical.map_global_captures(tokenizer=tokenizer)
            self.assertIs(MODULE.historical.map_global_captures, legacy_mapper)
        self.assertIs(MODULE.historical.require, original_require)

    def test_completed_history_requires_same_token_prefix_in_both_requests(self) -> None:
        tasks, binding, _tokenizer = self._prefix_fixture()
        previous = tasks[0]["captures"][0]
        current = tasks[0]["captures"][1]
        suffix = MODULE.GENERATION_SUFFIX_BY_THINKING[True]
        completed_history = previous["rendered"][: -len(suffix)]

        class BoundaryMergingTokenizer:
            @staticmethod
            def encode(text: str, *, add_special_tokens: bool) -> list[int]:
                if add_special_tokens:
                    raise AssertionError("V3 prefix audit added special tokens")
                if text == completed_history:
                    return [7]
                if text == previous["rendered"]:
                    return [8, 1]
                if text == current["rendered"]:
                    return [7, 2]
                raise AssertionError("unexpected boundary-tokenizer input")

        tokenizer = BoundaryMergingTokenizer()
        previous["token_ids"] = tokenizer.encode(
            previous["rendered"], add_special_tokens=False
        )
        current["token_ids"] = tokenizer.encode(
            current["rendered"], add_special_tokens=False
        )

        def legacy_mapper(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del kwargs
            MODULE.historical.require(False, MODULE.LEGACY_RENDERED_PREFIX_ERROR)
            MODULE.historical.require(False, MODULE.LEGACY_TOKEN_PREFIX_ERROR)
            return tasks, binding

        with (
            mock.patch.object(
                MODULE.historical, "map_global_captures", legacy_mapper
            ),
            MODULE._v3_capture_prefix_compatibility_patch(),
            self.assertRaisesRegex(ValueError, "previous canonical tokens"),
        ):
            MODULE.historical.map_global_captures(tokenizer=tokenizer)

    def test_deferred_prefix_failure_counts_must_equal_post_audit(self) -> None:
        tasks, binding, tokenizer = self._prefix_fixture()

        def dishonest_mapper(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del kwargs
            return tasks, binding

        with (
            mock.patch.object(
                MODULE.historical, "map_global_captures", dishonest_mapper
            ),
            MODULE._v3_capture_prefix_compatibility_patch(),
            self.assertRaisesRegex(ValueError, "assertion count differs"),
        ):
            MODULE.historical.map_global_captures(tokenizer=tokenizer)

    def test_prefix_patch_accepts_zero_fallback_cohort_and_restores(self) -> None:
        tasks = [
            {
                "instance_id": "astropy__astropy-no-fallback",
                "request_count_provenance": {"recovered": False},
                "captures": [{"local_index": 1, "global_index": 1}],
            }
        ]
        binding: dict[str, Any] = {"request_count_recoveries": []}

        def exact_mapper(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del kwargs
            return tasks, binding

        with mock.patch.object(
            MODULE.historical, "map_global_captures", exact_mapper
        ):
            with MODULE._v3_capture_prefix_compatibility_patch():
                mapped, audited = MODULE.historical.map_global_captures(
                    tokenizer=object()
                )
            self.assertIs(MODULE.historical.map_global_captures, exact_mapper)
        self.assertIs(mapped, tasks)
        self.assertEqual(
            audited["v3_prefix_chain_validation"][
                "completed_history_fallback_count"
            ],
            0,
        )

    def test_legacy_prefix_messages_are_unique_in_pinned_historical_source(self) -> None:
        payload = (
            ROOT / "scripts/materialize_swe_behavioral_probes.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(payload.count(f'"{MODULE.LEGACY_RENDERED_PREFIX_ERROR}"'), 1)
        self.assertEqual(payload.count(f'"{MODULE.LEGACY_TOKEN_PREFIX_ERROR}"'), 1)

    def test_authenticated_image_hashes_reach_source_summary_prompts_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roots = (Path(temporary) / "a", Path(temporary) / "b")
            for root in roots:
                root.mkdir()
            hashes = {roots[0].resolve(): "a" * 64, roots[1].resolve(): "b" * 64}

            def builder(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                return [], {"run": str(kwargs["run_root"])}

            def combiner(sources: Any, *, cohort_manifest_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                del sources, cohort_manifest_sha256
                prompts = [
                    {
                        "id": f"prompt-{index}",
                        "metadata": {
                            "cohort": {"index": index},
                            "provenance": {
                                "combination": {},
                                "prompt_record_payload_sha256": "0" * 64,
                            },
                        },
                    }
                    for index in range(2)
                ]
                return prompts, {
                    "cohorts": [{"index": 0}, {"index": 1}],
                    "prompts": [{"id": "prompt-0"}, {"id": "prompt-1"}],
                }

            with (
                mock.patch.object(MODULE.historical, "build_behavioral_bundle", builder),
                mock.patch.object(MODULE.historical, "combine_behavioral_bundles", combiner),
                MODULE._image_provenance_patch(hashes),
            ):
                source_a = MODULE.historical.build_behavioral_bundle(run_root=roots[0])
                source_b = MODULE.historical.build_behavioral_bundle(run_root=roots[1])
                prompts, summary = MODULE.historical.combine_behavioral_bundles(
                    [
                        {"summary": source_a[1]},
                        {"summary": source_b[1]},
                    ],
                    cohort_manifest_sha256="c" * 64,
                )
            self.assertEqual(source_a[1]["source_image_manifest_sha256"], "a" * 64)
            self.assertEqual(source_b[1]["source_image_manifest_sha256"], "b" * 64)
            for index, prompt in enumerate(prompts):
                expected = ("a" if index == 0 else "b") * 64
                metadata = prompt["metadata"]
                self.assertEqual(metadata["cohort"]["source_image_manifest_sha256"], expected)
                self.assertEqual(metadata["provenance"]["combination"]["source_image_manifest_sha256"], expected)
                self.assertEqual(
                    metadata["provenance"]["prompt_record_payload_sha256"],
                    MODULE.historical._prompt_record_payload_sha256(prompt),
                )
                self.assertEqual(summary["cohorts"][index]["source_image_manifest_sha256"], expected)
                self.assertEqual(
                    summary["prompts"][index]["prompt_record_payload_sha256"],
                    metadata["provenance"]["prompt_record_payload_sha256"],
                )

    def test_checker_declaration_and_run_provenance_precede_delegation_then_bundle_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            rows = [
                {"run_label": "a"},
                {"run_label": "b"},
            ]
            for row in rows:
                (runs / row["run_label"]).mkdir(parents=True)
            declaration = types.SimpleNamespace(cohort={"cohorts": rows})
            events: list[str] = []
            checker = types.SimpleNamespace(
                V3_RUNS_ROOT=runs,
                V3_OUTPUT_ROOT=MODULE.V3_CACHE_ROOT,
                validate_declaration=lambda path: events.append("declaration") or declaration,
                validate_run_image_provenance=lambda value: events.append("run_images") or types.SimpleNamespace(image_manifest_sha256s=("a" * 64, "b" * 64)),
                validate_materialized_bundle=lambda value, **kwargs: events.append("bundle"),
                capture_clean_source_freeze=lambda: events.append("source_freeze") or "a" * 40,
                build_materialization_receipt=lambda value, **kwargs: events.append("receipt_build") or {"receipt": True},
                validate_materialization_receipt=lambda value, **kwargs: events.append("receipt_validate"),
            )
            with (
                mock.patch.object(MODULE, "V3_RUNS_ROOT", runs),
                mock.patch.object(MODULE, "_load_pinned_checker", return_value=checker),
                mock.patch.object(MODULE, "_prepare_delegated_argv", return_value=(["delegated"], root / "out.json", root / "summary.json")),
                mock.patch.object(MODULE, "_ensure_cache_root"),
                mock.patch.object(MODULE, "_validate_new_receipt_path", return_value=root / "receipt.json"),
                mock.patch.object(MODULE, "_run_historical_materialization", side_effect=lambda **kwargs: events.append("historical") or 0),
                mock.patch.object(
                    MODULE,
                    "_write_new_json",
                    side_effect=lambda path, value: (
                        events.append("receipt_write"),
                        path.write_text("{}\n", encoding="ascii"),
                    ),
                ),
            ):
                self.assertEqual(MODULE.main([]), 0)
            self.assertEqual(
                events,
                [
                    "declaration",
                    "source_freeze",
                    "historical",
                    "bundle",
                    "receipt_build",
                    "receipt_write",
                    "receipt_validate",
                ],
            )

    def test_checker_loader_rejects_changed_bytes_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checker = Path(temporary) / "checker.py"
            marker = Path(temporary) / "executed"
            checker.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n", encoding="ascii")
            with (
                mock.patch.object(MODULE, "CHECKER_PATH", checker),
                mock.patch.object(MODULE, "CHECKER_SHA256", "0" * 64),
                self.assertRaisesRegex(ValueError, "SHA-256 changed"),
            ):
                MODULE._load_pinned_checker()
            self.assertFalse(marker.exists())

    def test_preloaded_mutated_c1_is_not_used_by_rematerialization_verifier(self) -> None:
        public_names = MODULE.PINNED_HISTORICAL_LOAD_ORDER
        missing = object()
        original_modules = {
            name: sys.modules.get(name, missing) for name in public_names
        }
        poison_calls: list[str] = []
        try:
            for name in reversed(public_names):
                sys.modules.pop(name, None)
            preloaded_c1 = importlib.import_module(
                "materialize_swe_multitask_c1_probes"
            )

            def poisoned_render_request(*args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                poison_calls.append("called")
                raise AssertionError("preloaded mutated C1 helper was used")

            with mock.patch.object(
                preloaded_c1, "render_request", poisoned_render_request
            ):
                isolated_spec = importlib.util.spec_from_file_location(
                    "_test_isolated_materialize_swe_state_interpreter_v3_probes",
                    ROOT / "scripts/materialize_swe_state_interpreter_v3_probes.py",
                )
                assert isolated_spec and isolated_spec.loader
                isolated = importlib.util.module_from_spec(isolated_spec)
                isolated_spec.loader.exec_module(isolated)

                self.assertIs(
                    sys.modules["materialize_swe_multitask_c1_probes"],
                    preloaded_c1,
                )
                self.assertIs(preloaded_c1.render_request, poisoned_render_request)
                self.assertIsNot(isolated.historical.C1, preloaded_c1)
                self.assertIs(isolated.historical.C1.C0, isolated.historical.C0)
                self.assertIsNot(
                    isolated.historical.C1.render_request,
                    poisoned_render_request,
                )

                class FakeTokenizer:
                    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
                        del args, kwargs
                        return "<|im_start|>assistant\n<think>\n"

                    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                        self.assert_encode_inputs(text, add_special_tokens)
                        return [7, 11]

                    @staticmethod
                    def assert_encode_inputs(text: str, add_special_tokens: bool) -> None:
                        if text != "<|im_start|>assistant\n<think>\n" or add_special_tokens:
                            raise AssertionError("fresh renderer received unexpected inputs")

                request = {
                    "messages": [],
                    "tools": [],
                    "chat_template_kwargs": {"enable_thinking": True},
                }
                expected_prompts = [
                    {
                        "normalized_count": 0,
                        "rendered": "<|im_start|>assistant\n<think>\n",
                        "token_ids": [7, 11],
                    }
                ]
                expected_summary = {"isolated_private_closure": True}
                render_json = lambda value: (
                    isolated.json.dumps(
                        value,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("ascii")

                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    cache = root / "cache"
                    runs = root / "runs"
                    cache.mkdir()
                    for label in ("a", "b"):
                        (runs / label).mkdir(parents=True)
                    prompts_path = cache / "prompts.json"
                    summary_path = cache / "prompts-summary.json"
                    prompts_path.write_bytes(render_json(expected_prompts))
                    summary_path.write_bytes(render_json(expected_summary))
                    declaration = types.SimpleNamespace(
                        cohort={
                            "cohorts": [
                                {"run_label": "a"},
                                {"run_label": "b"},
                            ]
                        },
                        campaign_paths=(root / "campaign-a.json", root / "campaign-b.json"),
                        cohort_path=root / "cohort.json",
                    )

                    def fake_historical_main(argv: list[str]) -> int:
                        rendered, token_ids, normalized_count, _messages = (
                            isolated.historical.C1.render_request(
                                FakeTokenizer(), request=request, template=""
                            )
                        )
                        prompts = [
                            {
                                "normalized_count": normalized_count,
                                "rendered": rendered,
                                "token_ids": token_ids,
                            }
                        ]
                        isolated.historical.atomic_write_json(
                            Path(argv[argv.index("--output") + 1]), prompts
                        )
                        isolated.historical.atomic_write_json(
                            Path(argv[argv.index("--summary") + 1]),
                            expected_summary,
                        )
                        return 0

                    checker = types.SimpleNamespace(
                        validate_run_image_provenance=lambda _declaration: types.SimpleNamespace(
                            image_manifest_sha256s=("a" * 64, "b" * 64)
                        ),
                        validate_materialized_bundle=lambda _declaration, **kwargs: {
                            "prompt_bundle_sha256": isolated.historical.sha256_file(
                                kwargs["prompts_path"]
                            ),
                            "summary_sha256": isolated.historical.sha256_file(
                                kwargs["summary_path"]
                            ),
                        },
                    )
                    receipt = {
                        "source_freeze_git_commit": "c" * 40,
                        "invocation": {
                            "all_probeable": True,
                            "require_official_outcomes": False,
                        },
                        "outputs": {
                            "prompt_bundle": {
                                "sha256": isolated.historical.sha256_file(
                                    prompts_path
                                )
                            },
                            "prompt_summary": {
                                "sha256": isolated.historical.sha256_file(
                                    summary_path
                                )
                            },
                        },
                    }
                    with (
                        mock.patch.object(isolated, "V3_CACHE_ROOT", cache),
                        mock.patch.object(isolated, "V3_RUNS_ROOT", runs),
                        mock.patch.object(
                            isolated.historical,
                            "main",
                            side_effect=fake_historical_main,
                        ),
                    ):
                        result = isolated.verify_frozen_materialization(
                            checker=checker,
                            declaration=declaration,
                            receipt=receipt,
                            prompts_path=prompts_path,
                            summary_path=summary_path,
                        )
                self.assertTrue(result["exact_match"])
                self.assertEqual(poison_calls, [])
        finally:
            for name, original in original_modules.items():
                if original is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

    def test_historical_materializer_is_not_modified_on_disk(self) -> None:
        historical_path = ROOT / "scripts/materialize_swe_behavioral_probes.py"
        before = hashlib.sha256(historical_path.read_bytes()).hexdigest()
        with MODULE._all_probeable_patch():
            pass
        with MODULE._v3_capture_prefix_compatibility_patch():
            pass
        after = hashlib.sha256(historical_path.read_bytes()).hexdigest()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
