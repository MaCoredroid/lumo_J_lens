from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "swe_task_state_v4_counterfactual_stage_a_runtime.py"
SPEC = importlib.util.spec_from_file_location("stage_a_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


class StageARuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = RUNTIME.load_and_validate_config()
        cls.source_paths = RUNTIME._validate_source_bindings(cls.config)
        (
            cls.capture_rows,
            cls.generation_rows,
            cls.prompt_summary,
        ) = RUNTIME.load_and_validate_prompt_bundles(cls.config, cls.source_paths)

    def test_contract_is_stage_a_prospective_and_all_internal_claims_stay_false(self) -> None:
        chronology = self.config["chronology"]
        self.assertFalse(chronology["gpu_execution_authorized_by_this_artifact"])
        self.assertFalse(chronology["stage_b_materialization_or_runtime_authorized"])
        self.assertFalse(chronology["condition_join_or_effect_analysis_authorized"])
        claims = self.config["claim_scope"]
        self.assertTrue(claims["cpu_preflight_contract_implemented"])
        self.assertTrue(
            all(
                value is False
                for key, value in claims.items()
                if key != "cpu_preflight_contract_implemented"
            )
        )
        for name in (
            "private_chain_of_thought_reconstructed",
            "cot_or_cot_like_decoding_established",
            "subjective_confidence_inferred",
            "subjective_doubt_inferred",
            "experienced_stress_inferred",
            "experienced_emotion_inferred",
            "causal_affect_or_state_effect_established",
        ):
            self.assertIs(claims[name], False)

    def test_exact_model_lens_tokenization_and_runtime_are_bound(self) -> None:
        model = self.config["model"]
        self.assertEqual(model["repo_id"], "nvidia/Qwen3.6-27B-NVFP4")
        self.assertEqual(model["revision"], "0893e1606ff3d5f97a441f405d5fc541a6bdf404")
        self.assertEqual(
            model["snapshot_tree_sha256"],
            "9e81d31df546344ad68696c3cfd6cadce4ad6d3952710a3dc2021c1c2d42414d",
        )
        self.assertEqual((model["snapshot_file_count"], model["snapshot_size_bytes"]), (17, 21941623844))
        self.assertEqual(model["vocabulary_size"], 248320)
        self.assertEqual(self.config["public_j_lens"]["sha256"], "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1")
        self.assertFalse(self.config["tokenization"]["retokenization_or_text_reconstruction_for_forward"])
        self.assertFalse(self.config["tokenization"]["input_truncation"])
        runtime = self.config["runtime"]
        self.assertEqual(runtime["max_model_len"], 49152)
        self.assertEqual(runtime["max_num_batched_tokens"], 4096)
        self.assertEqual(runtime["mamba_block_size"], 1024)
        self.assertEqual(runtime["kv_cache_dtype"], "fp8_e4m3")

    def test_prompt_bundles_are_opaque_exact_and_fit_without_truncation(self) -> None:
        self.assertEqual(len(self.capture_rows), 500)
        self.assertEqual(len(self.generation_rows), 240)
        self.assertEqual(self.prompt_summary["capture"]["minimum_token_count"], 13097)
        self.assertEqual(self.prompt_summary["capture"]["maximum_token_count"], 47413)
        self.assertEqual(self.prompt_summary["generation"]["minimum_token_count"], 13397)
        self.assertEqual(self.prompt_summary["generation"]["maximum_token_count"], 47413)
        self.assertEqual(self.prompt_summary["capture_only_row_count"], 260)
        self.assertTrue(self.prompt_summary["generation_rows_are_exact_capture_rows"])
        self.assertTrue(self.prompt_summary["no_input_truncation_capacity_passed"])
        capture = {row["id"]: row["token_ids"] for row in self.capture_rows}
        self.assertTrue(
            all(
                row["id"] in capture and row["token_ids"] == capture[row["id"]]
                for row in self.generation_rows
            )
        )
        self.assertTrue(
            all(set(row) == {"id", "token_ids"} for row in self.capture_rows)
        )
        self.assertTrue(
            all(RUNTIME.OPAQUE_ID_RE.fullmatch(row["id"]) for row in self.capture_rows)
        )

    def test_capture_geometry_and_lineage_are_full_not_schema_only(self) -> None:
        capture = self.config["capture"]
        self.assertEqual(capture["layers"], list(range(24, 48)))
        self.assertEqual(capture["positions_argument"], [-1])
        self.assertEqual(capture["position_semantics"], "each_already_truncated_causal_prefix_tail_only")
        self.assertEqual(capture["tensor_shape_per_boundary"], [24, 5120])
        self.assertEqual(capture["tensor_storage_dtype"], "little_endian_float32")
        self.assertEqual(capture["expected_shard_count"], 500)
        self.assertTrue(capture["require_all_64_layer_residual_manifest_per_boundary"])
        self.assertTrue(capture["require_reference_capture_residual_manifest_exact_equality"])
        self.assertTrue(self.config["postrun_authentication"]["schema_only_capture_or_generation_bundle_must_fail"])

    def test_output_schemas_are_exact_hash_bound_and_surface_contract_matches(self) -> None:
        paths = RUNTIME._validate_schema_bindings(self.config)
        self.assertEqual(
            set(paths),
            {
                "capture_manifest_schema",
                "generation_manifest_schema",
                "visible_completion_bundle_schema",
            },
        )
        surface = RUNTIME.load_json(paths["visible_completion_bundle_schema"])
        properties = surface["properties"]
        self.assertEqual(
            properties["kind"]["const"],
            "swe_task_state_v4_counterfactual_stage_a_visible_completion_bundle",
        )
        self.assertEqual(
            properties["status"]["const"],
            "stage_a_generation_complete_condition_key_not_joined",
        )
        self.assertEqual(
            properties["surface_policy"]["const"],
            "exact_experiment_completion_string_only_no_prompt_prefix_no_external_task_outcome",
        )
        self.assertEqual(properties["records"]["minItems"], 240)
        self.assertEqual(properties["records"]["maxItems"], 240)
        self.assertIs(properties["stage_b_records_present"]["const"], False)

    def test_runtime_has_no_gpu_execution_subcommand_or_model_runtime_import(self) -> None:
        parser = RUNTIME.build_parser()
        subparser_action = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            set(subparser_action.choices),
            {"preflight", "verify-capture", "verify-generation"},
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("from vllm import", source)
        self.assertNotIn("import vllm", source)
        self.assertNotIn("LLM(", source)
        self.assertNotIn("llm.generate(", source)

    def test_forbidden_paths_fail_before_resolve_or_read(self) -> None:
        for name in (
            "reserved/data.json",
            "validation/data.json",
            "split-manifest.json",
            "stage-a-condition-key.json",
            "stage-b-prompts.json",
            "semantic-answers.json",
            "control-expectations.json",
        ):
            with self.subTest(name=name):
                path = Path("/definitely/not/present") / name
                with mock.patch.object(Path, "resolve", side_effect=AssertionError("resolve called")):
                    with self.assertRaises(RUNTIME.StageARuntimeError):
                        RUNTIME.lexical_path_preflight((path,))

    def test_hash_only_manifest_is_never_parsed_and_runtime_reads_only_opaque_bundles(self) -> None:
        loaded: list[Path] = []
        original = RUNTIME.load_json

        def recording_load(path: Path):
            loaded.append(Path(path))
            return original(path)

        with mock.patch.object(RUNTIME, "load_json", side_effect=recording_load):
            RUNTIME._validate_source_bindings(self.config)
            RUNTIME._validate_schema_bindings(self.config)
            RUNTIME.load_and_validate_prompt_bundles(self.config, self.source_paths)
        loaded_names = {path.name for path in loaded}
        self.assertNotIn("materialization-manifest.json", loaded_names)
        self.assertNotIn("split-manifest.json", loaded_names)
        self.assertNotIn("stage-a-condition-key.json", loaded_names)
        self.assertNotIn("stage-a-condition-key", " ".join(str(path) for path in loaded))
        self.assertIn("stage-a-capture-prompts.json", loaded_names)
        self.assertIn("stage-a-generation-prompts.json", loaded_names)

    def test_config_mutation_and_prompt_metadata_injection_fail_closed(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["runtime"]["max_model_len"] = 65536
        with self.assertRaisesRegex(RUNTIME.StageARuntimeError, "config object changed"):
            RUNTIME.validate_config(mutated)
        rows = copy.deepcopy(self.capture_rows)
        rows[0]["condition_id"] = "forbidden"
        with self.assertRaisesRegex(RUNTIME.StageARuntimeError, "keys changed"):
            RUNTIME.validate_prompt_bundle(
                rows,
                spec=self.config["frozen_inputs"]["stage_a_capture_prompts"],
                label="capture",
                vocabulary_size=self.config["model"]["vocabulary_size"],
            )

    def test_schema_shaped_capture_without_actual_shards_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jlens-stage-a-schema-only-") as directory:
            root = Path(directory)
            reference = root / "reference-report.json"
            reference.write_text("{}\n", encoding="utf-8")
            sha = "1" * 64
            boundaries = []
            for index, prompt in enumerate(self.capture_rows):
                position = len(prompt["token_ids"]) - 1
                residual = {
                    "sha256": sha,
                    "tensor_count": 64,
                    "logical_bytes": 64 * 5120 * 4,
                    "token_position": position,
                }
                boundaries.append(
                    {
                        "index": index,
                        "prompt_id": prompt["id"],
                        "prompt_token_ids_sha256": RUNTIME.token_ids_sha256(prompt["token_ids"]),
                        "prompt_token_count": len(prompt["token_ids"]),
                        "token_position": position,
                        "reference_residual_manifest": residual,
                        "capture_residual_manifest": dict(residual),
                        "reference_residual_manifest_equal": True,
                        "shard": {
                            "path": f"shards/boundary-{index:06d}.safetensors",
                            "sha256": "2" * 64,
                            "size_bytes": 983041,
                            "tensor_keys": ["public_j_state", "raw_residual"],
                            "shape": [24, 5120],
                            "dtype": "little-endian-float32",
                            "raw_residual_logical_sha256": "3" * 64,
                            "public_j_state_logical_sha256": "4" * 64,
                            "reload_verified": True,
                        },
                        "forward_canary": {
                            "generated_token_id": 1,
                            "final_model_top1_matches_greedy": True,
                            "final_norm_reconstruction_within_tolerance": True,
                            "final_logits_reconstruction_within_tolerance": True,
                        },
                        "capture_valid": True,
                    }
                )
            model = self.config["model"]
            manifest = {
                "schema_version": 1,
                "kind": RUNTIME.CAPTURE_KIND,
                "status": "passed_authenticated_target_model_capture",
                "runtime_config_sha256": RUNTIME.CONFIG_SHA256,
                "capture_prompt_bundle": RUNTIME._expected_prompt_binding(
                    self.config, "stage_a_capture_prompts"
                ),
                "implementation": RUNTIME._expected_implementation(self.config),
                "model": {
                    "repo_id": model["repo_id"],
                    "revision": model["revision"],
                    "snapshot_tree_sha256_before": model["snapshot_tree_sha256"],
                    "snapshot_tree_sha256_after": model["snapshot_tree_sha256"],
                    "snapshot_file_count": model["snapshot_file_count"],
                    "snapshot_size_bytes": model["snapshot_size_bytes"],
                    "checkpoint_validated_before_model_load": True,
                    "checkpoint_validated_after_capture": True,
                },
                "lens": {
                    key: self.config["public_j_lens"][key]
                    for key in ("repo_id", "revision", "filename", "sha256", "size_bytes")
                },
                "runtime": {
                    "environment": {
                        "python_version": self.config["environment"]["python_version"],
                        "packages": self.config["environment"]["packages"],
                        "required_environment": self.config["environment"]["required_environment"],
                    },
                    "parameters": self.config["runtime"],
                    "gpu": self.config["environment"]["gpu"],
                    "started_at": "2026-07-20T00:00:00+00:00",
                    "completed_at": "2026-07-20T01:00:00+00:00",
                },
                "execution_order": {
                    "reference_completed_before_capture": True,
                    "reference_report": {
                        "path": reference.name,
                        "sha256": RUNTIME.sha256_file(reference),
                        "size_bytes": reference.stat().st_size,
                    },
                    "reference_and_capture_residual_manifests_equal": True,
                    "condition_key_read": False,
                    "split_manifest_read": False,
                    "stage_b_read": False,
                    "semantic_answers_read": False,
                    "join_conditions_read": False,
                },
                "boundaries": boundaries,
                "aggregate": {
                    "boundary_count": 500,
                    "all_capture_valid": True,
                    "all_shards_reload_verified": True,
                    "all_forward_canaries_passed": True,
                    "boundary_contract_sha256": "5" * 64,
                    "shard_set_sha256": "6" * 64,
                },
                "claim_scope": RUNTIME.OUTPUT_FALSE_CLAIMS,
                "reserved_validation_access_authorized": False,
            }
            manifest_path = root / "stage-a-capture-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RUNTIME.StageARuntimeError, "capture shard is missing"):
                RUNTIME.verify_capture_artifacts(
                    config=self.config,
                    capture_rows=self.capture_rows,
                    manifest_path=manifest_path,
                    capture_root=root,
                    verify_tensor_values=False,
                )

    def test_exact_local_snapshot_and_lens_preflight_without_model_load(self) -> None:
        snapshot, inventory = RUNTIME.resolve_and_verify_model(self.config)
        lens = RUNTIME.resolve_and_verify_lens(self.config)
        self.assertTrue(snapshot.is_dir())
        self.assertEqual(inventory["tree_sha256"], self.config["model"]["snapshot_tree_sha256"])
        self.assertEqual(inventory["file_count"], 17)
        self.assertEqual(inventory["size_bytes"], 21941623844)
        self.assertTrue(lens.resolve(strict=True).is_file())

    def test_environment_verifier_is_metadata_only_and_exact(self) -> None:
        spec = self.config["environment"]
        required = dict(spec["required_environment"])
        required["CUDA_HOME"] = str((ROOT / required["CUDA_HOME"]).resolve())
        with mock.patch.object(RUNTIME.sys, "executable", str((ROOT / spec["python_executable_repo_relative"]).resolve())):
            with mock.patch.dict(os.environ, required, clear=False):
                with mock.patch.object(RUNTIME, "_nvidia_smi", return_value=spec["gpu"]):
                    observed = RUNTIME.verify_runtime_environment(self.config)
        self.assertTrue(observed["exact_runtime_environment_passed"])
        self.assertEqual(observed["packages"], spec["packages"])


if __name__ == "__main__":
    unittest.main()
