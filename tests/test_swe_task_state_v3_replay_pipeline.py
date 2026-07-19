from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "swe_task_state_v3_replay_pipeline.py"


def load_module():
    name = "swe_task_state_v3_replay_pipeline_test"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


M = load_module()


def protocol() -> dict:
    return {
        "schema_version": 1,
        "id": "swe-task-state-interpreter-v3",
        "pins": {
            "v3_materializer_sha256": M.sha256_file(M.DEFAULT_MATERIALIZER),
            "historical_materializer_sha256": M.sha256_file(
                M.DEFAULT_HISTORICAL_MATERIALIZER
            ),
            "replay_pipeline_sha256": M.sha256_file(SCRIPT),
            "replay_shell_wrapper_sha256": M.sha256_file(M.REPLAY_SHELL_WRAPPER),
            "model": dict(M.EXPECTED_MODEL_PIN),
            "public_lens": {
                **M.EXPECTED_PUBLIC_LENS_PIN,
                "stored_dtype": "float16",
                "fit_precision": "unpublished",
            },
            "replay_runtime": {
                "target_checkpoint_quantization": "ModelOpt NVFP4 weights with FP8 runtime state",
                **M.EXPECTED_REPLAY_RUNTIME_PIN,
            },
        },
        "feature_contract": {"source_layers": list(M.LAYERS)},
        "bounded_memory_replay_contract": {
            "master_prompt_order": "exact_materialized_N60_prompt_bundle_order",
            "maximum_tasks_per_replay_chunk_inclusive": 15,
            "chunk_partition": "task_disjoint_and_collectively_exhaustive_in_master_task_order",
            "experiment_coverage": "every_master_prompt_experiment_exactly_once_in_master_prompt_order",
            "combined_report_schema_version": 3,
            "experiment_payload_merge": "semantic_and_canonical_payload_unchanged_with_each_per_record_canonical_sha256_preserved",
            "score_recomputation_or_averaging": "forbidden",
        },
    }


def prompt(task_id: str, task_index: int, prompt_index: int) -> dict:
    return {
        "id": f"{task_id}-prompt-{prompt_index}",
        "text": f"prompt text {task_id} {prompt_index}",
        "token_ids": [100 + task_index, 200 + prompt_index],
        "score_token_ids": [1000 + task_index, 2000 + prompt_index],
        "metadata": {
            "task": {"instance_id": task_id},
            "selection": {"task_request_index": prompt_index + 1},
            "provenance": {"prompt_record_payload_sha256": f"{task_index + 1:064x}"},
        },
    }


def experiment(source: dict, *, top1: bool = True, reconstruction: bool = True) -> dict:
    final_position = len(source["token_ids"]) - 1
    scored_ids = list(source["score_token_ids"])
    scored_tokens = [f" token-{token_id}" for token_id in scored_ids]
    top_token_id = 7 if top1 else 8

    def readout(*, fidelity: bool) -> dict:
        value = {
            "token_ids": [top_token_id, *range(20, 29)],
            "scores": [float(10 - index) for index in range(10)],
            "target_token_id": 7,
            "target_score": 1.0,
            "target_logprob": -1.0,
            "target_rank": 1,
            "scored_tokens": [
                {
                    "token_id": token_id,
                    "score": 0.5,
                    "logprob": -2.0,
                    "rank": index + 1,
                    "token": token,
                }
                for index, (token_id, token) in enumerate(
                    zip(scored_ids, scored_tokens, strict=True)
                )
            ],
            "tokens": [f"top-{index}" for index in range(10)],
            "target_token": "x",
        }
        if fidelity:
            value["final_distribution_fidelity"] = {
                "reference": "captured_block_63_final_model",
                "kl_final_to_readout": 0.1,
                "kl_readout_to_final": 0.1,
                "jensen_shannon_divergence": 0.1,
                "total_variation_distance": 0.1,
                "top1_matches_final": True,
                "top_k": 5,
                "top_k_overlap_count": 5,
                "top_k_overlap_fraction": 1.0,
            }
        return value

    return {
        "id": source["id"],
        "prompt": source["text"],
        "prompt_token_ids": list(source["token_ids"]),
        "prompt_tokens": ["a", "b"],
        "positions_requested": [-1],
        "positions_resolved": [final_position],
        "capture_positions_resolved": [final_position],
        "final_validation_position": final_position,
        "position_tokens": ["b"],
        "generated_token_id": 7,
        "generated_token": "x",
        "generated_text": "x",
        "generation_seconds": 0.1,
        "final_layer_top1_matches_greedy": top1,
        "scored_vocabulary": {"token_ids": scored_ids, "tokens": scored_tokens},
        "layers": [
            {
                "layer": layer,
                "layer_type": "linear_attention",
                "positions": [
                    {
                        "capture_index": 0,
                        "token_position": final_position,
                        "logit_lens": readout(fidelity=True),
                        "jacobian_lens": readout(fidelity=True),
                    }
                ],
            }
            for layer in M.LAYERS
        ],
        "final_model_readout": [readout(fidelity=True)],
        "captured_final_model_readout": [readout(fidelity=False)],
        "final_norm_reconstruction": {
            "max_abs_error": 0.0 if reconstruction else 0.2,
            "rms_error": 0.0 if reconstruction else 0.02,
            "reference_rms": 1.0,
            "relative_rms_error": 0.0 if reconstruction else 0.02,
            "max_abs_tolerance": 0.125,
            "rms_tolerance": 0.006,
            "within_tolerance": reconstruction,
        },
        "final_logits_reconstruction": {
            "max_abs_error": 0.0 if reconstruction else 0.2,
            "max_abs_tolerance": 0.0625,
            "rms_error": 0.0 if reconstruction else 0.02,
            "rms_tolerance": 0.01,
            "top_k_prefix": 5,
            "top_k_prefix_token_ids_match": reconstruction,
            "within_tolerance": reconstruction,
        },
        "cuda_max_memory_allocated_bytes": 0,
        "cuda_max_memory_reserved_bytes": 0,
        "readout_seconds": 0.2,
        "residual_capture_manifest": {
            "algorithm": (
                "SHA-256 over length-prefixed canonical layer/shape/dtype/"
                "token-position/byte-count headers and logical row-major FP32 bytes"
            ),
            "sha256": "a" * 64,
            "tensor_count": 64,
            "logical_bytes": 64 * 5120 * 4,
            "token_positions": [final_position],
        },
        "metadata": source["metadata"],
    }


def report_for_prompts(
    prompts: list[dict],
    *,
    fail_index: int | None = None,
) -> dict:
    experiments = [
        experiment(
            source,
            top1=index != fail_index,
            reconstruction=index != fail_index,
        )
        for index, source in enumerate(prompts)
    ]
    union_ids: list[int] = []
    union_tokens: list[str] = []
    for row in experiments:
        for token_id, token in zip(
            row["scored_vocabulary"]["token_ids"],
            row["scored_vocabulary"]["tokens"],
            strict=True,
        ):
            if token_id not in union_ids:
                union_ids.append(token_id)
                union_tokens.append(token)
    passed = fail_index is None
    return {
        "schema_version": 3,
        "score_encoding": "unrounded-float32",
        "status": "passed" if passed else "failed",
        "started_at": "2026-07-19T00:00:00+00:00",
        "completed_at": "2026-07-19T00:01:00+00:00",
        "elapsed_seconds": 60.0,
        "host": {"platform": "test", "python": "test", "gpu": {}, "packages": {}},
        "model": {**M.EXPECTED_MODEL_PIN, "extra": "preserved"},
        "lens": {**M.EXPECTED_PUBLIC_LENS_PIN, "extra": "preserved"},
        "runtime": {**M.EXPECTED_REPORT_RUNTIME, "model_load_seconds": 1.0},
        "scored_vocabulary": {
            "token_ids": [],
            "tokens": [],
            "scope": "global_plus_per_experiment",
            "union_token_ids": union_ids,
            "union_tokens": union_tokens,
        },
        "assertions": {
            "lens_hash_matches": True,
            "lens_metadata_matches": True,
            "model_architecture_matches": True,
            "all_final_layer_top1_match_greedy": passed,
            "all_final_adapter_reconstructions_within_tolerance": passed,
        },
        "experiments": experiments,
    }


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Workspace:
    def __init__(self, base: Path, *, task_count: int = 5, prompts_per_task: int = 2, chunk_size: int = 2):
        self.base = base
        self.parent = base / "artifacts"
        self.parent.mkdir()
        self.prompts_path = self.parent / "master.json"
        self.task_ids = [f"repo__task-{index}" for index in range(task_count)]
        self.prompts = [
            prompt(task_id, task_index, prompt_index)
            for task_index, task_id in enumerate(self.task_ids)
            for prompt_index in range(prompts_per_task)
        ]
        dump(self.prompts_path, self.prompts)
        self.receipt_path = self.parent / "materialization-receipt.json"
        dump(self.receipt_path, {"kind": "synthetic-materialization-receipt"})
        self.inputs = {
            "master_prompt_bundle": {"path": "synthetic/master.json", "sha256": M.sha256_file(self.prompts_path)},
            "master_prompt_summary": {"path": "synthetic/summary.json", "sha256": "1" * 64},
            "cohort_manifest": {"path": "synthetic/cohort.json", "sha256": "2" * 64},
            "interpreter_protocol": {"path": "synthetic/protocol.json", "sha256": "3" * 64},
            "action_protocol": {"path": "synthetic/action.json", "sha256": "4" * 64},
            "materialized_bundle_checker": {"path": "synthetic/checker.py", "sha256": "5" * 64},
            "v3_materializer": {"path": "synthetic/v3-materializer.py", "sha256": "a" * 64},
            "historical_materializer": {"path": "synthetic/historical-materializer.py", "sha256": "b" * 64},
            "materialization_receipt": {
                "path": "synthetic/materialization-receipt.json",
                "sha256": M.sha256_file(self.receipt_path),
            },
            "materialization_freeze": {
                "source_freeze_git_commit": "c" * 40,
                "data_freeze_git_commit": "d" * 40,
                "exact_child_receipt_only_commit_validated": True,
                "deterministic_rematerialization_required_before_split": True,
            },
            "replay_pipeline": {"path": "scripts/swe_task_state_v3_replay_pipeline.py", "sha256": "8" * 64},
            "replay_shell_wrapper": {"path": "scripts/run_swe_task_state_v3_replay.sh", "sha256": "9" * 64},
            "jlens_shell_runner": {"path": "scripts/run_jlens_nvfp4.sh", "sha256": "6" * 64},
            "jlens_python_runner": {"path": "scripts/run_jlens_nvfp4.py", "sha256": "7" * 64},
        }
        self.replay_root = self.parent / "replay"
        self.manifest_path = M.create_chunk_manifest(
            prompts_path=self.prompts_path,
            replay_root=self.replay_root,
            task_ids=self.task_ids,
            inputs=self.inputs,
            max_tasks_per_chunk=chunk_size,
        )
        self.partition = self.validate()

    def validate(self):
        return M.validate_chunk_manifest(
            manifest_path=self.manifest_path,
            replay_root=self.replay_root,
            prompts_path=self.prompts_path,
            expected_inputs=self.inputs,
            expected_task_ids=self.task_ids,
        )

    def write_complete_run(self, *, fail_chunk: int | None = None):
        reports = self.replay_root / "reports"
        logs = self.replay_root / "logs"
        reports.mkdir(exist_ok=True)
        logs.mkdir(exist_ok=True)
        run = M._run_manifest_base(self.partition)
        for chunk in self.partition.manifest["chunks"]:
            index = chunk["index"]
            chunk_path = self.replay_root / chunk["prompts_path"]
            chunk_prompts = list(M.iter_strict_json_array(chunk_path, "test chunk"))
            report_path = reports / f"{chunk['id']}-report.json"
            report = report_for_prompts(
                chunk_prompts,
                fail_index=0 if index == fail_chunk else None,
            )
            dump(report_path, report)
            log_path = logs / f"{chunk['id']}.stderr.log"
            log_path.write_text(f"chunk {index}\n", encoding="utf-8")
            exit_code = 1 if index == fail_chunk else 0
            audit = M.validate_report_file(
                report_path=report_path,
                prompts_path=chunk_path,
                prompt_records=M._chunk_prompt_records(chunk),
                protocol=protocol(),
                expected_exit_code=exit_code,
            )
            run["chunks"].append(
                {
                    "index": index,
                    "id": chunk["id"],
                    "prompts": {"path": chunk["prompts_path"], "sha256": chunk["prompts_sha256"]},
                    "report": {"path": f"reports/{report_path.name}", "sha256": audit.sha256},
                    "stderr_log": {"path": f"logs/{log_path.name}", "sha256": M.sha256_file(log_path)},
                    "exit_code": exit_code,
                    "accepted_terminal": True,
                    "report_status": audit.status,
                    "report_assertions": dict(audit.assertions),
                    "experiment_count": audit.experiment_count,
                    "master_task_range_inclusive": chunk["master_task_range_inclusive"],
                    "master_prompt_range_inclusive": chunk["master_prompt_range_inclusive"],
                    "experiment_payload_sha256s": list(audit.experiment_payload_sha256s),
                    "experiment_payload_sha256s_sha256": M.canonical_sha256(list(audit.experiment_payload_sha256s)),
                }
            )
        run["completed_chunk_count"] = len(run["chunks"])
        run["status"] = "complete"
        dump(self.replay_root / M.RUN_MANIFEST_NAME, run)
        return run


class ReplayPipelineTests(unittest.TestCase):
    def test_stream_split_whole_tasks_bounded_exhaustive_and_value_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=31, prompts_per_task=2, chunk_size=15)
            manifest = workspace.partition.manifest
            self.assertEqual([chunk["task_count"] for chunk in manifest["chunks"]], [15, 15, 1])
            self.assertEqual([chunk["prompt_count"] for chunk in manifest["chunks"]], [30, 30, 2])
            combined = []
            for chunk in manifest["chunks"]:
                combined.extend(M.iter_strict_json_array(workspace.replay_root / chunk["prompts_path"], "chunk"))
            self.assertEqual(combined, workspace.prompts)
            self.assertEqual(manifest["ordered_task_ids"], workspace.task_ids)

    def test_split_rejects_more_than_frozen_task_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            parent = base / "artifacts"
            parent.mkdir()
            prompts_path = parent / "master.json"
            dump(prompts_path, [prompt("repo__task-0", 0, 0)])
            with self.assertRaisesRegex(M.ReplayValidationError, "exceeds frozen maximum"):
                M.create_chunk_manifest(
                    prompts_path=prompts_path,
                    replay_root=parent / "replay",
                    task_ids=["repo__task-0"],
                    inputs={},
                    max_tasks_per_chunk=16,
                )

    def test_tampered_chunk_rejected_even_if_attacker_updates_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary))
            manifest = json.loads(workspace.manifest_path.read_text())
            chunk = manifest["chunks"][0]
            path = workspace.replay_root / chunk["prompts_path"]
            rows = json.loads(path.read_text())
            rows[0]["text"] = "tampered"
            dump(path, rows)
            chunk["prompts_sha256"] = M.sha256_file(path)
            chunk["prompts"][0]["canonical_sha256"] = M.canonical_sha256(rows[0])
            chunk["tasks"][0]["ordered_prompt_canonical_sha256s_sha256"] = M.canonical_sha256(
                [row["canonical_sha256"] for row in chunk["prompts"] if row["instance_id"] == chunk["tasks"][0]["instance_id"]]
            )
            manifest["ordered_prompt_canonical_sha256s_sha256"] = M.canonical_sha256(
                [row["canonical_sha256"] for item in manifest["chunks"] for row in item["prompts"]]
            )
            dump(workspace.manifest_path, manifest)
            with self.assertRaisesRegex(M.ReplayValidationError, "value changed"):
                workspace.validate()

    def test_split_manifest_self_pins_orchestrator_and_wrapper_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary))
            inputs = workspace.partition.manifest["inputs"]
            self.assertEqual(inputs["replay_pipeline"]["path"], "scripts/swe_task_state_v3_replay_pipeline.py")
            self.assertEqual(inputs["replay_shell_wrapper"]["path"], "scripts/run_swe_task_state_v3_replay.sh")
            workspace.inputs["replay_pipeline"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(M.ReplayValidationError, "upstream pins changed"):
                workspace.validate()

    def test_reordered_and_missing_chunk_rows_rejected(self):
        for mutation in ("reorder", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                workspace = Workspace(Path(temporary))
                manifest = json.loads(workspace.manifest_path.read_text())
                chunk = manifest["chunks"][0]
                path = workspace.replay_root / chunk["prompts_path"]
                rows = json.loads(path.read_text())
                if mutation == "reorder":
                    rows[0], rows[1] = rows[1], rows[0]
                else:
                    rows.pop()
                dump(path, rows)
                chunk["prompts_sha256"] = M.sha256_file(path)
                with self.assertRaises(M.ReplayValidationError):
                    workspace.validate()

    def test_chunk_symlink_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary))
            chunk = workspace.partition.manifest["chunks"][0]
            path = workspace.replay_root / chunk["prompts_path"]
            outside = workspace.parent / "outside.json"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(M.ReplayValidationError, "regular file"):
                workspace.validate()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary))
            manifest = json.loads(workspace.manifest_path.read_text())
            manifest["chunks"][0]["prompts_path"] = "../outside.json"
            dump(workspace.manifest_path, manifest)
            with self.assertRaisesRegex(M.ReplayValidationError, "safe relative path"):
                workspace.validate()

    def test_report_exit_zero_and_expected_strict_canary_exit_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=1, prompts_per_task=2, chunk_size=1)
            chunk = workspace.partition.manifest["chunks"][0]
            chunk_path = workspace.replay_root / chunk["prompts_path"]
            source = list(M.iter_strict_json_array(chunk_path, "chunk"))
            passed_path = workspace.parent / "passed.json"
            dump(passed_path, report_for_prompts(source))
            M.validate_report_file(
                report_path=passed_path,
                prompts_path=chunk_path,
                prompt_records=M._chunk_prompt_records(chunk),
                protocol=protocol(),
                expected_exit_code=0,
            )
            with self.assertRaisesRegex(M.ReplayValidationError, "exit code"):
                M.validate_report_file(
                    report_path=passed_path,
                    prompts_path=chunk_path,
                    prompt_records=M._chunk_prompt_records(chunk),
                    protocol=protocol(),
                    expected_exit_code=1,
                )
            failed_path = workspace.parent / "failed.json"
            dump(failed_path, report_for_prompts(source, fail_index=0))
            M.validate_report_file(
                report_path=failed_path,
                prompts_path=chunk_path,
                prompt_records=M._chunk_prompt_records(chunk),
                protocol=protocol(),
                expected_exit_code=1,
            )
            with self.assertRaisesRegex(M.ReplayValidationError, "exit code"):
                M.validate_report_file(
                    report_path=failed_path,
                    prompts_path=chunk_path,
                    prompt_records=M._chunk_prompt_records(chunk),
                    protocol=protocol(),
                    expected_exit_code=0,
                )

    def test_incomplete_duplicate_missing_reordered_and_tampered_report_rows_rejected(self):
        mutations = (
            "incomplete", "nested_incomplete", "duplicate", "missing",
            "reordered", "tampered", "numeric_type_tampered",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                workspace = Workspace(Path(temporary), task_count=1, prompts_per_task=3, chunk_size=1)
                chunk = workspace.partition.manifest["chunks"][0]
                chunk_path = workspace.replay_root / chunk["prompts_path"]
                source = list(M.iter_strict_json_array(chunk_path, "chunk"))
                value = report_for_prompts(source)
                if mutation == "incomplete":
                    del value["model"]
                elif mutation == "nested_incomplete":
                    del value["experiments"][0]["final_logits_reconstruction"][
                        "top_k_prefix_token_ids_match"
                    ]
                elif mutation == "duplicate":
                    value["experiments"][1] = value["experiments"][0]
                elif mutation == "missing":
                    value["experiments"].pop()
                elif mutation == "reordered":
                    value["experiments"][0], value["experiments"][1] = value["experiments"][1], value["experiments"][0]
                elif mutation == "numeric_type_tampered":
                    value["experiments"][0]["prompt_token_ids"][0] = float(
                        value["experiments"][0]["prompt_token_ids"][0]
                    )
                else:
                    value["experiments"][0]["scored_vocabulary"]["token_ids"][0] += 1
                path = workspace.parent / f"{mutation}.json"
                dump(path, value)
                with self.assertRaises(M.ReplayValidationError):
                    M.validate_report_file(
                        report_path=path,
                        prompts_path=chunk_path,
                        prompt_records=M._chunk_prompt_records(chunk),
                        protocol=protocol(),
                        expected_exit_code=0,
                    )

    def test_duplicate_json_key_in_report_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"schema_version":3,"schema_version":3}\n')
            prompts = Path(temporary) / "prompts.json"
            dump(prompts, [])
            with self.assertRaises(M.ReplayValidationError):
                M.validate_report_file(
                    report_path=path,
                    prompts_path=prompts,
                    prompt_records=[],
                    protocol=protocol(),
                    expected_exit_code=0,
                )

    def test_lossless_merge_preserves_exact_experiments_and_aggregates_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=5, prompts_per_task=2, chunk_size=2)
            run = workspace.write_complete_run(fail_chunk=1)
            source_experiments = []
            source_hashes = []
            for entry in run["chunks"]:
                source = json.loads((workspace.replay_root / entry["report"]["path"]).read_text())
                source_experiments.extend(source["experiments"])
                source_hashes.extend(M.canonical_sha256(row) for row in source["experiments"])
            report_path, merge_path = M.merge_replay_reports(
                partition=workspace.partition,
                protocol=protocol(),
            )
            merged = json.loads(report_path.read_text())
            merge = json.loads(merge_path.read_text())
            self.assertEqual(merged["status"], "failed")
            self.assertEqual(merged["experiments"], source_experiments)
            self.assertEqual([M.canonical_sha256(row) for row in merged["experiments"]], source_hashes)
            provenance = merged["combined_chunk_provenance"]
            self.assertEqual(provenance["source_chunk_count"], 3)
            self.assertEqual(provenance["source_experiment_count"], len(source_experiments))
            self.assertFalse(provenance["merge_contract"]["scores_metadata_and_token_ids_recomputed"])
            self.assertEqual(merge["ordered_experiment_payload_sha256s"], source_hashes)
            self.assertTrue(merge["value_identical_lossless_merge_validated"])

    def test_exact_validated_combined_report_orphan_can_finish_manifest_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=2, prompts_per_task=1, chunk_size=1)
            workspace.write_complete_run()
            report_path, merge_path = M.merge_replay_reports(
                partition=workspace.partition, protocol=protocol()
            )
            report_bytes = report_path.read_bytes()
            merge_path.unlink()
            recovered_report, recovered_merge = M.merge_replay_reports(
                partition=workspace.partition, protocol=protocol()
            )
            self.assertEqual(recovered_report.read_bytes(), report_bytes)
            self.assertTrue(recovered_merge.is_file())

    def test_forged_combined_provenance_and_rehashed_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=2, prompts_per_task=1, chunk_size=1)
            workspace.write_complete_run()
            report_path, merge_path = M.merge_replay_reports(
                partition=workspace.partition, protocol=protocol()
            )
            report = json.loads(report_path.read_text())
            report["combined_chunk_provenance"]["source_chunk_count"] += 1
            dump(report_path, report)
            merge = json.loads(merge_path.read_text())
            merge["combined_report"]["sha256"] = M.sha256_file(report_path)
            dump(merge_path, merge)
            with self.assertRaisesRegex(M.ReplayValidationError, "exact source experiments"):
                M.merge_replay_reports(
                    partition=workspace.partition, protocol=protocol()
                )

    def test_standalone_merge_receipt_revalidates_exact_canonical_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=2, prompts_per_task=1, chunk_size=1)
            workspace.write_complete_run()
            report_path, merge_path = M.merge_replay_reports(
                partition=workspace.partition, protocol=protocol()
            )
            summary_path = workspace.parent / "summary.json"
            dump(summary_path, {"synthetic": True})
            authenticated = M.AuthenticatedBundle(
                inputs=workspace.inputs,
                task_ids=tuple(workspace.task_ids),
                prompt_count=len(workspace.prompts),
                protocol=protocol(),
                materialization_receipt={"synthetic": True},
            )
            with (
                mock.patch.object(M, "V3_REPLAY_ROOT", workspace.replay_root),
                mock.patch.object(
                    M, "DEFAULT_MATERIALIZATION_RECEIPT", workspace.receipt_path
                ),
                mock.patch.object(
                    M, "authenticate_production_bundle", return_value=authenticated
                ),
                mock.patch.object(
                    M, "_production_partition", return_value=workspace.partition
                ),
            ):
                validated = M.validate_merge_receipt(
                    report_path=report_path,
                    merge_manifest_path=merge_path,
                    replay_root=workspace.replay_root,
                    prompts_path=workspace.prompts_path,
                    summary_path=summary_path,
                )
                self.assertEqual(validated.report_sha256, M.sha256_file(report_path))
                arbitrary = workspace.parent / "arbitrary-report.json"
                arbitrary.write_bytes(report_path.read_bytes())
                with self.assertRaisesRegex(M.ReplayValidationError, "not canonical"):
                    M.validate_merge_receipt(
                        report_path=arbitrary,
                        merge_manifest_path=merge_path,
                        replay_root=workspace.replay_root,
                        prompts_path=workspace.prompts_path,
                        summary_path=summary_path,
                    )

    def test_merge_rejects_report_tampered_after_run_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=2, prompts_per_task=1, chunk_size=1)
            run = workspace.write_complete_run()
            path = workspace.replay_root / run["chunks"][0]["report"]["path"]
            value = json.loads(path.read_text())
            value["experiments"][0]["generated_text"] = "tampered"
            dump(path, value)
            with self.assertRaisesRegex(M.ReplayValidationError, "SHA-256 changed"):
                M.merge_replay_reports(partition=workspace.partition, protocol=protocol())

    def test_run_manifest_rejects_cross_chunk_invariant_metadata_drift(self):
        mutations = (
            ("model", "extra", "changed", "invariant model"),
            ("lens", "extra", "changed", "invariant lens"),
            ("runtime", "capture_adapter", "changed", "invariant runtime"),
        )
        for section, key, replacement, expected_message in mutations:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as temporary:
                workspace = Workspace(Path(temporary), task_count=2, prompts_per_task=1, chunk_size=1)
                run = workspace.write_complete_run()
                entry = run["chunks"][1]
                report_path = workspace.replay_root / entry["report"]["path"]
                value = json.loads(report_path.read_text())
                value[section][key] = replacement
                dump(report_path, value)
                entry["report"]["sha256"] = M.sha256_file(report_path)
                dump(workspace.replay_root / M.RUN_MANIFEST_NAME, run)
                with self.assertRaisesRegex(M.ReplayValidationError, expected_message):
                    M.validate_run_manifest(
                        partition=workspace.partition,
                        protocol=protocol(),
                        require_complete=True,
                    )

    def test_model_load_timing_is_the_only_allowed_cross_chunk_runtime_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=2, prompts_per_task=1, chunk_size=1)
            run = workspace.write_complete_run()
            entry = run["chunks"][1]
            report_path = workspace.replay_root / entry["report"]["path"]
            value = json.loads(report_path.read_text())
            value["runtime"]["model_load_seconds"] = 99.0
            dump(report_path, value)
            entry["report"]["sha256"] = M.sha256_file(report_path)
            dump(workspace.replay_root / M.RUN_MANIFEST_NAME, run)
            checked, audits = M.validate_run_manifest(
                partition=workspace.partition,
                protocol=protocol(),
                require_complete=True,
            )
            self.assertEqual(checked["status"], "complete")
            self.assertEqual(len(audits), 2)

    def test_interrupted_unrecorded_chunk_artifacts_fail_closed_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=1, prompts_per_task=1, chunk_size=1)
            reports = workspace.replay_root / "reports"
            logs = workspace.replay_root / "logs"
            reports.mkdir()
            logs.mkdir()
            orphan = reports / "chunk-000-report.json"
            orphan.write_text("partial bytes retained for audit\n", encoding="utf-8")
            with mock.patch.object(M.subprocess, "run") as run_mock:
                with self.assertRaisesRegex(M.ReplayValidationError, "unrecorded/orphaned report"):
                    M.run_replay_chunks(
                        partition=workspace.partition,
                        protocol=protocol(),
                    )
            run_mock.assert_not_called()
            self.assertEqual(orphan.read_text(), "partial bytes retained for audit\n")
            state = json.loads((workspace.replay_root / M.RUN_MANIFEST_NAME).read_text())
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["completed_chunk_count"], 0)
            self.assertIn("manual move-aside", state["runner"]["interrupted_current_chunk_recovery"])

    def test_runner_invokes_exact_frozen_arguments_sequentially_and_discards_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Workspace(Path(temporary), task_count=2, prompts_per_task=1, chunk_size=1)
            calls = []

            def fake_run(command, **kwargs):
                calls.append((list(command), kwargs))
                prompt_path = Path(command[command.index("--prompts-file") + 1])
                report_path = Path(command[command.index("--output") + 1])
                source = list(M.iter_strict_json_array(prompt_path, "runner source"))
                dump(report_path, report_for_prompts(source))
                kwargs["stderr"].write(b"validated\n")
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(M.subprocess, "run", side_effect=fake_run):
                run_path = M.run_replay_chunks(
                    partition=workspace.partition,
                    protocol=protocol(),
                )
            self.assertEqual(len(calls), 2)
            for command, kwargs in calls:
                self.assertEqual(command[0], str(M.JLENS_SHELL_RUNNER))
                self.assertEqual(command[command.index("--layers") + 1], ",".join(str(layer) for layer in M.LAYERS))
                self.assertIn("--positions=-1", command)
                self.assertEqual(command[command.index("--top-k") + 1], "10")
                self.assertEqual(command[command.index("--max-model-len") + 1], "65536")
                self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.78")
                self.assertEqual(Path(kwargs["stdout"].name), Path(os.devnull))
            run = json.loads(run_path.read_text())
            self.assertEqual(run["status"], "complete")
            self.assertEqual(run["completed_chunk_count"], 2)

    def test_historical_replay_scripts_remain_byte_pinned(self):
        self.assertEqual(
            M.sha256_file(ROOT / "scripts" / "run_jlens_nvfp4.sh"),
            "89763dc20b09394e52f2296654b58408de40c902cb5dddc0a109026964eb2331",
        )
        self.assertEqual(
            M.sha256_file(ROOT / "scripts" / "run_jlens_nvfp4.py"),
            "18697bd8adc1159b7228390c526b9c418e87c25c64e7fa5d121d9f95906f7ee5",
        )


if __name__ == "__main__":
    unittest.main()
