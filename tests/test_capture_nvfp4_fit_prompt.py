from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "capture_nvfp4_fit_prompt.py"
SPEC = importlib.util.spec_from_file_location("capture_nvfp4_fit_prompt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FAKE_CAPTURE = r'''
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--prompt-manifest", type=Path, required=True)
parser.add_argument("--prompt-index", type=int, required=True)
parser.add_argument("--model-path", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tensor-output", type=Path, required=True)
parser.add_argument("--target-layers", required=True)
parser.add_argument("--capture-capacity", type=int, required=True)
args, _ = parser.parse_known_args()
raw = args.prompt_manifest.read_bytes()
manifest = json.loads(raw)
entry = manifest["prompts"][args.prompt_index]
provenance = {
    "manifest_path": str(args.prompt_manifest.resolve()),
    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    "manifest_index": args.prompt_index,
    "row_index": entry["row_index"],
    "text": entry["text"],
    "text_sha256": entry["text_sha256"],
    "token_count": entry["token_count"],
    "token_ids": entry["token_ids"],
}
generation = {
    "prompt_token_ids": entry["token_ids"],
    "generated_token_id": 2,
    "generated_text": "x",
    "finish_reason": "length",
    "top_logprobs": [],
}
observer = args.mode == "compiled-observer"
summary_count = 1120 if observer else 688
model_revision = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
model_path = str(args.model_path.resolve())
model_identity = {
    "policy": "pinned-nvidia-modelopt-snapshot-v1",
    "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "revision": model_revision,
    "resolved_path": model_path,
    "metadata_sha256": {
        "config.json": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
        "hf_quant_config.json": "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
        "model.safetensors.index.json": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
    },
    "strict_pinned_validation": True,
    "validator": "ModelOptCheckpoint(strict_pinned=True)",
}
capture = {
    "tensor_output": str(args.tensor_output),
    "install": {"capture_capacity": args.capture_capacity},
    "replay_parameter_provenance": {
        f"replay.p{i}": {} for i in range(785)
    },
    "tensor_summaries": {f"t{i}": {} for i in range(summary_count)},
}
if observer:
    capture.update({"missing_required": [], "truncated": []})
value = {
    "schema_version": 1,
    "status": "captured" if observer else "incomplete",
    "model": {
        "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
        "revision": model_revision,
        "resolved_path": model_path,
        "identity": model_identity,
    },
    "runtime": {
        "mode": args.mode,
        "target_profile": args.target_layers,
        "target_layers": list(range(64)),
        "mtp_enabled": False,
        "language_model_only": True,
    },
    "prompt": {
        "text": entry["text"],
        "token_ids": entry["token_ids"],
        "token_count": entry["token_count"],
        "provenance": provenance,
    },
    "capture": capture,
    "instrumented_generation": generation,
    "test_environment": {
        "VLLM_DISABLE_COMPILE_CACHE": os.environ.get("VLLM_DISABLE_COMPILE_CACHE"),
        "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
    },
}
if not observer:
    value["authoritative_compiled_generation"] = generation
args.output.write_text(json.dumps(value))
args.tensor_output.write_bytes((args.mode + ":" + str(args.prompt_index)).encode())
with (args.output.parent / "invocations.txt").open("a") as handle:
    handle.write(args.mode + "\n")
print(args.mode)
'''


FAKE_PROOF = r'''
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--baseline-json", type=Path, required=True)
parser.add_argument("--baseline-tensors", type=Path, required=True)
parser.add_argument("--observer-json", type=Path, required=True)
parser.add_argument("--observer-tensors", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args, _ = parser.parse_known_args()

def record(path):
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

observer = json.loads(args.observer_json.read_text())
value = {
    "status": "passed",
    "claim": {
        "mtp": "off",
        "observer_graph_modified": True,
        "observer_modification_discharged": True,
    },
    "generation_record_parity": {"exact": True},
    "shared_internal_tensor_parity": {
        "shared_tensor_count": 688,
        "all_shared_bit_exact": True,
    },
    "replay_parameter_parity": {
        "parameter_count": 785,
        "all_content_hashes_equal": True,
    },
    "observer_capture_completeness": {"required_missing": [], "truncated": []},
    "configuration": {"prompt": observer["prompt"]},
    "artifacts": {
        "baseline_json": record(args.baseline_json),
        "baseline_tensors": record(args.baseline_tensors),
        "observer_json": record(args.observer_json),
        "observer_tensors": record(args.observer_tensors),
    },
    "verifier": {
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    },
}
args.output.write_text(json.dumps(value))
with (args.output.parent / "invocations.txt").open("a") as handle:
    handle.write("proof\n")
print("proof")
'''


class PinnedPromptTests(unittest.TestCase):
    def test_pinned_manifest_and_index_are_bound(self) -> None:
        prompt = MODULE._load_pinned_prompt(MODULE.PINNED_MANIFEST, 9)
        self.assertEqual(prompt["manifest_sha256"], MODULE.PINNED_MANIFEST_SHA256)
        self.assertEqual(prompt["manifest_prompt_count"], 10)
        self.assertEqual(prompt["manifest_index"], 9)
        self.assertEqual(prompt["token_count"], 128)
        self.assertEqual(len(prompt["token_ids"]), 128)

    def test_modified_manifest_is_rejected_even_if_structurally_valid(self) -> None:
        value = json.loads(MODULE.PINNED_MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.json"
            path.write_text(json.dumps(value, sort_keys=True))
            with self.assertRaisesRegex(MODULE.OrchestrationError, "pinned"):
                MODULE._load_pinned_prompt(path, 0)

    def test_capture_commands_use_frozen_manifest_ids_and_disable_cache(self) -> None:
        prompt = MODULE._load_pinned_prompt(MODULE.PINNED_MANIFEST, 2)
        args = Namespace(
            python=ROOT / ".venv-vllm" / "bin" / "python",
            capture_script=ROOT / "scripts" / "check_nvfp4_runtime_capture.py",
            proof_script=ROOT / "scripts" / "prove_nvfp4_capture_pair.py",
            capture_capacity=128,
            gpu_memory_utilization=0.8,
            max_model_len=32768,
            max_num_batched_tokens=4096,
            max_num_seqs=2,
            hash_chunk_mib=32,
        )
        paths = MODULE._artifact_paths(Path("out"), 2)
        pinned_snapshot = MODULE._pinned_snapshot_path()
        command = MODULE._capture_command(
            args, prompt, paths, "compiled", pinned_snapshot
        )
        self.assertIn("--prompt-manifest", command)
        self.assertEqual(command[command.index("--prompt-index") + 1], "2")
        self.assertEqual(command[command.index("--target-layers") + 1], "all")
        self.assertNotIn("--prompt", command)
        self.assertEqual(
            command[command.index("--model-path") + 1], str(pinned_snapshot)
        )
        self.assertTrue(command[0].endswith(".venv-vllm/bin/python"))
        record = MODULE._command_record(
            command, environment=MODULE.ENVIRONMENT_OVERRIDES
        )
        self.assertEqual(record["environment_overrides"]["VLLM_DISABLE_COMPILE_CACHE"], "1")

    def test_production_orchestrator_rejects_model_path_override(self) -> None:
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                MODULE._parse_args(
                    [
                        "--prompt-index",
                        "0",
                        "--model-path",
                        "/tmp/same-shaped-alternate",
                    ]
                )


class OrchestratorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        def identity(snapshot: Path) -> dict[str, object]:
            resolved = MODULE._pinned_snapshot_path()
            self.assertEqual(snapshot.resolve(), resolved)
            return {
                "repo_id": MODULE.MODEL_REPO,
                "revision": MODULE.MODEL_REVISION,
                "resolved_path": str(resolved),
                "identity_policy": MODULE.MODEL_IDENTITY_POLICY,
                "metadata_sha256": MODULE.PINNED_METADATA_SHA256,
                "local_pinned_validation": True,
            }

        patcher = mock.patch.object(
            MODULE, "_validate_pinned_snapshot", side_effect=identity
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_args(
        self,
        root: Path,
        *,
        resume: bool = False,
        delete_baseline: bool = False,
    ) -> Namespace:
        capture = root / "fake_capture.py"
        proof = root / "fake_proof.py"
        if not capture.exists():
            capture.write_text(textwrap.dedent(FAKE_CAPTURE))
            proof.write_text(textwrap.dedent(FAKE_PROOF))
        return MODULE._parse_args(
            [
                "--prompt-index",
                "0",
                "--output-dir",
                str(root / "output"),
                "--python",
                sys.executable,
                "--capture-script",
                str(capture),
                "--proof-script",
                str(proof),
                "--hash-chunk-mib",
                "1",
                *( ["--resume"] if resume else [] ),
                *(
                    ["--delete-baseline-pt-after-proof"]
                    if delete_baseline
                    else []
                ),
            ]
        )

    def test_pipeline_resumes_without_rerun_and_deletes_only_baseline_pt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.make_args(root, delete_baseline=True)
            state = MODULE._execute(args)
            paths = MODULE._artifact_paths(args.output_dir, 0)

            self.assertEqual(state["status"], "complete")
            self.assertFalse(paths["baseline_tensors"].exists())
            self.assertTrue(paths["baseline_json"].is_file())
            self.assertTrue(paths["observer_tensors"].is_file())
            self.assertTrue(paths["proof"].is_file())
            self.assertEqual(
                state["retention"]["baseline_tensors"], "deleted_after_proof"
            )
            self.assertEqual(
                (args.output_dir / "invocations.txt").read_text().splitlines(),
                ["compiled", "compiled-observer", "proof"],
            )
            baseline = json.loads(paths["baseline_json"].read_text())
            observer = json.loads(paths["observer_json"].read_text())
            for metadata in (baseline, observer):
                self.assertEqual(
                    metadata["test_environment"]["VLLM_DISABLE_COMPILE_CACHE"],
                    "1",
                )
                self.assertEqual(
                    metadata["test_environment"]["VLLM_ENABLE_V1_MULTIPROCESSING"],
                    "0",
                )

            resumed = MODULE._execute(
                self.make_args(root, resume=True, delete_baseline=True)
            )
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(
                (args.output_dir / "invocations.txt").read_text().splitlines(),
                ["compiled", "compiled-observer", "proof"],
            )

    def test_fresh_run_and_resume_fail_closed_on_partial_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.make_args(root)
            paths = MODULE._artifact_paths(args.output_dir, 0)
            paths["baseline_json"].parent.mkdir(parents=True)
            paths["baseline_json"].write_text("{}")

            with self.assertRaises(FileExistsError):
                MODULE._execute(args)
            with self.assertRaisesRegex(MODULE.OrchestrationError, "partial baseline"):
                MODULE._execute(self.make_args(root, resume=True))

    def test_resume_rejects_changed_retained_observer_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.make_args(root)
            MODULE._execute(args)
            paths = MODULE._artifact_paths(args.output_dir, 0)
            paths["observer_tensors"].write_bytes(b"tampered")
            with self.assertRaisesRegex(MODULE.OrchestrationError, "changed"):
                MODULE._execute(self.make_args(root, resume=True))

    def test_resume_rejects_unpinned_model_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.make_args(root)
            MODULE._execute(args)
            paths = MODULE._artifact_paths(args.output_dir, 0)
            observer = json.loads(paths["observer_json"].read_text())
            observer["model"]["identity"]["metadata_sha256"]["config.json"] = (
                "0" * 64
            )
            paths["observer_json"].write_text(json.dumps(observer))
            with self.assertRaisesRegex(
                MODULE.OrchestrationError, "strict pinned model identity"
            ):
                MODULE._execute(self.make_args(root, resume=True))

    def test_resume_rejects_same_shaped_alternate_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.make_args(root)
            MODULE._execute(args)
            paths = MODULE._artifact_paths(args.output_dir, 0)
            observer = json.loads(paths["observer_json"].read_text())
            alternate = (
                root
                / "alternate-cache"
                / "models--nvidia--Qwen3.6-27B-NVFP4"
                / "snapshots"
                / MODULE.MODEL_REVISION
            )
            observer["model"]["resolved_path"] = str(alternate)
            observer["model"]["identity"]["resolved_path"] = str(alternate)
            paths["observer_json"].write_text(json.dumps(observer))
            with self.assertRaisesRegex(
                MODULE.OrchestrationError, "exact pinned snapshot"
            ):
                MODULE._execute(self.make_args(root, resume=True))


if __name__ == "__main__":
    unittest.main()
