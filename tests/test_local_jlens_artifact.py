#!/usr/bin/env python3
"""Fail-closed tests for completed local Jacobian Lens artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "download_jlens_local_test", SCRIPTS / "download_jlens.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

try:
    import torch
except ModuleNotFoundError:
    torch = None


def make_contract(*, d_model: int = 2, source_layers: tuple[int, ...] = (0, 1)):
    prompts = [
        {
            "id": "prompt-0",
            "text_sha256": "1" * 64,
            "token_count": 128,
            "token_ids": list(range(128)),
        }
    ]
    frozen_prompt_sha256 = MODULE.canonical_sha256(prompts)
    source_files = [
        {"path": path, "size": 1, "sha256": "2" * 64}
        for path in sorted(MODULE.LOCAL_FIT_SOURCE_FILES)
    ]
    source_identity = {
        "git_commit": "3" * 40,
        "files": source_files,
        "manifest_sha256": MODULE.canonical_sha256(source_files),
    }
    contract = {
        "model_id": MODULE.LOCAL_FIT_MODEL_REPO,
        "model_revision": MODULE.LOCAL_FIT_MODEL_REVISION,
        "model_artifacts": [],
        "quantization": dict(MODULE.LOCAL_FIT_QUANTIZATION),
        "estimator": {
            "name": MODULE.LOCAL_FIT_CONTRACT_ESTIMATOR,
            "source_layers": list(source_layers),
            "target_layer": len(source_layers),
            "max_seq_len": 128,
            "skip_first": 16,
            "cotangent_batch": MODULE.LOCAL_FIT_COTANGENT_BATCH,
            "row_limit": d_model,
            "input_batch": 1,
            "is_grads_batched": True,
        },
        "prompts_sha256": frozen_prompt_sha256,
        "prompt_manifest_sha256": MODULE.LOCAL_FIT_PROMPT_MANIFEST_SHA256,
        "seed": 0,
        "versions": MODULE.LOCAL_FIT_VERSIONS,
        "runtime_identity": {
            "versions": MODULE.LOCAL_FIT_VERSIONS,
            "cuda_runtime": "13.0",
            "nvidia_driver": "999.0",
            "gpu": {
                "name": "test GPU",
                "compute_capability": [12, 0],
                "total_memory_bytes": 32 * 2**30,
            },
            "qwen_modeling_source": {
                "path": "/test/modeling_qwen3_5.py",
                "size": 1,
                "sha256": "4" * 64,
            },
            "determinism": {
                "seed": 0,
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_allow_tf32": False,
                "cublas_workspace_config": ":4096:8",
            },
        },
        "source_identity": source_identity,
        "script_sha256": "2" * 64,
    }
    return contract, prompts, frozen_prompt_sha256


def make_provenance(
    artifact_sha256: str,
    *,
    d_model: int = 2,
    source_layers: tuple[int, ...] = (0, 1),
):
    contract, prompts, frozen_prompt_sha256 = make_contract(
        d_model=d_model, source_layers=source_layers
    )
    contract_sha256 = MODULE.canonical_sha256(contract)
    tensor_record = {
        "shape": [1],
        "dtype": "torch.uint8",
        "numel": 1,
        "nbytes": 1,
        "sha256": "5" * 64,
    }
    module_record = {
        "name": "linear",
        "in_features": 1,
        "out_features": 1,
        "weight_blocksize": 64,
        "quant_type": "nf4",
        "blocksize": 64,
        "nested": True,
        "nested_blocksize": 256,
        "original_shape": [1, 1],
        "original_dtype": "torch.bfloat16",
        "nested_dtype": "torch.float32",
        "tensors": {
            name: dict(tensor_record)
            for name in (
                "packed_weight",
                "absmax",
                "quant_map",
                "nested_absmax",
                "nested_quant_map",
                "nested_offset",
            )
        },
    }
    module_record["record_sha256"] = MODULE.canonical_sha256(module_record)
    quantized_weights = {
        "schema_version": 1,
        "module_count": 1,
        "modules": [module_record],
        "aggregate_sha256": MODULE.canonical_sha256([module_record]),
    }
    execution = {"quantized_weights": quantized_weights}
    return {
        "schema_version": 1,
        "status": "completed",
        "complete": True,
        "total_wall_seconds": 1.0,
        "invocations": [
            {
                "index": 0,
                "argv": ["--test"],
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:00:01+00:00",
                "elapsed_seconds": 1.0,
                "status": "completed",
                "source_observation": {},
            }
        ],
        "model": {
            "repo_id": MODULE.LOCAL_FIT_MODEL_REPO,
            "revision": MODULE.LOCAL_FIT_MODEL_REVISION,
            "loader": "AutoModelForCausalLM text-only",
            "execution_manifest_sha256": MODULE.canonical_sha256(execution),
            **execution,
        },
        "contract": contract,
        "contract_sha256": contract_sha256,
        "prompts": {
            "prompts": prompts,
            "frozen_prompt_sha256": frozen_prompt_sha256,
            "input_manifest_sha256": MODULE.LOCAL_FIT_PROMPT_MANIFEST_SHA256,
        },
        "result": {
            "path": "/relocatable/local.pt",
            "sha256": artifact_sha256,
            "storage_dtype": "float32",
            "n_prompts": len(prompts),
            "d_model": d_model,
            "layers": [
                {
                    "layer": layer,
                    "shape": [d_model, d_model],
                    "dtype": "float32",
                    "finite_count": d_model * d_model,
                    "min": 0.0,
                    "max": 1.0,
                    "frobenius_norm": 1.0,
                    "trace": 1.0,
                    "sha256": f"{layer + 3:064x}",
                    "published": {
                        "shape": [d_model, d_model],
                        "dtype": "float32",
                        "finite_count": d_model * d_model,
                        "min": 0.0,
                        "max": 1.0,
                        "frobenius_norm": 1.0,
                        "trace": 1.0,
                        "sha256": f"{layer + 3:064x}",
                    },
                }
                for layer in source_layers
            ],
        },
        "source": {
            "script_sha256": "2" * 64,
            "identity": contract["source_identity"],
            "start_observation": {
                "git_commit": contract["source_identity"]["git_commit"],
                "git_status_porcelain": "",
                "git_clean": True,
                "manifest_sha256": contract["source_identity"]["manifest_sha256"],
            },
        },
    }


SMALL_VERIFY = {
    "expected_n_prompts": 1,
    "expected_nf4_modules": 1,
    "expected_artifact_count": 0,
    "expected_artifacts_sha256": MODULE.canonical_sha256([]),
}


class LocalProvenanceTest(unittest.TestCase):
    def test_completed_provenance_binds_model_quantization_and_layers(self):
        artifact_sha256 = "a" * 64
        provenance = make_provenance(artifact_sha256)
        result = MODULE._verify_local_provenance(
            provenance,
            artifact_sha256=artifact_sha256,
            d_model=2,
            source_layers=(0, 1),
            target_layer=2,
            **SMALL_VERIFY,
        )
        self.assertEqual(result["n_prompts"], 1)
        self.assertEqual(
            result["fit_quantization"], MODULE.LOCAL_FIT_QUANTIZATION_LABEL
        )

    def test_incomplete_or_rewritten_contract_is_rejected(self):
        artifact_sha256 = "a" * 64
        provenance = make_provenance(artifact_sha256)
        provenance["status"] = "incomplete_diagnostic"
        with self.assertRaisesRegex(ValueError, "not completed"):
            MODULE._verify_local_provenance(
                provenance,
                artifact_sha256=artifact_sha256,
                d_model=2,
                source_layers=(0, 1),
                target_layer=2,
                **SMALL_VERIFY,
            )

        provenance = make_provenance(artifact_sha256)
        provenance["contract"]["quantization"]["type"] = "nvfp4"
        provenance["contract_sha256"] = MODULE.canonical_sha256(
            provenance["contract"]
        )
        with self.assertRaisesRegex(ValueError, "quantization mismatch"):
            MODULE._verify_local_provenance(
                provenance,
                artifact_sha256=artifact_sha256,
                d_model=2,
                source_layers=(0, 1),
                target_layer=2,
                **SMALL_VERIFY,
            )

    def test_provenance_must_bind_artifact_sha(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE._verify_local_provenance(
                make_provenance("a" * 64),
                artifact_sha256="b" * 64,
                d_model=2,
                source_layers=(0, 1),
                target_layer=2,
                **SMALL_VERIFY,
            )


@unittest.skipIf(torch is None, "torch is installed in .venv-vllm")
class LocalArtifactTest(unittest.TestCase):
    def materialize(
        self, directory: Path, *, finite: bool = True, storage_dtype: str = "float32"
    ):
        source_layers = (0, 1)
        contract, _, _ = make_contract(source_layers=source_layers)
        contract_sha256 = MODULE.canonical_sha256(contract)
        authoritative = {
            0: torch.eye(2, dtype=torch.float32),
            1: torch.ones(2, 2, dtype=torch.float32),
        }
        if not finite:
            authoritative[1][0, 0] = float("nan")
        matrices = {
            layer: value.to(
                dtype=torch.float16 if storage_dtype == "float16" else torch.float32
            )
            for layer, value in authoritative.items()
        }
        template = make_provenance("0" * 64, source_layers=source_layers)
        nf4_aggregate = template["model"]["quantized_weights"]["aggregate_sha256"]
        checkpoint = {
            "J": matrices,
            "n_prompts": 1,
            "d_model": 2,
            "source_layers": list(source_layers),
            "metadata": {
                "fit_model": MODULE.LOCAL_FIT_MODEL_REPO,
                "fit_model_revision": MODULE.LOCAL_FIT_MODEL_REVISION,
                "fit_quantization": MODULE.LOCAL_FIT_QUANTIZATION_LABEL,
                "estimator": MODULE.LOCAL_FIT_ESTIMATOR,
                "source_layers": list(source_layers),
                "target_layer": 2,
                "contract_sha256": contract_sha256,
                "storage_dtype": storage_dtype,
                "nf4_aggregate_sha256": nf4_aggregate,
            },
        }
        artifact = directory / "local.pt"
        torch.save(checkpoint, artifact)
        artifact_sha256 = MODULE.sha256_file(artifact)
        provenance = make_provenance(artifact_sha256, source_layers=source_layers)
        provenance["result"]["storage_dtype"] = storage_dtype
        for layer, record in zip(
            source_layers, provenance["result"]["layers"], strict=True
        ):
            authoritative_statistics = MODULE._tensor_statistics(
                authoritative[layer], dtype_name="float32"
            )
            published_statistics = MODULE._tensor_statistics(
                matrices[layer], dtype_name=storage_dtype
            )
            record.clear()
            record.update(
                {
                    "layer": layer,
                    **authoritative_statistics,
                    "published": published_statistics,
                }
            )
        sidecar = directory / "local.pt.provenance.json"
        sidecar.write_text(json.dumps(provenance), encoding="utf-8")
        return artifact, artifact_sha256, sidecar

    def test_complete_local_artifact_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, digest, sidecar = self.materialize(Path(temporary))
            result = MODULE.verify_local_fit_artifact(
                artifact,
                expected_sha256=digest,
                provenance_path=sidecar,
                d_model=2,
                source_layers=(0, 1),
                target_layer=2,
                **SMALL_VERIFY,
            )
        self.assertEqual(result["sha256"], digest)
        self.assertTrue(result["finite_checked"])

    def test_fp16_artifact_and_published_statistics_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, digest, sidecar = self.materialize(
                Path(temporary), storage_dtype="float16"
            )
            result = MODULE.verify_local_fit_artifact(
                artifact,
                expected_sha256=digest,
                provenance_path=sidecar,
                d_model=2,
                source_layers=(0, 1),
                target_layer=2,
                **SMALL_VERIFY,
            )
        self.assertEqual(result["tensor_dtype"], "torch.float16")

    def test_published_tensor_statistics_are_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, digest, sidecar = self.materialize(Path(temporary))
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            provenance["result"]["layers"][0]["published"]["trace"] += 1
            sidecar.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "published statistics trace"):
                MODULE.verify_local_fit_artifact(
                    artifact,
                    expected_sha256=digest,
                    provenance_path=sidecar,
                    d_model=2,
                    source_layers=(0, 1),
                    target_layer=2,
                    **SMALL_VERIFY,
                )

    def test_nonfinite_tensor_and_wrong_expected_sha_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, digest, sidecar = self.materialize(
                Path(temporary), finite=False
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                MODULE.verify_local_fit_artifact(
                    artifact,
                    expected_sha256=digest,
                    provenance_path=sidecar,
                    d_model=2,
                    source_layers=(0, 1),
                    target_layer=2,
                    **SMALL_VERIFY,
                )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.verify_local_fit_artifact(
                    artifact,
                    expected_sha256="f" * 64,
                    provenance_path=sidecar,
                    d_model=2,
                    source_layers=(0, 1),
                    target_layer=2,
                    **SMALL_VERIFY,
                )


if __name__ == "__main__":
    unittest.main()
