#!/usr/bin/env python3
"""Download and verify the pinned Qwen3.6-27B Jacobian lens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/"
    "Qwen3.6-27B_jacobian_lens_n1000.pt"
)
LENS_SHA256 = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
LENS_SIZE = 3_303_032_772

LOCAL_FIT_MODEL_REPO = "Qwen/Qwen3.6-27B"
LOCAL_FIT_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LOCAL_FIT_QUANTIZATION = {
    "method": "bitsandbytes",
    "type": "nf4",
    "double_quant": True,
    "compute_dtype": "bfloat16",
    "storage_dtype": "uint8",
    "blocksize": 64,
    "nested_blocksize": 256,
}
LOCAL_FIT_QUANTIZATION_LABEL = "bitsandbytes-nf4-double-quant-bfloat16"
LOCAL_FIT_ESTIMATOR = "anthropic-future-summed"
LOCAL_FIT_CONTRACT_ESTIMATOR = "anthropic_future_summed_vjp"
LOCAL_FIT_SOURCE_LAYERS = tuple(range(63))
LOCAL_FIT_TARGET_LAYER = 63
LOCAL_FIT_D_MODEL = 5120
LOCAL_FIT_N_PROMPTS = 10
LOCAL_FIT_PROMPT_MANIFEST_SHA256 = (
    "2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b"
)
LOCAL_FIT_MODEL_ARTIFACT_COUNT = 22
LOCAL_FIT_MODEL_ARTIFACTS_SHA256 = (
    "03759870202250dcf41726d3b312a7555ba0a7bfa25721cb384b20821dbec8f0"
)
LOCAL_FIT_VERSIONS = {
    "accelerate": "missing",
    "bitsandbytes": "0.49.2",
    "huggingface_hub": "1.23.0",
    "jlens": "0.1.0",
    "numpy": "2.5.1",
    "python": "3.12.13",
    "torch": "2.11.0+cu130",
    "transformers": "5.12.1",
    "jlens_direct_url": {
        "url": "https://github.com/anthropics/jacobian-lens.git",
        "vcs_info": {
            "commit_id": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "requested_revision": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "vcs": "git",
        },
    },
}
LOCAL_FIT_SOURCE_FILES = {
    "configs/jlens_nf4_fit_prompts.json",
    "docs/NF4_FIT_CONTRACT.md",
    "requirements-fit.txt",
    "scripts/check_fit.sh",
    "scripts/compare_jlens_artifacts.py",
    "scripts/download_jlens.py",
    "scripts/fit_jlens_nf4.py",
    "scripts/materialize_jlens_fit_prompts.py",
    "scripts/setup_fit.sh",
    "validation/fit-freeze.txt",
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} is not a 64-character hexadecimal SHA-256")
    return normalized


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _verify_source_identity(
    contract: dict[str, Any], provenance: dict[str, Any]
) -> None:
    identity = _require_mapping(
        contract.get("source_identity"), label="source identity"
    )
    commit = identity.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("source identity Git commit is invalid")
    files = identity.get("files")
    if not isinstance(files, list) or {
        record.get("path") for record in files if isinstance(record, dict)
    } != LOCAL_FIT_SOURCE_FILES:
        raise ValueError("source identity file set mismatch")
    for record in files:
        record = _require_mapping(record, label="source file record")
        if not isinstance(record.get("size"), int) or record["size"] <= 0:
            raise ValueError("source file size is invalid")
        normalize_sha256(record.get("sha256"), label="source file SHA-256")
    manifest_sha = canonical_sha256(files)
    if normalize_sha256(
        identity.get("manifest_sha256"), label="source manifest SHA-256"
    ) != manifest_sha:
        raise ValueError("source file manifest SHA-256 mismatch")
    script_record = next(
        record for record in files if record["path"] == "scripts/fit_jlens_nf4.py"
    )
    if contract.get("script_sha256") != script_record["sha256"]:
        raise ValueError("contract script SHA-256 is not bound to the source manifest")
    source = _require_mapping(provenance.get("source"), label="provenance source")
    if source.get("identity") != identity:
        raise ValueError("provenance source identity does not match the contract")
    if source.get("script_sha256") != script_record["sha256"]:
        raise ValueError("provenance script SHA-256 mismatch")
    observation = _require_mapping(
        source.get("start_observation"), label="start source observation"
    )
    if (
        observation.get("git_commit") != commit
        or observation.get("git_clean") is not True
        or observation.get("git_status_porcelain") != ""
        or observation.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("fit did not start from the recorded clean source identity")


def _verify_runtime_identity(contract: dict[str, Any]) -> None:
    if contract.get("versions") != LOCAL_FIT_VERSIONS:
        raise ValueError("local lens package-version contract mismatch")
    runtime = _require_mapping(
        contract.get("runtime_identity"), label="runtime identity"
    )
    if runtime.get("versions") != LOCAL_FIT_VERSIONS:
        raise ValueError("runtime package versions are not bound to the contract")
    if not isinstance(runtime.get("cuda_runtime"), str) or not isinstance(
        runtime.get("nvidia_driver"), str
    ):
        raise ValueError("CUDA runtime/driver identity is missing")
    gpu = _require_mapping(runtime.get("gpu"), label="runtime GPU")
    if (
        not isinstance(gpu.get("name"), str)
        or not isinstance(gpu.get("compute_capability"), list)
        or not isinstance(gpu.get("total_memory_bytes"), int)
    ):
        raise ValueError("runtime GPU identity is incomplete")
    qwen = _require_mapping(
        runtime.get("qwen_modeling_source"), label="Qwen source identity"
    )
    normalize_sha256(qwen.get("sha256"), label="Qwen source SHA-256")
    if not isinstance(qwen.get("size"), int) or qwen["size"] <= 0:
        raise ValueError("Qwen source size is invalid")
    determinism = _require_mapping(
        runtime.get("determinism"), label="runtime determinism"
    )
    expected_determinism = {
        "seed": 0,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    if determinism != expected_determinism:
        raise ValueError("runtime determinism contract mismatch")


def _verify_quantized_weights(
    model: dict[str, Any], *, expected_module_count: int
) -> str:
    declared_execution_sha = normalize_sha256(
        model.get("execution_manifest_sha256"),
        label="model execution manifest SHA-256",
    )
    execution = {
        key: value
        for key, value in model.items()
        if key
        not in {"repo_id", "revision", "loader", "execution_manifest_sha256"}
    }
    if canonical_sha256(execution) != declared_execution_sha:
        raise ValueError("model execution manifest SHA-256 mismatch")
    manifest = _require_mapping(
        execution.get("quantized_weights"), label="quantized-weight manifest"
    )
    modules = manifest.get("modules")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("module_count") != expected_module_count
        or not isinstance(modules, list)
        or len(modules) != expected_module_count
    ):
        raise ValueError("quantized-weight module manifest is incomplete")
    if normalize_sha256(
        manifest.get("aggregate_sha256"), label="quantized-weight aggregate SHA-256"
    ) != canonical_sha256(modules):
        raise ValueError("quantized-weight aggregate SHA-256 mismatch")
    names: set[str] = set()
    required_tensors = {
        "packed_weight",
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
        "nested_offset",
    }
    for module in modules:
        module = _require_mapping(module, label="quantized module record")
        name = module.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("quantized module names must be unique")
        names.add(name)
        unsigned = dict(module)
        declared_record_sha = normalize_sha256(
            unsigned.pop("record_sha256", None), label=f"quantized module {name} SHA-256"
        )
        if canonical_sha256(unsigned) != declared_record_sha:
            raise ValueError(f"quantized module record SHA-256 mismatch: {name}")
        if (
            module.get("quant_type") != "nf4"
            or module.get("blocksize") != 64
            or module.get("weight_blocksize") != 64
            or module.get("nested") is not True
            or module.get("nested_blocksize") != 256
        ):
            raise ValueError(f"quantized module block contract mismatch: {name}")
        tensors = _require_mapping(
            module.get("tensors"), label=f"quantized module {name} tensors"
        )
        if set(tensors) != required_tensors:
            raise ValueError(f"quantized module tensor set mismatch: {name}")
        for tensor_name, record in tensors.items():
            record = _require_mapping(
                record, label=f"quantized tensor {name}.{tensor_name}"
            )
            normalize_sha256(
                record.get("sha256"), label=f"quantized tensor {name}.{tensor_name} SHA-256"
            )
            if (
                not isinstance(record.get("shape"), list)
                or not isinstance(record.get("dtype"), str)
                or not isinstance(record.get("numel"), int)
                or not isinstance(record.get("nbytes"), int)
            ):
                raise ValueError(f"quantized tensor metadata is invalid: {name}.{tensor_name}")
    return manifest["aggregate_sha256"]


def _verify_local_provenance(
    provenance: dict[str, Any],
    *,
    artifact_sha256: str,
    d_model: int,
    source_layers: tuple[int, ...],
    target_layer: int,
    expected_n_prompts: int = LOCAL_FIT_N_PROMPTS,
    expected_nf4_modules: int = 496,
    expected_artifact_count: int = LOCAL_FIT_MODEL_ARTIFACT_COUNT,
    expected_artifacts_sha256: str = LOCAL_FIT_MODEL_ARTIFACTS_SHA256,
) -> dict[str, Any]:
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported local lens provenance schema")
    if provenance.get("status") != "completed":
        raise ValueError("local lens provenance status is not completed")
    if provenance.get("complete") is not True:
        raise ValueError("local lens provenance is not marked complete")
    total_wall = provenance.get("total_wall_seconds")
    if not isinstance(total_wall, (int, float)) or total_wall < 0:
        raise ValueError("local lens total wall time is invalid")
    invocations = provenance.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("local lens invocation history is missing")
    for index, invocation in enumerate(invocations):
        invocation = _require_mapping(invocation, label="fit invocation")
        if (
            invocation.get("index") != index
            or not isinstance(invocation.get("argv"), list)
            or not isinstance(invocation.get("started_at"), str)
            or not isinstance(invocation.get("elapsed_seconds"), (int, float))
            or invocation["elapsed_seconds"] < 0
            or not isinstance(invocation.get("source_observation"), dict)
        ):
            raise ValueError("local lens invocation history is invalid")
    if invocations[-1].get("status") != "completed":
        raise ValueError("local lens final invocation is not completed")

    model = _require_mapping(provenance.get("model"), label="provenance model")
    if model.get("repo_id") != LOCAL_FIT_MODEL_REPO:
        raise ValueError("local lens fit model repository mismatch")
    if model.get("revision") != LOCAL_FIT_MODEL_REVISION:
        raise ValueError("local lens fit model revision mismatch")
    nf4_aggregate_sha256 = _verify_quantized_weights(
        model, expected_module_count=expected_nf4_modules
    )

    contract = _require_mapping(
        provenance.get("contract"), label="provenance contract"
    )
    contract_sha256 = canonical_sha256(contract)
    declared_contract_sha256 = normalize_sha256(
        provenance.get("contract_sha256"), label="provenance contract_sha256"
    )
    if declared_contract_sha256 != contract_sha256:
        raise ValueError("local lens provenance contract SHA-256 mismatch")
    if contract.get("model_id") != LOCAL_FIT_MODEL_REPO:
        raise ValueError("local lens contract model repository mismatch")
    if contract.get("model_revision") != LOCAL_FIT_MODEL_REVISION:
        raise ValueError("local lens contract model revision mismatch")
    if contract.get("quantization") != LOCAL_FIT_QUANTIZATION:
        raise ValueError("local lens contract quantization mismatch")
    artifacts = contract.get("model_artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != expected_artifact_count
        or canonical_sha256(artifacts) != expected_artifacts_sha256
    ):
        raise ValueError("local lens pinned model-artifact manifest mismatch")
    if contract.get("seed") != 0:
        raise ValueError("local lens fit seed mismatch")
    _verify_runtime_identity(contract)
    _verify_source_identity(contract, provenance)

    estimator = _require_mapping(
        contract.get("estimator"), label="local lens estimator contract"
    )
    if estimator.get("name") != LOCAL_FIT_CONTRACT_ESTIMATOR:
        raise ValueError("local lens estimator name mismatch")
    if estimator.get("source_layers") != list(source_layers):
        raise ValueError("local lens estimator source layers mismatch")
    if estimator.get("target_layer") != target_layer:
        raise ValueError("local lens estimator target layer mismatch")
    if estimator.get("max_seq_len") != 128 or estimator.get("skip_first") != 16:
        raise ValueError("local lens estimator token window mismatch")
    if (
        estimator.get("cotangent_batch") != 4
        or estimator.get("input_batch") != 1
        or estimator.get("is_grads_batched") is not True
    ):
        raise ValueError("local lens estimator execution flags mismatch")
    if estimator.get("row_limit") != d_model:
        raise ValueError("local lens estimator is not a complete dense fit")

    prompts = _require_mapping(
        provenance.get("prompts"), label="local lens prompt provenance"
    )
    prompt_records = prompts.get("prompts")
    if not isinstance(prompt_records, list) or not prompt_records:
        raise ValueError("local lens prompt provenance is empty")
    frozen_prompt_sha256 = canonical_sha256(prompt_records)
    if normalize_sha256(
        prompts.get("frozen_prompt_sha256"), label="frozen prompt SHA-256"
    ) != frozen_prompt_sha256:
        raise ValueError("local lens frozen prompt SHA-256 mismatch")
    if contract.get("prompts_sha256") != frozen_prompt_sha256:
        raise ValueError("local lens contract does not bind the frozen prompts")
    if (
        prompts.get("input_manifest_sha256")
        != LOCAL_FIT_PROMPT_MANIFEST_SHA256
        or contract.get("prompt_manifest_sha256")
        != LOCAL_FIT_PROMPT_MANIFEST_SHA256
    ):
        raise ValueError("local lens prompt-manifest identity mismatch")

    result = _require_mapping(provenance.get("result"), label="provenance result")
    if normalize_sha256(
        result.get("sha256"), label="provenance result SHA-256"
    ) != artifact_sha256:
        raise ValueError("local lens provenance does not match the artifact SHA-256")
    if result.get("d_model") != d_model:
        raise ValueError("local lens provenance d_model mismatch")
    n_prompts = result.get("n_prompts")
    if not isinstance(n_prompts, int) or isinstance(n_prompts, bool) or n_prompts <= 0:
        raise ValueError("local lens provenance n_prompts must be positive")
    if n_prompts != expected_n_prompts or len(prompt_records) != n_prompts:
        raise ValueError("local lens prompt records do not match n_prompts")
    storage_dtype = result.get("storage_dtype")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("unsupported local lens storage dtype")

    layer_records = result.get("layers")
    if not isinstance(layer_records, list) or len(layer_records) != len(source_layers):
        raise ValueError("local lens provenance does not contain every layer")
    expected_finite = d_model * d_model
    for layer, record in zip(source_layers, layer_records, strict=True):
        record = _require_mapping(record, label=f"layer {layer} provenance")
        if record.get("layer") != layer:
            raise ValueError(f"local lens provenance layer order mismatch at {layer}")
        if record.get("shape") != [d_model, d_model]:
            raise ValueError(f"local lens provenance shape mismatch at layer {layer}")
        if record.get("dtype") != "float32":
            raise ValueError(f"local lens authoritative dtype mismatch at layer {layer}")
        if record.get("finite_count") != expected_finite:
            raise ValueError(f"local lens finite count mismatch at layer {layer}")
        normalize_sha256(record.get("sha256"), label=f"layer {layer} SHA-256")
        published = _require_mapping(
            record.get("published"), label=f"layer {layer} published statistics"
        )
        if (
            published.get("shape") != [d_model, d_model]
            or published.get("dtype") != storage_dtype
            or published.get("finite_count") != expected_finite
        ):
            raise ValueError(f"local lens published statistics mismatch at layer {layer}")
        normalize_sha256(
            published.get("sha256"), label=f"layer {layer} published SHA-256"
        )

    return {
        "contract_sha256": contract_sha256,
        "fit_model": model["repo_id"],
        "fit_model_revision": model["revision"],
        "fit_quantization": LOCAL_FIT_QUANTIZATION_LABEL,
        "n_prompts": n_prompts,
        "storage_dtype": storage_dtype,
        "layer_records": layer_records,
        "nf4_aggregate_sha256": nf4_aggregate_sha256,
    }


def _tensor_statistics(tensor: Any, *, dtype_name: str) -> dict[str, Any]:
    import torch

    finite = 0
    minimum = math.inf
    maximum = -math.inf
    squared_norm = 0.0
    digest = hashlib.sha256()
    rows = tensor.shape[0]
    for row in range(0, rows, 64):
        chunk = tensor[row : min(row + 64, rows)].detach().cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
        finite += int(torch.isfinite(chunk).sum())
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
        squared_norm += float(torch.square(chunk.double()).sum())
    return {
        "shape": list(tensor.shape),
        "dtype": dtype_name,
        "finite_count": finite,
        "min": minimum,
        "max": maximum,
        "frobenius_norm": math.sqrt(squared_norm),
        "trace": float(torch.trace(tensor.double())),
        "sha256": digest.hexdigest(),
    }


def _compare_tensor_statistics(
    actual: dict[str, Any], declared: dict[str, Any], *, label: str
) -> None:
    for key in ("shape", "dtype", "finite_count", "sha256"):
        if actual[key] != declared.get(key):
            raise ValueError(f"{label} {key} does not match the published tensor")
    for key in ("min", "max", "frobenius_norm", "trace"):
        value = declared.get(key)
        if not isinstance(value, (int, float)) or not math.isclose(
            actual[key], float(value), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(f"{label} {key} does not match the published tensor")


def _verify_local_checkpoint(
    path: Path,
    provenance_metadata: dict[str, Any],
    *,
    check_finite: bool,
    d_model: int,
    source_layers: tuple[int, ...],
    target_layer: int,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected_keys = {"J", "d_model", "metadata", "n_prompts", "source_layers"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_keys:
        keys = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint)
        raise ValueError(f"unexpected local lens checkpoint keys: {keys}")
    if checkpoint["d_model"] != d_model:
        raise ValueError(f"unexpected local lens d_model: {checkpoint['d_model']}")
    if checkpoint["n_prompts"] != provenance_metadata["n_prompts"]:
        raise ValueError("local lens checkpoint/provenance prompt count mismatch")
    if checkpoint["source_layers"] != list(source_layers):
        raise ValueError("local lens checkpoint source layers mismatch")

    metadata = _require_mapping(
        checkpoint["metadata"], label="local lens checkpoint metadata"
    )
    required_metadata = {
        "fit_model": LOCAL_FIT_MODEL_REPO,
        "fit_model_revision": LOCAL_FIT_MODEL_REVISION,
        "fit_quantization": LOCAL_FIT_QUANTIZATION_LABEL,
        "estimator": LOCAL_FIT_ESTIMATOR,
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "contract_sha256": provenance_metadata["contract_sha256"],
        "storage_dtype": provenance_metadata["storage_dtype"],
        "nf4_aggregate_sha256": provenance_metadata["nf4_aggregate_sha256"],
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"local lens checkpoint metadata mismatch: {key}")

    jacobians = checkpoint["J"]
    if not isinstance(jacobians, dict) or set(jacobians) != set(source_layers):
        raise ValueError("local lens Jacobian keys are not exactly the source layers")
    expected_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[provenance_metadata["storage_dtype"]]
    for layer, layer_record in zip(
        source_layers, provenance_metadata["layer_records"], strict=True
    ):
        jacobian = jacobians[layer]
        if not torch.is_tensor(jacobian):
            raise ValueError(f"local lens layer {layer} is not a tensor")
        if jacobian.shape != (d_model, d_model):
            raise ValueError(
                f"local lens layer {layer} has shape {tuple(jacobian.shape)}"
            )
        if jacobian.dtype != expected_dtype:
            raise ValueError(
                f"local lens layer {layer} has dtype {jacobian.dtype}, "
                f"expected {expected_dtype}"
            )
        if check_finite and not bool(torch.isfinite(jacobian).all()):
            raise ValueError(f"local lens layer {layer} contains non-finite values")
        actual_statistics = _tensor_statistics(
            jacobian, dtype_name=provenance_metadata["storage_dtype"]
        )
        published = _require_mapping(
            layer_record.get("published"),
            label=f"layer {layer} published statistics",
        )
        _compare_tensor_statistics(
            actual_statistics, published, label=f"layer {layer} published statistics"
        )
        if provenance_metadata["storage_dtype"] == "float32":
            _compare_tensor_statistics(
                actual_statistics,
                layer_record,
                label=f"layer {layer} authoritative statistics",
            )

    return {
        "checkpoint_keys": sorted(checkpoint),
        "d_model": d_model,
        "n_prompts": checkpoint["n_prompts"],
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "tensor_dtype": str(expected_dtype),
        "tensor_shape": [d_model, d_model],
        "finite_checked": check_finite,
        "fit_model": metadata["fit_model"],
        "fit_model_revision": metadata["fit_model_revision"],
        "fit_quantization": metadata["fit_quantization"],
        "fit_estimator": metadata["estimator"],
        "contract_sha256": metadata["contract_sha256"],
    }


def verify_local_fit_artifact(
    path: Path,
    *,
    expected_sha256: str,
    provenance_path: Path,
    check_finite: bool = True,
    d_model: int = LOCAL_FIT_D_MODEL,
    source_layers: tuple[int, ...] = LOCAL_FIT_SOURCE_LAYERS,
    target_layer: int = LOCAL_FIT_TARGET_LAYER,
    expected_n_prompts: int = LOCAL_FIT_N_PROMPTS,
    expected_nf4_modules: int = 496,
    expected_artifact_count: int = LOCAL_FIT_MODEL_ARTIFACT_COUNT,
    expected_artifacts_sha256: str = LOCAL_FIT_MODEL_ARTIFACTS_SHA256,
) -> dict[str, object]:
    """Verify a completed local fit before applying it to the serving model."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    expected_sha256 = normalize_sha256(
        expected_sha256, label="expected local lens SHA-256"
    )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "local lens SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("local lens provenance is not valid JSON") from exc
    provenance = _require_mapping(provenance, label="local lens provenance")
    provenance_metadata = _verify_local_provenance(
        provenance,
        artifact_sha256=actual_sha256,
        d_model=d_model,
        source_layers=source_layers,
        target_layer=target_layer,
        expected_n_prompts=expected_n_prompts,
        expected_nf4_modules=expected_nf4_modules,
        expected_artifact_count=expected_artifact_count,
        expected_artifacts_sha256=expected_artifacts_sha256,
    )
    checkpoint_metadata = _verify_local_checkpoint(
        path,
        provenance_metadata,
        check_finite=check_finite,
        d_model=d_model,
        source_layers=source_layers,
        target_layer=target_layer,
    )
    return {
        "kind": "local_fit",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_size_bytes": provenance_path.stat().st_size,
        "provenance_sha256": sha256_file(provenance_path),
        **checkpoint_metadata,
    }


def verify_file(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    if size != LENS_SIZE:
        raise ValueError(f"lens size mismatch: expected {LENS_SIZE}, got {size}")
    digest = sha256_file(path)
    if digest != LENS_SHA256:
        raise ValueError(f"lens SHA-256 mismatch: expected {LENS_SHA256}, got {digest}")
    return {"path": str(path.resolve()), "size_bytes": size, "sha256": digest}


def verify_checkpoint(path: Path, *, check_finite: bool) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(checkpoint) != {"J", "d_model", "n_prompts", "source_layers"}:
        raise ValueError(f"unexpected checkpoint keys: {sorted(checkpoint)}")
    expected_layers = list(range(63))
    if checkpoint["d_model"] != 5120:
        raise ValueError(f"unexpected d_model: {checkpoint['d_model']}")
    if checkpoint["n_prompts"] != 1000:
        raise ValueError(f"unexpected n_prompts: {checkpoint['n_prompts']}")
    if checkpoint["source_layers"] != expected_layers:
        raise ValueError("source_layers are not exactly 0..62")
    if sorted(checkpoint["J"]) != expected_layers:
        raise ValueError("Jacobian keys are not exactly 0..62")

    for layer in expected_layers:
        jacobian = checkpoint["J"][layer]
        if jacobian.shape != (5120, 5120):
            raise ValueError(f"layer {layer} has shape {tuple(jacobian.shape)}")
        if jacobian.dtype != torch.float16:
            raise ValueError(f"layer {layer} has dtype {jacobian.dtype}")
        if check_finite and not bool(torch.isfinite(jacobian).all()):
            raise ValueError(f"layer {layer} contains non-finite values")

    return {
        "checkpoint_keys": sorted(checkpoint),
        "d_model": checkpoint["d_model"],
        "n_prompts": checkpoint["n_prompts"],
        "source_layers": expected_layers,
        "tensor_dtype": "torch.float16",
        "tensor_shape": [5120, 5120],
        "finite_checked": check_finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-finite-check",
        action="store_true",
        help="skip the full 3.3 GB tensor finiteness scan",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=LENS_REPO,
            filename=LENS_FILENAME,
            revision=LENS_REVISION,
        )
    )
    result = {
        "repo_id": LENS_REPO,
        "revision": LENS_REVISION,
        "filename": LENS_FILENAME,
        **verify_file(path),
        **verify_checkpoint(path, check_finite=not args.no_finite_check),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
