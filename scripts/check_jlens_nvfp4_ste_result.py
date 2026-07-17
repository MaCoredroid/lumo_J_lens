#!/usr/bin/env python3
"""Validate the completed native NVFP4/FP8-STE Jacobian Lens evidence.

This is an offline evidence checker.  It does not need the 6.6 GB checkpoint,
the model weights, or a GPU.  It binds the compact fit records to the exact
production artifact, reuses the production verifier for every prompt/chunk
proof, and independently recomputes the paired native/public held-out report.

Adapter reconstruction is intentionally a separate certificate.  A valid
report may record a failed adapter certificate; that status is never promoted
to a Jacobian-Lens quality verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_jlens_nvfp4_reports as paired_compare  # noqa: E402
import verify_nvfp4_ste_artifact as artifact_verify  # noqa: E402


VALIDATION = ROOT / "validation"

D_MODEL = 5120
SOURCE_LAYERS = list(range(63))
TARGET_LAYER = 63
N_PROMPTS = 10
FIT_TOKEN_COUNT = 128
IO_ROWS = 64
FIT_CHUNK_COUNT = 20

ARTIFACT_SHA256 = "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057"
ARTIFACT_SIZE = 6_606_046_478
STATE_SHA256 = "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6"
STATE_SIZE = 329_400
PROVENANCE_SHA256 = "289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601"
PROVENANCE_SIZE = 988_263
RUN_PROGRESS_SHA256 = "da1510e27c365c5e6b5202a8fbaa2c16e82ae196067aa98cf66a78fdfa4e1328"
RUN_PROGRESS_SIZE = 565_120
VERIFICATION_SHA256 = "5541e993dd9ba2603118ec2ab55eaeabd27e1a20648990bedd26ede12bd5d1e6"
VERIFICATION_SIZE = 2_157
EXPORT_SHA256 = "499032f7fbcd7b1e6c75303648e7f059ebf292d64a1d70787c30dbdb86b939b5"
EXPORT_SIZE = 1_355
UPSTREAM_LOAD_SHA256 = "b1575cc6828f518a40adb4accb872d86314c337c0a80b894f20f5c03c6dd5616"
UPSTREAM_LOAD_SIZE = 705
GEOMETRY_SHA256 = "43b7431bdc006c1e097e7c187a23671d95f4807c0fe701078ee7345fafdb6fa2"
GEOMETRY_SIZE = 94_335
NATIVE_REPORT_SHA256 = "17ecf282aadd26db281bc2ac5817769ddd1a82ba6b7d0474db386117490e9b90"
NATIVE_REPORT_SIZE = 2_437_812
PUBLIC_REPORT_SHA256 = "fb94cf4f84d110d2b52f473695a675e3e88341836c949658e76fae29a6ebc486"
PUBLIC_REPORT_SIZE = 2_435_472
PAIRED_REPORT_SHA256 = "2fe0d1e6e564119dcb757ec7073da9df051d5c496ed494bb38cd870e32ff6f02"
PAIRED_REPORT_SIZE = 767_523

RUN_ID = "20e4bc8c-9fed-4513-b548-9727f9686222"
CONTRACT_SHA256 = "7944ea163b548edc3372fa67242fbbcfbe0a5abbe95c04ce4a378107ebe03dd0"
SOURCE_FILES_SHA256 = "dbe1f28bbd829fa30cb48b4c593419de205c440d195c12bed398c0036ed16400"
FIT_PROMPT_MANIFEST_SHA256 = "2c36f17dee7287c096f7d1fdb7f8d7ecb8372c6cf8d13d7af8cfaed820439d3b"
FIT_PROMPT_ENTRIES_SHA256 = "ca06a4904a378964cecec933a08df0451efaaf7151a6fa938e5489d6e55784a7"
COMMITTED_PROMPTS_SHA256 = "a1690ab9e88cff53a2eba407195ced52e6908208fedffed68819ee47c1a888c1"
LAYER_AGGREGATE_SHA256 = "a4c2adc7be15232db0e5a8840a6442248caa80a363c0c5239a1ee248f36fb3b4"
METADATA_PAYLOAD_SHA256 = "7b96a49e209e2e3008531fc7d7ac46de1582eb824cba50c95c3b1d7302bb8b66"

MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
FIT_QUANTIZATION = "nvidia-modelopt-nvfp4-fp8-exact-forward-identity-ste"
FIT_ESTIMATOR = "anthropic-future-summed-vjp"

EVAL_PROMPTS_SHA256 = "cd0fe64e800c7b937fcd891196eed6d7c30a8ff1246b9555dc2962bf61c9a56b"
EVAL_ROWS = [3, 18, 42, 49]
EVAL_POSITIONS = [16, 32, 64, 96]
EVAL_CAPTURE_POSITIONS = [16, 32, 64, 96, 127]
EVAL_TOKEN_COUNT = 128
EVAL_TOP_K = 10
EVAL_MAX_MODEL_LEN = 256
EVAL_GPU_MEMORY_UTILIZATION = 0.82
EVAL_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EVAL_OBSERVATIONS = len(EVAL_ROWS) * len(EVAL_POSITIONS) * len(SOURCE_LAYERS)

PUBLIC_REPO = "neuronpedia/jacobian-lens"
PUBLIC_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
PUBLIC_SHA256 = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
PUBLIC_SIZE = 3_303_032_772
PUBLIC_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/"
    "Qwen3.6-27B_jacobian_lens_n1000.pt"
)

FILES = {
    "verification": "jlens-nvfp4-ste-artifact-verification-2026-07-17.json",
    "export": "jlens-nvfp4-ste-export-2026-07-17.json",
    "upstream_load": "jlens-nvfp4-ste-upstream-load-2026-07-17.json",
    "final_metadata": "jlens-nvfp4-ste-final-metadata-2026-07-17.json",
    "fit_state": "jlens-nvfp4-ste-fit-state-2026-07-17.json",
    "run_progress": "jlens-nvfp4-ste-run-progress-2026-07-17.json",
    "geometry": "jlens-nvfp4-ste-vs-public-2026-07-17.json",
    "native_report": "jlens-native-nvfp4-ste-on-nvfp4-heldout-2026-07-17.json",
    "public_report": "jlens-public-schema3-on-nvfp4-heldout-2026-07-17.json",
    "paired_report": "jlens-nvfp4-ste-vs-public-heldout-2026-07-17.json",
}


@dataclass(frozen=True)
class Evidence:
    verification: dict[str, Any]
    export: dict[str, Any]
    upstream_load: dict[str, Any]
    final_metadata: dict[str, Any]
    fit_state: dict[str, Any]
    run_progress: dict[str, Any]
    geometry: dict[str, Any]
    native_report: dict[str, Any]
    public_report: dict[str, Any]
    paired_report: dict[str, Any]
    eval_prompts: dict[str, Any]
    file_sha256: dict[str, str]
    file_size: dict[str, int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], str, int]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict), f"{path.name}: root must be an object")
    return value, _sha256_bytes(raw), len(raw)


def load_evidence(validation_dir: Path = VALIDATION) -> Evidence:
    records: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for label, filename in FILES.items():
        records[label], digests[label], sizes[label] = _load_json(
            validation_dir / filename
        )
    prompts, prompt_digest, prompt_size = _load_json(
        ROOT / "configs" / "jlens_nf4_eval_prompts.json"
    )
    digests["eval_prompts"] = prompt_digest
    sizes["eval_prompts"] = prompt_size
    return Evidence(
        verification=records["verification"],
        export=records["export"],
        upstream_load=records["upstream_load"],
        final_metadata=records["final_metadata"],
        fit_state=records["fit_state"],
        run_progress=records["run_progress"],
        geometry=records["geometry"],
        native_report=records["native_report"],
        public_report=records["public_report"],
        paired_report=records["paired_report"],
        eval_prompts=prompts,
        file_sha256=digests,
        file_size=sizes,
    )


def _parse_timestamp(value: Any, label: str) -> None:
    require(isinstance(value, str) and value, f"{label}: timestamp missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label}: invalid timestamp") from error
    require(parsed.tzinfo is not None, f"{label}: timestamp must include a timezone")


def _finite_tree(value: Any, label: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"{label}: non-finite numeric value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
        return
    raise ValueError(f"{label}: unsupported JSON value")


def _suffix(path: Any, expected: Sequence[str], label: str) -> None:
    require(isinstance(path, str) and path, f"{label}: path missing")
    parts = PurePosixPath(path).parts
    require(tuple(parts[-len(expected) :]) == tuple(expected), f"{label}: path mismatch")


def _exact_file(
    evidence: Evidence, label: str, expected_sha256: str, expected_size: int
) -> None:
    require(
        evidence.file_sha256.get(label) == expected_sha256,
        f"{label}: committed file SHA-256 mismatch",
    )
    require(
        evidence.file_size.get(label) == expected_size,
        f"{label}: committed file size mismatch",
    )


def _check_verification_record(evidence: Evidence) -> None:
    record = evidence.verification
    expected = {
        "checkpoint_keys": ["J", "d_model", "n_prompts", "source_layers"],
        "committed_prompts_sha256": COMMITTED_PROMPTS_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "d_model": D_MODEL,
        "finite_checked": True,
        "fit_estimator": FIT_ESTIMATOR,
        "fit_model": MODEL_REPO,
        "fit_model_revision": MODEL_REVISION,
        "fit_quantization": FIT_QUANTIZATION,
        "kind": "native_nvfp4_ste_fit",
        "layer_aggregate_sha256": LAYER_AGGREGATE_SHA256,
        "n_prompts": N_PROMPTS,
        "provenance_sha256": PROVENANCE_SHA256,
        "provenance_size_bytes": PROVENANCE_SIZE,
        "run_id": RUN_ID,
        "sha256": ARTIFACT_SHA256,
        "size_bytes": ARTIFACT_SIZE,
        "source_layers": SOURCE_LAYERS,
        "state_sha256": STATE_SHA256,
        "state_size_bytes": STATE_SIZE,
        "surrogate_backward": (
            "identity STE; not the literal derivative of quantized rounding"
        ),
        "target_layer": TARGET_LAYER,
        "tensor_dtype": "torch.float32",
        "tensor_shape": [D_MODEL, D_MODEL],
        "verification_scope": "exact pinned production run; not a generic portable fit",
    }
    require(
        set(record) == set(expected) | {"path", "provenance_path", "state_path"},
        "artifact verification fields mismatch",
    )
    for key, value in expected.items():
        require(record.get(key) == value, f"artifact verification {key} mismatch")
    _suffix(record["path"], ["Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt"], "artifact")
    _suffix(
        record["provenance_path"],
        ["nvfp4_ste_fit", "final-mean", "metadata.json"],
        "artifact provenance",
    )
    _suffix(record["state_path"], ["nvfp4_ste_fit", "state.json"], "fit state")


def _check_export_and_upstream(evidence: Evidence) -> None:
    export = evidence.export
    require(
        set(export)
        == {
            "artifact_type",
            "checkpoint_keys",
            "d_model",
            "n_prompts",
            "path",
            "sha256",
            "size_bytes",
            "source",
            "source_layers",
            "tensor_dtype",
            "tensor_shape",
        },
        "export record fields mismatch",
    )
    for key, value in {
        "artifact_type": "upstream-jacobian-lens-torch-checkpoint",
        "checkpoint_keys": ["J", "d_model", "n_prompts", "source_layers"],
        "d_model": D_MODEL,
        "n_prompts": N_PROMPTS,
        "sha256": ARTIFACT_SHA256,
        "size_bytes": ARTIFACT_SIZE,
        "source_layers": SOURCE_LAYERS,
        "tensor_dtype": "torch.float32",
        "tensor_shape": [D_MODEL, D_MODEL],
    }.items():
        require(export.get(key) == value, f"export {key} mismatch")
    _suffix(export["path"], ["Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt"], "export")
    source = export["source"]
    require(
        source.get("contract_sha256") == CONTRACT_SHA256
        and source.get("layer_aggregate_sha256") == LAYER_AGGREGATE_SHA256
        and source.get("metadata_sha256") == PROVENANCE_SHA256,
        "export source binding mismatch",
    )
    _suffix(source.get("final_mean"), ["nvfp4_ste_fit", "final-mean"], "export source")

    upstream = evidence.upstream_load
    require(
        set(upstream)
        == {
            "artifact_sha256",
            "artifact_size_bytes",
            "generated_at",
            "jlens_direct_url",
            "jlens_version",
            "loaders",
            "schema_version",
            "status",
        },
        "upstream loader record fields mismatch",
    )
    require(
        upstream.get("schema_version") == 1
        and upstream.get("status") == "passed"
        and upstream.get("artifact_sha256") == ARTIFACT_SHA256
        and upstream.get("artifact_size_bytes") == ARTIFACT_SIZE
        and upstream.get("jlens_version") == "0.1.0",
        "upstream loader artifact identity mismatch",
    )
    require(
        upstream.get("jlens_direct_url")
        == {
            "url": "https://github.com/anthropics/jacobian-lens.git",
            "vcs_info": {
                "commit_id": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
                "requested_revision": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
                "vcs": "git",
            },
        },
        "upstream Jacobian Lens source pin mismatch",
    )
    require(
        upstream.get("loaders")
        == [
            {"loader": "JacobianLens.load", "status": "passed"},
            {"loader": "JacobianLens.from_pretrained", "status": "passed"},
        ],
        "upstream loader smoke results mismatch",
    )
    _parse_timestamp(upstream.get("generated_at"), "upstream load")


def _check_fit_evidence(evidence: Evidence) -> dict[str, Any]:
    _exact_file(evidence, "verification", VERIFICATION_SHA256, VERIFICATION_SIZE)
    _exact_file(evidence, "export", EXPORT_SHA256, EXPORT_SIZE)
    _exact_file(evidence, "upstream_load", UPSTREAM_LOAD_SHA256, UPSTREAM_LOAD_SIZE)
    _exact_file(evidence, "final_metadata", PROVENANCE_SHA256, PROVENANCE_SIZE)
    _exact_file(evidence, "fit_state", STATE_SHA256, STATE_SIZE)
    _exact_file(evidence, "run_progress", RUN_PROGRESS_SHA256, RUN_PROGRESS_SIZE)
    _check_verification_record(evidence)
    _check_export_and_upstream(evidence)

    layer_records = artifact_verify._verify_final_metadata(
        evidence.final_metadata,
        expected_contract_sha256=CONTRACT_SHA256,
        expected_source_files_sha256=SOURCE_FILES_SHA256,
        expected_prompt_manifest_sha256=FIT_PROMPT_MANIFEST_SHA256,
        expected_prompt_entries_sha256=FIT_PROMPT_ENTRIES_SHA256,
        d_model=D_MODEL,
        source_layers=SOURCE_LAYERS,
        target_layer=TARGET_LAYER,
        n_prompts=N_PROMPTS,
        io_rows=IO_ROWS,
        source_root=ROOT,
    )
    artifact_verify._verify_fit_state(
        evidence.fit_state,
        state_file_sha256=evidence.file_sha256["fit_state"],
        expected_state_sha256=STATE_SHA256,
        metadata=evidence.final_metadata,
        metadata_file_sha256=evidence.file_sha256["final_metadata"],
        expected_contract_sha256=CONTRACT_SHA256,
        d_model=D_MODEL,
        source_layers=SOURCE_LAYERS,
        n_prompts=N_PROMPTS,
        io_rows=IO_ROWS,
    )

    require(
        evidence.final_metadata.get("run_id") == RUN_ID
        and evidence.final_metadata.get("metadata_sha256") == METADATA_PAYLOAD_SHA256
        and evidence.final_metadata.get("layer_aggregate_sha256")
        == LAYER_AGGREGATE_SHA256
        and evidence.final_metadata.get("committed_prompts_sha256")
        == COMMITTED_PROMPTS_SHA256,
        "final metadata production identity mismatch",
    )
    embedded_progress = evidence.final_metadata["metadata"]["progress"]
    progress_payload = {
        key: evidence.run_progress[key]
        for key in (
            "schema_version",
            "contract_sha256",
            "prompts",
            "max_cuda_peak_allocated_bytes",
            "max_cuda_peak_reserved_bytes",
        )
    }
    require(
        progress_payload == embedded_progress,
        "standalone run progress payload differs from final provenance",
    )
    require(
        evidence.run_progress.get("schema_version") == 1
        and evidence.run_progress.get("status") == "completed"
        and evidence.run_progress.get("contract_sha256") == CONTRACT_SHA256,
        "run progress is not the completed production run",
    )
    _parse_timestamp(evidence.run_progress.get("created_at"), "run progress creation")
    _parse_timestamp(evidence.run_progress.get("updated_at"), "run progress update")
    _parse_timestamp(evidence.run_progress.get("completed_at"), "run progress completion")

    prompt_progress = evidence.run_progress.get("prompts")
    require(
        isinstance(prompt_progress, dict)
        and set(prompt_progress) == {str(index) for index in range(N_PROMPTS)},
        "run progress does not contain exactly ten prompt proofs",
    )
    for index in range(N_PROMPTS):
        prompt = prompt_progress[str(index)]
        require(
            set(prompt) == {"capture_binding", "capture_invocations", "chunks", "commit"},
            f"prompt {index} progress fields mismatch",
        )
        require(
            len(prompt.get("chunks", [])) == FIT_CHUNK_COUNT,
            f"prompt {index} does not contain exactly 20 chunks",
        )
        proof = prompt["capture_binding"]["proof_claim"]
        require(
            proof["generation_record_parity"]["exact"] is True
            and proof["shared_internal_tensor_parity"]["all_shared_bit_exact"] is True
            and proof["replay_parameter_parity"]["all_content_hashes_equal"] is True
            and proof["observer_capture_completeness"]["required_missing"] == [],
            f"prompt {index} exact-forward proof is incomplete",
        )

    return {
        "run_id": RUN_ID,
        "prompts": N_PROMPTS,
        "chunks_per_prompt": FIT_CHUNK_COUNT,
        "layers": len(layer_records),
        "artifact_sha256": ARTIFACT_SHA256,
        "state_sha256": STATE_SHA256,
        "provenance_sha256": PROVENANCE_SHA256,
    }


def _check_public_artifact(record: Mapping[str, Any], label: str) -> None:
    expected = {
        "checkpoint_keys": ["J", "d_model", "n_prompts", "source_layers"],
        "d_model": D_MODEL,
        "filename": PUBLIC_FILENAME,
        "finite_checked": False,
        "kind": "pinned_public",
        "n_prompts": 1000,
        "repo_id": PUBLIC_REPO,
        "revision": PUBLIC_REVISION,
        "sha256": PUBLIC_SHA256,
        "size_bytes": PUBLIC_SIZE,
        "source_layers": SOURCE_LAYERS,
        "tensor_dtype": "torch.float16",
        "tensor_shape": [D_MODEL, D_MODEL],
    }
    require(set(record) == set(expected) | {"path"}, f"{label}: public fields mismatch")
    for key, value in expected.items():
        require(record.get(key) == value, f"{label}: public {key} mismatch")
    require(isinstance(record.get("path"), str) and record["path"], f"{label}: path missing")


def _check_quantile_summary(
    record: Mapping[str, Any], *, count: int, label: str
) -> None:
    require(
        set(record) == {"count", "mean", "standard_deviation", "quantiles"},
        f"{label}: quantile summary fields mismatch",
    )
    require(record.get("count") == count, f"{label}: quantile count mismatch")
    require(
        set(record.get("quantiles", {}))
        == {"p000", "p005", "p025", "p050", "p075", "p095", "p100"},
        f"{label}: quantile grid mismatch",
    )
    _finite_tree(record, label)


def _check_geometry(evidence: Evidence) -> dict[str, Any]:
    _exact_file(evidence, "geometry", GEOMETRY_SHA256, GEOMETRY_SIZE)
    geometry = evidence.geometry
    require(
        set(geometry)
        == {
            "aggregate",
            "artifacts",
            "configuration",
            "generated_at",
            "layers",
            "schema_version",
            "scope",
        },
        "geometry report fields mismatch",
    )
    require(geometry.get("schema_version") == 1, "geometry schema mismatch")
    require(
        geometry.get("scope")
        == "numeric matrix comparison only; no held-out token or logit evaluation",
        "geometry scope mismatch",
    )
    _parse_timestamp(geometry.get("generated_at"), "geometry generation")
    require(
        geometry.get("configuration")
        == {
            "d_model": D_MODEL,
            "source_layers": SOURCE_LAYERS,
            "row_chunk": 16,
            "row_cosine_quantiles": [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
            "relative_difference_reference": "public lens",
        },
        "geometry configuration mismatch",
    )
    artifacts = geometry.get("artifacts")
    require(isinstance(artifacts, dict), "geometry artifacts missing")
    require(
        artifacts.get("local") == evidence.verification,
        "geometry is not bound to the exact native artifact record",
    )
    _check_public_artifact(artifacts.get("public", {}), "geometry")

    layers = geometry.get("layers")
    require(isinstance(layers, list) and len(layers) == 63, "geometry layer count mismatch")
    require([item.get("layer") for item in layers] == SOURCE_LAYERS, "geometry layer order mismatch")
    for layer, record in enumerate(layers):
        require(
            set(record)
            == {
                "difference_frobenius_norm",
                "frobenius_cosine",
                "layer",
                "local",
                "public",
                "relative_difference_denominator",
                "relative_frobenius_difference",
                "row_wise_cosine",
            },
            f"geometry layer {layer} fields mismatch",
        )
        require(
            record.get("relative_difference_denominator") == "public_frobenius_norm",
            f"geometry layer {layer} denominator mismatch",
        )
        _check_quantile_summary(
            record.get("row_wise_cosine", {}), count=D_MODEL, label=f"geometry layer {layer} rows"
        )
        for side in ("local", "public"):
            side_record = record.get(side, {})
            require(
                set(side_record) == {"frobenius_norm", "trace", "best_scalar_identity"},
                f"geometry layer {layer} {side} fields mismatch",
            )
            require(
                float(side_record.get("frobenius_norm", 0)) > 0,
                f"geometry layer {layer} {side} norm is not positive",
            )
            require(
                set(side_record.get("best_scalar_identity", {}))
                == {"scalar", "frobenius_norm", "energy_fraction", "residual_relative_norm"},
                f"geometry layer {layer} {side} identity fields mismatch",
            )
        _finite_tree(record, f"geometry layer {layer}")

    aggregate = geometry.get("aggregate", {})
    require(aggregate.get("layer_count") == 63, "geometry aggregate layer count mismatch")
    _check_quantile_summary(
        aggregate.get("all_rows_cosine", {}), count=63 * D_MODEL, label="geometry all rows"
    )
    _check_quantile_summary(
        aggregate.get("per_layer_frobenius_cosine", {}), count=63, label="geometry layer cosine"
    )
    _check_quantile_summary(
        aggregate.get("per_layer_relative_frobenius_difference", {}),
        count=63,
        label="geometry layer relative difference",
    )
    _finite_tree(aggregate, "geometry aggregate")

    # These are consistency identities, not quality thresholds.
    local_total = math.sqrt(sum(float(item["local"]["frobenius_norm"]) ** 2 for item in layers))
    public_total = math.sqrt(sum(float(item["public"]["frobenius_norm"]) ** 2 for item in layers))
    difference_total = math.sqrt(
        sum(float(item["difference_frobenius_norm"]) ** 2 for item in layers)
    )
    for key, derived in (
        ("local_total_frobenius_norm", local_total),
        ("public_total_frobenius_norm", public_total),
        ("difference_total_frobenius_norm", difference_total),
    ):
        require(
            math.isclose(float(aggregate.get(key)), derived, rel_tol=1e-12, abs_tol=1e-12),
            f"geometry aggregate {key} is not derived from layers",
        )
    require(
        math.isclose(
            float(aggregate.get("global_relative_frobenius_difference")),
            difference_total / public_total,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "geometry aggregate relative difference is inconsistent",
    )
    inner = (local_total**2 + public_total**2 - difference_total**2) / 2
    require(
        math.isclose(
            float(aggregate.get("global_frobenius_cosine")),
            max(-1.0, min(1.0, inner / (local_total * public_total))),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "geometry aggregate cosine is inconsistent",
    )
    return {
        "global_relative_frobenius_difference": aggregate[
            "global_relative_frobenius_difference"
        ],
        "global_frobenius_cosine": aggregate["global_frobenius_cosine"],
    }


def _check_eval_prompt_manifest(evidence: Evidence) -> list[dict[str, Any]]:
    require(
        evidence.file_sha256.get("eval_prompts") == EVAL_PROMPTS_SHA256,
        "held-out prompt manifest SHA-256 mismatch",
    )
    manifest = evidence.eval_prompts
    require(
        set(manifest)
        == {"dataset", "prompts", "schema_version", "selection", "tokenizer"},
        "held-out prompt manifest fields mismatch",
    )
    require(manifest.get("schema_version") == 1, "held-out prompt schema mismatch")
    require(
        manifest.get("selection")
        == {
            "minimum_stripped_characters": 600,
            "order": "dataset row order",
            "required_token_count": EVAL_TOKEN_COUNT,
            "take": len(EVAL_ROWS),
        },
        "held-out prompt selection contract mismatch",
    )
    require(
        manifest.get("dataset")
        == {
            "config": "wikitext-103-raw-v1",
            "repo": "Salesforce/wikitext",
            "revision": EVAL_DATASET_REVISION,
            "split": "validation",
        },
        "held-out dataset identity mismatch",
    )
    require(
        manifest.get("tokenizer")
        == {
            "add_special_tokens": True,
            "force_bos_when_supported": True,
            "repo": "Qwen/Qwen3.6-27B",
            "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "truncation": "right",
        },
        "held-out tokenizer identity mismatch",
    )
    prompts = manifest.get("prompts")
    require(isinstance(prompts, list) and len(prompts) == 4, "held-out prompt count mismatch")
    require([item.get("row_index") for item in prompts] == EVAL_ROWS, "held-out row order mismatch")
    for item in prompts:
        row = item["row_index"]
        require(
            set(item) == {"row_index", "text", "text_sha256", "token_count", "token_ids"},
            f"held-out row {row} fields mismatch",
        )
        require(item.get("token_count") == EVAL_TOKEN_COUNT, f"held-out row {row} length mismatch")
        token_ids = item.get("token_ids")
        require(
            isinstance(token_ids, list)
            and len(token_ids) == EVAL_TOKEN_COUNT
            and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in token_ids),
            f"held-out row {row} token IDs mismatch",
        )
        text = item.get("text")
        require(isinstance(text, str), f"held-out row {row} text missing")
        require(
            _sha256_bytes(text.encode("utf-8")) == item.get("text_sha256"),
            f"held-out row {row} text SHA-256 mismatch",
        )
    return prompts


def _check_top_k_readout(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label}: readout missing")
    for field in ("token_ids", "tokens", "scores"):
        require(
            isinstance(value.get(field), list) and len(value[field]) == EVAL_TOP_K,
            f"{label}: expected exact top-{EVAL_TOP_K} {field}",
        )


def _check_report_prompt_contract(
    report: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]], label: str
) -> None:
    require(report.get("schema_version") == 3, f"{label}: schema must be 3")
    require(
        report.get("score_encoding") == "unrounded-float32",
        f"{label}: scores are not marked as unrounded float32",
    )
    runtime = report.get("runtime", {})
    require(
        runtime.get("max_model_len") == EVAL_MAX_MODEL_LEN
        and runtime.get("gpu_memory_utilization") == EVAL_GPU_MEMORY_UTILIZATION,
        f"{label}: runtime evaluation grid mismatch",
    )
    experiments = report.get("experiments")
    require(isinstance(experiments, list) and len(experiments) == 4, f"{label}: prompt count mismatch")
    for prompt, experiment in zip(prompts, experiments, strict=True):
        row = prompt["row_index"]
        prompt_id = f"wikitext-validation-row-{row}"
        require(experiment.get("id") == prompt_id, f"{label}: prompt ID mismatch for row {row}")
        require(
            experiment.get("prompt_token_ids") == prompt["token_ids"],
            f"{label}: frozen token IDs mismatch for row {row}",
        )
        require(len(experiment.get("prompt_tokens", [])) == EVAL_TOKEN_COUNT, f"{label}: token decode count mismatch for row {row}")
        require(
            experiment.get("positions_requested") == EVAL_POSITIONS
            and experiment.get("positions_resolved") == EVAL_POSITIONS
            and experiment.get("capture_positions_resolved") == EVAL_CAPTURE_POSITIONS
            and experiment.get("final_validation_position") == 127,
            f"{label}: position contract mismatch for row {row}",
        )
        for index, readout in enumerate(experiment.get("final_model_readout", [])):
            _check_top_k_readout(readout, f"{label}.{prompt_id}.final_model[{index}]")
        for index, readout in enumerate(experiment.get("captured_final_model_readout", [])):
            _check_top_k_readout(readout, f"{label}.{prompt_id}.captured_final[{index}]")
        require(
            len(experiment.get("final_model_readout", [])) == len(EVAL_CAPTURE_POSITIONS)
            and len(experiment.get("captured_final_model_readout", []))
            == len(EVAL_CAPTURE_POSITIONS),
            f"{label}: final readout capture grid mismatch for row {row}",
        )
        layers = experiment.get("layers")
        require(
            isinstance(layers, list)
            and [record.get("layer") for record in layers] == SOURCE_LAYERS,
            f"{label}: layer grid mismatch for row {row}",
        )
        for layer in layers:
            positions = layer.get("positions")
            require(
                isinstance(positions, list)
                and [record.get("token_position") for record in positions]
                == EVAL_POSITIONS,
                f"{label}: layer position grid mismatch for row {row}, layer {layer.get('layer')}",
            )
            for position in positions:
                _check_top_k_readout(
                    position.get("jacobian_lens"),
                    f"{label}.{prompt_id}.layer-{layer['layer']}.jacobian",
                )
                _check_top_k_readout(
                    position.get("logit_lens"),
                    f"{label}.{prompt_id}.layer-{layer['layer']}.logit",
                )


def _check_native_lens_binding(evidence: Evidence) -> None:
    lens = evidence.native_report.get("lens")
    require(isinstance(lens, dict), "native report lens record missing")
    require(
        set(lens) == set(evidence.verification) | {"application"},
        "native report lens fields differ from artifact verification",
    )
    for key, value in evidence.verification.items():
        require(lens.get(key) == value, f"native report is not bound to artifact field {key}")
    require(
        lens.get("application")
        == (
            f"{FIT_QUANTIZATION} fitted lens applied to strictly rehashed "
            "NVIDIA NVFP4/FP8 residuals"
        ),
        "native report lens application mismatch",
    )


def _check_paired_report(
    evidence: Evidence, derived: Mapping[str, Any]
) -> dict[str, Any]:
    paired = evidence.paired_report
    expected_keys = set(derived) | {"generated_at", "input_files"}
    require(set(paired) == expected_keys, "paired report fields mismatch")
    for key, value in derived.items():
        require(paired.get(key) == value, f"paired report {key} was not independently derived")
    _parse_timestamp(paired.get("generated_at"), "paired report generation")
    input_files = paired.get("input_files")
    require(
        isinstance(input_files, dict) and set(input_files) == {"native", "public"},
        "paired input-file records missing",
    )
    for side, evidence_label, expected_name in (
        ("native", "native_report", FILES["native_report"]),
        ("public", "public_report", FILES["public_report"]),
    ):
        record = input_files[side]
        require(
            set(record) == {"path", "sha256", "size_bytes"},
            f"paired {side} input fields mismatch",
        )
        require(
            record.get("sha256") == evidence.file_sha256[evidence_label]
            and record.get("size_bytes") == evidence.file_size[evidence_label],
            f"paired {side} input digest mismatch",
        )
        _suffix(record.get("path"), [expected_name], f"paired {side} input")

    adapters = paired.get("adapter_certificates")
    require(isinstance(adapters, dict), "paired adapter certificates missing")
    require(
        adapters.get("paired_diagnostics_identical") is True
        and adapters.get("paired_report_status_identical") is True
        and adapters.get("paired_assertions_identical") is True,
        "paired adapter certificates are not independently aligned",
    )
    require(
        "status" not in paired and "adapter_certificates" not in paired.get("metrics", {}),
        "adapter certificate was conflated with the lens metrics",
    )
    return {
        "native": adapters["native"]["report_status"],
        "public": adapters["public"]["report_status"],
    }


def _check_heldout_evidence(evidence: Evidence) -> dict[str, Any]:
    _exact_file(
        evidence, "native_report", NATIVE_REPORT_SHA256, NATIVE_REPORT_SIZE
    )
    _exact_file(
        evidence, "public_report", PUBLIC_REPORT_SHA256, PUBLIC_REPORT_SIZE
    )
    _exact_file(
        evidence, "paired_report", PAIRED_REPORT_SHA256, PAIRED_REPORT_SIZE
    )
    prompts = _check_eval_prompt_manifest(evidence)
    _check_native_lens_binding(evidence)

    # This validates the exact ModelOpt metadata and all three shard digests,
    # pinned host/package/runtime identity, raw finite readouts, residual hashes,
    # public artifact provenance, and independently derived adapter assertions.
    derived = paired_compare.compare_reports(
        evidence.native_report, evidence.public_report
    )
    _check_report_prompt_contract(evidence.native_report, prompts, "native")
    _check_report_prompt_contract(evidence.public_report, prompts, "public")

    pairing = derived.get("pairing", {})
    require(
        pairing.get("prompt_ids")
        == [f"wikitext-validation-row-{row}" for row in EVAL_ROWS]
        and pairing.get("prompt_count") == len(EVAL_ROWS)
        and pairing.get("source_layers") == SOURCE_LAYERS
        and pairing.get("positions_by_prompt")
        == {f"wikitext-validation-row-{row}": EVAL_POSITIONS for row in EVAL_ROWS}
        and pairing.get("top_k") == EVAL_TOP_K
        and pairing.get("score_encoding") == "unrounded-float32"
        and pairing.get("observation_count") == EVAL_OBSERVATIONS,
        "paired held-out grid mismatch",
    )
    _finite_tree(derived.get("metrics"), "paired lens metrics")
    adapter_status = _check_paired_report(evidence, derived)
    return {
        "prompt_count": len(EVAL_ROWS),
        "observation_count": EVAL_OBSERVATIONS,
        "adapter_status": adapter_status,
        "overall": derived["metrics"]["overall"],
        "macro_layer_comparisons": derived["metrics"]["macro_layer_comparisons"],
    }


def validate_evidence(evidence: Evidence) -> dict[str, Any]:
    fit = _check_fit_evidence(evidence)
    geometry = _check_geometry(evidence)
    heldout = _check_heldout_evidence(evidence)
    return {"fit": fit, "geometry": geometry, "heldout": heldout}


def main() -> int:
    summary = validate_evidence(load_evidence())
    overall = summary["heldout"]["overall"]
    comparison = overall["comparisons"]["native_vs_public_jacobian_lens"]
    native_method = overall["methods"]["native_jacobian_lens"]
    public_method = overall["methods"]["public_jacobian_lens"]
    print(
        json.dumps(
            {
                "status": "validated",
                "scope": (
                    "exact native NVFP4 forward with declared identity-STE surrogate "
                    "backward; adapter certificate reported separately"
                ),
                "fit": summary["fit"],
                "geometry": summary["geometry"],
                "heldout": {
                    "prompt_count": summary["heldout"]["prompt_count"],
                    "observation_count": summary["heldout"]["observation_count"],
                    "adapter_status": summary["heldout"]["adapter_status"],
                    "native_jacobian_lens": {
                        "target_top1_rate": native_method["target_top1_rate"],
                        "target_top5_rate": native_method["target_top5_rate"],
                    },
                    "public_jacobian_lens": {
                        "target_top1_rate": public_method["target_top1_rate"],
                        "target_top5_rate": public_method["target_top5_rate"],
                    },
                    "native_vs_public": {
                        "top1_agreement_rate": comparison["top1_agreement_rate"],
                        "top5_overlap_mean_fraction": comparison[
                            "top5_overlap_mean_fraction"
                        ],
                        "target_rank_spearman": comparison["target_rank"][
                            "spearman"
                        ],
                    },
                },
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
