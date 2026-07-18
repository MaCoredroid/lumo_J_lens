#!/usr/bin/env python3
"""Analyze the frozen N=20 greedy-next-token J-lens transport control.

The three lens reports are intentionally read one at a time.  Each large
experiment object is reduced to the identity, numerical-certification, rank,
and log-probability fields needed by the frozen protocol before the next
report is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "swe_next_token_transport_protocol.json"
PROTOCOL_SHA256 = "2474b31630f3074daea03577a64f32ce74332e3d4f028a3990b0817e50d6a331"
REPORT_LABELS = ("public", "nf4", "native")
JACOBIAN_METHOD = {
    "public": "public_jacobian",
    "nf4": "nf4_jacobian",
    "native": "native_jacobian",
}
METHODS = (
    "ordinary_logit",
    "public_jacobian",
    "nf4_jacobian",
    "native_jacobian",
)
COMPARISONS = (
    ("public_jacobian_minus_ordinary_logit", "public_jacobian", "ordinary_logit"),
    ("public_jacobian_minus_native_jacobian", "public_jacobian", "native_jacobian"),
    ("public_jacobian_minus_nf4_jacobian", "public_jacobian", "nf4_jacobian"),
)
STRICT_FINAL_NORM_MAX_ABS = 0.125
STRICT_FINAL_NORM_RMS = 0.006
STRICT_FINAL_LOGITS_MAX_ABS = 0.0625
STRICT_FINAL_LOGITS_RMS = 0.01
TOP_K_PREFIX = 5
REPORT_SCHEMA_VERSION = 3
ARRAY_MARKER = re.compile(rb'(?m)^\s*"experiments"\s*:\s*\[')
MODEL_PIN = {
    "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
    "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}
LENS_PINS = {
    "public": {
        "repo_id": "neuronpedia/jacobian-lens",
        "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
        "sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
        "n_prompts": 1000,
    },
    "nf4": {
        "kind": "local_fit",
        "sha256": "54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f",
        "provenance_sha256": "08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7",
        "n_prompts": 10,
    },
    "native": {
        "kind": "native_nvfp4_ste_fit",
        "sha256": "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057",
        "provenance_sha256": "289e93a0c99579a0d5637cb37b42c4575b73eb2c38d35d47963de85178e90601",
        "state_sha256": "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6",
        "n_prompts": 10,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def percentile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "percentile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_protocol(value: Any, *, protocol_sha256: str) -> Mapping[str, Any]:
    protocol = mapping(value, "next-token transport protocol")
    require(protocol_sha256 == PROTOCOL_SHA256, "frozen transport protocol hash changed")
    require(protocol.get("schema_version") == 2, "transport protocol schema changed")
    require(
        protocol.get("id") == "swe-n20-greedy-next-token-transport-v2",
        "transport protocol identity changed",
    )
    require(
        protocol.get("methods") == list(METHODS),
        "transport method order changed",
    )
    readout = mapping(protocol.get("readout"), "transport readout")
    require(
        readout.get("layers") == list(range(24, 48))
        and readout.get("position") == -1
        and readout.get("layer_selection") == "none",
        "fixed transport readout changed",
    )
    inference = mapping(protocol.get("inference"), "transport inference")
    require(
        inference.get("algorithm")
        == "paired_hierarchical_repository_then_task_percentile_v1"
        and inference.get("same_draw_for_both_methods") is True
        and inference.get("resample_repositories_then_tasks_within_repository")
        is True
        and inference.get("resample_layers_or_checkpoints") is False,
        "transport inference contract changed",
    )
    return protocol


def _stream_json_experiments(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, int]:
    """Read a report without retaining its large experiment payloads together."""

    header = bytearray()
    with path.open("rb") as handle:
        initial_stat = os.fstat(handle.fileno())
        source_digest = hashlib.sha256()
        source_bytes = 0

        def read_hashed(size: int = -1) -> bytes:
            nonlocal source_bytes
            value = handle.read(size)
            source_digest.update(value)
            source_bytes += len(value)
            return value

        match = None
        while match is None:
            chunk = read_hashed(1024 * 1024)
            require(bool(chunk), f"{path} has no experiments array")
            header.extend(chunk)
            match = ARRAY_MARKER.search(header)
            require(
                len(header) <= 16 * 1024 * 1024,
                f"{path} experiments array was not found in the report header",
            )

        array_open = match.end() - 1
        metadata_prefix = bytes(header[:array_open])
        pending = bytes(header[match.end() :])
        compact_rows: list[dict[str, Any]] = []
        current = bytearray()
        depth = 0
        in_string = False
        escaped = False
        started = False
        finished = False
        suffix = b""

        while not finished:
            if not pending:
                pending = read_hashed(1024 * 1024)
                require(bool(pending), f"{path} experiments array is truncated")
            index = 0
            while index < len(pending):
                byte = pending[index]
                if not started:
                    if byte in b" \t\r\n,":
                        index += 1
                        continue
                    if byte == ord("]"):
                        suffix = pending[index + 1 :] + read_hashed()
                        finished = True
                        break
                    require(byte == ord("{"), f"{path} has malformed experiment array")
                    started = True
                    depth = 1
                    current.append(byte)
                    index += 1
                    continue

                current.append(byte)
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                elif byte == ord('"'):
                    in_string = True
                elif byte == ord("{"):
                    depth += 1
                elif byte == ord("}"):
                    depth -= 1
                    if depth == 0:
                        experiment = mapping(
                            json.loads(current), f"{path} experiment"
                        )
                        compact_rows.append(_compact_experiment(experiment, path.name))
                        current.clear()
                        started = False
                index += 1
            pending = b"" if not finished else pending
        final_stat = os.fstat(handle.fileno())

    require(
        initial_stat.st_dev == final_stat.st_dev
        and initial_stat.st_ino == final_stat.st_ino
        and initial_stat.st_size == final_stat.st_size == source_bytes
        and initial_stat.st_mtime_ns == final_stat.st_mtime_ns,
        f"{path} changed while it was being reduced",
    )

    metadata_value = json.loads(metadata_prefix + b"[]" + suffix)
    metadata = dict(mapping(metadata_value, f"{path} report metadata"))
    return metadata, compact_rows, source_digest.hexdigest(), source_bytes


def _readout_record(value: Any, *, label: str, vocabulary_size: int) -> dict[str, Any]:
    record = mapping(value, label)
    target_id = integer(record.get("target_token_id"), f"{label} target token ID")
    require(target_id < vocabulary_size, f"{label} target token ID exceeds vocabulary")
    rank = integer(record.get("target_rank"), f"{label} target rank", minimum=1)
    require(rank <= vocabulary_size, f"{label} target rank exceeds vocabulary")
    logprob = finite(record.get("target_logprob"), f"{label} target logprob")
    return {
        "target_token_id": target_id,
        "target_rank": rank,
        "target_logprob": logprob,
        "record_sha256": sha256_json(record),
    }


def _numeric_diagnostics(
    experiment: Mapping[str, Any], *, generated_token_id: int, label: str
) -> dict[str, Any]:
    final = sequence(experiment.get("final_model_readout"), f"{label} final readout")
    captured = sequence(
        experiment.get("captured_final_model_readout"), f"{label} captured readout"
    )
    require(len(final) == len(captured) == 1, f"{label} must capture final position only")
    final_record = mapping(final[0], f"{label} reconstructed final readout")
    captured_record = mapping(captured[0], f"{label} captured final readout")
    for name, record in (("reconstructed", final_record), ("captured", captured_record)):
        require(
            integer(record.get("target_token_id"), f"{label} {name} target token")
            == generated_token_id,
            f"{label} {name} final target differs from generated token",
        )
        finite(record.get("target_logprob"), f"{label} {name} target logprob")
        integer(record.get("target_rank"), f"{label} {name} target rank", minimum=1)

    reconstructed_top = sequence(final_record.get("token_ids"), f"{label} final top-k")
    captured_top = sequence(captured_record.get("token_ids"), f"{label} captured top-k")
    require(
        len(reconstructed_top) >= TOP_K_PREFIX and len(captured_top) >= TOP_K_PREFIX,
        f"{label} final top-k is shorter than {TOP_K_PREFIX}",
    )
    explicit_top1 = (
        reconstructed_top[0] == generated_token_id
        and captured_top[0] == generated_token_id
    )
    require(
        experiment.get("final_layer_top1_matches_greedy") is explicit_top1,
        f"{label} final greedy assertion disagrees with captured records",
    )

    norm = mapping(experiment.get("final_norm_reconstruction"), f"{label} final norm")
    require(
        finite(norm.get("max_abs_tolerance"), f"{label} norm max tolerance")
        == STRICT_FINAL_NORM_MAX_ABS
        and finite(norm.get("rms_tolerance"), f"{label} norm RMS tolerance")
        == STRICT_FINAL_NORM_RMS,
        f"{label} final norm tolerance changed",
    )
    norm_max = finite(norm.get("max_abs_error"), f"{label} norm max error")
    norm_rms = finite(norm.get("rms_error"), f"{label} norm RMS error")
    norm_pass = norm_max <= STRICT_FINAL_NORM_MAX_ABS and norm_rms <= STRICT_FINAL_NORM_RMS
    require(
        norm.get("within_tolerance") is norm_pass,
        f"{label} final norm certification is inconsistent",
    )

    logits = mapping(
        experiment.get("final_logits_reconstruction"), f"{label} final logits"
    )
    require(
        finite(logits.get("max_abs_tolerance"), f"{label} logits max tolerance")
        == STRICT_FINAL_LOGITS_MAX_ABS
        and finite(logits.get("rms_tolerance"), f"{label} logits RMS tolerance")
        == STRICT_FINAL_LOGITS_RMS
        and integer(logits.get("top_k_prefix"), f"{label} top-k prefix", minimum=1)
        == TOP_K_PREFIX,
        f"{label} final-logit certification contract changed",
    )
    logits_max = finite(logits.get("max_abs_error"), f"{label} logits max error")
    logits_rms = finite(logits.get("rms_error"), f"{label} logits RMS error")
    top5_pass = (
        logits.get("top_k_prefix_token_ids_match") is True
        and reconstructed_top[:TOP_K_PREFIX] == captured_top[:TOP_K_PREFIX]
    )
    strict_logits_pass = (
        logits_max <= STRICT_FINAL_LOGITS_MAX_ABS
        and logits_rms <= STRICT_FINAL_LOGITS_RMS
        and top5_pass
    )
    require(
        logits.get("within_tolerance") is strict_logits_pass,
        f"{label} final-logit certification is inconsistent",
    )
    strict = explicit_top1 and norm_pass and strict_logits_pass
    sensitivity = (
        explicit_top1
        and top5_pass
        and norm_pass
        and logits_rms <= 0.02
        and logits_max <= 0.125
    )
    return {
        "captured_and_reconstructed_greedy_top1_match": explicit_top1,
        "captured_and_reconstructed_top5_prefix_match": top5_pass,
        "final_norm_within_strict_tolerances": norm_pass,
        "final_logits_max_abs_error": logits_max,
        "final_logits_rms_error": logits_rms,
        "strict_certified": strict,
        "sensitivity_certified": sensitivity,
    }


def _compact_experiment(experiment: Mapping[str, Any], report_label: str) -> dict[str, Any]:
    prompt_id = text(experiment.get("id"), f"{report_label} prompt ID")
    generated = integer(
        experiment.get("generated_token_id"), f"{prompt_id} generated token ID"
    )
    require(generated < 248320, f"{prompt_id} generated token ID exceeds vocabulary")
    prompt_token_ids = sequence(
        experiment.get("prompt_token_ids"), f"{prompt_id} prompt token IDs"
    )
    require(bool(prompt_token_ids), f"{prompt_id} has an empty prompt")
    require(
        all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in prompt_token_ids),
        f"{prompt_id} prompt token IDs are malformed",
    )
    prompt = experiment.get("prompt")
    require(isinstance(prompt, str), f"{prompt_id} prompt must be text")
    final_position = integer(
        experiment.get("final_validation_position"), f"{prompt_id} final position"
    )
    require(final_position == len(prompt_token_ids) - 1, f"{prompt_id} final position changed")
    require(
        experiment.get("positions_requested") == [-1]
        and experiment.get("positions_resolved") == [final_position]
        and experiment.get("capture_positions_resolved") == [final_position],
        f"{prompt_id} is not the frozen final-position readout",
    )

    metadata = mapping(experiment.get("metadata"), f"{prompt_id} metadata")
    task = mapping(metadata.get("task"), f"{prompt_id} task metadata")
    cohort = mapping(metadata.get("cohort"), f"{prompt_id} cohort metadata")
    task_id = text(task.get("instance_id"), f"{prompt_id} task ID")
    repo = text(task.get("repo"), f"{prompt_id} repository")
    cohort_id = text(cohort.get("id"), f"{prompt_id} cohort ID")
    cohort_manifest = text(
        cohort.get("cohort_manifest_sha256"), f"{prompt_id} cohort manifest SHA"
    )

    residual = mapping(
        experiment.get("residual_capture_manifest"), f"{prompt_id} residual manifest"
    )
    require(
        residual.get("token_positions") == [final_position]
        and integer(residual.get("tensor_count"), f"{prompt_id} residual count", minimum=1)
        == 64,
        f"{prompt_id} residual capture identity changed",
    )

    vocabulary_size = 248320
    layer_values = sequence(experiment.get("layers"), f"{prompt_id} layers")
    require(len(layer_values) == 24, f"{prompt_id} must have 24 fixed layers")
    layers: dict[str, Any] = {}
    observed_layers: list[int] = []
    for layer_value in layer_values:
        layer = mapping(layer_value, f"{prompt_id} layer")
        layer_id = integer(layer.get("layer"), f"{prompt_id} layer ID")
        observed_layers.append(layer_id)
        positions = sequence(layer.get("positions"), f"{prompt_id} layer {layer_id} positions")
        require(len(positions) == 1, f"{prompt_id} layer {layer_id} has extra positions")
        position = mapping(positions[0], f"{prompt_id} layer {layer_id} position")
        require(
            position.get("token_position") == final_position,
            f"{prompt_id} layer {layer_id} position differs from final boundary",
        )
        ordinary = _readout_record(
            position.get("logit_lens"),
            label=f"{prompt_id} layer {layer_id} ordinary logit",
            vocabulary_size=vocabulary_size,
        )
        jacobian = _readout_record(
            position.get("jacobian_lens"),
            label=f"{prompt_id} layer {layer_id} Jacobian",
            vocabulary_size=vocabulary_size,
        )
        require(
            ordinary["target_token_id"] == generated
            and jacobian["target_token_id"] == generated,
            f"{prompt_id} layer {layer_id} target differs from generated token",
        )
        layers[str(layer_id)] = {"ordinary_logit": ordinary, "jacobian": jacobian}
    require(observed_layers == list(range(24, 48)), f"{prompt_id} fixed layer order changed")

    return {
        "id": prompt_id,
        "task_id": task_id,
        "repo": repo,
        "cohort_id": cohort_id,
        "cohort_manifest_sha256": cohort_manifest,
        "metadata_identity_sha256": sha256_json(metadata),
        "generated_token_id": generated,
        "prompt_identity_sha256": sha256_json(
            {
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "positions_resolved": [final_position],
            }
        ),
        "residual_identity_sha256": sha256_json(residual),
        "diagnostics": _numeric_diagnostics(
            experiment, generated_token_id=generated, label=prompt_id
        ),
        "layers": layers,
    }


def load_compact_report(path: Path, *, label: str) -> dict[str, Any]:
    require(label in REPORT_LABELS, f"unknown report label: {label}")
    metadata, rows, source_sha256, source_bytes = _stream_json_experiments(path)
    require(metadata.get("schema_version") == REPORT_SCHEMA_VERSION, f"{label} report schema changed")
    require(metadata.get("score_encoding") == "unrounded-float32", f"{label} score encoding changed")
    model = mapping(metadata.get("model"), f"{label} model")
    runtime = mapping(metadata.get("runtime"), f"{label} runtime")
    require(
        all(model.get(key) == expected for key, expected in MODEL_PIN.items()),
        f"{label} model pin changed",
    )
    require(
        runtime.get("mtp_enabled") is False
        and runtime.get("enforce_eager") is True
        and runtime.get("language_model_only") is True
        and runtime.get("transport_dtype") == "torch.float32"
        and runtime.get("readout_dtype") == "torch.bfloat16",
        f"{label} replay was not eager with MTP disabled",
    )
    assertions = mapping(metadata.get("assertions"), f"{label} assertions")
    require(
        assertions.get("lens_hash_matches") is True
        and assertions.get("lens_metadata_matches") is True
        and assertions.get("model_architecture_matches") is True,
        f"{label} report integrity assertion failed",
    )
    model_identity = {
        key: model.get(key)
        for key in (
            "repo_id",
            "revision",
            "config_sha256",
            "index_sha256",
            "quant_method",
            "quant_algo",
        )
    }
    runtime_identity = dict(runtime)
    model_load_seconds = finite(
        runtime_identity.pop("model_load_seconds", None),
        f"{label} model-load duration",
    )
    require(model_load_seconds > 0.0, f"{label} model-load duration must be positive")
    lens = mapping(metadata.get("lens"), f"{label} lens")
    require(
        lens.get("d_model") == 5120
        and lens.get("source_layers") == list(range(63))
        and lens.get("tensor_shape") == [5120, 5120],
        f"{label} lens shape or source layers changed",
    )
    require(
        all(lens.get(key) == expected for key, expected in LENS_PINS[label].items()),
        f"{label} lens pin changed",
    )
    return {
        "label": label,
        "path": str(path),
        "bytes": source_bytes,
        "sha256": source_sha256,
        "report_status": metadata.get("status"),
        "model_identity": model_identity,
        "runtime_identity": runtime_identity,
        "lens": {
            key: lens.get(key)
            for key in (
                "kind",
                "repo_id",
                "revision",
                "sha256",
                "provenance_sha256",
                "state_sha256",
                "application",
                "n_prompts",
            )
            if lens.get(key) is not None
        },
        "rows": rows,
    }


def pair_reports(reports: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(set(reports) == set(REPORT_LABELS), "exactly three labeled reports are required")
    report_list = [reports[label] for label in REPORT_LABELS]
    require(
        len({sha256_json(report["model_identity"]) for report in report_list}) == 1,
        "replay model identity differs across reports",
    )
    require(
        len({sha256_json(report["runtime_identity"]) for report in report_list}) == 1,
        "replay runtime identity differs across reports",
    )
    id_lists = [[row["id"] for row in report["rows"]] for report in report_list]
    require(id_lists[0] == id_lists[1] == id_lists[2], "report prompt order or coverage differs")
    require(len(id_lists[0]) == len(set(id_lists[0])), "report prompt IDs are not unique")

    paired: list[dict[str, Any]] = []
    cohort_sha = mapping(protocol.get("scope"), "transport scope").get("cohort_manifest_sha256")
    for index, rows in enumerate(zip(*(report["rows"] for report in report_list), strict=True)):
        public, nf4, native = rows
        prompt_id = public["id"]
        for field in (
            "id",
            "task_id",
            "repo",
            "cohort_id",
            "cohort_manifest_sha256",
            "metadata_identity_sha256",
            "generated_token_id",
            "prompt_identity_sha256",
            "residual_identity_sha256",
        ):
            require(
                public[field] == nf4[field] == native[field],
                f"{prompt_id} exact {field} pairing failed",
            )
        require(
            public["cohort_manifest_sha256"] == cohort_sha,
            f"{prompt_id} cohort manifest differs from frozen scope",
        )

        methods: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
        for layer in range(24, 48):
            key = str(layer)
            ordinary_records = [row["layers"][key]["ordinary_logit"] for row in rows]
            require(
                ordinary_records[0] == ordinary_records[1] == ordinary_records[2],
                f"{prompt_id} layer {layer} ordinary-logit pairing failed",
            )
            ordinary = ordinary_records[0]
            require(
                ordinary["target_token_id"] == public["generated_token_id"],
                f"{prompt_id} layer {layer} ordinary target identity failed",
            )
            methods["ordinary_logit"].append({"layer": layer, **ordinary})
            for report_label, row in zip(REPORT_LABELS, rows, strict=True):
                jacobian = row["layers"][key]["jacobian"]
                require(
                    jacobian["target_token_id"] == public["generated_token_id"],
                    f"{prompt_id} layer {layer} {report_label} target identity failed",
                )
                methods[JACOBIAN_METHOD[report_label]].append(
                    {"layer": layer, **jacobian}
                )

        paired.append(
            {
                "checkpoint_index": index,
                "id": prompt_id,
                "task_id": public["task_id"],
                "repo": public["repo"],
                "cohort_id": public["cohort_id"],
                "generated_token_id": public["generated_token_id"],
                "prompt_identity_sha256": public["prompt_identity_sha256"],
                "metadata_identity_sha256": public["metadata_identity_sha256"],
                "strict_eligible": all(row["diagnostics"]["strict_certified"] for row in rows),
                "sensitivity_eligible": all(
                    row["diagnostics"]["sensitivity_certified"] for row in rows
                ),
                "report_diagnostics": {
                    label: row["diagnostics"]
                    for label, row in zip(REPORT_LABELS, rows, strict=True)
                },
                "methods": methods,
            }
        )

    scope = mapping(protocol.get("scope"), "transport scope")
    require(len(paired) == integer(scope.get("checkpoint_count"), "checkpoint count", minimum=1), "checkpoint count differs from frozen scope")
    tasks = {row["task_id"] for row in paired}
    repos = {row["repo"] for row in paired}
    require(len(tasks) == integer(scope.get("task_count"), "task count", minimum=1), "task count differs from frozen scope")
    require(len(repos) == 11, "repository count differs from frozen scope")
    counts = {task: sum(row["task_id"] == task for row in paired) for task in tasks}
    require(set(counts.values()) == {8}, "frozen cohort must contain eight checkpoints per task")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in paired:
        by_task.setdefault(row["task_id"], []).append(row)
    for task_rows in by_task.values():
        for ordinal, row in enumerate(
            sorted(task_rows, key=lambda record: record["checkpoint_index"])
        ):
            row["checkpoint_ordinal"] = ordinal
    return paired


def load_prompt_identities(path: Path) -> tuple[list[dict[str, Any]], str, int]:
    payload = path.read_bytes()
    prompts = sequence(json.loads(payload), "frozen prompt bundle")
    identities: list[dict[str, Any]] = []
    for index, raw_prompt in enumerate(prompts):
        prompt = mapping(raw_prompt, f"prompt[{index}]")
        prompt_id = text(prompt.get("id"), f"prompt[{index}] ID")
        prompt_text = prompt.get("text")
        require(isinstance(prompt_text, str), f"{prompt_id} text must be a string")
        token_ids = sequence(prompt.get("token_ids"), f"{prompt_id} token IDs")
        require(bool(token_ids), f"{prompt_id} token IDs are empty")
        require(
            all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and 0 <= token_id < 248320
                for token_id in token_ids
            ),
            f"{prompt_id} token IDs exceed the vocabulary",
        )
        metadata = mapping(prompt.get("metadata"), f"{prompt_id} metadata")
        task = mapping(metadata.get("task"), f"{prompt_id} task")
        cohort = mapping(metadata.get("cohort"), f"{prompt_id} cohort")
        identities.append(
            {
                "id": prompt_id,
                "task_id": text(task.get("instance_id"), f"{prompt_id} task ID"),
                "repo": text(task.get("repo"), f"{prompt_id} repository"),
                "cohort_id": text(cohort.get("id"), f"{prompt_id} cohort ID"),
                "prompt_identity_sha256": sha256_json(
                    {
                        "prompt": prompt_text,
                        "prompt_token_ids": token_ids,
                        "positions_resolved": [len(token_ids) - 1],
                    }
                ),
                "metadata_identity_sha256": sha256_json(metadata),
            }
        )
    return identities, hashlib.sha256(payload).hexdigest(), len(payload)


def validate_prompt_scope(
    paired: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> None:
    require(len(paired) == len(expected), "frozen prompt/report row counts differ")
    fields = (
        "id",
        "task_id",
        "repo",
        "cohort_id",
        "prompt_identity_sha256",
        "metadata_identity_sha256",
    )
    for row, prompt in zip(paired, expected, strict=True):
        for field in fields:
            require(
                row[field] == prompt[field],
                f"{row['id']} report differs from frozen prompt field {field}",
            )


def _checkpoint_metrics(layer_rows: Sequence[Mapping[str, Any]], vocabulary_size: int) -> dict[str, float]:
    require(len(layer_rows) == 24, "checkpoint method must have 24 layer observations")
    ranks = [integer(row["target_rank"], "target rank", minimum=1) for row in layer_rows]
    require(all(rank <= vocabulary_size for rank in ranks), "target rank exceeds vocabulary")
    logprobs = [finite(row["target_logprob"], "target logprob") for row in layer_rows]
    utilities = [math.log(vocabulary_size / rank) / math.log(vocabulary_size) for rank in ranks]
    return {
        "normalized_rank_utility": math.fsum(utilities) / len(utilities),
        "mean_target_logprob": math.fsum(logprobs) / len(logprobs),
        "mean_log_target_rank": math.fsum(math.log(rank) for rank in ranks) / len(ranks),
        "rank_at_most_100_fraction": sum(rank <= 100 for rank in ranks) / len(ranks),
        "rank_at_most_1000_fraction": sum(rank <= 1000 for rank in ranks) / len(ranks),
        "rank_at_most_10000_fraction": sum(rank <= 10000 for rank in ranks) / len(ranks),
    }


def _mean_metrics(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    require(bool(records), "cannot average empty metric records")
    keys = (
        "normalized_rank_utility",
        "mean_target_logprob",
        "mean_log_target_rank",
        "rank_at_most_100_fraction",
        "rank_at_most_1000_fraction",
        "rank_at_most_10000_fraction",
    )
    result = {key: math.fsum(float(record[key]) for record in records) / len(records) for key in keys}
    result["geometric_mean_target_rank"] = math.exp(result["mean_log_target_rank"])
    return result


def _public_metrics(value: Mapping[str, float]) -> dict[str, float]:
    return {key: float(value[key]) for key in (
        "normalized_rank_utility",
        "mean_target_logprob",
        "geometric_mean_target_rank",
        "rank_at_most_100_fraction",
        "rank_at_most_1000_fraction",
        "rank_at_most_10000_fraction",
    )}


def _descriptive_progression(
    eligible: Sequence[Mapping[str, Any]], *, vocabulary_size: int
) -> dict[str, Any]:
    """Task-equal views that are never used for selection or classification."""

    layer_records: list[dict[str, Any]] = []
    for layer in range(24, 48):
        task_values: dict[str, dict[str, list[float]]] = {}
        for row in eligible:
            task_methods = task_values.setdefault(
                row["task_id"], {method: [] for method in METHODS}
            )
            for method in METHODS:
                layer_row = row["methods"][method][layer - 24]
                require(layer_row["layer"] == layer, "descriptive layer order changed")
                rank = integer(layer_row["target_rank"], "descriptive target rank", minimum=1)
                task_methods[method].append(
                    math.log(vocabulary_size / rank) / math.log(vocabulary_size)
                )
        utilities = {}
        for method in METHODS:
            task_means = [
                math.fsum(values[method]) / len(values[method])
                for values in task_values.values()
            ]
            utilities[method] = (
                math.fsum(task_means) / len(task_means) if task_means else None
            )
        layer_records.append(
            {
                "layer": layer,
                "eligible_checkpoint_count": len(eligible),
                "eligible_task_count": len(task_values),
                "normalized_rank_utility": utilities,
            }
        )

    ordinal_records: list[dict[str, Any]] = []
    for ordinal in range(8):
        rows = [row for row in eligible if row["checkpoint_ordinal"] == ordinal]
        utilities: dict[str, float | None] = {}
        for method in METHODS:
            values = [
                _checkpoint_metrics(row["methods"][method], vocabulary_size)[
                    "normalized_rank_utility"
                ]
                for row in rows
            ]
            utilities[method] = math.fsum(values) / len(values) if values else None
        ordinal_records.append(
            {
                "checkpoint_ordinal": ordinal,
                "eligible_checkpoint_count": len(rows),
                "eligible_task_count": len({row["task_id"] for row in rows}),
                "normalized_rank_utility": utilities,
            }
        )
    return {
        "status": "descriptive_only_no_selection_no_decision_role",
        "best_layer_selection_forbidden": True,
        "used_by_classification": False,
        "aggregation": "checkpoint_to_task_then_equal_task_mean",
        "by_fixed_layer": layer_records,
        "by_checkpoint_ordinal": ordinal_records,
    }


def _descriptive_cohort_metrics(
    task_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    for cohort_id in sorted({record["cohort_id"] for record in task_records}):
        records = [record for record in task_records if record["cohort_id"] == cohort_id]
        methods = {
            method: _public_metrics(
                _mean_metrics([record["methods"][method] for record in records])
            )
            for method in METHODS
        }
        comparisons = {
            name: math.fsum(
                record["methods"][candidate]["normalized_rank_utility"]
                - record["methods"][reference]["normalized_rank_utility"]
                for record in records
            )
            / len(records)
            for name, candidate, reference in COMPARISONS
        }
        cohorts[cohort_id] = {
            "task_count": len(records),
            "methods": methods,
            "normalized_rank_utility_comparisons": comparisons,
        }
    return {
        "status": "descriptive_only_not_independent_replication_tests",
        "used_by_classification": False,
        "cohorts": cohorts,
    }


def _hierarchical_bootstrap(
    task_deltas: Sequence[Mapping[str, Any]], *, samples: int, seed: int, confidence_level: float, minimum_valid_fraction: float
) -> dict[str, Any]:
    by_repo: dict[str, dict[str, float]] = {}
    for record in task_deltas:
        repo = text(record.get("repo"), "bootstrap repository")
        task_id = text(record.get("task_id"), "bootstrap task")
        delta = finite(record.get("delta"), "bootstrap delta")
        require(task_id not in by_repo.setdefault(repo, {}), "duplicate bootstrap task")
        by_repo[repo][task_id] = delta
    require(bool(by_repo) and samples > 0, "bootstrap needs tasks and positive samples")
    repositories = sorted(by_repo)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw: list[float] = []
        for _ in repositories:
            repository = repositories[rng.randrange(len(repositories))]
            tasks = sorted(by_repo[repository])
            for _ in tasks:
                task = tasks[rng.randrange(len(tasks))]
                draw.append(by_repo[repository][task])
        if draw:
            estimate = math.fsum(draw) / len(draw)
            if math.isfinite(estimate):
                estimates.append(estimate)
    valid_fraction = len(estimates) / samples
    available = valid_fraction >= minimum_valid_fraction
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "algorithm": "paired_hierarchical_repository_then_task_percentile_v1",
        "unit": "task_level_delta_resample_repositories_then_tasks",
        "same_draw_for_both_methods": True,
        "resampled_layers_or_checkpoints": False,
        "samples_requested": samples,
        "samples_valid": len(estimates),
        "valid_fraction": valid_fraction,
        "minimum_valid_fraction": minimum_valid_fraction,
        "seed": seed,
        "confidence_level": confidence_level,
        "status": "available" if available else "insufficient_valid_bootstrap_fraction",
        "confidence_interval": (
            {
                "lower": percentile(estimates, alpha),
                "upper": percentile(estimates, 1.0 - alpha),
            }
            if available
            else None
        ),
    }


def _support(paired: Sequence[Mapping[str, Any]], eligible: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], *, sensitivity: bool) -> dict[str, Any]:
    gates_key = "support_gates"
    gates = mapping(
        mapping(protocol.get("paired_stable_reconstruction_sensitivity"), "sensitivity").get(gates_key)
        if sensitivity
        else protocol.get(gates_key),
        "transport support gates",
    )
    expected_tasks = sorted({row["task_id"] for row in paired})
    expected_repos = sorted({row["repo"] for row in paired})
    counts = {task: sum(row["task_id"] == task for row in eligible) for task in expected_tasks}
    eligible_tasks = {row["task_id"] for row in eligible}
    eligible_repos = {row["repo"] for row in eligible}
    minimum_count = int(
        gates[
            "minimum_jointly_eligible_checkpoints"
            if sensitivity
            else "minimum_jointly_certified_checkpoints"
        ]
    )
    minimum_fraction = float(
        gates[
            "minimum_jointly_eligible_fraction"
            if sensitivity
            else "minimum_jointly_certified_fraction"
        ]
    )
    minimum_per_task = int(
        gates[
            "minimum_jointly_eligible_checkpoints_per_task"
            if sensitivity
            else "minimum_jointly_certified_checkpoints_per_task"
        ]
    )
    checks = {
        "minimum_checkpoint_count_pass": len(eligible) >= minimum_count,
        "minimum_checkpoint_fraction_pass": len(eligible) / len(paired) >= minimum_fraction,
        "all_20_tasks_represented_pass": eligible_tasks == set(expected_tasks) and len(expected_tasks) == 20,
        "all_11_repositories_represented_pass": eligible_repos == set(expected_repos) and len(expected_repos) == 11,
        "minimum_checkpoints_per_task_pass": all(count >= minimum_per_task for count in counts.values()),
        "fixed_24_layer_observations_per_method_pass": all(
            all(len(row["methods"][method]) == 24 for method in METHODS) for row in eligible
        ),
        "all_target_and_pairing_identity_checks_pass": True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "all_gates_pass": all(checks.values()),
        "eligible_checkpoint_count": len(eligible),
        "total_checkpoint_count": len(paired),
        "eligible_fraction": len(eligible) / len(paired),
        "eligible_task_count": len(eligible_tasks),
        "eligible_repository_count": len(eligible_repos),
        "eligible_checkpoints_by_task": counts,
        "thresholds": dict(gates),
        "checks": checks,
    }


def _classify(track_support: Mapping[str, Any], comparisons: Mapping[str, Any], protocol: Mapping[str, Any], *, sensitivity: bool) -> dict[str, Any]:
    prefix = "sensitivity_" if sensitivity else ""
    if not track_support["all_gates_pass"]:
        return {"classification": f"{prefix}insufficient_support", "reason_codes": ["support_gate_failed"]}
    public_logit = comparisons["public_jacobian_minus_ordinary_logit"]
    public_native = comparisons["public_jacobian_minus_native_jacobian"]
    if public_logit["bootstrap"]["status"] != "available" or public_native["bootstrap"]["status"] != "available":
        return {"classification": f"{prefix}insufficient_support", "reason_codes": ["bootstrap_unavailable"]}
    thresholds = mapping(protocol.get("decision_thresholds"), "decision thresholds")
    public_rule = mapping(
        thresholds.get("public_positive_control"), "public positive-control threshold"
    )
    native_rule = mapping(
        thresholds.get("native_refit_capacity_candidate"),
        "native capacity threshold",
    )
    equivalence_rule = mapping(
        thresholds.get("no_material_native_deficit"), "native equivalence threshold"
    )
    failure_rule = mapping(
        thresholds.get("readout_control_failure"), "readout failure threshold"
    )
    logit_ci = public_logit["bootstrap"]["confidence_interval"]
    native_ci = public_native["bootstrap"]["confidence_interval"]
    public_control = (
        public_logit["estimate"]
        >= float(public_rule["public_minus_logit_point_minimum_inclusive"])
        and logit_ci["lower"]
        > float(public_rule["confidence_interval_lower_minimum_exclusive"])
    )
    native_deficit = (
        public_control
        and public_native["estimate"]
        >= float(native_rule["public_minus_native_point_minimum_inclusive"])
        and native_ci["lower"]
        > float(native_rule["confidence_interval_lower_minimum_exclusive"])
    )
    native_equivalent = (
        public_control
        and native_ci["upper"]
        < float(
            equivalence_rule[
                "public_minus_native_confidence_interval_upper_maximum_exclusive"
            ]
        )
    )
    readout_failure = (
        logit_ci["upper"]
        <= float(
            failure_rule[
                "public_minus_logit_confidence_interval_upper_maximum_inclusive"
            ]
        )
    )
    audit = {
        "public_positive_control_pass": public_control,
        "material_native_deficit_pass": native_deficit,
        "no_material_native_deficit_pass": native_equivalent,
        "readout_control_failure_pass": readout_failure,
    }
    if public_control and native_deficit:
        classification = "native_refit_capacity_candidate"
        reasons = ["public_control_positive", "public_materially_beats_native"]
    elif public_control and native_equivalent:
        classification = "no_material_native_deficit"
        reasons = ["public_control_positive", "native_deficit_ci_below_material_threshold"]
    elif readout_failure:
        classification = "readout_control_failure"
        reasons = ["public_minus_logit_ci_upper_nonpositive"]
    else:
        classification = "insufficient_support"
        reasons = ["predeclared_effect_rules_inconclusive"]
    return {
        "classification": f"{prefix}{classification}",
        "reason_codes": reasons,
        "rule_audit": audit,
    }


def _nf4_diagnostic(
    comparisons: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    support_pass: bool,
) -> dict[str, Any]:
    threshold = float(mapping(protocol.get("nf4_diagnostic"), "NF4 diagnostic")["threshold"])
    if not support_pass:
        return {
            "classification": "insufficient_support_not_interpretable",
            "threshold": threshold,
            "public_materially_beats_native": None,
            "public_materially_beats_nf4": None,
            "decision_override_forbidden": True,
        }

    def materially_public_better(name: str) -> bool:
        comparison = comparisons[name]
        interval = comparison["bootstrap"].get("confidence_interval")
        return bool(
            comparison["bootstrap"].get("status") == "available"
            and comparison.get("estimate") is not None
            and comparison["estimate"] >= threshold
            and interval is not None
            and interval["lower"] > 0.0
        )

    native = materially_public_better("public_jacobian_minus_native_jacobian")
    nf4 = materially_public_better("public_jacobian_minus_nf4_jacobian")
    if native and nf4:
        classification = "shared_local_fit_capacity_or_fit_corpus_limitation_more_plausible"
    elif native and not nf4:
        classification = "nvfp4_ste_specific_deficit_more_plausible_not_causally_isolated"
    elif not native and not nf4:
        classification = "semantic_probe_and_readout_design_remain_priority"
    else:
        classification = "mixed_public_beats_nf4_only_not_predeclared"
    return {
        "classification": classification,
        "threshold": threshold,
        "public_materially_beats_native": native,
        "public_materially_beats_nf4": nf4,
        "decision_override_forbidden": True,
    }


def build_track(paired: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], *, sensitivity: bool, bootstrap_samples: int | None = None) -> dict[str, Any]:
    eligibility_key = "sensitivity_eligible" if sensitivity else "strict_eligible"
    eligible = [row for row in paired if row[eligibility_key]]
    support = _support(paired, eligible, protocol, sensitivity=sensitivity)
    vocabulary_size = integer(mapping(protocol.get("readout"), "readout").get("scored_vocabulary_size"), "vocabulary size", minimum=2)
    by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in eligible:
        by_task.setdefault((row["repo"], row["task_id"]), []).append(row)

    task_records: list[dict[str, Any]] = []
    for (repo, task_id), checkpoints in sorted(by_task.items()):
        method_metrics: dict[str, Any] = {}
        for method in METHODS:
            checkpoint_metrics = [_checkpoint_metrics(row["methods"][method], vocabulary_size) for row in checkpoints]
            method_metrics[method] = _mean_metrics(checkpoint_metrics)
        task_records.append(
            {
                "repo": repo,
                "task_id": task_id,
                "cohort_id": text(checkpoints[0]["cohort_id"], "cohort ID"),
                "eligible_checkpoint_count": len(checkpoints),
                "methods": method_metrics,
            }
        )

    method_summary: dict[str, Any] = {}
    if task_records:
        for method in METHODS:
            aggregate = _mean_metrics([record["methods"][method] for record in task_records])
            method_summary[method] = {
                "status": "available",
                "task_equal_weighting": True,
                "metrics": _public_metrics(aggregate),
            }
    else:
        method_summary = {method: {"status": "insufficient_no_eligible_tasks", "metrics": None} for method in METHODS}

    inference = mapping(protocol.get("inference"), "inference")
    samples = integer(inference.get("samples"), "bootstrap samples", minimum=1) if bootstrap_samples is None else bootstrap_samples
    require(samples > 0, "bootstrap samples must be positive")
    comparisons: dict[str, Any] = {}
    for name, candidate, reference in COMPARISONS:
        deltas = [
            {
                "repo": record["repo"],
                "task_id": record["task_id"],
                "delta": record["methods"][candidate]["normalized_rank_utility"]
                - record["methods"][reference]["normalized_rank_utility"],
            }
            for record in task_records
        ]
        if deltas:
            estimate = math.fsum(record["delta"] for record in deltas) / len(deltas)
            bootstrap = _hierarchical_bootstrap(
                deltas,
                samples=samples,
                seed=integer(inference.get("seed"), "bootstrap seed"),
                confidence_level=finite(inference.get("confidence_level"), "confidence level"),
                minimum_valid_fraction=finite(inference.get("minimum_valid_fraction"), "minimum valid fraction"),
            )
        else:
            estimate = None
            bootstrap = {"status": "insufficient_no_eligible_tasks", "confidence_interval": None}
        comparisons[name] = {
            "candidate": candidate,
            "reference": reference,
            "estimate": estimate,
            "unit": "equal_task_mean_normalized_rank_utility_delta",
            "task_deltas": deltas,
            "bootstrap": bootstrap,
        }

    classification = _classify(support, comparisons, protocol, sensitivity=sensitivity)
    return {
        "status": "available" if task_records else "insufficient_no_eligible_tasks",
        "role": (
            "post_public_numerical_diagnostic_amendment_sensitivity_cannot_override_primary"
            if sensitivity
            else "strict_primary"
        ),
        "support": support,
        "methods": method_summary,
        "task_records": [
            {
                **{key: record[key] for key in ("repo", "task_id", "cohort_id", "eligible_checkpoint_count")},
                "methods": {method: _public_metrics(record["methods"][method]) for method in METHODS},
            }
            for record in task_records
        ],
        "descriptive_transport_emergence": _descriptive_progression(
            eligible, vocabulary_size=vocabulary_size
        ),
        "descriptive_cohort_metrics": _descriptive_cohort_metrics(task_records),
        "comparisons": comparisons,
        "classification": classification,
        "nf4_diagnostic": _nf4_diagnostic(
            comparisons, protocol, support_pass=bool(support["all_gates_pass"])
        ),
        "decision_overrides_behavioral_semantic_analysis": False,
    }


def _behavioral_context(path: Path | None, input_hashes: Mapping[str, str], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if path is None:
        return {"status": "not_available", "primary_semantic_decision_override_forbidden": True}
    value = mapping(json.loads(path.read_bytes()), "behavioral analysis")
    require(value.get("kind") == "swe_verified_behavioral_task_held_out_analysis", "behavioral analysis kind changed")
    inputs = mapping(value.get("inputs"), "behavioral analysis inputs")
    for label in REPORT_LABELS:
        require(
            inputs.get(f"{label}_report") == input_hashes[f"{label}_report"],
            f"behavioral analysis does not bind the {label} report",
        )
    campaign = mapping(value.get("campaign"), "behavioral campaign")
    require(
        campaign.get("combined_cohort_manifest_sha256")
        == mapping(protocol.get("scope"), "transport scope").get("cohort_manifest_sha256"),
        "behavioral analysis cohort differs from transport scope",
    )
    decision = mapping(value.get("scientific_decision"), "behavioral scientific decision")
    return {
        "status": "available_hash_bound_same_replay",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "operational_status": value.get("operational_status", value.get("status")),
        "scientific_classification": decision.get("classification"),
        "scientific_reason_codes": decision.get("reason_codes", decision.get("reasons", [])),
        "primary_semantic_decision_override_forbidden": True,
    }


def build_analysis(
    reports: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    prompts_path: Path,
    behavioral_analysis_path: Path | None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    scope = mapping(protocol.get("scope"), "transport scope")
    expected_prompts, prompt_sha, prompt_bytes = load_prompt_identities(prompts_path)
    require(prompt_sha == scope.get("prompt_bundle_sha256"), "prompt bundle differs from frozen scope")
    paired = pair_reports(reports, protocol)
    validate_prompt_scope(paired, expected_prompts)
    input_hashes = {f"{label}_report": reports[label]["sha256"] for label in REPORT_LABELS}
    strict = build_track(paired, protocol, sensitivity=False, bootstrap_samples=bootstrap_samples)
    sensitivity = build_track(paired, protocol, sensitivity=True, bootstrap_samples=bootstrap_samples)
    context = _behavioral_context(behavioral_analysis_path, input_hashes, protocol)
    checkpoint_audit = [
        {
            "checkpoint_index": row["checkpoint_index"],
            "id": row["id"],
            "task_id": row["task_id"],
            "repo": row["repo"],
            "cohort_id": row["cohort_id"],
            "generated_token_id": row["generated_token_id"],
            "checkpoint_ordinal": row["checkpoint_ordinal"],
            "strict_eligible": row["strict_eligible"],
            "sensitivity_eligible": row["sensitivity_eligible"],
            "report_diagnostics": row["report_diagnostics"],
        }
        for row in paired
    ]
    return {
        "schema_version": 1,
        "kind": "swe_verified_greedy_next_token_transport_analysis",
        "analysis_version": "strict-and-amended-sensitivity-v2",
        "status": "complete",
        "interpretation": dict(mapping(protocol.get("interpretation"), "interpretation")),
        "protocol": {
            "path": str(DEFAULT_PROTOCOL.relative_to(ROOT)),
            "sha256": protocol_sha256,
            "id": protocol.get("id"),
            "frozen_at_utc": protocol.get("frozen_at_utc"),
            "inference": dict(mapping(protocol.get("inference"), "inference")),
            "decision_thresholds": dict(mapping(protocol.get("decision_thresholds"), "thresholds")),
        },
        "inputs": {
            "prompts": {
                "path": str(prompts_path),
                "sha256": prompt_sha,
                "bytes": prompt_bytes,
            },
            "reports": {
                label: {
                    key: reports[label][key]
                    for key in ("path", "sha256", "bytes", "report_status", "lens")
                }
                for label in REPORT_LABELS
            },
        },
        "pairing": {
            "status": "passed_exact_fail_closed",
            "checkpoint_count": len(paired),
            "task_count": len({row["task_id"] for row in paired}),
            "repository_count": len({row["repo"] for row in paired}),
            "same_generated_token_across_reports": True,
            "same_target_token_at_every_fixed_layer_and_method": True,
            "exact_prompt_pairing": True,
            "exact_residual_pairing": True,
            "exact_ordinary_logit_pairing": True,
            "captured_final_model_top1_match_required_for_eligibility": True,
        },
        "checkpoint_eligibility_audit": checkpoint_audit,
        "tracks": {"strict_primary": strict, "paired_stable_reconstruction_sensitivity": sensitivity},
        "behavioral_semantic_analysis": context,
        "decision_role": "supplemental_fit_capacity_diagnostic_only",
        "primary_semantic_decision_override_forbidden": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--nf4-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--behavioral-analysis", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "prompts": args.prompts.expanduser().resolve(strict=True),
        "public": args.public_report.expanduser().resolve(strict=True),
        "nf4": args.nf4_report.expanduser().resolve(strict=True),
        "native": args.native_report.expanduser().resolve(strict=True),
        "protocol": args.protocol.expanduser().resolve(strict=True),
    }
    protocol_bytes = paths["protocol"].read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = validate_protocol(json.loads(protocol_bytes), protocol_sha256=protocol_sha)
    reports: dict[str, Any] = {}
    for label in REPORT_LABELS:
        reports[label] = load_compact_report(paths[label], label=label)
    behavioral = args.behavioral_analysis
    if behavioral is None:
        candidate = paths["public"].parent / "analysis.json"
        behavioral = candidate if candidate.is_file() else None
    elif behavioral is not None:
        behavioral = behavioral.expanduser().resolve(strict=True)
    analysis = build_analysis(
        reports,
        protocol,
        protocol_sha256=protocol_sha,
        prompts_path=paths["prompts"],
        behavioral_analysis_path=behavioral,
    )
    output = args.output.expanduser().resolve()
    atomic_write_json(output, analysis)
    print(
        f"wrote {output} (sha256={sha256_file(output)}, "
        f"strict={analysis['tracks']['strict_primary']['classification']['classification']}, "
        f"sensitivity={analysis['tracks']['paired_stable_reconstruction_sensitivity']['classification']['classification']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
