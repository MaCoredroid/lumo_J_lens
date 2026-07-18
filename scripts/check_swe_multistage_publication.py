#!/usr/bin/env python3
"""Fail-closed integrity checks for the published SWE multistage experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence
import zlib


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "validation/jlens-swe-multistage-2026-07-18"
EVIDENCE = ROOT / "validation/swe-multistage-django-13297"
LEGACY = BUNDLE / "legacy-lexically-primed-audit"

CURRENT_MANIFEST_SHA256 = (
    "c977fd5197466eb21b44e509f8c36016df0f3556e3bf30055d890f314f7ca6bd"
)
FRESH_MANIFEST_SHA256 = (
    "104d9ba5d422be3a489f8fc225fce3724791efdfe270231a8b172af973ce689f"
)
LEGACY_MANIFEST_SHA256 = (
    "bf0cb4e9e47c165f7fc5d49e8110fe03303471fb592508a9fea9407ddb0b6503"
)
EVIDENCE_LEDGER_SHA256 = (
    "6ca599fc730e51b45b2d2fc491a341650265b2c528a7811e9aa2bf4c98a47591"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LEDGER_ROW_RE = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
DETERMINISTIC_GZIP_HEADER = bytes.fromhex("1f8b0800000000000203")
MAX_LEGACY_OUTPUT_BYTES = 20_000_000

INPUT_KEYS = {
    "pilot",
    "materializer",
    "augmenter",
    "analyzer",
    "jlens_runner",
    "jlens_python_runner",
    "lifecycle_protocol",
    "trajectory_manifest",
    "action_protocol",
}
ARTIFACT_KEYS = {
    "prompts",
    "prompts_summary",
    "action_prompts",
    "action_prompts_summary",
    "public_report",
    "nf4_report",
    "native_report",
    "analysis",
}
REPORT_KEYS = {"public_report", "nf4_report", "native_report"}
ARTIFACT_FILENAMES = {
    "prompts": "prompts.json",
    "prompts_summary": "prompts_summary.json",
    "action_prompts": "action_prompts.json",
    "action_prompts_summary": "action_prompts_summary.json",
    "public_report": "public-report.json",
    "nf4_report": "nf4-report.json",
    "native_report": "native-report.json",
    "analysis": "analysis.json",
}
PHASE_STATUSES = {
    "materialize": "0",
    "augment": "0",
    "public": "1",
    "nf4": "1",
    "native": "1",
    "analyze": "0",
}
FRESH_SOURCE_PATHS = {
    "pilot": "run_swe_multistage_pilot.fresh-local.sh",
    "materializer": "fresh-sources/materialize_swe_multistage_probes.py",
    "augmenter": "fresh-sources/augment_swe_multistage_action_probes.py",
    "analyzer": "fresh-sources/analyze_swe_multistage_probes.py",
    "jlens_runner": "fresh-sources/run_jlens_nvfp4.sh",
    "jlens_python_runner": "fresh-sources/run_jlens_nvfp4.py",
    "lifecycle_protocol": "fresh-sources/swe_multistage_protocol.json",
    "trajectory_manifest": "fresh-sources/swe_multistage_trajectory_manifest.json",
    "action_protocol": "fresh-sources/swe_stage_action_probes.json",
}
EXPECTED_RUNTIME = {
    "enable_prefix_caching": True,
    "gpu_memory_utilization": 0.78,
    "kv_cache_dtype": "fp8_e4m3",
    "kv_offloading_backend": "native",
    "kv_offloading_size_gib": 8,
    "layers": list(range(16, 48)),
    "lens_sha256": {
        "native": "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057",
        "nf4": "54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f",
        "public": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
    },
    "mamba_block_size": 1024,
    "max_model_len": 49152,
    "max_num_batched_tokens": 4096,
    "model": {
        "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
        "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
        "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
        "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    },
    "mtp_enabled": False,
    "positions": [-1],
    "stream_final_only": True,
    "top_k": 10,
}
EXPECTED_REPORT_LENSES = {
    "public_report": {
        "sha256": EXPECTED_RUNTIME["lens_sha256"]["public"],
        "n_prompts": 1000,
        "repo_id": "neuronpedia/jacobian-lens",
        "revision": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
        "kind": None,
    },
    "nf4_report": {
        "sha256": EXPECTED_RUNTIME["lens_sha256"]["nf4"],
        "n_prompts": 10,
        "repo_id": None,
        "revision": None,
        "kind": "local_fit",
    },
    "native_report": {
        "sha256": EXPECTED_RUNTIME["lens_sha256"]["native"],
        "n_prompts": 10,
        "repo_id": None,
        "revision": None,
        "kind": "native_nvfp4_ste_fit",
    },
}


class PublicationError(ValueError):
    """Raised when published evidence is missing, unsafe, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}",
    )


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(value: bytes, label: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PublicationError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"invalid JSON in {label}: {error}") from error


def strict_json_file(path: Path, label: str) -> Any:
    try:
        value = path.read_bytes()
    except OSError as error:
        raise PublicationError(f"cannot read {label}: {error}") from error
    return strict_json_bytes(value, label)


def canonical_relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be nonempty text")
    require("\\" not in value and "\x00" not in value, f"unsafe {label}: {value!r}")
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in ("", ".", "..") for part in path.parts),
        f"non-canonical or unsafe {label}: {value!r}",
    )
    return value


def regular_file(root: Path, relative: str, label: str) -> Path:
    relative = canonical_relative(relative, label)
    root = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        require(not current.is_symlink(), f"{label} traverses a symlink: {relative}")
    try:
        mode = current.lstat().st_mode
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise PublicationError(f"missing {label}: {relative}: {error}") from error
    require(resolved.is_relative_to(root), f"{label} escapes its root: {relative}")
    require(stat.S_ISREG(mode), f"{label} is not a regular file: {relative}")
    return current


def validate_digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def verify_sidecar(sidecar: Path, target_name: str, expected: str) -> None:
    validate_digest(expected, "expected sidecar digest")
    expected_bytes = f"{expected}  {target_name}\n".encode("ascii")
    sidecar = regular_file(sidecar.parent, sidecar.name, "manifest sidecar")
    try:
        actual = sidecar.read_bytes()
    except OSError as error:
        raise PublicationError(f"cannot read sidecar {sidecar}: {error}") from error
    require(actual == expected_bytes, f"sidecar grammar or pin mismatch: {sidecar}")
    target = regular_file(sidecar.parent, target_name, "sidecar target")
    require(sha256_file(target) == expected, f"sidecar target hash mismatch: {target}")


def verify_record(root: Path, value: Any, *, label: str, report: bool) -> Path:
    record = mapping(value, label)
    expected_keys = {"path", "path_base", "bytes", "sha256"}
    if report:
        expected_keys.add("report_status")
    exact_keys(record, expected_keys, label)
    require(record["path_base"] == "repository_root", f"{label} path base mismatch")
    path = regular_file(root, record["path"], f"{label} path")
    byte_count = record["bytes"]
    require(
        isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count > 0,
        f"{label} byte count is invalid",
    )
    expected_sha = validate_digest(record["sha256"], f"{label} digest")
    require(path.stat().st_size == byte_count, f"{label} byte count mismatch")
    require(sha256_file(path) == expected_sha, f"{label} SHA-256 mismatch")
    if report:
        require(record["report_status"] == "failed", f"{label} status changed")
    return path


def verify_phases(value: Any, *, suffix: str = "") -> None:
    phases = mapping(value, "run phases")
    exact_keys(phases, set(PHASE_STATUSES), "run phases")
    for phase, expected_status in PHASE_STATUSES.items():
        record = mapping(phases[phase], f"phase {phase}")
        exact_keys(record, {"status", "status_file_sha256"}, f"phase {phase}")
        require(record["status"] == expected_status, f"phase {phase} status mismatch")
        expected_sha = validate_digest(
            record["status_file_sha256"], f"phase {phase} status digest"
        )
        status_name = f"{phase}{suffix}.exit_status"
        status_path = regular_file(BUNDLE, status_name, f"phase {phase} status file")
        require(
            status_path.read_bytes() == f"{expected_status}\n".encode("ascii")
            and sha256_file(status_path) == expected_sha,
            f"phase {phase} status file mismatch",
        )


def verify_current_manifest() -> dict[str, Path]:
    manifest_path = BUNDLE / "run_manifest.json"
    verify_sidecar(BUNDLE / "run_manifest.sha256", manifest_path.name, CURRENT_MANIFEST_SHA256)
    manifest = mapping(strict_json_file(manifest_path, "current run manifest"), "manifest")
    exact_keys(
        manifest,
        {
            "schema_version",
            "kind",
            "status",
            "mode",
            "path_contract",
            "inputs",
            "artifacts",
            "phases",
            "runtime_contract",
        },
        "current run manifest",
    )
    require(
        manifest["schema_version"] == 2
        and manifest["kind"] == "swe_verified_multistage_pilot_run"
        and manifest["status"] == "complete"
        and manifest["mode"] == "fresh_replay",
        "current run identity mismatch",
    )
    require(
        manifest["path_contract"]
        == {
            "absolute_paths_embedded": False,
            "output_directory": "directory containing this manifest",
            "repository_root": "directory containing this repository",
        },
        "current path contract mismatch",
    )
    inputs = mapping(manifest["inputs"], "current inputs")
    artifacts = mapping(manifest["artifacts"], "current artifacts")
    exact_keys(inputs, INPUT_KEYS, "current inputs")
    exact_keys(artifacts, ARTIFACT_KEYS, "current artifacts")
    paths: dict[str, Path] = {}
    seen: set[Path] = set()
    for group, records in (("input", inputs), ("artifact", artifacts)):
        for key, record in records.items():
            path = verify_record(
                ROOT,
                record,
                label=f"current {group} {key}",
                report=group == "artifact" and key in REPORT_KEYS,
            )
            require(path not in seen, f"current manifest aliases a path: {path}")
            seen.add(path)
            paths[key] = path
    verify_phases(manifest["phases"])
    require(manifest["runtime_contract"] == EXPECTED_RUNTIME, "runtime contract mismatch")
    return paths


def parse_ledger(value: bytes) -> list[tuple[str, str]]:
    require(value.endswith(b"\n") and b"\r" not in value, "ledger newline contract failed")
    try:
        lines = value.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise PublicationError("evidence ledger is not ASCII") from error
    require(len(lines) == 45, f"evidence ledger must have 45 rows, got {len(lines)}")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines, start=1):
        match = LEDGER_ROW_RE.fullmatch(line)
        require(match is not None, f"invalid evidence ledger row {index}")
        assert match is not None
        relative = canonical_relative(match.group(2), f"ledger row {index} path")
        require(relative not in seen, f"duplicate evidence path: {relative}")
        seen.add(relative)
        rows.append((match.group(1), relative))
    return rows


def verify_evidence_ledger() -> None:
    ledger = regular_file(EVIDENCE, "evidence.sha256", "evidence ledger")
    require(sha256_file(ledger) == EVIDENCE_LEDGER_SHA256, "evidence ledger hash mismatch")
    rows = parse_ledger(ledger.read_bytes())
    declared = {relative for _, relative in rows}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in EVIDENCE.rglob("*")
        if path.is_file() and path != ledger
    }
    require(declared == actual, "evidence ledger is not directory-complete")
    for expected, relative in rows:
        path = regular_file(ROOT, relative, "evidence ledger target")
        require(sha256_file(path) == expected, f"evidence hash mismatch: {relative}")
    chats = sorted(
        PurePosixPath(relative).name
        for relative in declared
        if "/proxy_dumps/chat_" in relative
    )
    require(
        chats == [f"chat_{index:04d}.json" for index in range(1, 26)],
        "evidence request sequence is not contiguous",
    )


def decompress_one_gzip(value: bytes, label: str) -> bytes:
    require(value.startswith(DETERMINISTIC_GZIP_HEADER), f"{label} gzip header is not deterministic")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        output = decoder.decompress(value, MAX_LEGACY_OUTPUT_BYTES + 1)
        output += decoder.flush()
    except zlib.error as error:
        raise PublicationError(f"invalid gzip stream for {label}: {error}") from error
    require(len(output) <= MAX_LEGACY_OUTPUT_BYTES, f"{label} exceeds decompression bound")
    require(decoder.eof, f"{label} gzip stream is truncated")
    require(not decoder.unused_data and not decoder.unconsumed_tail, f"{label} has trailing data")
    return output


def verify_legacy() -> dict[str, tuple[str, int]]:
    verify_sidecar(LEGACY / "manifest.sha256", "manifest.json", LEGACY_MANIFEST_SHA256)
    value = mapping(strict_json_file(LEGACY / "manifest.json", "legacy manifest"), "legacy")
    exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "compression",
            "invalidated_claim",
            "replacement_status",
            "artifacts",
        },
        "legacy manifest",
    )
    require(
        value["schema_version"] == 1
        and value["kind"] == "invalidated_lexically_primed_multistage_audit"
        and value["compression"] == "gzip -n -9",
        "legacy manifest identity mismatch",
    )
    records = sequence(value["artifacts"], "legacy artifacts")
    require(len(records) == 8, "legacy manifest must preserve all eight artifacts")
    originals: dict[str, tuple[str, int]] = {}
    compressed_seen: set[str] = set()
    for index, raw_record in enumerate(records):
        record = mapping(raw_record, f"legacy artifact {index}")
        exact_keys(
            record,
            {"compressed_path", "compressed_sha256", "original_path", "original_sha256"},
            f"legacy artifact {index}",
        )
        compressed = canonical_relative(record["compressed_path"], "legacy compressed path")
        original = canonical_relative(record["original_path"], "legacy original path")
        require("/" not in compressed and "/" not in original, "legacy names must be basenames")
        require(compressed == f"{original}.gz", "legacy compressed/original names disagree")
        require(compressed not in compressed_seen and original not in originals, "duplicate legacy artifact")
        compressed_seen.add(compressed)
        path = regular_file(LEGACY, compressed, "legacy compressed artifact")
        expected_compressed = validate_digest(
            record["compressed_sha256"], "legacy compressed digest"
        )
        expected_original = validate_digest(record["original_sha256"], "legacy original digest")
        require(sha256_file(path) == expected_compressed, f"legacy compressed hash mismatch: {compressed}")
        output = decompress_one_gzip(path.read_bytes(), compressed)
        require(sha256_bytes(output) == expected_original, f"legacy original hash mismatch: {original}")
        strict_json_bytes(output, f"legacy {original}")
        originals[original] = (expected_original, len(output))
    require(
        set(originals)
        == {
            "prompts.json",
            "prompts_summary.json",
            "action_prompts.json",
            "action_prompts_summary.json",
            "public-report.json",
            "nf4-report.json",
            "native-report.json",
            "analysis.json",
        },
        "legacy artifact names differ from the original fresh run",
    )
    return originals


def verify_fresh_manifest(legacy: Mapping[str, tuple[str, int]]) -> None:
    path = BUNDLE / "run_manifest.fresh-local.json"
    verify_sidecar(BUNDLE / "run_manifest.fresh-local.sha256", path.name, FRESH_MANIFEST_SHA256)
    value = mapping(strict_json_file(path, "fresh-local manifest"), "fresh-local")
    exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "status",
            "mode",
            "inputs",
            "artifacts",
            "phases",
            "runtime_contract",
        },
        "fresh-local manifest",
    )
    require(
        value["schema_version"] == 1
        and value["kind"] == "swe_verified_multistage_pilot_run"
        and value["status"] == "complete"
        and value["mode"] == "fresh_replay",
        "fresh-local identity mismatch",
    )
    inputs = mapping(value["inputs"], "fresh-local inputs")
    artifacts = mapping(value["artifacts"], "fresh-local artifacts")
    exact_keys(inputs, INPUT_KEYS, "fresh-local inputs")
    exact_keys(artifacts, ARTIFACT_KEYS, "fresh-local artifacts")
    for key, relative in FRESH_SOURCE_PATHS.items():
        record = mapping(inputs[key], f"fresh-local input {key}")
        exact_keys(record, {"path", "bytes", "sha256"}, f"fresh-local input {key}")
        archived = regular_file(BUNDLE, relative, f"fresh-local archived input {key}")
        require(
            archived.stat().st_size == record["bytes"]
            and sha256_file(archived) == record["sha256"],
            f"fresh-local archived input mismatch: {key}",
        )
    duplicate_pilot = regular_file(
        BUNDLE, "fresh-sources/run_swe_multistage_pilot.sh", "duplicate archived pilot"
    )
    require(
        sha256_file(duplicate_pilot) == inputs["pilot"]["sha256"],
        "duplicate archived pilot mismatch",
    )
    for key, raw_record in artifacts.items():
        record = mapping(raw_record, f"fresh-local artifact {key}")
        expected_keys = {"path", "bytes", "sha256"}
        if key in REPORT_KEYS:
            expected_keys.add("report_status")
        exact_keys(record, expected_keys, f"fresh-local artifact {key}")
        raw_path = record["path"]
        require(
            isinstance(raw_path, str)
            and bool(raw_path)
            and "\\" not in raw_path
            and "\x00" not in raw_path
            and PurePosixPath(raw_path).is_absolute(),
            f"fresh-local artifact path is not a canonical absolute POSIX path: {key}",
        )
        logical = PurePosixPath(raw_path).name
        require(logical == ARTIFACT_FILENAMES[key], f"fresh-local artifact name mismatch: {key}")
        byte_count = record["bytes"]
        require(
            isinstance(byte_count, int)
            and not isinstance(byte_count, bool)
            and byte_count > 0,
            f"fresh-local artifact byte count is invalid: {key}",
        )
        validate_digest(record["sha256"], f"fresh-local artifact digest {key}")
        require(logical in legacy, f"fresh-local artifact is absent from legacy archive: {logical}")
        digest, byte_count = legacy[logical]
        require(
            record["sha256"] == digest and record["bytes"] == byte_count,
            f"fresh-local artifact binding mismatch: {key}",
        )
        if key in REPORT_KEYS:
            require(record["report_status"] == "failed", f"fresh-local report status changed: {key}")
    verify_phases(value["phases"], suffix=".fresh")
    require(value["runtime_contract"] == EXPECTED_RUNTIME, "fresh-local runtime mismatch")


def verify_semantic_contract(paths: Mapping[str, Path]) -> None:
    summary = mapping(strict_json_file(paths["prompts_summary"], "prompt summary"), "summary")
    require(
        summary.get("prompt_count") == 8
        and summary.get("hidden_prompt_count") == 0
        and summary.get("explicit_control_prompt_count") == 8,
        "corrected prompt visibility counts regressed",
    )
    prompts = sequence(strict_json_file(paths["prompts"], "prompt bundle"), "prompts")
    require(len(prompts) == 8, "corrected prompt bundle must contain eight prompts")
    expected_stages = [f"S{index}" for index in range(8)]
    expected_ids = [
        f"swe-s{index}-000-django__django-13297" for index in range(8)
    ]
    prompt_ids: list[str] = []
    prompt_stages: list[str] = []
    for index, raw_prompt in enumerate(prompts):
        prompt = mapping(raw_prompt, f"prompt {index}")
        metadata = mapping(prompt.get("metadata"), f"prompt {index} metadata")
        stage = mapping(metadata.get("stage"), f"prompt {index} stage")
        prompt_ids.append(str(prompt.get("id")))
        prompt_stages.append(str(stage.get("id")))
        require(
            metadata.get("analysis_role") == "explicit_contaminated_control",
            f"prompt {index} is not an explicit contaminated control",
        )
        audit = mapping(metadata.get("visibility_audit"), f"prompt {index} visibility audit")
        records = sequence(audit.get("records"), f"prompt {index} visibility records")
        target_records = [
            mapping(record, "visibility record")
            for record in records
            if mapping(record, "visibility record").get("subject") == "target"
        ]
        require(
            target_records and all(record.get("exposed") is True for record in target_records),
            f"prompt {index} no longer records target exposure",
        )
    require(prompt_ids == expected_ids, "corrected prompt IDs/order no longer cover S0-S7")
    require(prompt_stages == expected_stages, "corrected prompt stages/order no longer cover S0-S7")
    summary_prompts = sequence(summary.get("prompts"), "prompt summary rows")
    require(
        [mapping(row, "prompt summary row").get("id") for row in summary_prompts]
        == expected_ids
        and [mapping(row, "prompt summary row").get("stage_id") for row in summary_prompts]
        == expected_stages,
        "prompt summary does not match the exact S0-S7 bundle",
    )

    source_sha = sha256_file(paths["prompts"])
    action_protocol_sha = sha256_file(paths["action_protocol"])
    action_prompts = sequence(
        strict_json_file(paths["action_prompts"], "action prompt bundle"),
        "action prompts",
    )
    require(len(action_prompts) == 8, "action prompt bundle must contain eight prompts")
    for index, (raw_source, raw_action) in enumerate(zip(prompts, action_prompts, strict=True)):
        source = mapping(raw_source, f"source prompt {index}")
        action = mapping(raw_action, f"action prompt {index}")
        action_metadata = mapping(action.get("metadata"), f"action prompt {index} metadata")
        binding = mapping(
            action_metadata.get("stage_action_probe"),
            f"action prompt {index} binding",
        )
        source_metadata = mapping(source.get("metadata"), f"source prompt {index} metadata")
        inherited_action_metadata = {
            key: value
            for key, value in action_metadata.items()
            if key != "stage_action_probe"
        }
        require(
            action.get("id") == expected_ids[index]
            and action.get("text") == source.get("text")
            and action.get("token_ids") == source.get("token_ids")
            and binding.get("source_prompt_bundle_sha256") == source_sha
            and binding.get("action_protocol_sha256") == action_protocol_sha
            and binding.get("exact_prompt_text_preserved") is True
            and binding.get("exact_prompt_token_ids_preserved") is True,
            f"action prompt {index} is not exactly bound to its source prompt",
        )
        require(
            inherited_action_metadata == source_metadata
            and set(action_metadata) == set(source_metadata) | {"stage_action_probe"},
            f"action prompt {index} changed inherited source metadata",
        )

    action_sha = sha256_file(paths["action_prompts"])
    action_summary = mapping(
        strict_json_file(paths["action_prompts_summary"], "action prompt summary"),
        "action prompt summary",
    )
    action_summary_rows = sequence(
        action_summary.get("prompts"), "action prompt summary rows"
    )
    require(
        action_summary.get("schema_version") == 1
        and action_summary.get("kind")
        == "swe_verified_stage_action_probe_materialization"
        and action_summary.get("source_prompt_bundle_sha256") == source_sha
        and action_summary.get("prompt_bundle_sha256") == action_sha
        and action_summary.get("action_protocol_sha256") == action_protocol_sha
        and action_summary.get("prompt_count") == 8
        and action_summary.get("available_action_label_count") == 8
        and action_summary.get("missing_action_label_count") == 0
        and action_summary.get("action_class_counts")
        == {"inspect": 5, "edit": 1, "validate": 1, "finalize": 1}
        and [mapping(row, "action summary row").get("id") for row in action_summary_rows]
        == expected_ids
        and [
            mapping(row, "action summary row").get("stage_id")
            for row in action_summary_rows
        ]
        == expected_stages,
        "action prompt summary is stale or not bound to S0-S7",
    )
    analysis = mapping(strict_json_file(paths["analysis"], "analysis"), "analysis")
    contract = mapping(analysis.get("interpretation_contract"), "analysis contract")
    require(
        analysis.get("gold_probe_status") == "no_hidden_gold_eligible_prompts"
        and contract.get("hidden_gold_eligible_prompt_count") == 0
        and contract.get("explicit_control_prompt_count") == 8,
        "analysis hidden-gold status regressed",
    )
    require(
        analysis.get("source_prompt_bundle_sha256") == source_sha
        and analysis.get("augmented_prompt_bundle_sha256") == action_sha
        and analysis.get("action_protocol_sha256") == action_protocol_sha,
        "analysis lineage does not match the published prompt bundles/protocol",
    )
    details = sequence(analysis.get("prompt_details"), "analysis prompt details")
    require(
        [mapping(row, "analysis prompt detail").get("id") for row in details]
        == expected_ids
        and [mapping(row, "analysis prompt detail").get("stage_id") for row in details]
        == expected_stages,
        "analysis prompt ordering does not match S0-S7",
    )

    for report_key in ("public_report", "nf4_report", "native_report"):
        report = mapping(strict_json_file(paths[report_key], report_key), report_key)
        model = mapping(report.get("model"), f"{report_key} model")
        runtime = mapping(report.get("runtime"), f"{report_key} runtime")
        lens = mapping(report.get("lens"), f"{report_key} lens")
        lens_expected = EXPECTED_REPORT_LENSES[report_key]
        require(
            report.get("schema_version") == 3
            and report.get("status") == "failed"
            and model.get("repo_id") == EXPECTED_RUNTIME["model"]["repo_id"]
            and model.get("revision") == EXPECTED_RUNTIME["model"]["revision"]
            and model.get("config_sha256") == EXPECTED_RUNTIME["model"]["config_sha256"]
            and model.get("index_sha256") == EXPECTED_RUNTIME["model"]["index_sha256"]
            and model.get("quant_method") == "modelopt"
            and model.get("quant_algo") == "MIXED_PRECISION",
            f"{report_key} model identity mismatch",
        )
        require(
            runtime.get("enforce_eager") is True
            and runtime.get("language_model_only") is True
            and runtime.get("mtp_enabled") is False
            and runtime.get("capture_adapter") == "vLLM apply_model forward hooks"
            and runtime.get("enable_prefix_caching") is True
            and runtime.get("gpu_memory_utilization")
            == EXPECTED_RUNTIME["gpu_memory_utilization"]
            and runtime.get("kv_cache_dtype") == EXPECTED_RUNTIME["kv_cache_dtype"]
            and runtime.get("kv_offloading_backend")
            == EXPECTED_RUNTIME["kv_offloading_backend"]
            and runtime.get("kv_offloading_size")
            == EXPECTED_RUNTIME["kv_offloading_size_gib"]
            and runtime.get("mamba_block_size") == EXPECTED_RUNTIME["mamba_block_size"]
            and runtime.get("max_model_len") == EXPECTED_RUNTIME["max_model_len"]
            and runtime.get("max_num_batched_tokens")
            == EXPECTED_RUNTIME["max_num_batched_tokens"]
            and runtime.get("stream_final_only") is True
            and runtime.get("transport_dtype") == "torch.float32"
            and runtime.get("readout_dtype") == "torch.bfloat16",
            f"{report_key} runtime identity mismatch",
        )
        require(
            lens.get("sha256") == lens_expected["sha256"]
            and lens.get("n_prompts") == lens_expected["n_prompts"]
            and lens.get("repo_id") == lens_expected["repo_id"]
            and lens.get("revision") == lens_expected["revision"]
            and lens.get("kind") == lens_expected["kind"]
            and lens.get("d_model") == 5120
            and lens.get("source_layers") == list(range(63))
            and lens.get("tensor_shape") == [5120, 5120],
            f"{report_key} lens identity mismatch",
        )
        experiments = sequence(report.get("experiments"), f"{report_key} experiments")
        require(len(experiments) == 8, f"{report_key} must contain eight experiments")
        for index, (raw_experiment, raw_action) in enumerate(
            zip(experiments, action_prompts, strict=True)
        ):
            experiment = mapping(raw_experiment, f"{report_key} experiment {index}")
            action = mapping(raw_action, f"action prompt {index}")
            vocabulary = mapping(
                experiment.get("scored_vocabulary"),
                f"{report_key} experiment {index} vocabulary",
            )
            require(
                experiment.get("id") == expected_ids[index]
                and experiment.get("prompt") == action.get("text")
                and experiment.get("prompt_token_ids") == action.get("token_ids")
                and experiment.get("metadata") == action.get("metadata")
                and vocabulary.get("token_ids") == action.get("score_token_ids"),
                f"{report_key} experiment {index} is not paired to the action prompt",
            )


def verify_publication() -> dict[str, int]:
    paths = verify_current_manifest()
    verify_evidence_ledger()
    legacy = verify_legacy()
    verify_fresh_manifest(legacy)
    verify_semantic_contract(paths)
    return {
        "current_records": len(INPUT_KEYS) + len(ARTIFACT_KEYS),
        "evidence_rows": 45,
        "legacy_artifacts": len(legacy),
        "prompts": 8,
    }


def main() -> int:
    try:
        result = verify_publication()
    except PublicationError as error:
        raise SystemExit(f"SWE multistage publication verification failed: {error}") from error
    print(
        "SWE multistage publication verified: "
        f"{result['current_records']} current records, {result['evidence_rows']} evidence "
        f"rows, {result['legacy_artifacts']} archived artifacts, and "
        f"{result['prompts']} explicit controls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
