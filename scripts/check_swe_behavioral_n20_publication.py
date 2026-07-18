#!/usr/bin/env python3
"""Fail-closed checks for the compact N=20 SWE behavioral publication."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "validation/jlens-swe-behavioral-n20-2026-07-18/publication"
CURRENT_PUBLICATION_MANIFEST_SHA256 = (
    "9c5077eb5f538e3630b78a460b8d2e4be7d231d86113bfcab4da409c74e2f5f2"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

INCLUDED_PATHS = {
    "analysis": "analysis.json",
    "next_token_transport_analysis": "next-token-transport-analysis.json",
    "action_layer_readout": "action-layer-readout.json",
    "campaign_evidence": "campaign_evidence.json",
    "prompts_summary": "prompts_summary.json",
    "run_manifest": "run_manifest.json",
    "analysis_checksum": "checksums/analysis.sha256",
    "prompts_checksum": "checksums/prompts.sha256",
    "public_report_checksum": "checksums/public-report.sha256",
    "nf4_report_checksum": "checksums/nf4-report.sha256",
    "native_report_checksum": "checksums/native-report.sha256",
    "run_manifest_checksum": "checksums/run_manifest.sha256",
    "official_outcomes_development": "official-outcomes/development.json",
    "official_outcomes_replication": "official-outcomes/replication.json",
}
OMITTED_PATHS = {
    "prompts": "prompts.json",
    "public_report": "public-report.json",
    "nf4_report": "nf4-report.json",
    "native_report": "native-report.json",
}
SOURCE_INPUT_PATHS = {
    "pilot": "scripts/run_swe_behavioral_pilot.sh",
    "materializer": "scripts/materialize_swe_behavioral_probes.py",
    "analyzer": "scripts/analyze_swe_behavioral_probes.py",
    "jlens_runner": "scripts/run_jlens_nvfp4.sh",
    "jlens_python_runner": "scripts/run_jlens_nvfp4.py",
    "model_checkpoint_verifier": "scripts/modelopt_checkpoint.py",
    "action_protocol": "configs/swe_stage_action_probes.json",
    "readout_protocol": "configs/swe_behavioral_readout_protocol.json",
    "cohort_manifest": "configs/swe_behavioral_n20_cohort.json",
}
SUPPLEMENTAL_SOURCES = {
    "next_token_transport": {
        "artifact_key": "next_token_transport_analysis",
        "source_attempt_relative_path": "next-token-transport-analysis.json",
        "decision_role": "supplemental_fit_capacity_diagnostic_only",
        "analyzer": {
            "path": "scripts/analyze_swe_next_token_transport.py",
            "bytes": 58761,
            "sha256": "154f463794c61f49bd8dc9e2d6530bb2a31e93af5614e679c1355f05ef0c90af",
        },
        "test": {
            "path": "tests/test_analyze_swe_next_token_transport.py",
            "bytes": 14460,
            "sha256": "9479c1650440d713b6d3d4ac211e005e60577ad8eb398d5ad4459a7cb5f551d0",
        },
        "protocol": {
            "path": "configs/swe_next_token_transport_protocol.json",
            "bytes": 8751,
            "sha256": "2474b31630f3074daea03577a64f32ce74332e3d4f028a3990b0817e50d6a331",
        },
    },
    "action_layer_readout": {
        "artifact_key": "action_layer_readout",
        "source_attempt_relative_path": "action-layer-readout.json",
        "decision_role": "supplemental_readout_refinement_only",
        "analyzer": {
            "path": "scripts/analyze_swe_action_layer_readout.py",
            "bytes": 73582,
            "sha256": "2d8b5ba9115a58a4086af27ae4170f1fd91fd7e396df5430a093bcd1a0acaf34",
        },
        "test": {
            "path": "tests/test_analyze_swe_action_layer_readout.py",
            "bytes": 13084,
            "sha256": "50f820eb7ef9e002fef7908a80e38f3f468939d52d91d49ffbaa2ec6c7cbe00c",
        },
        "protocol": {
            "path": "configs/swe_action_layer_readout_protocol.json",
            "bytes": 6636,
            "sha256": "b3119725902d567a9c7a2b916fea9de8f1cc00b1290e95304b8809c0e77eecc5",
        },
    },
}
EXPECTED_TASK_IDS = {
    "development": [
        "astropy__astropy-14539",
        "django__django-10914",
        "matplotlib__matplotlib-25332",
        "mwaskom__seaborn-3069",
        "psf__requests-6028",
        "pydata__xarray-4094",
        "pylint-dev__pylint-4970",
        "pytest-dev__pytest-8399",
        "scikit-learn__scikit-learn-12585",
        "sphinx-doc__sphinx-8269",
    ],
    "replication": [
        "astropy__astropy-14508",
        "django__django-16938",
        "matplotlib__matplotlib-26291",
        "psf__requests-5414",
        "pydata__xarray-4687",
        "pylint-dev__pylint-8898",
        "pytest-dev__pytest-10356",
        "scikit-learn__scikit-learn-25973",
        "sphinx-doc__sphinx-11510",
        "sympy__sympy-24066",
    ],
}
EXPECTED_OUTCOME_COUNTS = {
    "development": {"empty": 0, "error": 0, "resolved": 5, "unresolved": 5},
    "replication": {"empty": 0, "error": 0, "resolved": 4, "unresolved": 6},
}
EXPECTED_CAMPAIGN_METRICS = {
    "development": {
        "requests": 267,
        "metric_lines": 171,
        "accepted": 32650,
        "drafted": 37756,
        "fatal_tasks": [],
    },
    "replication": {
        "requests": 432,
        "metric_lines": 307,
        "accepted": 59139,
        "drafted": 67277,
        "fatal_tasks": [
            "pydata__xarray-4687",
            "pylint-dev__pylint-8898",
            "pytest-dev__pytest-10356",
            "sphinx-doc__sphinx-11510",
        ],
    },
}
EXPECTED_PHASE_STATUSES = {
    "initialization": "0",
    "campaign_validation": "0",
    "materialize": "0",
    "lens_artifact_validation": "0",
    "public_preflight": "0",
    "public": "1",
    "nf4_preflight": "0",
    "nf4": "1",
    "native_preflight": "0",
    "native": "1",
    "analyze": "0",
}
EXPECTED_CLAIM_BOUNDARY = {
    "raw_artifacts_committed": False,
    "omitted_hashes_are_identity_bindings_not_content_availability": True,
    "independent_reanalysis_requires_omitted_artifacts": True,
    "compact_bundle_supports": [
        "inspection of the exact published analysis and materialization summary",
        "inspection of the exact supplemental next-token transport and learned action-layer analyses",
        "verification of official binary outcome coverage and counts",
        "verification of source-manifest, analyzer, focused-test, protocol, sidecar, and omitted-artifact hash lineage",
    ],
    "not_available_in_this_bundle": [
        "the full prompt bundle and three layer-by-layer J-lens reports",
        "raw Qwen Code proxy captures, task workspaces, and vLLM server logs",
        "SWE-bench harness logs and per-instance reports",
        "model checkpoints and fitted lens tensors",
    ],
}


class PublicationError(ValueError):
    """Raised when the compact publication is unsafe, stale, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def require_close(actual: Any, expected: float, label: str) -> None:
    require(
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12),
        f"{label} changed",
    )


def require_attempt_path(
    value: Any, publication: Mapping[str, Any], filename: str, label: str
) -> None:
    require(isinstance(value, str), f"{label} must be text")
    source = mapping(publication["source_attempt"], "source attempt")
    parts = PurePosixPath(value).parts
    require(
        len(parts) >= 3
        and tuple(parts[-3:]) == ("attempts", source["id"], filename),
        f"{label} does not identify the immutable source attempt",
    )


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
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise PublicationError(f"missing root for {label}: {error}") from error
    current = resolved_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        require(not current.is_symlink(), f"{label} traverses a symlink: {relative}")
    try:
        mode = current.lstat().st_mode
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise PublicationError(f"missing {label}: {relative}: {error}") from error
    require(resolved.is_relative_to(resolved_root), f"{label} escapes its root: {relative}")
    require(stat.S_ISREG(mode), f"{label} is not a regular file: {relative}")
    return current


def verify_sidecar(path: Path, target_name: str, expected_digest: str) -> None:
    expected_digest = validate_digest(expected_digest, "sidecar target digest")
    expected = f"{expected_digest}  {target_name}\n".encode("ascii")
    try:
        actual = path.read_bytes()
    except OSError as error:
        raise PublicationError(f"cannot read sidecar {path}: {error}") from error
    require(actual == expected, f"sidecar grammar or target binding mismatch: {path}")


def verify_publication_manifest() -> tuple[Mapping[str, Any], dict[str, Path]]:
    manifest_path = regular_file(BUNDLE, "publication_manifest.json", "publication manifest")
    digest = sha256_file(manifest_path)
    require(
        digest == CURRENT_PUBLICATION_MANIFEST_SHA256,
        "publication manifest hash differs from the source pin",
    )
    sidecar = regular_file(
        BUNDLE, "publication_manifest.sha256", "publication manifest sidecar"
    )
    verify_sidecar(sidecar, manifest_path.name, digest)
    value = mapping(strict_json_file(manifest_path, "publication manifest"), "manifest")
    exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "source_attempt",
            "included_artifacts",
            "omitted_artifacts",
            "supplemental_analyses",
            "claim_boundary",
        },
        "publication manifest",
    )
    require(
        value["schema_version"] == 2
        and value["kind"] == "swe_verified_behavioral_n20_compact_publication",
        "publication manifest identity mismatch",
    )
    source = mapping(value["source_attempt"], "source attempt")
    exact_keys(
        source,
        {"id", "path", "run_manifest_sha256", "execution_status", "scientific_status"},
        "source attempt",
    )
    require(
        source["id"] == "attempt-20260718T184758Z-9CsrJ9"
        and source["path"] == f"attempts/{source['id']}"
        and source["execution_status"] == "complete"
        and source["scientific_status"] == "insufficient_support",
        "source attempt identity or status mismatch",
    )
    validate_digest(source["run_manifest_sha256"], "source run manifest digest")

    included = mapping(value["included_artifacts"], "included artifacts")
    exact_keys(included, set(INCLUDED_PATHS), "included artifacts")
    paths: dict[str, Path] = {}
    for key, expected_path in INCLUDED_PATHS.items():
        record = mapping(included[key], f"included artifact {key}")
        exact_keys(record, {"path", "bytes", "sha256"}, f"included artifact {key}")
        require(record["path"] == expected_path, f"included artifact path mismatch: {key}")
        require(
            isinstance(record["bytes"], int)
            and not isinstance(record["bytes"], bool)
            and record["bytes"] > 0,
            f"included artifact byte count is invalid: {key}",
        )
        validate_digest(record["sha256"], f"included artifact digest {key}")
        path = regular_file(BUNDLE, expected_path, f"included artifact {key}")
        require(path.stat().st_size == record["bytes"], f"included size mismatch: {key}")
        require(sha256_file(path) == record["sha256"], f"included hash mismatch: {key}")
        paths[key] = path

    supplements = mapping(value["supplemental_analyses"], "supplemental analyses")
    exact_keys(supplements, set(SUPPLEMENTAL_SOURCES), "supplemental analyses")
    for key, expected in SUPPLEMENTAL_SOURCES.items():
        record = mapping(supplements[key], f"supplemental analysis {key}")
        exact_keys(
            record,
            {
                "artifact_key",
                "source_attempt_relative_path",
                "decision_role",
                "analyzer",
                "test",
                "protocol",
            },
            f"supplemental analysis {key}",
        )
        artifact_key = record["artifact_key"]
        require(
            isinstance(artifact_key, str)
            and artifact_key in INCLUDED_PATHS
            and artifact_key == expected["artifact_key"]
            and record["source_attempt_relative_path"]
            == expected["source_attempt_relative_path"]
            and record["decision_role"] == expected["decision_role"]
            and INCLUDED_PATHS[artifact_key]
            == record["source_attempt_relative_path"],
            f"supplemental output identity changed: {key}",
        )
        canonical_relative(
            record["source_attempt_relative_path"],
            f"supplemental source-attempt path {key}",
        )
        for source_kind in ("analyzer", "test", "protocol"):
            source = mapping(record[source_kind], f"supplemental {key} {source_kind}")
            exact_keys(
                source,
                {"path", "bytes", "sha256"},
                f"supplemental {key} {source_kind}",
            )
            require(
                source == expected[source_kind],
                f"supplemental {source_kind} identity changed: {key}",
            )
            source_path = regular_file(
                ROOT, source["path"], f"supplemental {key} {source_kind}"
            )
            require(
                source_path.stat().st_size == source["bytes"]
                and sha256_file(source_path) == source["sha256"],
                f"supplemental {source_kind} source changed: {key}",
            )

    omitted = mapping(value["omitted_artifacts"], "omitted artifacts")
    exact_keys(omitted, set(OMITTED_PATHS), "omitted artifacts")
    for key, expected_path in OMITTED_PATHS.items():
        record = mapping(omitted[key], f"omitted artifact {key}")
        expected_keys = {"path", "bytes", "sha256", "reason"}
        if key != "prompts":
            expected_keys.add("report_status")
        exact_keys(record, expected_keys, f"omitted artifact {key}")
        require(record["path"] == expected_path, f"omitted artifact path mismatch: {key}")
        require(
            isinstance(record["bytes"], int)
            and not isinstance(record["bytes"], bool)
            and record["bytes"] > 0,
            f"omitted artifact byte count is invalid: {key}",
        )
        validate_digest(record["sha256"], f"omitted artifact digest {key}")
        if key == "prompts":
            require(
                record["reason"] == "not_committed_compact_bundle_policy",
                "prompt omission reason changed",
            )
        else:
            require(
                record["report_status"] == "failed"
                and record["reason"] == "not_committed_exceeds_github_single_file_limit"
                and record["bytes"] > 100_000_000,
                f"report omission contract changed: {key}",
            )
        require(
            not (BUNDLE / expected_path).exists(),
            f"omitted artifact is unexpectedly present: {key}",
        )

    require(value["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY, "claim boundary changed")
    declared = set(INCLUDED_PATHS.values()) | {
        "publication_manifest.json",
        "publication_manifest.sha256",
    }
    actual: set[str] = set()
    for path in BUNDLE.rglob("*"):
        require(not path.is_symlink(), f"publication contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(BUNDLE).as_posix())
    require(actual == declared, "publication directory is not manifest-complete")
    return value, paths


def _source_record_matches(
    source: Mapping[str, Any], published: Mapping[str, Any], *, label: str
) -> None:
    require(source.get("path_base") == "output_directory", f"{label} path base changed")
    require(
        source.get("bytes") == published.get("bytes")
        and source.get("sha256") == published.get("sha256"),
        f"{label} source-manifest binding mismatch",
    )


def verify_run_manifest(
    publication: Mapping[str, Any], paths: Mapping[str, Path]
) -> Mapping[str, Any]:
    run_manifest = mapping(strict_json_file(paths["run_manifest"], "run manifest"), "run manifest")
    exact_keys(
        run_manifest,
        {
            "schema_version",
            "kind",
            "status",
            "execution_status",
            "scientific_status",
            "mode",
            "failure",
            "path_contract",
            "attempt",
            "cohorts",
            "inputs",
            "artifacts",
            "phases",
            "runtime_contract",
            "integrity_contract",
        },
        "run manifest",
    )
    source_attempt = mapping(publication["source_attempt"], "source attempt")
    require(
        run_manifest["schema_version"] == 1
        and run_manifest["kind"] == "swe_verified_behavioral_combined_n20_pilot_run"
        and run_manifest["status"] == "complete"
        and run_manifest["execution_status"] == "complete"
        and run_manifest["scientific_status"] == "insufficient_support"
        and run_manifest["mode"] == "fresh_replay"
        and run_manifest["failure"] is None,
        "run manifest completion identity mismatch",
    )
    require(
        run_manifest["attempt"]
        == {"id": source_attempt["id"], "path": source_attempt["path"]},
        "run attempt mismatch",
    )
    require(
        sha256_file(paths["run_manifest"]) == source_attempt["run_manifest_sha256"],
        "source attempt does not bind the copied run manifest",
    )
    verify_sidecar(
        paths["run_manifest_checksum"],
        "run_manifest.json",
        source_attempt["run_manifest_sha256"],
    )

    artifacts = mapping(run_manifest["artifacts"], "run artifacts")
    included = mapping(publication["included_artifacts"], "included artifacts")
    omitted = mapping(publication["omitted_artifacts"], "omitted artifacts")
    for key in ("analysis", "campaign_evidence", "prompts_summary"):
        _source_record_matches(
            mapping(artifacts.get(key), f"run artifact {key}"),
            mapping(included[key], f"included artifact {key}"),
            label=key,
        )
    for key in (
        "analysis_checksum",
        "prompts_checksum",
        "public_report_checksum",
        "nf4_report_checksum",
        "native_report_checksum",
    ):
        _source_record_matches(
            mapping(artifacts.get(key), f"run artifact {key}"),
            mapping(included[key], f"included artifact {key}"),
            label=key,
        )
    for key in OMITTED_PATHS:
        source = mapping(artifacts.get(key), f"run artifact {key}")
        published = mapping(omitted[key], f"omitted artifact {key}")
        _source_record_matches(source, published, label=key)
        require(source.get("path") == published["path"], f"omitted source path mismatch: {key}")
        if key != "prompts":
            require(
                source.get("report_status") == published["report_status"],
                f"report status mismatch: {key}",
            )

    sidecars = {
        "analysis_checksum": ("analysis.json", included["analysis"]["sha256"]),
        "prompts_checksum": ("prompts.json", omitted["prompts"]["sha256"]),
        "public_report_checksum": (
            "public-report.json",
            omitted["public_report"]["sha256"],
        ),
        "nf4_report_checksum": ("nf4-report.json", omitted["nf4_report"]["sha256"]),
        "native_report_checksum": (
            "native-report.json",
            omitted["native_report"]["sha256"],
        ),
    }
    for key, (target, digest) in sidecars.items():
        verify_sidecar(paths[key], target, digest)

    phases = mapping(run_manifest["phases"], "run phases")
    exact_keys(phases, set(EXPECTED_PHASE_STATUSES), "run phases")
    for phase, expected_status in EXPECTED_PHASE_STATUSES.items():
        require(
            mapping(phases[phase], f"phase {phase}").get("status") == expected_status,
            f"phase status mismatch: {phase}",
        )

    runtime = mapping(run_manifest["runtime_contract"], "runtime contract")
    require(
        runtime.get("model")
        == {
            "repo_id": "nvidia/Qwen3.6-27B-NVFP4",
            "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "config_sha256": "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
            "index_sha256": "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
        }
        and runtime.get("layers") == list(range(24, 48))
        and runtime.get("positions") == [-1]
        and runtime.get("top_k") == 10
        and runtime.get("bootstrap_samples") == 5000
        and runtime.get("max_model_len") == 65536
        and runtime.get("stream_final_only") is True,
        "run runtime contract mismatch",
    )
    mtp_scope = mapping(runtime.get("mtp_scope"), "MTP scope")
    require(
        mapping(mtp_scope.get("trajectory_generation"), "generation MTP").get("enabled") is True
        and mapping(mtp_scope.get("eager_residual_capture"), "capture MTP").get("enabled") is False,
        "generation and residual-capture MTP scopes are not separated",
    )
    require(
        all(
            value is True
            for value in mapping(
                run_manifest["integrity_contract"], "integrity contract"
            ).values()
        ),
        "run integrity contract contains a failed assertion",
    )

    inputs = mapping(run_manifest["inputs"], "run inputs")
    for key, expected_path in SOURCE_INPUT_PATHS.items():
        record = mapping(inputs.get(key), f"run input {key}")
        require(
            record.get("path_base") == "repository_root" and record.get("path") == expected_path,
            f"run input path mismatch: {key}",
        )
        source_path = regular_file(ROOT, expected_path, f"run input {key}")
        require(
            source_path.stat().st_size == record.get("bytes")
            and sha256_file(source_path) == record.get("sha256"),
            f"run input changed since replay: {key}",
        )
    return run_manifest


def verify_official_outcomes(
    cohort: str, value: Any, expected_sha256: str
) -> Mapping[str, str]:
    outcome_file = mapping(value, f"{cohort} official outcomes")
    require(
        outcome_file.get("schema_version") == 1
        and outcome_file.get("kind") == "swe_verified_behavioral_official_outcomes",
        f"{cohort} official outcome identity mismatch",
    )
    expected_ids = EXPECTED_TASK_IDS[cohort]
    require(outcome_file.get("instance_ids") == expected_ids, f"{cohort} task order changed")
    rows = sequence(outcome_file.get("outcomes"), f"{cohort} outcome rows")
    require(
        [mapping(row, "outcome row").get("instance_id") for row in rows] == expected_ids,
        f"{cohort} outcome row coverage changed",
    )
    counts = Counter(mapping(row, "outcome row").get("outcome") for row in rows)
    recomputed = {
        name: counts.get(name, 0) for name in ("empty", "error", "resolved", "unresolved")
    }
    require(
        recomputed == EXPECTED_OUTCOME_COUNTS[cohort]
        and outcome_file.get("counts") == EXPECTED_OUTCOME_COUNTS[cohort],
        f"{cohort} official counts changed",
    )
    require(
        all(
            mapping(row, "outcome row").get("outcome") in {"resolved", "unresolved"}
            and SHA256_RE.fullmatch(str(mapping(row, "outcome row").get("patch_sha256")))
            for row in rows
        ),
        f"{cohort} has a nonbinary or unbound official outcome",
    )
    validate_digest(expected_sha256, f"{cohort} expected official digest")
    return {
        str(mapping(row, "outcome row")["instance_id"]): str(mapping(row, "outcome row")["outcome"])
        for row in rows
    }


def verify_semantic_contract(
    publication: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    campaign_evidence: Mapping[str, Any],
    official: Mapping[str, Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> None:
    included = mapping(publication["included_artifacts"], "included artifacts")
    omitted = mapping(publication["omitted_artifacts"], "omitted artifacts")
    cohorts = sequence(run_manifest.get("cohorts"), "run cohorts")
    require(
        [mapping(row, "run cohort").get("name") for row in cohorts]
        == ["development", "replication"],
        "run cohort order changed",
    )

    official_maps: dict[str, Mapping[str, str]] = {}
    for index, cohort in enumerate(("development", "replication")):
        run_cohort = mapping(cohorts[index], f"run cohort {cohort}")
        campaign_record = mapping(run_cohort.get("campaign"), f"{cohort} campaign record")
        expected_campaign_path = (
            "configs/swe_behavioral_campaign.json"
            if cohort == "development"
            else "configs/swe_behavioral_replication_campaign.json"
        )
        require(
            campaign_record.get("path_base") == "repository_root"
            and campaign_record.get("path") == expected_campaign_path,
            f"{cohort} campaign source path changed",
        )
        campaign_path = regular_file(ROOT, expected_campaign_path, f"{cohort} campaign")
        require(
            campaign_path.stat().st_size == campaign_record.get("bytes")
            and sha256_file(campaign_path) == campaign_record.get("sha256"),
            f"{cohort} campaign source changed",
        )
        official_record = mapping(run_cohort.get("official_outcomes"), f"{cohort} official record")
        pub_key = f"official_outcomes_{cohort}"
        require(
            official_record.get("sha256") == included[pub_key]["sha256"]
            and official_record.get("bytes") == included[pub_key]["bytes"],
            f"{cohort} official source-manifest binding mismatch",
        )
        official_maps[cohort] = verify_official_outcomes(
            cohort, official[cohort], str(official_record["sha256"])
        )

    require(
        summary.get("schema_version") == 1
        and summary.get("kind") == "swe_verified_behavioral_probe_combination"
        and summary.get("cohort_count") == 2
        and summary.get("task_count") == 20
        and summary.get("prompt_count") == 160
        and summary.get("global_request_count") == 699
        and summary.get("dynamic_target_count") == 44
        and summary.get("eligible_target_checkpoint_count") == 4
        and summary.get("prompt_bundle_sha256") == omitted["prompts"]["sha256"],
        "prompt summary identity or counts changed",
    )
    summary_cohorts = sequence(summary.get("cohorts"), "summary cohorts")
    require(
        [mapping(row, "summary cohort").get("id") for row in summary_cohorts]
        == ["development", "replication"],
        "summary cohort order changed",
    )
    campaign_hashes: list[str] = []
    for index, cohort in enumerate(("development", "replication")):
        row = mapping(summary_cohorts[index], f"summary cohort {cohort}")
        campaign_hash = mapping(cohorts[index], f"run cohort {cohort}")["campaign"]["sha256"]
        campaign_hashes.append(str(campaign_hash))
        require(
            row.get("source_task_instance_ids") == EXPECTED_TASK_IDS[cohort]
            and row.get("source_task_count") == 10
            and row.get("source_prompt_count") == 80
            and row.get("source_global_request_count")
            == EXPECTED_CAMPAIGN_METRICS[cohort]["requests"]
            and row.get("campaign_sha256") == campaign_hash,
            f"summary cohort contract changed: {cohort}",
        )
    require(
        summary.get("campaign_sha256s") == campaign_hashes
        and summary.get("source_campaign_sha256s") == campaign_hashes,
        "summary campaign lineage changed",
    )
    prompt_rows = sequence(summary.get("prompts"), "summary prompt rows")
    require(len(prompt_rows) == 160, "summary must contain 160 prompt records")
    prompt_ids = [mapping(row, "summary prompt row").get("id") for row in prompt_rows]
    require(len(set(prompt_ids)) == 160, "summary prompt IDs are not unique")
    prompt_tasks = Counter(
        mapping(row, "summary prompt row").get("instance_id") for row in prompt_rows
    )
    require(
        prompt_tasks
        == Counter(
            {task: 8 for tasks in EXPECTED_TASK_IDS.values() for task in tasks}
        ),
        "summary does not preserve eight checkpoints per task",
    )

    require(
        campaign_evidence.get("schema_version") == 1
        and campaign_evidence.get("kind") == "swe_behavioral_campaign_replay_evidence"
        and campaign_evidence.get("task_count") == 20,
        "campaign evidence identity mismatch",
    )
    evidence_cohorts = sequence(campaign_evidence.get("cohorts"), "evidence cohorts")
    require(
        [mapping(row, "evidence cohort").get("cohort") for row in evidence_cohorts]
        == ["development", "replication"],
        "campaign evidence order changed",
    )
    for index, cohort in enumerate(("development", "replication")):
        row = mapping(evidence_cohorts[index], f"campaign evidence {cohort}")
        expected = EXPECTED_CAMPAIGN_METRICS[cohort]
        proxy = mapping(row.get("proxy_capture_binding"), f"{cohort} proxy binding")
        mtp = mapping(row.get("mtp_speculative_decoding"), f"{cohort} MTP evidence")
        fatal = sequence(row.get("fatal_turn_limit_tasks"), f"{cohort} fatal tasks")
        require(
            row.get("instance_ids") == EXPECTED_TASK_IDS[cohort]
            and proxy.get("algorithm") == "filename_length_payload_sha256_v1"
            and proxy.get("request_count") == expected["requests"]
            and mtp.get("metric_format") == "vllm_spec_decoding_metrics_v1"
            and mtp.get("metric_line_count") == expected["metric_lines"]
            and mtp.get("accepted_tokens") == expected["accepted"]
            and mtp.get("drafted_tokens") == expected["drafted"]
            and math.isclose(
                float(mtp.get("weighted_acceptance_rate")),
                expected["accepted"] / expected["drafted"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and [mapping(item, "fatal task").get("instance_id") for item in fatal]
            == expected["fatal_tasks"],
            f"campaign generation/MTP evidence changed: {cohort}",
        )

    require(
        analysis.get("schema_version") == 1
        and analysis.get("kind") == "swe_verified_behavioral_task_held_out_analysis"
        and analysis.get("analysis_version") == "task-held-out-paired-decision-v2"
        and analysis.get("status") == "insufficient_split_or_class_support"
        and analysis.get("operational_status") == "insufficient_split_or_class_support",
        "analysis identity or operational status changed",
    )
    expected_analysis_inputs = {
        "prompts": omitted["prompts"]["sha256"],
        "public_report": omitted["public_report"]["sha256"],
        "nf4_report": omitted["nf4_report"]["sha256"],
        "native_report": omitted["native_report"]["sha256"],
        "protocol": mapping(run_manifest["inputs"], "run inputs")["readout_protocol"]["sha256"],
    }
    require(analysis.get("inputs") == expected_analysis_inputs, "analysis input hashes changed")
    campaign = mapping(analysis.get("campaign"), "analysis campaign")
    protocol = mapping(analysis.get("protocol"), "analysis protocol")
    require(
        campaign.get("campaign_count") == 2
        and campaign.get("task_count") == 20
        and campaign.get("prompt_count") == 160
        and campaign.get("repository_count") == 11
        and campaign.get("campaign_sha256s") == campaign_hashes
        and protocol.get("fixed_layers") == list(range(24, 48))
        and mapping(protocol.get("bootstrap"), "analysis bootstrap").get("samples") == 5000,
        "analysis campaign or protocol scope changed",
    )
    expected_certification = {
        "experiment_count": 160,
        "final_logits_within_tolerance": 70,
        "final_norm_within_tolerance": 157,
        "final_top5_match": 159,
        "fully_certified": 68,
        "greedy_top1_match": 159,
        "report_status": "failed",
    }
    eligibility = mapping(analysis.get("numerical_eligibility"), "numerical eligibility")
    require(
        eligibility
        == {
            "public": expected_certification,
            "nf4": expected_certification,
            "native": expected_certification,
        },
        "report numerical certification counts changed",
    )
    pairing = mapping(analysis.get("pairing"), "analysis pairing")
    require(
        pairing.get("prompt_count") == 160
        and pairing.get("reports") == ["public", "nf4", "native"]
        and all(
            pairing.get(field) is True
            for field in (
                "exact_prompt_id_text_tokens_metadata_equal",
                "final_model_readouts_equal",
                "final_reconstruction_evidence_equal",
                "fixed_band_ordinary_logit_readouts_equal",
                "residual_capture_manifests_equal",
                "runtime_identity_equal",
            )
        ),
        "paired replay identity changed",
    )
    decision = mapping(analysis.get("scientific_decision"), "scientific decision")
    support = mapping(decision.get("support"), "decision support")
    coverage = mapping(support.get("joint_coverage"), "joint coverage")
    require(
        decision.get("classification") == "insufficient_support"
        and decision.get("probe_valid") is False
        and decision.get("next_step")
        == "collect the missing jointly certified task-level controls; do not refit"
        and decision.get("reason_codes")
        == ["predeclared_coverage_or_paired_inference_gate_failed"]
        and support.get("complete") is False
        and coverage.get("all_predeclared_joint_coverage_gates_pass") is False,
        "scientific decision or next step changed",
    )
    future = mapping(coverage.get("future_identifier"), "future coverage")
    action = mapping(coverage.get("next_action"), "action coverage")
    outcome = mapping(coverage.get("official_outcome"), "outcome coverage")
    require(
        future.get("eligible_target_row_count") == 4
        and future.get("repository_count") == 1
        and future.get("task_averaged_observation_count") == 1
        and action.get("jointly_certified_labeled_row_count") == 66
        and action.get("selected_row_count") == 160
        and action.get("jointly_certified_selected_row_fraction") == 0.4125
        and outcome.get("jointly_certified_binary_outcome_task_count") == 8
        and outcome.get("selected_task_count") == 20
        and outcome.get("nonbinary_error_empty_or_missing_imputed_as_failure") is False,
        "predeclared coverage failure evidence changed",
    )
    tracks = mapping(analysis.get("tracks"), "analysis tracks")
    require(
        len(
            sequence(
                mapping(tracks.get("next_action"), "next action track").get("rows"),
                "next action rows",
            )
        )
        == 154
        and len(
            sequence(
                mapping(tracks.get("official_outcome"), "outcome track").get(
                    "rows"
                ),
                "outcome rows",
            )
        )
        == 20
        and len(
            sequence(
                mapping(
                    tracks.get("future_target_vs_fixed_hidden_same_task_foil"),
                    "future track",
                ).get("rows"),
                "future rows",
            )
        )
        == 4,
        "analysis track row coverage changed",
    )

    audits = sequence(summary.get("task_audits"), "task audits")
    ordered_tasks = EXPECTED_TASK_IDS["development"] + EXPECTED_TASK_IDS["replication"]
    require(
        [mapping(row, "task audit").get("instance_id") for row in audits]
        == ordered_tasks,
        "task audit order changed",
    )
    for raw in audits:
        audit = mapping(raw, "task audit")
        cohort = str(audit.get("cohort_id"))
        instance_id = str(audit.get("instance_id"))
        outcome_audit = mapping(audit.get("official_outcome"), "task official outcome")
        expected_verdict = official_maps[cohort][instance_id]
        require(
            outcome_audit.get("status") == "available"
            and outcome_audit.get("verdict") == expected_verdict
            and outcome_audit.get("class_id")
            == ("success" if expected_verdict == "resolved" else "failure")
            and outcome_audit.get("official_outcomes_sha256")
            == included[f"official_outcomes_{cohort}"]["sha256"],
            f"task outcome binding changed: {instance_id}",
        )


def verify_next_token_transport(
    publication: Mapping[str, Any], value: Mapping[str, Any]
) -> None:
    exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "analysis_version",
            "status",
            "decision_role",
            "primary_semantic_decision_override_forbidden",
            "inputs",
            "protocol",
            "pairing",
            "checkpoint_eligibility_audit",
            "tracks",
            "behavioral_semantic_analysis",
            "interpretation",
        },
        "next-token transport analysis",
    )
    require(
        value["schema_version"] == 1
        and value["kind"] == "swe_verified_greedy_next_token_transport_analysis"
        and value["analysis_version"] == "strict-and-amended-sensitivity-v2"
        and value["status"] == "complete"
        and value["decision_role"] == "supplemental_fit_capacity_diagnostic_only"
        and value["primary_semantic_decision_override_forbidden"] is True,
        "next-token transport identity or decision role changed",
    )
    included = mapping(publication["included_artifacts"], "included artifacts")
    omitted = mapping(publication["omitted_artifacts"], "omitted artifacts")
    supplements = mapping(publication["supplemental_analyses"], "supplements")
    supplement = mapping(supplements["next_token_transport"], "transport supplement")

    inputs = mapping(value["inputs"], "transport inputs")
    exact_keys(inputs, {"prompts", "reports"}, "transport inputs")
    prompts = mapping(inputs["prompts"], "transport prompts input")
    require(
        prompts.get("bytes") == omitted["prompts"]["bytes"]
        and prompts.get("sha256") == omitted["prompts"]["sha256"],
        "transport prompt input binding changed",
    )
    require_attempt_path(prompts.get("path"), publication, "prompts.json", "transport prompts")
    reports = mapping(inputs["reports"], "transport reports")
    exact_keys(reports, {"public", "nf4", "native"}, "transport reports")
    for method, artifact_key in (
        ("public", "public_report"),
        ("nf4", "nf4_report"),
        ("native", "native_report"),
    ):
        report = mapping(reports[method], f"transport {method} report")
        source = mapping(omitted[artifact_key], f"omitted {artifact_key}")
        require(
            report.get("bytes") == source["bytes"]
            and report.get("sha256") == source["sha256"]
            and report.get("report_status") == source["report_status"],
            f"transport raw report binding changed: {method}",
        )
        require_attempt_path(
            report.get("path"), publication, f"{method}-report.json", f"transport {method} report"
        )

    behavioral = mapping(
        value["behavioral_semantic_analysis"], "transport behavioral analysis"
    )
    require(
        behavioral.get("bytes") == included["analysis"]["bytes"]
        and behavioral.get("sha256") == included["analysis"]["sha256"]
        and behavioral.get("status") == "available_hash_bound_same_replay"
        and behavioral.get("operational_status")
        == "insufficient_split_or_class_support"
        and behavioral.get("scientific_classification") == "insufficient_support"
        and behavioral.get("scientific_reason_codes")
        == ["predeclared_coverage_or_paired_inference_gate_failed"]
        and behavioral.get("primary_semantic_decision_override_forbidden") is True,
        "transport binding to the frozen behavioral decision changed",
    )
    require_attempt_path(
        behavioral.get("path"), publication, "analysis.json", "transport behavioral analysis"
    )
    protocol = mapping(value["protocol"], "transport protocol")
    protocol_source = mapping(supplement["protocol"], "transport protocol source")
    require(
        protocol.get("id") == "swe-n20-greedy-next-token-transport-v2"
        and protocol.get("path") == protocol_source["path"]
        and protocol.get("sha256") == protocol_source["sha256"]
        and mapping(protocol.get("inference"), "transport inference").get("samples")
        == 5000,
        "transport protocol identity changed",
    )
    interpretation = mapping(value["interpretation"], "transport interpretation")
    require(
        interpretation.get("decision_role")
        == "supplemental fit-capacity diagnostic only"
        and interpretation.get("primary_semantic_decision_override_forbidden") is True
        and "chain of thought"
        in sequence(interpretation.get("does_not_test"), "transport exclusions"),
        "transport sensitivity-only interpretation changed",
    )
    pairing = mapping(value["pairing"], "transport pairing")
    require(
        pairing.get("status") == "passed_exact_fail_closed"
        and pairing.get("checkpoint_count") == 160
        and pairing.get("task_count") == 20
        and pairing.get("repository_count") == 11
        and all(
            pairing.get(key) is True
            for key in (
                "exact_prompt_pairing",
                "exact_residual_pairing",
                "exact_ordinary_logit_pairing",
                "same_generated_token_across_reports",
                "same_target_token_at_every_fixed_layer_and_method",
                "captured_final_model_top1_match_required_for_eligibility",
            )
        ),
        "transport pairing contract changed",
    )
    eligibility = sequence(
        value["checkpoint_eligibility_audit"], "transport eligibility audit"
    )
    require(
        len(eligibility) == 160
        and len({mapping(row, "transport eligibility row").get("id") for row in eligibility})
        == 160
        and sum(
            mapping(row, "transport eligibility row").get("strict_eligible") is True
            for row in eligibility
        )
        == 68
        and sum(
            mapping(row, "transport eligibility row").get("sensitivity_eligible") is True
            for row in eligibility
        )
        == 155,
        "transport checkpoint eligibility changed",
    )

    tracks = mapping(value["tracks"], "transport tracks")
    exact_keys(
        tracks,
        {"strict_primary", "paired_stable_reconstruction_sensitivity"},
        "transport tracks",
    )
    strict = mapping(tracks["strict_primary"], "strict transport track")
    strict_support = mapping(strict.get("support"), "strict transport support")
    strict_classification = mapping(
        strict.get("classification"), "strict transport classification"
    )
    require(
        strict.get("status") == "available"
        and strict.get("role") == "strict_primary"
        and strict.get("decision_overrides_behavioral_semantic_analysis") is False
        and strict_support.get("status") == "failed"
        and strict_support.get("all_gates_pass") is False
        and strict_support.get("eligible_checkpoint_count") == 68
        and strict_support.get("eligible_fraction") == 0.425
        and strict_support.get("eligible_task_count") == 20
        and strict_support.get("eligible_repository_count") == 11
        and strict_classification
        == {"classification": "insufficient_support", "reason_codes": ["support_gate_failed"]},
        "strict transport insufficient-support gate changed",
    )
    strict_checks = mapping(strict_support.get("checks"), "strict transport checks")
    require(
        strict_checks.get("minimum_checkpoint_count_pass") is False
        and strict_checks.get("minimum_checkpoint_fraction_pass") is False
        and strict_checks.get("minimum_checkpoints_per_task_pass") is False,
        "strict transport failure reasons changed",
    )

    sensitivity = mapping(
        tracks["paired_stable_reconstruction_sensitivity"],
        "transport sensitivity track",
    )
    sensitivity_support = mapping(
        sensitivity.get("support"), "transport sensitivity support"
    )
    sensitivity_classification = mapping(
        sensitivity.get("classification"), "transport sensitivity classification"
    )
    require(
        sensitivity.get("status") == "available"
        and sensitivity.get("role")
        == "post_public_numerical_diagnostic_amendment_sensitivity_cannot_override_primary"
        and sensitivity.get("decision_overrides_behavioral_semantic_analysis") is False
        and sensitivity_support.get("status") == "passed"
        and sensitivity_support.get("all_gates_pass") is True
        and sensitivity_support.get("eligible_checkpoint_count") == 155
        and sensitivity_support.get("eligible_fraction") == 0.96875
        and sensitivity_support.get("eligible_task_count") == 20
        and sensitivity_support.get("eligible_repository_count") == 11
        and sensitivity_classification.get("classification")
        == "sensitivity_readout_control_failure"
        and sensitivity_classification.get("reason_codes")
        == ["public_minus_logit_ci_upper_nonpositive"],
        "transport sensitivity classification or support changed",
    )
    rule_audit = mapping(
        sensitivity_classification.get("rule_audit"), "transport rule audit"
    )
    require(
        rule_audit
        == {
            "material_native_deficit_pass": False,
            "no_material_native_deficit_pass": False,
            "public_positive_control_pass": False,
            "readout_control_failure_pass": True,
        },
        "transport decision rule audit changed",
    )
    expected_utilities = {
        "ordinary_logit": 0.6114097870811301,
        "public_jacobian": 0.3769213265084506,
        "native_jacobian": 0.30778315950156665,
        "nf4_jacobian": 0.23768552182219588,
    }
    methods = mapping(sensitivity.get("methods"), "transport sensitivity methods")
    exact_keys(methods, set(expected_utilities), "transport sensitivity methods")
    for method, expected in expected_utilities.items():
        record = mapping(methods[method], f"transport sensitivity method {method}")
        require(
            record.get("status") == "available"
            and record.get("task_equal_weighting") is True,
            f"transport method status changed: {method}",
        )
        require_close(
            mapping(record.get("metrics"), f"transport metrics {method}").get(
                "normalized_rank_utility"
            ),
            expected,
            f"transport normalized rank utility {method}",
        )
    expected_comparisons = {
        "public_jacobian_minus_ordinary_logit": (
            -0.23448846057267952,
            -0.26189783341346545,
            -0.20506711487358711,
        ),
        "public_jacobian_minus_native_jacobian": (
            0.06913816700688398,
            0.06313168094866745,
            0.07519090159678049,
        ),
        "public_jacobian_minus_nf4_jacobian": (
            0.13923580468625474,
            0.13168685255182727,
            0.14722053567721483,
        ),
    }
    comparisons = mapping(sensitivity.get("comparisons"), "transport comparisons")
    exact_keys(comparisons, set(expected_comparisons), "transport comparisons")
    for name, (estimate, lower, upper) in expected_comparisons.items():
        comparison = mapping(comparisons[name], f"transport comparison {name}")
        bootstrap = mapping(comparison.get("bootstrap"), f"transport bootstrap {name}")
        interval = mapping(
            bootstrap.get("confidence_interval"), f"transport interval {name}"
        )
        require(
            bootstrap.get("status") == "available"
            and bootstrap.get("samples_requested") == 5000
            and bootstrap.get("samples_valid") == 5000
            and bootstrap.get("valid_fraction") == 1.0,
            f"transport bootstrap status changed: {name}",
        )
        require_close(comparison.get("estimate"), estimate, f"transport estimate {name}")
        require_close(interval.get("lower"), lower, f"transport CI lower {name}")
        require_close(interval.get("upper"), upper, f"transport CI upper {name}")


def _verify_action_track(
    track: Mapping[str, Any], *, sensitivity: bool
) -> None:
    label = "action sensitivity" if sensitivity else "strict action"
    support = mapping(track.get("support"), f"{label} support")
    expected_rows = 149 if sensitivity else 66
    expected_support = (
        {"inspect": 99, "edit": 12, "validate": 25, "finalize": 13}
        if sensitivity
        else {"inspect": 48, "edit": 4, "validate": 8, "finalize": 6}
    )
    require(
        support.get("complete") is False
        and support.get("row_count") == expected_rows
        and support.get("task_count") == 20
        and support.get("repository_count") == 11
        and support.get("class_support") == expected_support,
        f"{label} support changed",
    )
    checks = mapping(support.get("checks"), f"{label} support checks")
    require(
        checks.get("all_classes") is True
        and checks.get("all_methods_complete_nested_crossfit") is True
        and checks.get("minimum_repositories") is True
        and checks.get("minimum_tasks") is True
        and (
            checks.get("minimum_rows_per_task") is False
            if sensitivity
            else checks.get("minimum_rows") is False
        ),
        f"{label} insufficient-support reason changed",
    )
    expected_accuracy = (
        {
            "native_jacobian": 0.6497571872571872,
            "nf4_jacobian": 0.7099222999222999,
            "ordinary_logit": 0.6699222999222999,
            "public_jacobian": 0.6321814296814297,
        }
        if sensitivity
        else {
            "native_jacobian": 0.75,
            "nf4_jacobian": 0.78125,
            "ordinary_logit": 0.8072916666666667,
            "public_jacobian": 0.6822916666666667,
        }
    )
    methods = mapping(track.get("methods"), f"{label} methods")
    exact_keys(methods, set(expected_accuracy), f"{label} methods")
    for method, expected in expected_accuracy.items():
        record = mapping(methods[method], f"{label} method {method}")
        require(
            record.get("status") == "available"
            and record.get("row_count") == expected_rows
            and record.get("repository_count") == 11
            and record.get("fold_count") == 11
            and record.get("successful_fold_count") == 11
            and record.get("complete_prediction_coverage") is True,
            f"{label} nested crossfit coverage changed: {method}",
        )
        require_close(
            mapping(record.get("metrics"), f"{label} metrics {method}").get(
                "balanced_accuracy"
            ),
            expected,
            f"{label} balanced accuracy {method}",
        )
    signals = mapping(track.get("signals"), f"{label} signals")
    require(
        signals
        == {
            "actionable": False,
            "native_fit_capacity_signal": False,
            "no_native_refit_signal": False,
            "readout_refinement_signal": False,
        },
        f"{label} signals changed",
    )
    ordinal = mapping(
        track.get("posthoc_descriptive_checkpoint_ordinal_prior"),
        f"{label} ordinal prior",
    )
    require(
        ordinal.get("status") == "POSTHOC_DESCRIPTIVE_ONLY"
        and ordinal.get("used_for_feature_or_hyperparameter_selection") is False
        and ordinal.get("used_for_material_effect_signals_or_classification") is False
        and ordinal.get("used_for_paired_primary_comparisons") is False,
        f"{label} ordinal baseline role changed",
    )
    require_close(
        mapping(ordinal.get("metrics"), f"{label} ordinal metrics").get(
            "balanced_accuracy"
        ),
        0.5923232323232324 if sensitivity else 0.6354166666666666,
        f"{label} ordinal balanced accuracy",
    )


def _verify_action_comparison(
    comparisons: Mapping[str, Any],
    name: str,
    expected: tuple[float, float, float],
    label: str,
) -> None:
    comparison = mapping(comparisons.get(name), f"{label} comparison {name}")
    interval = mapping(
        mapping(comparison.get("intervals"), f"{label} intervals {name}").get(
            "balanced_accuracy_gain"
        ),
        f"{label} balanced-accuracy interval {name}",
    )
    require(
        comparison.get("status") == "available"
        and comparison.get("samples_requested") == 5000
        and comparison.get("valid_fraction", 0.0) >= 0.8,
        f"{label} bootstrap status changed: {name}",
    )
    require_close(
        mapping(
            comparison.get("observed_benefit_deltas"),
            f"{label} observed deltas {name}",
        ).get("balanced_accuracy_gain"),
        expected[0],
        f"{label} balanced-accuracy delta {name}",
    )
    require_close(interval.get("lower"), expected[1], f"{label} CI lower {name}")
    require_close(interval.get("upper"), expected[2], f"{label} CI upper {name}")


def verify_action_layer_readout(
    publication: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    value: Mapping[str, Any],
) -> None:
    exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "status",
            "classification",
            "next_step",
            "inputs",
            "protocol",
            "decision_audit",
            "raw_report_revalidation",
            "eligibility_ledger",
            "tracks",
        },
        "action-layer readout analysis",
    )
    require(
        value["schema_version"] == 1
        and value["kind"] == "swe_n20_action_layer_nested_readout_analysis"
        and value["status"] == "complete"
        and value["classification"] == "insufficient_strict_support"
        and value["next_step"]
        == "treat sensitivity as descriptive and collect strict numerical coverage",
        "action-layer identity or strict decision changed",
    )
    included = mapping(publication["included_artifacts"], "included artifacts")
    omitted = mapping(publication["omitted_artifacts"], "omitted artifacts")
    supplements = mapping(publication["supplemental_analyses"], "supplements")
    action_source = mapping(supplements["action_layer_readout"], "action supplement")
    transport_source = mapping(
        supplements["next_token_transport"], "transport supplement"
    )
    inputs = mapping(value["inputs"], "action inputs")
    require(
        inputs
        == {
            "action_protocol": action_source["protocol"]["sha256"],
            "behavioral_analysis": included["analysis"]["sha256"],
            "behavioral_protocol": mapping(run_manifest["inputs"], "run inputs")[
                "readout_protocol"
            ]["sha256"],
            "native_report": omitted["native_report"]["sha256"],
            "nf4_report": omitted["nf4_report"]["sha256"],
            "prompts": omitted["prompts"]["sha256"],
            "public_report": omitted["public_report"]["sha256"],
            "transport_protocol": transport_source["protocol"]["sha256"],
        },
        "action-layer input and raw-report hashes changed",
    )
    protocol = mapping(value["protocol"], "action protocol")
    require(
        protocol.get("id") == "swe-n20-action-layer-readout-v1"
        and protocol.get("sha256") == action_source["protocol"]["sha256"]
        and protocol.get("layers") == list(range(24, 48))
        and protocol.get("class_ids") == ["inspect", "edit", "validate", "finalize"]
        and protocol.get("feature_count") == 96
        and mapping(protocol.get("bootstrap"), "action bootstrap").get("samples")
        == 5000
        and mapping(protocol.get("bootstrap"), "action bootstrap").get(
            "models_refit_inside_bootstrap"
        )
        is False,
        "action-layer protocol changed",
    )
    audit = mapping(value["decision_audit"], "action decision audit")
    require(
        audit.get("strict_track_controls_classification") is True
        and audit.get("sensitivity_track_can_override_primary") is False
        and audit.get("best_layer_or_feature_selection_performed") is False
        and audit.get("outer_heldout_repository_used_for_scaling_class_weights_or_C_selection")
        is False
        and audit.get("bootstrap_refit_models") is False
        and audit.get("cohort_subgroups_are_independent_replication") is False
        and audit.get("interpretation")
        == "behavioral vocabulary-feature readout, not hidden prose, chain of thought, or a replacement Jacobian-lens fit",
        "action-layer decision boundary changed",
    )
    revalidation = mapping(
        value["raw_report_revalidation"], "action raw-report revalidation"
    )
    pairing = mapping(revalidation.get("pairing"), "action report pairing")
    require(
        revalidation.get("required_for_sensitivity_rows_omitted_by_strict_behavioral_analysis")
        is True
        and revalidation.get("strict_features_exactly_equal_to_behavioral_analysis")
        is True
        and pairing.get("prompt_count") == 160
        and pairing.get("reports") == ["public", "nf4", "native"]
        and all(
            pairing.get(key) is True
            for key in (
                "exact_prompt_id_text_tokens_metadata_equal",
                "final_model_readouts_equal",
                "final_reconstruction_evidence_equal",
                "fixed_band_ordinary_logit_readouts_equal",
                "residual_capture_manifests_equal",
                "runtime_identity_equal",
            )
        ),
        "action raw-report revalidation changed",
    )
    ledger = sequence(value["eligibility_ledger"], "action eligibility ledger")
    require(
        len(ledger) == 154
        and len({mapping(row, "action eligibility row").get("row_id") for row in ledger})
        == 154
        and sum(
            mapping(row, "action eligibility row").get("strict_feature_retained") is True
            for row in ledger
        )
        == 66
        and sum(
            mapping(row, "action eligibility row").get("sensitivity_feature_retained")
            is True
            for row in ledger
        )
        == 149,
        "action eligibility coverage changed",
    )
    tracks = mapping(value["tracks"], "action tracks")
    exact_keys(
        tracks,
        {"strict_primary", "paired_stable_reconstruction_sensitivity"},
        "action tracks",
    )
    strict = mapping(tracks["strict_primary"], "strict action track")
    sensitivity = mapping(
        tracks["paired_stable_reconstruction_sensitivity"],
        "action sensitivity track",
    )
    require(
        strict.get("role") == "primary_supplemental_readout_refinement_track"
        and sensitivity.get("role") == "post_public_numeric_diagnostic_sensitivity_only"
        and sensitivity.get("primary_decision_override_forbidden") is True,
        "action strict/sensitivity roles changed",
    )
    _verify_action_track(strict, sensitivity=False)
    _verify_action_track(sensitivity, sensitivity=True)
    strict_comparisons = mapping(
        strict.get("paired_comparisons"), "strict action comparisons"
    )
    sensitivity_comparisons = mapping(
        sensitivity.get("paired_comparisons"), "action sensitivity comparisons"
    )
    _verify_action_comparison(
        strict_comparisons,
        "learned_public_jacobian_minus_learned_ordinary_logit",
        (-0.125, -0.33785962301587286, -0.009615384615384692),
        "strict action",
    )
    _verify_action_comparison(
        strict_comparisons,
        "learned_public_jacobian_minus_frozen_public_jacobian_readout",
        (0.20833333333333337, -0.04665570175438585, 0.4634845020325204),
        "strict action",
    )
    _verify_action_comparison(
        sensitivity_comparisons,
        "learned_public_jacobian_minus_learned_ordinary_logit",
        (-0.03774087024087025, -0.12247646947575014, 0.04192493508439029),
        "action sensitivity",
    )
    _verify_action_comparison(
        sensitivity_comparisons,
        "learned_public_jacobian_minus_frozen_public_jacobian_readout",
        (0.20884421134421133, 0.031704075318246164, 0.39886800699300695),
        "action sensitivity",
    )


def verify_publication() -> dict[str, int]:
    publication, paths = verify_publication_manifest()
    run_manifest = verify_run_manifest(publication, paths)
    values = {
        "summary": mapping(strict_json_file(paths["prompts_summary"], "prompt summary"), "summary"),
        "campaign_evidence": mapping(
            strict_json_file(paths["campaign_evidence"], "campaign evidence"),
            "campaign evidence",
        ),
        "analysis": mapping(strict_json_file(paths["analysis"], "analysis"), "analysis"),
        "next_token_transport": mapping(
            strict_json_file(
                paths["next_token_transport_analysis"],
                "next-token transport analysis",
            ),
            "next-token transport analysis",
        ),
        "action_layer_readout": mapping(
            strict_json_file(paths["action_layer_readout"], "action-layer readout"),
            "action-layer readout",
        ),
        "development": mapping(
            strict_json_file(
                paths["official_outcomes_development"], "development outcomes"
            ),
            "development outcomes",
        ),
        "replication": mapping(
            strict_json_file(
                paths["official_outcomes_replication"], "replication outcomes"
            ),
            "replication outcomes",
        ),
    }
    verify_semantic_contract(
        publication,
        run_manifest,
        values["summary"],
        values["campaign_evidence"],
        {"development": values["development"], "replication": values["replication"]},
        values["analysis"],
    )
    verify_next_token_transport(publication, values["next_token_transport"])
    verify_action_layer_readout(
        publication, run_manifest, values["action_layer_readout"]
    )
    return {
        "included_artifacts": len(INCLUDED_PATHS),
        "omitted_artifacts": len(OMITTED_PATHS),
        "tasks": 20,
        "prompts": 160,
    }


def main() -> int:
    try:
        result = verify_publication()
    except PublicationError as error:
        raise SystemExit(f"SWE behavioral N=20 publication verification failed: {error}") from error
    print(
        "SWE behavioral N=20 compact publication verified: "
        f"{result['tasks']} tasks, {result['prompts']} checkpoints, "
        f"{result['included_artifacts']} included artifacts, and "
        f"{result['omitted_artifacts']} hash-bound omitted artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
