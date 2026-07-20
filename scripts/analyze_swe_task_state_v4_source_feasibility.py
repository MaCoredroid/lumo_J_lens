#!/usr/bin/env python3
"""Authenticate a V4 identity-only source universe without selecting a cohort.

This design-only analyzer consumes exact official Parquet bytes for the full
SWE-bench and SWE-bench Verified test splits.  It projects only ``instance_id``
and ``repo``, proves that Verified is an exact subset of the full test identity
universe, and authenticates the aggregate complement.  It never emits task
identifiers and does not authorize selection, generation, power analysis,
confirmatory interpretation, or reserved-validation access.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/swe_task_state_v4_source_feasibility.json"
DEFAULT_OUTPUT_ROOT = ROOT / ".cache/swe_state_interpreter_v4_design"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "swe-task-state-v4-source-feasibility.json"

SCHEMA_VERSION = 1
CONFIG_ID = "swe-task-state-v4-source-feasibility-v1"
OUTPUT_ID = "swe-task-state-v4-source-feasibility-result-v1"
CONFIG_SHA256 = "165f8c5d179c24855e8edfec30d22c078706b3703969f655af9a062de725afb8"
IDENTITY_COLUMNS = ("instance_id", "repo")
INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PATH_TOKENS = frozenset({"reserved", "validation"})

FULL_REPOSITORY_COUNTS = {
    "astropy/astropy": 95,
    "django/django": 850,
    "matplotlib/matplotlib": 184,
    "mwaskom/seaborn": 22,
    "pallets/flask": 11,
    "psf/requests": 44,
    "pydata/xarray": 110,
    "pylint-dev/pylint": 57,
    "pytest-dev/pytest": 119,
    "scikit-learn/scikit-learn": 229,
    "sphinx-doc/sphinx": 187,
    "sympy/sympy": 386,
}
VERIFIED_REPOSITORY_COUNTS = {
    "astropy/astropy": 22,
    "django/django": 231,
    "matplotlib/matplotlib": 34,
    "mwaskom/seaborn": 2,
    "pallets/flask": 1,
    "psf/requests": 8,
    "pydata/xarray": 22,
    "pylint-dev/pylint": 10,
    "pytest-dev/pytest": 19,
    "scikit-learn/scikit-learn": 32,
    "sphinx-doc/sphinx": 44,
    "sympy/sympy": 75,
}
COMPLEMENT_REPOSITORY_COUNTS = {
    "astropy/astropy": 73,
    "django/django": 619,
    "matplotlib/matplotlib": 150,
    "mwaskom/seaborn": 20,
    "pallets/flask": 10,
    "psf/requests": 36,
    "pydata/xarray": 88,
    "pylint-dev/pylint": 47,
    "pytest-dev/pytest": 100,
    "scikit-learn/scikit-learn": 197,
    "sphinx-doc/sphinx": 143,
    "sympy/sympy": 311,
}

EXPECTED_SOURCE_SPECS = {
    "full_test": {
        "dataset_repo_id": "princeton-nlp/SWE-bench",
        "revision": "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6",
        "split": "test",
        "file_logical_path": "data/test-00000-of-00001.parquet",
        "file_format": "parquet",
        "file_sha256": (
            "db4f70ef735b3162c74801ddcdf8d7bae8d704193788c6d844f898c20b571cbb"
        ),
        "instance_count": 2294,
        "instance_ids_set_sha256": (
            "8348567d58d34d7749213678ac7b3e08cc21c14839262839db45c8c8f4aa4369"
        ),
        "repository_counts": FULL_REPOSITORY_COUNTS,
    },
    "verified": {
        "dataset_repo_id": "princeton-nlp/SWE-bench_Verified",
        "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        "split": "test",
        "file_logical_path": "data/test-00000-of-00001.parquet",
        "file_format": "parquet",
        "file_sha256": (
            "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
        ),
        "file_size_bytes": 2_096_679,
        "instance_count": 500,
        "instance_ids_set_sha256": (
            "33e18be7a9bd9f674790b63ed4d0b3fb17c176994802e3062b7d5a430a4e7d16"
        ),
        "repository_counts": VERIFIED_REPOSITORY_COUNTS,
    },
}

EXPECTED_DERIVATION = {
    "operation": "full_test_identity_set_minus_verified_identity_set",
    "verified_exact_subset_of_full_test_required": True,
    "complement_instance_count": 1794,
    "complement_instance_ids_set_sha256": (
        "953b83337651cfa8e68f812f30e3ba1394a8a08e1f66980680832a1d6bd02861"
    ),
    "complement_repository_counts": COMPLEMENT_REPOSITORY_COUNTS,
}

EXPECTED_DATA_POLICY = {
    "identity_columns_read": list(IDENTITY_COLUMNS),
    "selection_performed": False,
    "selection_authorized": False,
    "generation_performed": False,
    "generation_authorized": False,
    "task_payload_fields_read": False,
    "reserved_membership_accessed": False,
    "reserved_validation_data_accessed": False,
    "reserved_validation_allowed": False,
    "confirmatory_interpretation": False,
    "operational_reliability_claim": False,
    "independent_v4_development_result": False,
    "power_analysis_performed": False,
}


class SourceFeasibilityError(ValueError):
    """Raised whenever the identity-only source contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceFeasibilityError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> Any:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    try:
        return json.loads(
            path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SourceFeasibilityError(
                    f"non-finite JSON number in {label}: {token}"
                )
            ),
        )
    except SourceFeasibilityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceFeasibilityError(f"cannot read {label}: {error}") from error


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise SourceFeasibilityError(
            f"value is not compact canonical JSON: {error}"
        ) from error


def sha256_identity_set(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(sorted(set(values)))).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be one lowercase SHA-256",
    )
    return value


def _validate_repository_counts(
    value: Any,
    *,
    expected: Mapping[str, int],
    expected_total: int,
    label: str,
) -> dict[str, int]:
    observed = _mapping(value, label)
    _require(list(observed) == sorted(observed), f"{label} keys must be lexical")
    result: dict[str, int] = {}
    for repository, count in observed.items():
        _require(
            isinstance(repository, str) and bool(repository),
            f"{label} contains an invalid repository",
        )
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count >= 1,
            f"{label} count is invalid: {repository}",
        )
        result[repository] = count
    _require(result == dict(expected), f"{label} changed")
    _require(sum(result.values()) == expected_total, f"{label} total changed")
    return result


def validate_source_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "source-feasibility config"))
    _require(
        set(config)
        == {
            "schema_version",
            "id",
            "status",
            "identity_projection",
            "hash_contract",
            "sources",
            "derivation",
            "output_contract",
            "data_policy",
        },
        "source-feasibility config fields changed",
    )
    _require(
        config.get("schema_version") == SCHEMA_VERSION
        and config.get("id") == CONFIG_ID
        and config.get("status")
        == "design_only_identity_source_feasibility_not_selection",
        "source-feasibility config identity changed",
    )
    _require(
        dict(_mapping(config.get("identity_projection"), "identity projection"))
        == {
            "columns_in_order": list(IDENTITY_COLUMNS),
            "unique_instance_ids_required": True,
            "verified_repository_must_match_full_test": True,
            "task_payload_columns_must_not_be_read": True,
        },
        "identity projection contract changed",
    )
    _require(
        dict(_mapping(config.get("hash_contract"), "hash contract"))
        == {
            "algorithm": "sha256",
            "file_hash_input": "exact_file_bytes",
            "identity_set_hash_input": (
                "compact_ascii_canonical_json_of_lexically_sorted_unique_instance_ids"
            ),
            "canonical_json": (
                "sort_keys_true_separators_comma_colon_ensure_ascii_true_allow_nan_false"
            ),
        },
        "hash contract changed",
    )

    sources = _mapping(config.get("sources"), "source bindings")
    _require(list(sources) == ["full_test", "verified"], "source order changed")
    for source_id, expected in EXPECTED_SOURCE_SPECS.items():
        source = dict(_mapping(sources.get(source_id), f"{source_id} source"))
        _require(set(source) == set(expected), f"{source_id} source fields changed")
        for key, expected_value in expected.items():
            if key == "repository_counts":
                _validate_repository_counts(
                    source.get(key),
                    expected=expected_value,
                    expected_total=int(expected["instance_count"]),
                    label=f"{source_id} repository counts",
                )
            else:
                _require(
                    source.get(key) == expected_value,
                    f"{source_id} source binding changed: {key}",
                )
        _validate_sha256(source.get("file_sha256"), f"{source_id} file hash")
        _validate_sha256(
            source.get("instance_ids_set_sha256"),
            f"{source_id} identity-set hash",
        )

    derivation = dict(_mapping(config.get("derivation"), "derivation contract"))
    _require(set(derivation) == set(EXPECTED_DERIVATION), "derivation fields changed")
    for key, expected_value in EXPECTED_DERIVATION.items():
        if key == "complement_repository_counts":
            _validate_repository_counts(
                derivation.get(key),
                expected=expected_value,
                expected_total=1794,
                label="complement repository counts",
            )
        else:
            _require(
                derivation.get(key) == expected_value,
                f"derivation contract changed: {key}",
            )
    _validate_sha256(
        derivation.get("complement_instance_ids_set_sha256"),
        "complement identity-set hash",
    )

    output = dict(_mapping(config.get("output_contract"), "output contract"))
    _require(
        output
        == {
            "dedicated_root": ".cache/swe_state_interpreter_v4_design",
            "default_filename": "swe-task-state-v4-source-feasibility.json",
            "new_output_no_clobber": True,
            "aggregate_only_no_instance_ids": True,
            "forbidden_casefolded_path_tokens": ["reserved", "validation"],
        },
        "output contract changed",
    )
    _require(
        dict(_mapping(config.get("data_policy"), "data policy"))
        == EXPECTED_DATA_POLICY,
        "closed-data policy changed",
    )
    return config


IdentityLoader = Callable[
    [Path, str, tuple[str, str]], Sequence[tuple[Any, Any]]
]


def _project_identity_columns_pyarrow(
    path: Path,
    file_format: str,
    columns: tuple[str, str],
) -> list[tuple[Any, Any]]:
    """Project only identity buffers, importing readers only after byte checks."""

    _require(columns == IDENTITY_COLUMNS, "identity projection columns changed")
    _require(file_format == "parquet", f"unsupported source format: {file_format}")
    rows: list[tuple[Any, Any]] = []
    try:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise SourceFeasibilityError(
                "identity projection requires PyArrow; no task source was read"
            ) from error
        table = parquet.read_table(
            str(path),
            columns=["instance_id", "repo"],
            memory_map=True,
            use_threads=False,
        )
        _require(
            tuple(table.column_names) == columns,
            "Parquet identity projection returned extra or reordered columns",
        )
        instance_ids = table.column(columns[0]).to_pylist()
        repositories = table.column(columns[1]).to_pylist()
        rows.extend(zip(instance_ids, repositories, strict=True))
    except SourceFeasibilityError:
        raise
    except Exception as error:
        raise SourceFeasibilityError(
            f"identity-only projection failed for {file_format}: {error}"
        ) from error
    return rows


def load_identity_rows(
    path: Path,
    *,
    file_format: str,
    loader: IdentityLoader | None = None,
) -> list[tuple[Any, Any]]:
    projector = loader or _project_identity_columns_pyarrow
    rows = list(projector(path, file_format, IDENTITY_COLUMNS))
    for index, row in enumerate(rows):
        _require(
            isinstance(row, (tuple, list)) and len(row) == 2,
            f"identity projector returned malformed row {index}",
        )
    return [(row[0], row[1]) for row in rows]


def _identity_map(
    rows: Sequence[tuple[Any, Any]],
    *,
    label: str,
) -> dict[str, str]:
    identities: dict[str, str] = {}
    for index, (instance_id, repository) in enumerate(rows):
        _require(
            isinstance(instance_id, str)
            and INSTANCE_ID_RE.fullmatch(instance_id) is not None,
            f"{label} identity row {index} has an invalid instance ID",
        )
        _require(
            isinstance(repository, str) and bool(repository) and "/" in repository,
            f"{label} identity row {index} has an invalid repository",
        )
        _require(
            instance_id not in identities,
            f"{label} repeats an instance ID",
        )
        identities[instance_id] = repository
    _require(bool(identities), f"{label} identity projection is empty")
    return identities


def _aggregate_identity_map(values: Mapping[str, str]) -> dict[str, Any]:
    repositories = Counter(values.values())
    return {
        "instance_count": len(values),
        "instance_ids_set_sha256": sha256_identity_set(set(values)),
        "repository_counts": dict(sorted(repositories.items())),
    }


def summarize_identity_sources(
    full_rows: Sequence[tuple[Any, Any]],
    verified_rows: Sequence[tuple[Any, Any]],
) -> dict[str, Any]:
    """Derive aggregate-only summaries; no identifier is returned."""

    full = _identity_map(full_rows, label="full test")
    verified = _identity_map(verified_rows, label="Verified")
    missing = set(verified) - set(full)
    _require(not missing, "Verified is not an exact subset of full test")
    _require(
        all(full[instance_id] == repository for instance_id, repository in verified.items()),
        "Verified repository identity differs from full test",
    )
    complement = {
        instance_id: repository
        for instance_id, repository in full.items()
        if instance_id not in verified
    }
    return {
        "full_test": _aggregate_identity_map(full),
        "verified": _aggregate_identity_map(verified),
        "complement": _aggregate_identity_map(complement),
        "verified_exact_subset_of_full_test": True,
        "full_verified_repository_identity_exact": True,
    }


def validate_identity_summary(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    observed = dict(_mapping(summary, "identity summary"))
    _require(
        set(observed)
        == {
            "full_test",
            "verified",
            "complement",
            "verified_exact_subset_of_full_test",
            "full_verified_repository_identity_exact",
        },
        "identity summary fields changed",
    )
    _require(
        observed["verified_exact_subset_of_full_test"] is True
        and observed["full_verified_repository_identity_exact"] is True,
        "identity subset/repository proof failed",
    )
    expected_rows = {
        "full_test": config["sources"]["full_test"],
        "verified": config["sources"]["verified"],
        "complement": {
            "instance_count": config["derivation"]["complement_instance_count"],
            "instance_ids_set_sha256": config["derivation"][
                "complement_instance_ids_set_sha256"
            ],
            "repository_counts": config["derivation"][
                "complement_repository_counts"
            ],
        },
    }
    for source_id, expected in expected_rows.items():
        aggregate = dict(_mapping(observed[source_id], f"{source_id} aggregate"))
        _require(
            set(aggregate)
            == {"instance_count", "instance_ids_set_sha256", "repository_counts"},
            f"{source_id} aggregate fields changed",
        )
        for key in ("instance_count", "instance_ids_set_sha256", "repository_counts"):
            _require(
                aggregate[key] == expected[key],
                f"{source_id} aggregate changed: {key}",
            )
    _require(
        int(observed["full_test"]["instance_count"])
        - int(observed["verified"]["instance_count"])
        == int(observed["complement"]["instance_count"]),
        "identity complement count arithmetic failed",
    )
    return observed


def validate_input_file_hashes(
    config: Mapping[str, Any],
    *,
    full_file_sha256: str,
    verified_file_sha256: str,
    verified_file_size_bytes: int,
) -> None:
    _validate_sha256(full_file_sha256, "observed full-test file hash")
    _validate_sha256(verified_file_sha256, "observed Verified file hash")
    _require(
        isinstance(verified_file_size_bytes, int)
        and not isinstance(verified_file_size_bytes, bool)
        and verified_file_size_bytes >= 0,
        "observed Verified file size is invalid",
    )
    _require(
        full_file_sha256 == config["sources"]["full_test"]["file_sha256"],
        "full-test source bytes differ from the pinned hash",
    )
    _require(
        verified_file_sha256 == config["sources"]["verified"]["file_sha256"],
        "Verified source bytes differ from the pinned hash",
    )
    _require(
        verified_file_size_bytes
        == config["sources"]["verified"]["file_size_bytes"],
        "Verified source byte size differs from the pinned size",
    )


def build_source_feasibility_report(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    config_sha256: str,
    full_file_sha256: str,
    verified_file_sha256: str,
    verified_file_size_bytes: int,
    analyzer_sha256: str,
) -> dict[str, Any]:
    source_bindings: dict[str, Any] = {}
    for source_id, observed_hash, observed_size in (
        ("full_test", full_file_sha256, None),
        ("verified", verified_file_sha256, verified_file_size_bytes),
    ):
        source = config["sources"][source_id]
        source_binding = {
            "dataset_repo_id": source["dataset_repo_id"],
            "revision": source["revision"],
            "split": source["split"],
            "file_logical_path": source["file_logical_path"],
            "file_format": source["file_format"],
            "file_sha256": observed_hash,
            **dict(summary[source_id]),
        }
        if observed_size is not None:
            source_binding["file_size_bytes"] = observed_size
        source_bindings[source_id] = source_binding
    return {
        "schema_version": SCHEMA_VERSION,
        "id": OUTPUT_ID,
        "status": "completed_design_source_feasibility_not_selection",
        "source_feasibility_passed": True,
        "candidate_universe": "full_test_identity_set_minus_verified_identity_set",
        "source_binding": {
            "config_path": str(DEFAULT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "analyzer_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "analyzer_sha256": analyzer_sha256,
            "sources": source_bindings,
        },
        "identity_projection": {
            "columns_in_order": list(IDENTITY_COLUMNS),
            "only_identity_column_values_converted": True,
            "unique_instance_ids": True,
            "verified_exact_subset_of_full_test": True,
            "full_verified_repository_identity_exact": True,
        },
        "complement": dict(summary["complement"]),
        "fresh_cohort_selection_performed": False,
        "fresh_cohort_selection_authorized": False,
        "generation_performed": False,
        "generation_authorized": False,
        "task_payload_fields_read": False,
        "raw_instance_ids_emitted": False,
        "reserved_membership_accessed": False,
        "reserved_validation_data_accessed": False,
        "reserved_validation_accessed": False,
        "reserved_validation_allowed": False,
        "confirmatory_interpretation": False,
        "operational_reliability_claim": False,
        "independent_v4_development_result": False,
        "power_analysis_performed": False,
        "data_policy": dict(config["data_policy"]),
    }


def _contains_forbidden_path_token(path: Path) -> bool:
    return any(
        token in part.casefold()
        for part in path.parts
        for token in FORBIDDEN_PATH_TOKENS
    )


def _regular_nonsymlink_path(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    _require(
        lexical.is_file() and not lexical.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} traverses a symlink")
    _require(
        not _contains_forbidden_path_token(resolved),
        f"{label} uses a forbidden reserved or validation path token",
    )
    return resolved


def validate_cli_paths(
    *,
    config_path: Path,
    full_test_path: Path,
    verified_path: Path,
    output_path: Path,
    canonical_config: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    config = _regular_nonsymlink_path(config_path, "source-feasibility config")
    expected_config = _regular_nonsymlink_path(
        canonical_config, "canonical source-feasibility config"
    )
    _require(config == expected_config, "source-feasibility config path is not canonical")
    full_test = _regular_nonsymlink_path(full_test_path, "full-test source")
    verified = _regular_nonsymlink_path(verified_path, "Verified source")
    _require(full_test != verified, "full-test and Verified sources must differ")
    _require(full_test.suffix == ".parquet", "full-test source must be one Parquet file")
    _require(
        verified.suffix == ".parquet",
        "Verified source must be one Parquet file",
    )

    root_lexical = Path(os.path.abspath(output_root.expanduser()))
    _require(
        root_lexical.is_dir() and not root_lexical.is_symlink(),
        "dedicated V4 design output root is unavailable",
    )
    resolved_root = root_lexical.resolve(strict=True)
    _require(resolved_root == root_lexical, "output root traverses a symlink")
    output_lexical = Path(os.path.abspath(output_path.expanduser()))
    _require(
        output_lexical.parent == resolved_root and output_lexical.suffix == ".json",
        "output must be one JSON file directly under the dedicated V4 design root",
    )
    _require(
        not output_lexical.exists() and not output_lexical.is_symlink(),
        "source-feasibility output already exists; no-clobber is mandatory",
    )
    _require(
        not _contains_forbidden_path_token(output_lexical),
        "output uses a forbidden reserved or validation path token",
    )
    return {
        "config": config,
        "full_test": full_test,
        "verified": verified,
        "output": output_lexical,
    }


def write_json_no_clobber(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--full-test", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = validate_cli_paths(
            config_path=args.config,
            full_test_path=args.full_test,
            verified_path=args.verified,
            output_path=args.output,
        )
        config_sha256 = sha256_file(paths["config"])
        _require(
            config_sha256 == CONFIG_SHA256,
            "source-feasibility config bytes differ from the bound hash",
        )
        config = validate_source_config(
            _read_json(paths["config"], "source-feasibility config")
        )
        full_file_sha256 = sha256_file(paths["full_test"])
        verified_file_sha256 = sha256_file(paths["verified"])
        verified_file_size_bytes = paths["verified"].stat().st_size
        validate_input_file_hashes(
            config,
            full_file_sha256=full_file_sha256,
            verified_file_sha256=verified_file_sha256,
            verified_file_size_bytes=verified_file_size_bytes,
        )
        full_rows = load_identity_rows(
            paths["full_test"],
            file_format=config["sources"]["full_test"]["file_format"],
        )
        verified_rows = load_identity_rows(
            paths["verified"],
            file_format=config["sources"]["verified"]["file_format"],
        )
        summary = validate_identity_summary(
            summarize_identity_sources(full_rows, verified_rows), config
        )
        report = build_source_feasibility_report(
            config,
            summary,
            config_sha256=config_sha256,
            full_file_sha256=full_file_sha256,
            verified_file_sha256=verified_file_sha256,
            verified_file_size_bytes=verified_file_size_bytes,
            analyzer_sha256=sha256_file(Path(__file__).resolve()),
        )
        write_json_no_clobber(paths["output"], report)
    except (SourceFeasibilityError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_feasibility_passed": True,
                "complement_instance_count": report["complement"]["instance_count"],
                "selection_authorized": False,
                "generation_authorized": False,
                "reserved_validation_accessed": False,
                "output": str(paths["output"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
