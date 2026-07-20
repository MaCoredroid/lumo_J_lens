#!/usr/bin/env python3
"""Expectation-blind deterministic candidate-unit catalogs for semantic V3.

The sole construction API, :func:`build_candidate_catalog`, accepts one
authenticated completion packet and no labels, expectations, outcomes,
activations, or semantic annotations.  Units are derived only from the exact
``assistant_text`` Unicode string using a frozen standard-library algorithm.

This module does not create controls, read score keys, launch a model, or write
artifacts.  It wraps the audited V3 candidate-bundle builder/authenticator and
adds frozen segmentation, coverage, cap, and aggregate-manifest provenance.
Full validated-packet hashes remain host-side provenance and are never added to
the model projection, which remains limited to authenticated text and opaque
candidate IDs under the audited V3 runner.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_annotation_runner_v3 as runner  # noqa: E402


SCHEMA_VERSION = 1
INTERFACE_VERSION = 3
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_candidate_catalog_v3.json"
)
CONFIG_KIND = "swe_task_state_v4_epistemic_chain_candidate_catalog_config_v3"
CATALOG_KIND = "swe_task_state_v4_epistemic_chain_candidate_catalog_v3"
MANIFEST_KIND = "swe_task_state_v4_epistemic_chain_candidate_catalog_manifest_v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FROZEN_LINE_BOUNDARY_SEQUENCES = (
    "\r\n",
    "\n",
    "\r",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
)


class CandidateCatalogError(RuntimeError):
    """Raised when a frozen catalog contract or authenticated input is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateCatalogError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be an array",
    )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateCatalogError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    return runner.canonical_json_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return runner.sha256_bytes(value)


def sha256_text(value: str) -> str:
    return runner.sha256_text(value)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_repo_file(binding: Mapping[str, Any], *, label: str) -> Path:
    _require(
        set(binding) == {"path", "sha256"}
        and isinstance(binding["path"], str)
        and SHA256_RE.fullmatch(str(binding["sha256"])) is not None,
        f"{label} binding invalid",
    )
    logical = ROOT / str(binding["path"])
    try:
        resolved = logical.resolve(strict=True)
        resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise CandidateCatalogError(f"{label} must resolve inside repository") from error
    _require(sha256_file(resolved) == binding["sha256"], f"{label} hash changed")
    return resolved


def validate_catalog_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "candidate catalog config"))
    _require(
        set(config)
        == {
            "schema_version",
            "interface_version",
            "kind",
            "status",
            "scope",
            "algorithm",
            "caps",
            "bindings",
        },
        "candidate catalog config fields invalid",
    )
    _require(
        config["schema_version"] == SCHEMA_VERSION
        and config["interface_version"] == INTERFACE_VERSION
        and config["kind"] == CONFIG_KIND
        and config["status"]
        == "prospective_expectation_blind_catalog_contract_before_v3_controls",
        "candidate catalog config identity invalid",
    )
    scope = dict(_mapping(config["scope"], "catalog scope"))
    _require(
        scope
        == {
            "development_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "expectations_labels_outcomes_activations_accepted": False,
            "private_chain_of_thought_ground_truth_claimed": False,
            "affect_emotion_confidence_doubt_or_stress_targeted": False,
        },
        "candidate catalog scope invalid",
    )
    algorithm = dict(_mapping(config["algorithm"], "catalog algorithm"))
    _require(
        set(algorithm)
        == {
            "name",
            "version",
            "coordinate_system",
            "normalization",
            "edge_whitespace",
            "newline_boundary",
            "line_boundary_sequences",
            "sentence_terminal_boundaries",
            "sentence_closing_characters",
            "semicolon_boundary",
            "colon_fence_grouping",
            "fenced_code_delimiters",
            "inline_code_delimiter",
            "url_regex",
            "decimal_regex",
            "initialism_regex",
            "abbreviations",
            "empty_text_action",
            "whitespace_only_action",
            "truncation",
            "fuzzy_or_normalized_fallback",
        },
        "catalog algorithm fields invalid",
    )
    line_boundary_sequences = _sequence(
        algorithm["line_boundary_sequences"],
        "algorithm line_boundary_sequences",
    )
    _require(
        algorithm["name"] == "stdlib_exact_visible_unit_segmentation"
        and algorithm["version"] == 1
        and algorithm["coordinate_system"] == "python_unicode_codepoints"
        and algorithm["normalization"] == "none"
        and algorithm["edge_whitespace"] == "trim_from_unit_edges_only"
        and algorithm["newline_boundary"]
        == "longest_match_split_outside_protected_ranges"
        and tuple(line_boundary_sequences) == FROZEN_LINE_BOUNDARY_SEQUENCES
        and algorithm["sentence_terminal_boundaries"] == [".", "?", "!"]
        and algorithm["semicolon_boundary"]
        == "preserve_within_line_split_only_at_newline"
        and algorithm["colon_fence_grouping"]
        == "group_colon_introducer_with_immediately_following_fenced_block"
        and algorithm["fenced_code_delimiters"] == ["```", "~~~"]
        and algorithm["inline_code_delimiter"] == "`"
        and algorithm["empty_text_action"]
        == "usable_authenticated_zero_unit_bundle"
        and algorithm["whitespace_only_action"]
        == "structured_unusable_without_bundle"
        and algorithm["truncation"] == "forbidden"
        and algorithm["fuzzy_or_normalized_fallback"] == "forbidden",
        "catalog algorithm contract invalid",
    )
    for field in (
        "url_regex",
        "decimal_regex",
        "initialism_regex",
        "sentence_closing_characters",
    ):
        _require(isinstance(algorithm[field], str), f"algorithm {field} invalid")
    for field in ("fenced_code_delimiters", "abbreviations"):
        items = _sequence(algorithm[field], f"algorithm {field}")
        _require(
            bool(items)
            and all(isinstance(item, str) and bool(item) for item in items)
            and len(items) == len(set(items)),
            f"algorithm {field} invalid",
        )
    try:
        re.compile(str(algorithm["url_regex"]))
        re.compile(str(algorithm["decimal_regex"]))
        re.compile(str(algorithm["initialism_regex"]))
    except re.error as error:
        raise CandidateCatalogError(f"catalog regex invalid: {error}") from error

    caps = dict(_mapping(config["caps"], "catalog caps"))
    _require(
        set(caps)
        == {
            "max_units_per_packet",
            "max_executable_schema_bytes",
            "overflow_action",
        }
        and isinstance(caps["max_units_per_packet"], int)
        and not isinstance(caps["max_units_per_packet"], bool)
        and 3 <= caps["max_units_per_packet"] <= 4096
        and isinstance(caps["max_executable_schema_bytes"], int)
        and not isinstance(caps["max_executable_schema_bytes"], bool)
        and 1024 <= caps["max_executable_schema_bytes"] <= 1024 * 1024
        and caps["overflow_action"]
        == "explicit_failure_without_truncation",
        "catalog caps invalid",
    )
    bindings = dict(_mapping(config["bindings"], "catalog bindings"))
    _require(
        set(bindings) == {"builder", "runner_v3", "codebook_v2"},
        "catalog bindings invalid",
    )
    for name in ("builder", "runner_v3", "codebook_v2"):
        _bound_repo_file(_mapping(bindings[name], name), label=name)
    _require(
        Path(str(bindings["builder"]["path"])).name == Path(__file__).name,
        "catalog config binds a different builder",
    )
    return config


def load_catalog_config() -> tuple[dict[str, Any], str]:
    try:
        raw = json.loads(
            CONFIG_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateCatalogError(f"cannot load catalog config: {error}") from error
    return validate_catalog_config(raw), sha256_file(CONFIG_PATH)


def _validated_completion_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return runner.v2.legacy.validate_packet(
            value, annotation_pass="completion_chain"
        )
    except (
        runner.v2.legacy.AnnotationRunnerError,
        runner.v2.legacy.packet_contract.AnnotationPacketError,
    ) as error:
        raise CandidateCatalogError(str(error)) from error


def _line_boundary_at(
    text: str, index: int, sequences: Sequence[str]
) -> str | None:
    """Return the frozen longest boundary beginning at ``index``."""

    for sequence in sequences:
        if text.startswith(sequence, index):
            return sequence
    return None


def _next_line_boundary_start(
    text: str, start: int, sequences: Sequence[str]
) -> int:
    index = start
    while index < len(text):
        if _line_boundary_at(text, index, sequences) is not None:
            return index
        index += 1
    return len(text)


def _line_ranges(
    text: str, sequences: Sequence[str]
) -> list[tuple[int, int, int]]:
    """Return line start, content end, and physical end codepoint offsets."""

    ranges: list[tuple[int, int, int]] = []
    line_start = 0
    index = 0
    while index < len(text):
        boundary = _line_boundary_at(text, index, sequences)
        if boundary is None:
            index += 1
            continue
        physical_end = index + len(boundary)
        ranges.append((line_start, index, physical_end))
        line_start = physical_end
        index = physical_end
    if line_start < len(text):
        ranges.append((line_start, len(text), len(text)))
    return ranges


def _fenced_ranges(
    text: str,
    delimiters: Sequence[str],
    line_boundaries: Sequence[str],
) -> tuple[list[tuple[int, int]], list[int]]:
    ranges: list[tuple[int, int]] = []
    openings: list[int] = []
    active: str | None = None
    active_start: int | None = None
    for line_start, content_end, _physical_end in _line_ranges(
        text, line_boundaries
    ):
        content = text[line_start:content_end]
        stripped = content.lstrip(" \t")
        if active is None:
            delimiter = next(
                (item for item in delimiters if stripped.startswith(item)), None
            )
            if delimiter is None:
                continue
            active = delimiter
            active_start = line_start
            openings.append(line_start)
            after_open = stripped[len(delimiter) :]
            if delimiter in after_open:
                ranges.append((active_start, content_end))
                active = None
                active_start = None
        elif stripped.startswith(active):
            _require(active_start is not None, "fence state invalid")
            ranges.append((active_start, content_end))
            active = None
            active_start = None
    if active is not None:
        _require(active_start is not None, "unclosed fence state invalid")
        ranges.append((active_start, len(text)))
    return ranges, openings


def _mark(mask: list[bool], start: int, end: int) -> None:
    for index in range(max(0, start), min(len(mask), end)):
        mask[index] = True


def _protected_mask(
    text: str, algorithm: Mapping[str, Any]
) -> tuple[list[bool], dict[str, int]]:
    mask = [False] * len(text)
    counts = {
        "fenced_code": 0,
        "colon_fence_join": 0,
        "inline_code": 0,
        "url": 0,
        "decimal": 0,
        "abbreviation_or_initialism": 0,
    }
    line_boundaries = tuple(algorithm["line_boundary_sequences"])
    fence_ranges, openings = _fenced_ranges(
        text,
        list(algorithm["fenced_code_delimiters"]),
        line_boundaries,
    )
    for start, end in fence_ranges:
        _mark(mask, start, end)
        counts["fenced_code"] += 1
    for opening in openings:
        cursor = opening - 1
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        if cursor >= 0 and text[cursor] == ":":
            _mark(mask, cursor + 1, opening)
            counts["colon_fence_join"] += 1

    delimiter = str(algorithm["inline_code_delimiter"])
    index = 0
    while index < len(text):
        if text[index] != delimiter or mask[index]:
            index += 1
            continue
        run_end = index
        while run_end < len(text) and text[run_end] == delimiter:
            run_end += 1
        ticks = text[index:run_end]
        line_end = _next_line_boundary_start(
            text, run_end, line_boundaries
        )
        closing = text.find(ticks, run_end, line_end)
        protected_end = closing + len(ticks) if closing >= 0 else line_end
        _mark(mask, index, protected_end)
        counts["inline_code"] += 1
        index = max(run_end, protected_end)

    url_re = re.compile(str(algorithm["url_regex"]))
    for match in url_re.finditer(text):
        end = match.end()
        while end > match.start() and text[end - 1] in ".,;:!?":
            end -= 1
        if end > match.start():
            _mark(mask, match.start(), end)
            counts["url"] += 1
    for regex_name, count_name in (
        ("decimal_regex", "decimal"),
        ("initialism_regex", "abbreviation_or_initialism"),
    ):
        regex = re.compile(str(algorithm[regex_name]))
        for match in regex.finditer(text):
            _mark(mask, match.start(), match.end())
            counts[count_name] += 1
    for abbreviation in algorithm["abbreviations"]:
        for match in re.finditer(re.escape(str(abbreviation)), text, re.IGNORECASE):
            _mark(mask, match.start(), match.end())
            counts["abbreviation_or_initialism"] += 1
    return mask, counts


def _segment_spans(
    text: str, algorithm: Mapping[str, Any]
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    _require(isinstance(text, str), "assistant text must be a string")
    if text == "":
        return [], {
            "fenced_code": 0,
            "colon_fence_join": 0,
            "inline_code": 0,
            "url": 0,
            "decimal": 0,
            "abbreviation_or_initialism": 0,
        }
    protected, counts = _protected_mask(text, algorithm)
    line_boundaries = tuple(algorithm["line_boundary_sequences"])
    boundaries: set[int] = {len(text)}
    index = 0
    while index < len(text):
        line_boundary = _line_boundary_at(text, index, line_boundaries)
        if line_boundary is not None:
            end = index + len(line_boundary)
            if not any(protected[index:end]):
                boundaries.add(end)
            index = end
            continue
        if (
            text[index] in algorithm["sentence_terminal_boundaries"]
            and not protected[index]
        ):
            end = index + 1
            closers = str(algorithm["sentence_closing_characters"])
            while end < len(text) and text[end] in closers and not protected[end]:
                end += 1
            if end == len(text) or text[end].isspace():
                boundaries.add(end)
        # Semicolons are deliberately preserved within a line.  C02-style
        # hypotheses depend on both clauses around the semicolon.
        index += 1

    spans: list[tuple[int, int]] = []
    cursor = 0
    for boundary in sorted(boundaries):
        start = cursor
        end = boundary
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
        cursor = boundary
    return spans, counts


def verify_exact_nonwhitespace_coverage(
    *, assistant_text: str, spans: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    """Return explicit coverage/gap status without normalization or repair."""

    _require(isinstance(assistant_text, str), "assistant text must be a string")
    validated: list[tuple[int, int]] = []
    previous_end = 0
    for index, raw in enumerate(_sequence(spans, "candidate spans")):
        _require(
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes))
            and len(raw) == 2,
            f"candidate span {index} invalid",
        )
        start, end = raw
        _require(
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= len(assistant_text),
            f"candidate span {index} bounds invalid",
        )
        if start < previous_end:
            return {
                "coverage_status": "invalid_overlap",
                "non_whitespace_char_count": sum(
                    not char.isspace() for char in assistant_text
                ),
                "covered_non_whitespace_char_count": None,
                "gap_count": None,
                "gaps": None,
            }
        previous_end = end
        validated.append((start, end))

    covered = [False] * len(assistant_text)
    for start, end in validated:
        for index in range(start, end):
            if not assistant_text[index].isspace():
                covered[index] = True
    missing = [
        index
        for index, char in enumerate(assistant_text)
        if not char.isspace() and not covered[index]
    ]
    gaps: list[dict[str, Any]] = []
    if missing:
        gap_start = missing[0]
        previous = missing[0]
        for current in missing[1:]:
            if current != previous + 1:
                text = assistant_text[gap_start : previous + 1]
                gaps.append(
                    {
                        "char_start": gap_start,
                        "char_end": previous + 1,
                        "text": text,
                        "text_sha256": sha256_text(text),
                    }
                )
                gap_start = current
            previous = current
        text = assistant_text[gap_start : previous + 1]
        gaps.append(
            {
                "char_start": gap_start,
                "char_end": previous + 1,
                "text": text,
                "text_sha256": sha256_text(text),
            }
        )
    non_whitespace_count = sum(not char.isspace() for char in assistant_text)
    return {
        "coverage_status": "complete" if not gaps else "coverage_gap",
        "non_whitespace_char_count": non_whitespace_count,
        "covered_non_whitespace_char_count": non_whitespace_count - len(missing),
        "gap_count": len(gaps),
        "gaps": gaps,
    }


def _load_bound_codebook(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["bindings"]["codebook_v2"], "codebook binding")
    path = _bound_repo_file(binding, label="codebook_v2")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateCatalogError(f"cannot load bound codebook: {error}") from error
    try:
        return runner.v2.validate_v2_codebook(raw)
    except runner.v2.QuoteFirstRunnerError as error:
        raise CandidateCatalogError(str(error)) from error


def catalog_result_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def build_candidate_catalog(*, packet: Mapping[str, Any]) -> dict[str, Any]:
    """Build one exact catalog solely from authenticated ``assistant_text``."""

    _require(
        tuple(inspect.signature(build_candidate_catalog).parameters) == ("packet",),
        "candidate construction API changed",
    )
    config, config_sha256 = load_catalog_config()
    validated = _validated_completion_packet(packet)
    text_record = _mapping(
        validated["materialized_assistant_text"], "materialized assistant text"
    )
    assistant_text = str(text_record["text"])
    algorithm = _mapping(config["algorithm"], "catalog algorithm")
    spans, protection_counts = _segment_spans(assistant_text, algorithm)
    coverage = verify_exact_nonwhitespace_coverage(
        assistant_text=assistant_text, spans=spans
    )
    builder_sha256 = str(config["bindings"]["builder"]["sha256"])
    common = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": CATALOG_KIND,
        "packet_id_sha256": validated["packet_id_sha256"],
        "source_id_sha256": validated["source_id_sha256"],
        "materialized_assistant_text_sha256": text_record["sha256"],
        "catalog_config_sha256": config_sha256,
        "catalog_builder_sha256": builder_sha256,
        "segmentation_algorithm": {
            "name": algorithm["name"],
            "version": algorithm["version"],
            "normalization_applied": False,
            "fuzzy_fallback_applied": False,
            "truncation_applied": False,
            "protection_counts": protection_counts,
            "spans_sha256": sha256_bytes(canonical_json_bytes(spans)),
        },
        "coverage": coverage,
    }
    if assistant_text and coverage["non_whitespace_char_count"] == 0:
        return {
            **common,
            "catalog_status": "whitespace_only_unusable",
            "catalog_usable": False,
            "catalog_failure": {
                "code": "authenticated_text_contains_only_unicode_whitespace",
                "fail_closed": True,
            },
            "unit_count": 0,
            "executable_schema_bytes": None,
            "candidate_unit_bundle": None,
            "candidate_unit_bundle_sha256": None,
            "ordered_unit_sha256s": None,
            "ordered_unit_sha256s_sha256": None,
        }
    if coverage["coverage_status"] != "complete":
        return {
            **common,
            "catalog_status": "coverage_gap",
            "catalog_usable": False,
            "unit_count": len(spans),
            "executable_schema_bytes": None,
            "candidate_unit_bundle": None,
            "candidate_unit_bundle_sha256": None,
            "ordered_unit_sha256s": None,
            "ordered_unit_sha256s_sha256": None,
        }

    try:
        bundle = runner.build_candidate_unit_bundle(packet=validated, spans=spans)
        bundle_sha256 = runner.candidate_unit_bundle_sha256(bundle)
        authenticated = runner.authenticate_candidate_unit_bundle(
            value=bundle,
            packet=validated,
            expected_bundle_sha256=bundle_sha256,
        )
    except runner.BoundedIdRunnerError as error:
        raise CandidateCatalogError(str(error)) from error
    unit_hashes = [
        sha256_bytes(canonical_json_bytes(unit)) for unit in bundle["units"]
    ]
    codebook = _load_bound_codebook(config)
    schemas = [runner.completion_decision_response_schema(authenticated)]
    if len(authenticated.units) >= 3:
        schemas.append(
            runner.completion_chain_detail_response_schema(codebook, authenticated)
        )
    schema_bytes = max(len(canonical_json_bytes(schema)) for schema in schemas)
    caps = _mapping(config["caps"], "catalog caps")
    if len(spans) > int(caps["max_units_per_packet"]):
        status = "unit_count_overflow"
    elif schema_bytes > int(caps["max_executable_schema_bytes"]):
        status = "schema_bytes_overflow"
    else:
        status = "available"
    return {
        **common,
        "catalog_status": status,
        "catalog_usable": status == "available",
        "unit_count": len(spans),
        "executable_schema_bytes": schema_bytes,
        "candidate_unit_bundle": bundle,
        "candidate_unit_bundle_sha256": bundle_sha256,
        "ordered_unit_sha256s": unit_hashes,
        "ordered_unit_sha256s_sha256": sha256_bytes(
            canonical_json_bytes(unit_hashes)
        ),
    }


def build_catalog_manifest(
    *,
    packets: Sequence[Mapping[str, Any]],
    catalogs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-authenticate and bind catalogs to exact expectation-blind packets."""

    packet_values = list(_sequence(packets, "authenticated packets"))
    catalog_values = [
        dict(_mapping(item, "catalog result"))
        for item in _sequence(catalogs, "catalog results")
    ]
    _require(bool(packet_values), "catalog manifest requires at least one packet")
    _require(
        len(packet_values) == len(catalog_values),
        "catalog manifest packet and catalog counts differ",
    )

    packets_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_packet in enumerate(packet_values):
        validated = _validated_completion_packet(
            _mapping(raw_packet, f"authenticated packet {index}")
        )
        packet_id = validated["packet_id_sha256"]
        _require(
            packet_id not in packets_by_id,
            "catalog manifest contains a duplicate authenticated packet id",
        )
        packets_by_id[packet_id] = validated

    catalogs_by_id: dict[str, dict[str, Any]] = {}
    for index, catalog in enumerate(catalog_values):
        packet_id = catalog.get("packet_id_sha256")
        _require(
            isinstance(packet_id, str)
            and SHA256_RE.fullmatch(packet_id) is not None,
            f"catalog result {index} packet id invalid",
        )
        _require(
            packet_id not in catalogs_by_id,
            "catalog manifest contains a duplicate catalog packet id",
        )
        catalogs_by_id[packet_id] = catalog
    _require(
        set(packets_by_id) == set(catalogs_by_id),
        "catalog manifest packet and catalog identities differ",
    )

    authenticated: list[tuple[dict[str, Any], str]] = []
    for packet_id in sorted(packets_by_id):
        validated_packet = packets_by_id[packet_id]
        rebuilt = build_candidate_catalog(packet=validated_packet)
        supplied = catalogs_by_id[packet_id]
        try:
            exact_match = canonical_json_bytes(supplied) == canonical_json_bytes(
                rebuilt
            )
        except (TypeError, ValueError) as error:
            raise CandidateCatalogError(
                "catalog result is not canonically serializable"
            ) from error
        _require(
            exact_match,
            "catalog result differs from current authenticated recomputation",
        )
        coverage = _mapping(rebuilt.get("coverage"), "catalog coverage")
        _require(
            coverage.get("coverage_status") == "complete"
            and coverage.get("gap_count") == 0
            and rebuilt.get("catalog_status") == "available"
            and rebuilt.get("catalog_usable") is True,
            "catalog manifest requires an available exact-coverage catalog",
        )
        unit_count = rebuilt.get("unit_count")
        schema_bytes = rebuilt.get("executable_schema_bytes")
        _require(
            isinstance(unit_count, int)
            and not isinstance(unit_count, bool)
            and unit_count >= 0
            and isinstance(schema_bytes, int)
            and not isinstance(schema_bytes, bool)
            and schema_bytes >= 0,
            "catalog manifest count fields invalid",
        )
        for field in (
            "candidate_unit_bundle_sha256",
            "ordered_unit_sha256s_sha256",
        ):
            _require(
                isinstance(rebuilt.get(field), str)
                and SHA256_RE.fullmatch(rebuilt[field]) is not None,
                f"catalog manifest {field} invalid",
            )
        _require(
            isinstance(rebuilt.get("candidate_unit_bundle"), Mapping)
            and isinstance(rebuilt.get("ordered_unit_sha256s"), Sequence)
            and not isinstance(rebuilt.get("ordered_unit_sha256s"), (str, bytes))
            and len(rebuilt["candidate_unit_bundle"].get("units", []))
            == unit_count
            and len(rebuilt["ordered_unit_sha256s"]) == unit_count,
            "catalog manifest unit collection invalid",
        )
        authenticated.append(
            (
                rebuilt,
                sha256_bytes(canonical_json_bytes(validated_packet)),
            )
        )

    ordered = [item for item, _packet_hash in authenticated]
    packet_hashes = [packet_hash for _item, packet_hash in authenticated]
    config_hashes = {item["catalog_config_sha256"] for item in ordered}
    builder_hashes = {item["catalog_builder_sha256"] for item in ordered}
    _require(
        len(config_hashes) == 1 and len(builder_hashes) == 1,
        "catalog manifest mixes config or builder identities",
    )
    entries = [
        {
            "packet_id_sha256": item["packet_id_sha256"],
            "authenticated_packet_sha256": packet_hashes[index],
            "source_id_sha256": item["source_id_sha256"],
            "materialized_assistant_text_sha256": item[
                "materialized_assistant_text_sha256"
            ],
            "catalog_status": item["catalog_status"],
            "catalog_result_sha256": catalog_result_sha256(item),
            "candidate_unit_bundle_sha256": item[
                "candidate_unit_bundle_sha256"
            ],
            "ordered_unit_sha256s_sha256": item[
                "ordered_unit_sha256s_sha256"
            ],
            "unit_count": item["unit_count"],
        }
        for index, item in enumerate(ordered)
    ]
    statuses: dict[str, int] = {}
    for entry in entries:
        status = str(entry["catalog_status"])
        statuses[status] = statuses.get(status, 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "kind": MANIFEST_KIND,
        "catalog_config_sha256": next(iter(config_hashes)),
        "catalog_builder_sha256": next(iter(builder_hashes)),
        "catalog_count": len(entries),
        "total_unit_count": sum(int(item["unit_count"]) for item in entries),
        "status_counts": statuses,
        "all_catalogs_usable": statuses == {"available": len(entries)},
        "ordered_entries": entries,
        "ordered_authenticated_packet_sha256s_sha256": sha256_bytes(
            canonical_json_bytes(packet_hashes)
        ),
        "ordered_catalog_result_sha256s_sha256": sha256_bytes(
            canonical_json_bytes(
                [entry["catalog_result_sha256"] for entry in entries]
            )
        ),
    }
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
    }


def authenticate_catalog_manifest(
    *,
    value: Mapping[str, Any],
    expected_manifest_sha256: str,
    packets: Sequence[Mapping[str, Any]],
    catalogs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate a supplied manifest against an external frozen hash.

    The caller must supply the expected hash independently; a hash stored next
    to the manifest is never treated as its own trust root.  Exact packets and
    catalogs are revalidated and rebuilt through :func:`build_catalog_manifest`
    before the authenticated manifest is returned.
    """

    supplied = dict(_mapping(value, "catalog manifest envelope"))
    _require(
        set(supplied) == {"manifest", "manifest_sha256"},
        "catalog manifest envelope fields invalid",
    )
    _require(
        isinstance(expected_manifest_sha256, str)
        and SHA256_RE.fullmatch(expected_manifest_sha256) is not None,
        "expected catalog manifest hash invalid",
    )
    supplied_manifest = dict(_mapping(supplied["manifest"], "catalog manifest"))
    supplied_hash = supplied["manifest_sha256"]
    _require(
        isinstance(supplied_hash, str)
        and SHA256_RE.fullmatch(supplied_hash) is not None,
        "self-reported catalog manifest hash invalid",
    )
    try:
        actual_supplied_hash = sha256_bytes(
            canonical_json_bytes(supplied_manifest)
        )
    except (TypeError, ValueError) as error:
        raise CandidateCatalogError(
            "catalog manifest is not canonically serializable"
        ) from error
    _require(
        supplied_hash == actual_supplied_hash,
        "self-reported catalog manifest hash does not match its payload",
    )
    _require(
        actual_supplied_hash == expected_manifest_sha256,
        "catalog manifest does not match the external expected hash",
    )

    rebuilt = build_catalog_manifest(packets=packets, catalogs=catalogs)
    _require(
        rebuilt["manifest_sha256"] == expected_manifest_sha256,
        "authenticated recomputation differs from the external expected hash",
    )
    _require(
        canonical_json_bytes(supplied_manifest)
        == canonical_json_bytes(rebuilt["manifest"]),
        "catalog manifest differs from authenticated recomputation",
    )
    return copy.deepcopy(rebuilt["manifest"])
