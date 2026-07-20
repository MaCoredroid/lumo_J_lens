#!/usr/bin/env python3
"""Build, lock, and score sealed quote-first V2 semantic controls.

This module never runs a model and never reads target annotations.  It creates
one model-input JSONL/manifest pair and a physically separate expectation
JSONL/manifest pair.  The model-input manifest has no reference to the
expectation artifacts.  Model outputs must be hash-locked, with exact packet
coverage, before the scorer is allowed to read the expectations.

The checked-in suite config intentionally starts with an unfrozen V2 codebook
binding.  ``freeze-helper`` prints the exact binding that an integration owner
must review and patch prospectively.  ``build`` fails closed until that binding
is explicit and authenticates byte size, file SHA-256, and canonical SHA-256.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import swe_task_state_v4_epistemic_chain_annotation_runner_v2 as quote_runner  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_controls_v2.json"
LEGACY_CODEBOOK_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_codebook.json"
)
PROPOSAL_CODEBOOK_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_codebook_v2_proposal.json"
)
SCHEMA_VERSION = 1
SUITE_KIND = "swe_task_state_v4_epistemic_chain_sealed_controls_v2"
SUITE_ID = "visible-semantic-chain-quote-first-sealed-controls-v2"
FROZEN_SUITE_STATUS = "prospective_content_and_codebook_binding_frozen_for_build"
MODEL_INPUT_KIND = "swe_task_state_v4_epistemic_chain_control_model_input_v2"
MODEL_INPUT_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_model_input_manifest_v2"
)
EXPECTATION_KIND = "swe_task_state_v4_epistemic_chain_control_expectation_v2"
EXPECTATION_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_expectation_manifest_v2"
)
OUTPUT_RECORD_KIND = "swe_task_state_v4_epistemic_chain_control_output_v2"
LOCKED_OUTPUT_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_control_locked_output_manifest_v2"
)
REPORT_KIND = "swe_task_state_v4_epistemic_chain_control_report_v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_ID_RE = re.compile(r"^(?:C(?:0[1-9]|[12][0-9]|3[0-2])|R0[1-4]|V0[1-8]|A(?:0[1-9]|1[0-2]))$")
PASS_NAMES = ("completion", "novelty", "adjudication")
RESULT_PROJECTION_FIELDS = (
    "raw_semantic_decision",
    "raw_semantic_proposal",
    "semantic_validation_status",
    "semantic_validation_error",
    "materialization_status",
    "interface_unknown_reason",
    "quote_resolution",
    "annotation_record",
)


class ControlSuiteError(RuntimeError):
    """Raised when suite content, separation, locking, or scoring fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlSuiteError(message)


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
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ControlSuiteError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlSuiteError(f"cannot load strict JSON {path}: {error}") from error


def _guard_path(path: Path, *, must_exist: bool, label: str) -> Path:
    logical = Path(path).absolute()
    lowered_parts = {part.casefold() for part in logical.parts}
    _require(
        "validation" not in lowered_parts and ".git" not in lowered_parts,
        f"{label} may not use a reserved path",
    )
    _require(not logical.is_symlink(), f"{label} may not be a logical symlink")
    try:
        repository_root = ROOT.resolve(strict=True)
        if must_exist:
            resolved = logical.resolve(strict=True)
        else:
            parent = logical.parent.resolve(strict=True)
            resolved = parent / logical.name
        resolved.relative_to(repository_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlSuiteError(f"{label} must remain inside repository") from error
    if must_exist:
        _require(resolved.is_file(), f"{label} must be a regular file")
    return resolved


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path = _guard_path(path, must_exist=False, label="JSONL output")
    _require(not path.exists(), f"refusing to overwrite {path}")
    temporary = Path(str(path) + ".tmp")
    _require(not temporary.exists(), f"temporary output already exists: {temporary}")
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json_text(row) + "\n")
                count += 1
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return count, sha256_file(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> str:
    path = _guard_path(path, must_exist=False, label="JSON output")
    _require(not path.exists(), f"refusing to overwrite {path}")
    temporary = Path(str(path) + ".tmp")
    _require(not temporary.exists(), f"temporary output already exists: {temporary}")
    try:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return sha256_file(path)


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    path = _guard_path(path, must_exist=True, label=label)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"blank {label} line {line_number}")
            try:
                value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, ControlSuiteError) as error:
                raise ControlSuiteError(
                    f"invalid {label} JSON line {line_number}: {error}"
                ) from error
            rows.append(dict(_mapping(value, f"{label} line {line_number}")))
    return rows


def codebook_freeze_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact current binding without mutating config or files."""

    binding = _mapping(config.get("codebook_binding"), "codebook binding")
    path = Path(str(binding.get("path")))
    if not path.is_absolute():
        path = ROOT / path
    resolved = _guard_path(path, must_exist=True, label="V2 codebook")
    value = load_json_strict(resolved)
    quote_runner.validate_v2_codebook(value)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "canonical_sha256": sha256_bytes(canonical_json_bytes(value)),
    }


def with_current_codebook_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an in-memory frozen copy for tests/review; never write it."""

    result = copy.deepcopy(dict(config))
    result["codebook_binding"] = codebook_freeze_binding(config)
    result["status"] = "prospective_content_and_codebook_binding_frozen_for_build"
    return result


def _authenticate_codebook(
    config: Mapping[str, Any], *, require_frozen: bool
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    binding = dict(_mapping(config.get("codebook_binding"), "codebook binding"))
    expected_fields = {"path", "size_bytes", "sha256", "canonical_sha256"}
    _require(set(binding) == expected_fields, "codebook binding fields changed")
    frozen = (
        isinstance(binding.get("size_bytes"), int)
        and not isinstance(binding.get("size_bytes"), bool)
        and int(binding["size_bytes"]) > 0
        and SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None
        and SHA256_RE.fullmatch(str(binding.get("canonical_sha256"))) is not None
    )
    if require_frozen:
        _require(frozen, "codebook binding is not explicitly frozen")
    path = Path(str(binding.get("path")))
    if not path.is_absolute():
        path = ROOT / path
    resolved = _guard_path(path, must_exist=True, label="V2 codebook")
    value = dict(_mapping(load_json_strict(resolved), "V2 codebook"))
    codebook = quote_runner.validate_v2_codebook(value)
    actual = {
        "path": str(resolved.relative_to(ROOT)),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "canonical_sha256": sha256_bytes(canonical_json_bytes(value)),
    }
    if frozen:
        _require(binding == actual, "frozen V2 codebook binding drifted")
    return codebook, resolved, actual


def _no_chain_proposal() -> dict[str, Any]:
    return quote_runner.deterministic_no_chain_proposal()


def _unknown_proposal() -> dict[str, Any]:
    result = _no_chain_proposal()
    result["decision"] = "unknown"
    result["unknown_reason"] = "completion_semantics_ambiguous"
    return result


def _completion_proposal(control: Mapping[str, Any]) -> dict[str, Any]:
    expected = _mapping(control.get("expected"), "completion expectation")
    if "proposal" in expected:
        proposal = dict(_mapping(expected["proposal"], "completion proposal"))
    elif expected.get("category") == "no_chain":
        proposal = _no_chain_proposal()
    elif expected.get("category") == "unknown":
        proposal = _unknown_proposal()
    else:
        raise ControlSuiteError("completion expectation lacks proposal or category")
    return quote_runner.validate_model_proposal(proposal)


def _result_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(value, "completion result")
    _require(
        all(field in source for field in RESULT_PROJECTION_FIELDS),
        "completion result lacks quote-runner fields",
    )
    return {field: copy.deepcopy(source[field]) for field in RESULT_PROJECTION_FIELDS}


def _materialized_expectation(
    *, proposal: Mapping[str, Any], assistant_text: str, empty_control: bool = False
) -> dict[str, Any]:
    result = quote_runner.materialize_completion_proposal(
        proposal=proposal,
        assistant_text=assistant_text,
        decision_source=(
            "deterministic_empty_visible_prose" if empty_control else "model"
        ),
    )
    return _result_projection(result)


def _resolve_candidate_spec(
    spec: Mapping[str, Any], *, completion_control: Mapping[str, Any]
) -> dict[str, Any]:
    source = spec.get("source")
    if source == "gold":
        proposal = _completion_proposal(completion_control)
    elif source == "no_chain":
        proposal = _no_chain_proposal()
    elif source == "unknown":
        proposal = _unknown_proposal()
    elif source == "explicit":
        proposal = dict(_mapping(spec.get("proposal"), "explicit candidate proposal"))
    else:
        raise ControlSuiteError("candidate source invalid")
    overrides = spec.get("overrides", {})
    _require(isinstance(overrides, Mapping), "candidate overrides must be an object")
    proposal.update(dict(overrides))
    return quote_runner.validate_model_proposal(proposal)


def _teaching_texts(codebook: Mapping[str, Any]) -> tuple[set[str], set[tuple[str, str]]]:
    completion: set[str] = set()
    novelty: set[tuple[str, str]] = set()
    for key in ("positive_teaching_examples", "negative_teaching_examples"):
        for item in _sequence(codebook.get(key, []), key):
            row = _mapping(item, f"{key} item")
            if isinstance(row.get("assistant_text"), str):
                completion.add(str(row["assistant_text"]))
    for item in _sequence(
        codebook.get("novelty_teaching_examples", []), "novelty teaching examples"
    ):
        row = _mapping(item, "novelty teaching item")
        prefix = row.get("visible_prefix")
        hypothesis = row.get("locked_hypothesis_quote")
        if isinstance(prefix, str) and isinstance(hypothesis, str):
            novelty.add((prefix, hypothesis))
    return completion, novelty


def _historical_teaching_texts() -> tuple[set[str], set[tuple[str, str]]]:
    """Load non-V2 prompt examples that controls must also remain distinct from."""

    legacy_path = _guard_path(
        LEGACY_CODEBOOK_PATH, must_exist=True, label="legacy epistemic codebook"
    )
    legacy = _mapping(load_json_strict(legacy_path), "legacy epistemic codebook")
    completion: set[str] = set()
    novelty: set[tuple[str, str]] = set()
    for key in ("positive_examples", "negative_examples"):
        for item in _sequence(legacy.get(key, []), f"legacy {key}"):
            row = _mapping(item, f"legacy {key} item")
            if isinstance(row.get("assistant_text"), str):
                completion.add(str(row["assistant_text"]))
    for item in _sequence(legacy.get("novelty_examples", []), "legacy novelty examples"):
        row = _mapping(item, "legacy novelty item")
        prefix = row.get("visible_prefix")
        hypothesis = row.get("locked_hypothesis_text")
        if isinstance(prefix, str) and isinstance(hypothesis, str):
            novelty.add((prefix, hypothesis))

    proposal_path = _guard_path(
        PROPOSAL_CODEBOOK_PATH,
        must_exist=True,
        label="V2 codebook proposal",
    )
    proposal = _mapping(load_json_strict(proposal_path), "V2 codebook proposal")
    for item in _sequence(
        proposal.get("prospective_semantic_controls", []),
        "prospective semantic controls",
    ):
        row = _mapping(item, "prospective semantic control")
        if isinstance(row.get("assistant_text"), str):
            completion.add(str(row["assistant_text"]))
    return completion, novelty


def validate_config(
    value: Any, *, require_frozen_codebook: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = dict(_mapping(value, "control suite config"))
    _require(
        set(config)
        == {
            "schema_version",
            "kind",
            "id",
            "status",
            "scope",
            "codebook_binding",
            "sealing_contract",
            "completion_controls",
            "resolver_controls",
            "novelty_controls",
            "adjudication_controls",
        },
        "control suite config fields changed",
    )
    _require(
        config.get("schema_version") == SCHEMA_VERSION
        and config.get("kind") == SUITE_KIND
        and config.get("id") == SUITE_ID
        and config.get("status") == FROZEN_SUITE_STATUS,
        "control suite identity invalid",
    )
    scope = _mapping(config.get("scope"), "control suite scope")
    _require(
        scope.get("development_controls_only") is True
        and scope.get("reserved_validation_closed") is True
        and scope.get("reserved_validation_accessed") is False
        and scope.get("target_annotations_present") is False
        and scope.get("private_chain_of_thought_ground_truth_claimed") is False
        and scope.get("affect_emotion_confidence_doubt_or_stress_targeted") is False,
        "control suite scope invalid",
    )
    sealing = _mapping(config.get("sealing_contract"), "sealing contract")
    _require(
        sealing.get("model_inputs_and_expectations_physically_separate") is True
        and sealing.get("model_input_manifest_may_not_reference_expectations") is True
        and sealing.get("expectations_joined_only_after_output_lock") is True
        and sealing.get("packet_ids_are_blinded_hashes") is True
        and sealing.get("teaching_examples_forbidden_as_controls") is True
        and sealing.get("any_prompt_schema_codebook_model_or_control_change_resets_gate")
        is True,
        "sealing contract weakened",
    )
    required_counts = dict(_mapping(sealing.get("required_counts"), "required counts"))
    expected_counts = {
        "completion": 32,
        "resolver": 4,
        "novelty": 8,
        "adjudication": 12,
    }
    _require(required_counts == expected_counts, "required control counts changed")
    collections = {
        "completion": list(_sequence(config.get("completion_controls"), "completion controls")),
        "resolver": list(_sequence(config.get("resolver_controls"), "resolver controls")),
        "novelty": list(_sequence(config.get("novelty_controls"), "novelty controls")),
        "adjudication": list(_sequence(config.get("adjudication_controls"), "adjudication controls")),
    }
    _require(
        {name: len(rows) for name, rows in collections.items()} == expected_counts,
        "control counts differ from sealed contract",
    )
    all_ids: list[str] = []
    for rows in collections.values():
        for item in rows:
            row = _mapping(item, "control row")
            control_id = row.get("id")
            _require(
                isinstance(control_id, str)
                and CONTROL_ID_RE.fullmatch(control_id) is not None,
                "control id invalid",
            )
            all_ids.append(control_id)
    _require(len(set(all_ids)) == 56, "control ids are duplicated")

    codebook, _codebook_path, actual_binding = _authenticate_codebook(
        config, require_frozen=require_frozen_codebook
    )
    teaching_completion, teaching_novelty = _teaching_texts(codebook)
    historical_completion, historical_novelty = _historical_teaching_texts()
    teaching_completion.update(historical_completion)
    teaching_novelty.update(historical_novelty)
    completion_by_id: dict[str, dict[str, Any]] = {}
    ontology_seen = {
        "evidence_kind": set(),
        "belief_edge": set(),
        "hypothesis_domain": set(),
        "action_intent": set(),
    }
    for raw in collections["completion"]:
        row = dict(_mapping(raw, "completion control"))
        control_id = str(row["id"])
        text = row.get("assistant_text")
        _require(isinstance(text, str), f"{control_id} assistant text invalid")
        # The empty string is a required deterministic interface sentinel and is
        # not a semantic teaching example.  Every nonempty model-generated case
        # must be unseen relative to active, legacy, and proposed demonstrations.
        _require(
            text == "" or text not in teaching_completion,
            f"{control_id} duplicates an existing or proposed teaching example",
        )
        proposal = _completion_proposal(row)
        expected_result = _materialized_expectation(
            proposal=proposal,
            assistant_text=text,
            empty_control=(text == ""),
        )
        expected = _mapping(row["expected"], f"{control_id} expectation")
        if expected.get("expected_materialization_status") is not None:
            _require(
                expected_result["materialization_status"]
                == expected["expected_materialization_status"],
                f"{control_id} materialization expectation is wrong",
            )
        if "expected_interface_unknown_reason" in expected:
            _require(
                expected_result["interface_unknown_reason"]
                == expected["expected_interface_unknown_reason"],
                f"{control_id} interface reason expectation is wrong",
            )
        if "expected_valid_ordered_tuple_count" in expected:
            resolution = _mapping(
                expected_result.get("quote_resolution"), f"{control_id} quote resolution"
            )
            _require(
                resolution.get("valid_ordered_tuple_count")
                == expected["expected_valid_ordered_tuple_count"],
                f"{control_id} ordered tuple expectation is wrong",
            )
        if proposal["decision"] == "chain":
            for name in ontology_seen:
                ontology_seen[name].add(proposal[name])
        completion_by_id[control_id] = row
    ontology = _mapping(codebook.get("ontology"), "V2 ontology")
    for name, observed in ontology_seen.items():
        _require(
            observed == set(ontology[name]),
            f"completion controls do not cover every {name}",
        )
    _require(
        _completion_proposal(completion_by_id["C01"])["decision"] == "chain"
        and _completion_proposal(completion_by_id["C02"])["decision"] == "chain"
        and _completion_proposal(completion_by_id["C01"])[
            "relation_marker_present"
        ]
        is True
        and _completion_proposal(completion_by_id["C02"])[
            "relation_marker_present"
        ]
        is False,
        "marker-present/absent semantic pair invalid",
    )

    for raw in collections["resolver"]:
        row = _mapping(raw, "resolver control")
        if "assistant_text_ref" in row:
            ref = str(row["assistant_text_ref"])
            _require(ref in completion_by_id, "resolver assistant text ref invalid")
            assistant_text = str(completion_by_id[ref]["assistant_text"])
        else:
            assistant_text = row.get("assistant_text")
            _require(isinstance(assistant_text, str), "resolver assistant text invalid")
        if "proposal_ref" in row:
            ref = str(row["proposal_ref"])
            _require(ref in completion_by_id, "resolver proposal ref invalid")
            proposal = _completion_proposal(completion_by_id[ref])
        else:
            proposal = quote_runner.validate_model_proposal(row.get("proposal"))
        result = quote_runner.materialize_completion_proposal(
            proposal=proposal, assistant_text=str(assistant_text)
        )
        expected = _mapping(row.get("expected"), "resolver expectation")
        resolution = _mapping(result.get("quote_resolution"), "resolver quote resolution")
        _require(
            result.get("materialization_status") == expected.get("materialization_status")
            and result.get("interface_unknown_reason")
            == expected.get("interface_unknown_reason")
            and resolution.get("valid_ordered_tuple_count")
            == expected.get("valid_ordered_tuple_count"),
            f"{row['id']} resolver expectation invalid",
        )

    novelty_statuses: list[str] = []
    for raw in collections["novelty"]:
        row = _mapping(raw, "novelty control")
        prefix = row.get("visible_prefix")
        hypothesis = row.get("locked_hypothesis")
        status = row.get("expected_status")
        _require(
            isinstance(prefix, str)
            and isinstance(hypothesis, str)
            and bool(hypothesis)
            and status in {"novel", "prefix_exposed", "ambiguous"},
            "novelty control invalid",
        )
        _require(
            (prefix, hypothesis) not in teaching_novelty,
            f"{row['id']} duplicates an existing novelty teaching example",
        )
        novelty_statuses.append(str(status))
    _require(
        {name: novelty_statuses.count(name) for name in set(novelty_statuses)}
        == {"novel": 2, "prefix_exposed": 4, "ambiguous": 2},
        "novelty status balance changed",
    )

    position_classes: list[str] = []
    for raw in collections["adjudication"]:
        row = _mapping(raw, "adjudication control")
        ref = str(row.get("completion_ref"))
        _require(ref in completion_by_id, "adjudication completion ref invalid")
        completion = completion_by_id[ref]
        candidate_1 = _resolve_candidate_spec(
            _mapping(row.get("candidate_1"), "candidate 1"),
            completion_control=completion,
        )
        candidate_2 = _resolve_candidate_spec(
            _mapping(row.get("candidate_2"), "candidate 2"),
            completion_control=completion,
        )
        expected_proposal = _resolve_candidate_spec(
            _mapping(row.get("expected"), "adjudication expected"),
            completion_control=completion,
        )
        expected_class = row.get("expected_position_class")
        _require(
            expected_class
            in {"candidate_1_correct", "candidate_2_correct", "neither_correct"},
            "adjudication position class invalid",
        )
        truth = (candidate_1 == expected_proposal, candidate_2 == expected_proposal)
        required_truth = {
            "candidate_1_correct": (True, False),
            "candidate_2_correct": (False, True),
            "neither_correct": (False, False),
        }[str(expected_class)]
        _require(truth == required_truth, f"{row['id']} position class is false")
        position_classes.append(str(expected_class))
    declared_balance = dict(
        _mapping(sealing.get("candidate_presentation_classes"), "candidate balance")
    )
    observed_balance = {
        name: position_classes.count(name)
        for name in ("candidate_1_correct", "candidate_2_correct", "neither_correct")
    }
    _require(
        declared_balance == observed_balance == {
            "candidate_1_correct": 4,
            "candidate_2_correct": 4,
            "neither_correct": 4,
        },
        "adjudication candidate balance changed",
    )
    return config, codebook, actual_binding


def load_config(
    path: Path = CONFIG_PATH, *, require_frozen_codebook: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = _guard_path(path, must_exist=True, label="control suite config")
    return validate_config(
        load_json_strict(path), require_frozen_codebook=require_frozen_codebook
    )


def _packet_id(pass_name: str, control_id: str, *, nonce: int = 0) -> str:
    return sha256_text(
        f"quote-first-sealed-controls-v2\0{pass_name}\0{control_id}\0{nonce}"
    )


def _packet_id_for_candidate_order(
    *, control_id: str, candidate_1: Mapping[str, Any], candidate_2: Mapping[str, Any]
) -> tuple[str, int, list[dict[str, Any]]]:
    desired = [dict(candidate_1), dict(candidate_2)]
    for nonce in range(100_000):
        packet_id = _packet_id("adjudication", control_id, nonce=nonce)
        ordered = quote_runner.blind_candidate_order(
            packet_id_sha256=packet_id, left=candidate_1, right=candidate_2
        )
        if ordered == desired:
            return packet_id, nonce, ordered
    raise ControlSuiteError("unable to realize sealed candidate presentation order")


def _model_manifest_has_forbidden_reference(value: Any) -> bool:
    forbidden = ("expectation", "expected", "gold", "label", "reason")
    if isinstance(value, Mapping):
        return any(
            any(token in str(key).casefold() for token in forbidden)
            or _model_manifest_has_forbidden_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_model_manifest_has_forbidden_reference(item) for item in value)
    if isinstance(value, str):
        return any(token in value.casefold() for token in forbidden)
    return False


def build_suite(
    *,
    config: Mapping[str, Any],
    output_directory: Path,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Build separated artifacts; refuse an unfrozen or drifting codebook."""

    config, _codebook, codebook_binding = validate_config(
        config, require_frozen_codebook=True
    )
    config_path = _guard_path(
        config_path, must_exist=True, label="control suite config"
    )
    physical_config = load_json_strict(config_path)
    _require(
        canonical_json_bytes(physical_config) == canonical_json_bytes(config),
        "in-memory suite config differs from the bound physical config",
    )
    output_directory = Path(output_directory).absolute()
    _require(
        output_directory.exists() and output_directory.is_dir(),
        "output directory must already exist",
    )
    _guard_path(output_directory / "placeholder", must_exist=False, label="output directory")
    completion_by_id = {
        str(item["id"]): dict(item) for item in config["completion_controls"]
    }

    model_rows: list[dict[str, Any]] = []
    expectation_rows: list[dict[str, Any]] = []
    for raw in config["completion_controls"]:
        row = dict(raw)
        control_id = str(row["id"])
        packet_id = _packet_id("completion", control_id)
        text = str(row["assistant_text"])
        proposal = _completion_proposal(row)
        expected_result = _materialized_expectation(
            proposal=proposal, assistant_text=text, empty_control=(text == "")
        )
        model_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": MODEL_INPUT_KIND,
                "pass": "completion",
                "packet_id_sha256": packet_id,
                "payload": {"assistant_text": text},
            }
        )
        expectation_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": EXPECTATION_KIND,
                "control_id": control_id,
                "pass": "completion",
                "packet_id_sha256": packet_id,
                "input_payload_sha256": sha256_bytes(
                    canonical_json_bytes({"assistant_text": text})
                ),
                "reason": row["expected"].get("reason"),
                "expected_semantic_proposal": proposal,
                "expected_result_projection": expected_result,
            }
        )

    for raw in config["novelty_controls"]:
        row = dict(raw)
        control_id = str(row["id"])
        packet_id = _packet_id("novelty", control_id)
        payload = {
            "visible_prefix": row["visible_prefix"],
            "locked_hypothesis": row["locked_hypothesis"],
        }
        model_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": MODEL_INPUT_KIND,
                "pass": "novelty",
                "packet_id_sha256": packet_id,
                "payload": payload,
            }
        )
        expectation_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": EXPECTATION_KIND,
                "control_id": control_id,
                "pass": "novelty",
                "packet_id_sha256": packet_id,
                "input_payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
                "reason": row["reason"],
                "expected_novelty_decision": row["expected_status"],
            }
        )

    for raw in config["adjudication_controls"]:
        row = dict(raw)
        control_id = str(row["id"])
        completion = completion_by_id[str(row["completion_ref"])]
        candidate_1 = _resolve_candidate_spec(
            _mapping(row["candidate_1"], "candidate 1"),
            completion_control=completion,
        )
        candidate_2 = _resolve_candidate_spec(
            _mapping(row["candidate_2"], "candidate 2"),
            completion_control=completion,
        )
        expected_proposal = _resolve_candidate_spec(
            _mapping(row["expected"], "adjudication expected"),
            completion_control=completion,
        )
        packet_id, order_nonce, candidates = _packet_id_for_candidate_order(
            control_id=control_id,
            candidate_1=candidate_1,
            candidate_2=candidate_2,
        )
        text = str(completion["assistant_text"])
        payload = {
            "assistant_text": text,
            "candidate_annotations": candidates,
        }
        expected_result = _materialized_expectation(
            proposal=expected_proposal, assistant_text=text
        )
        model_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": MODEL_INPUT_KIND,
                "pass": "adjudication",
                "packet_id_sha256": packet_id,
                "payload": payload,
            }
        )
        expectation_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": EXPECTATION_KIND,
                "control_id": control_id,
                "pass": "adjudication",
                "packet_id_sha256": packet_id,
                "input_payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
                "candidate_order_nonce": order_nonce,
                "expected_position_class": row["expected_position_class"],
                "expected_semantic_proposal": expected_proposal,
                "expected_result_projection": expected_result,
            }
        )

    for raw in config["resolver_controls"]:
        row = dict(raw)
        if "assistant_text_ref" in row:
            text = str(completion_by_id[str(row["assistant_text_ref"])]["assistant_text"])
        else:
            text = str(row["assistant_text"])
        if "proposal_ref" in row:
            proposal = _completion_proposal(
                completion_by_id[str(row["proposal_ref"])]
            )
        else:
            proposal = quote_runner.validate_model_proposal(row["proposal"])
        result = quote_runner.materialize_completion_proposal(
            proposal=proposal, assistant_text=text
        )
        expectation_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": EXPECTATION_KIND,
                "control_id": row["id"],
                "pass": "resolver",
                "packet_id_sha256": None,
                "resolver_input": {
                    "assistant_text": text,
                    "proposal": proposal,
                },
                "expected_result_projection": _result_projection(result),
            }
        )

    _require(len(model_rows) == 52, "model-input row count changed")
    _require(len(expectation_rows) == 56, "expectation row count changed")
    _require(
        len({row["packet_id_sha256"] for row in model_rows}) == 52,
        "model packet ids are not unique",
    )
    model_path = output_directory / "epistemic-chain-controls-v2-model-inputs.jsonl"
    model_manifest_path = (
        output_directory / "epistemic-chain-controls-v2-model-inputs-manifest.json"
    )
    expectation_path = output_directory / "epistemic-chain-controls-v2-expectations.jsonl"
    expectation_manifest_path = (
        output_directory / "epistemic-chain-controls-v2-expectations-manifest.json"
    )
    model_count, model_sha = _write_jsonl_atomic(model_path, model_rows)
    model_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MODEL_INPUT_MANIFEST_KIND,
        "status": "model_inputs_sealed",
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
        },
        "passes": list(PASS_NAMES),
        "counts": {
            "completion": 32,
            "novelty": 8,
            "adjudication": 12,
            "total": model_count,
        },
        "records": {
            "path": model_path.name,
            "sha256": model_sha,
            "count": model_count,
        },
        "codebook": codebook_binding,
    }
    _require(
        not _model_manifest_has_forbidden_reference(model_manifest),
        "model-input manifest contains an expectation or answer reference",
    )
    model_manifest_sha = _write_json_atomic(model_manifest_path, model_manifest)

    expectation_count, expectation_sha = _write_jsonl_atomic(
        expectation_path, expectation_rows
    )
    config_canonical_sha = sha256_bytes(canonical_json_bytes(config))
    expectation_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPECTATION_MANIFEST_KIND,
        "status": "expectations_sealed_not_model_input",
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_present": False,
        },
        "counts": {
            "completion": 32,
            "resolver": 4,
            "novelty": 8,
            "adjudication": 12,
            "total": expectation_count,
        },
        "records": {
            "path": expectation_path.name,
            "sha256": expectation_sha,
            "count": expectation_count,
        },
        "model_inputs": {
            "path": model_manifest_path.name,
            "sha256": model_manifest_sha,
        },
        "suite_config": {
            "path": str(config_path.relative_to(ROOT)),
            "canonical_sha256": config_canonical_sha,
        },
        "codebook": codebook_binding,
        "implementation": {
            "controls_builder_sha256": sha256_file(Path(__file__).resolve()),
            "quote_runner_v2_sha256": sha256_file(
                Path(quote_runner.__file__).resolve()
            ),
        },
    }
    expectation_manifest_sha = _write_json_atomic(
        expectation_manifest_path, expectation_manifest
    )
    return {
        "status": "sealed_artifacts_built_no_model_run",
        "model_input_manifest": {
            "path": str(model_manifest_path),
            "sha256": model_manifest_sha,
        },
        "expectation_manifest": {
            "path": str(expectation_manifest_path),
            "sha256": expectation_manifest_sha,
        },
    }


def _load_bound_records(
    manifest_path: Path, manifest: Mapping[str, Any], *, label: str
) -> list[dict[str, Any]]:
    binding = _mapping(manifest.get("records"), f"{label} record binding")
    record_path = manifest_path.parent / str(binding.get("path"))
    record_path = _guard_path(record_path, must_exist=True, label=f"{label} records")
    _require(
        SHA256_RE.fullmatch(str(binding.get("sha256"))) is not None
        and sha256_file(record_path) == binding["sha256"],
        f"{label} records hash changed",
    )
    rows = _load_jsonl(record_path, label=f"{label} records")
    _require(len(rows) == binding.get("count"), f"{label} record count changed")
    return rows


def _load_model_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _guard_path(path, must_exist=True, label="model-input manifest")
    manifest = dict(_mapping(load_json_strict(path), "model-input manifest"))
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == MODEL_INPUT_MANIFEST_KIND
        and manifest.get("status") == "model_inputs_sealed",
        "model-input manifest identity invalid",
    )
    _require(
        not _model_manifest_has_forbidden_reference(manifest),
        "model-input manifest references answers",
    )
    rows = _load_bound_records(path, manifest, label="model input")
    expected_counts = {
        "completion": 32,
        "novelty": 8,
        "adjudication": 12,
        "total": 52,
    }
    _require(
        manifest.get("counts") == expected_counts and len(rows) == 52,
        "model-input manifest counts changed",
    )
    packet_ids: set[str] = set()
    observed_counts = {name: 0 for name in PASS_NAMES}
    for row in rows:
        _require(
            set(row)
            == {"schema_version", "kind", "pass", "packet_id_sha256", "payload"}
            and row.get("schema_version") == SCHEMA_VERSION
            and row.get("kind") == MODEL_INPUT_KIND
            and row.get("pass") in PASS_NAMES
            and isinstance(row.get("payload"), Mapping),
            "model-input packet fields or identity invalid",
        )
        packet_id = row.get("packet_id_sha256")
        _require(
            isinstance(packet_id, str)
            and SHA256_RE.fullmatch(packet_id) is not None
            and packet_id not in packet_ids,
            "model-input packet id invalid or duplicated",
        )
        packet_ids.add(packet_id)
        observed_counts[str(row["pass"])] += 1
    _require(
        observed_counts == {"completion": 32, "novelty": 8, "adjudication": 12},
        "model-input pass coverage changed",
    )
    return manifest, rows


def lock_outputs(
    *, model_input_manifest_path: Path, output_records_path: Path, output_manifest_path: Path
) -> dict[str, Any]:
    """Lock exact output bytes and packet coverage without opening expectations."""

    model_input_manifest_path = _guard_path(
        model_input_manifest_path, must_exist=True, label="model-input manifest"
    )
    _manifest, inputs = _load_model_manifest(model_input_manifest_path)
    output_records_path = _guard_path(
        output_records_path, must_exist=True, label="control output records"
    )
    outputs = _load_jsonl(output_records_path, label="control output")
    expected = {
        str(row["packet_id_sha256"]): str(row["pass"]) for row in inputs
    }
    observed: dict[str, str] = {}
    for row in outputs:
        _require(
            set(row)
            == {"schema_version", "kind", "pass", "packet_id_sha256", "result"}
            and row.get("schema_version") == SCHEMA_VERSION
            and row.get("kind") == OUTPUT_RECORD_KIND,
            "control output record fields or identity invalid",
        )
        packet_id = row.get("packet_id_sha256")
        pass_name = row.get("pass")
        _require(
            isinstance(packet_id, str)
            and SHA256_RE.fullmatch(packet_id) is not None
            and packet_id not in observed
            and pass_name in PASS_NAMES
            and isinstance(row.get("result"), Mapping),
            "control output record invalid or duplicated",
        )
        observed[packet_id] = str(pass_name)
    _require(observed == expected, "control outputs do not exactly cover sealed inputs")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": LOCKED_OUTPUT_MANIFEST_KIND,
        "status": "locked_before_expectation_join",
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
        },
        "model_inputs": {
            "path": str(model_input_manifest_path),
            "sha256": sha256_file(model_input_manifest_path),
        },
        "records": {
            "path": str(output_records_path),
            "sha256": sha256_file(output_records_path),
            "count": len(outputs),
        },
    }
    _write_json_atomic(output_manifest_path, manifest)
    return manifest


def _load_locked_outputs(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = _guard_path(path, must_exist=True, label="locked output manifest")
    manifest = dict(_mapping(load_json_strict(path), "locked output manifest"))
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == LOCKED_OUTPUT_MANIFEST_KIND
        and manifest.get("status") == "locked_before_expectation_join",
        "outputs were not locked before expectation join",
    )
    rows = _load_bound_records(path, manifest, label="locked output")
    by_packet: dict[str, dict[str, Any]] = {}
    for row in rows:
        packet_id = row.get("packet_id_sha256")
        _require(
            set(row)
            == {"schema_version", "kind", "pass", "packet_id_sha256", "result"}
            and row.get("schema_version") == SCHEMA_VERSION
            and row.get("kind") == OUTPUT_RECORD_KIND
            and row.get("pass") in PASS_NAMES
            and isinstance(packet_id, str)
            and SHA256_RE.fullmatch(packet_id) is not None
            and packet_id not in by_packet
            and isinstance(row.get("result"), Mapping),
            "locked output packet fields, identity, or id invalid",
        )
        by_packet[packet_id] = row
    return manifest, by_packet


def _load_expectations(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _guard_path(path, must_exist=True, label="expectation manifest")
    manifest = dict(_mapping(load_json_strict(path), "expectation manifest"))
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == EXPECTATION_MANIFEST_KIND
        and manifest.get("status") == "expectations_sealed_not_model_input",
        "expectation manifest identity invalid",
    )
    rows = _load_bound_records(path, manifest, label="expectation")
    return manifest, rows


def _novelty_decision(result: Mapping[str, Any]) -> str | None:
    for field in ("decision", "raw_semantic_decision", "novelty_status"):
        value = result.get(field)
        if value in {"novel", "prefix_exposed", "ambiguous", "unknown"}:
            return str(value)
    return None


def score_locked_outputs(
    *,
    locked_output_manifest_path: Path,
    expectation_manifest_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Join sealed answers only after authenticating a prior output lock."""

    # Authenticate the lock, its exact bytes, the sealed model manifest, and
    # packet coverage before opening the physically separate expectation file.
    locked_manifest, outputs = _load_locked_outputs(locked_output_manifest_path)
    locked_model = _mapping(locked_manifest.get("model_inputs"), "locked model inputs")
    locked_model_path = Path(str(locked_model.get("path")))
    if not locked_model_path.is_absolute():
        locked_model_path = Path(locked_output_manifest_path).parent / locked_model_path
    locked_model_path = _guard_path(
        locked_model_path, must_exist=True, label="locked model-input manifest"
    )
    _require(
        SHA256_RE.fullmatch(str(locked_model.get("sha256"))) is not None
        and sha256_file(locked_model_path) == locked_model.get("sha256"),
        "locked model-input manifest hash changed",
    )
    model_manifest, model_inputs = _load_model_manifest(locked_model_path)
    locked_coverage = {
        packet_id: str(row["pass"]) for packet_id, row in outputs.items()
    }
    sealed_coverage = {
        str(row["packet_id_sha256"]): str(row["pass"]) for row in model_inputs
    }
    _require(
        locked_coverage == sealed_coverage,
        "locked output coverage differs from sealed model inputs",
    )

    expectation_manifest, expectations = _load_expectations(expectation_manifest_path)
    expected_model = _mapping(
        expectation_manifest.get("model_inputs"), "expectation model inputs"
    )
    expected_model_path = expectation_manifest_path.parent / str(
        expected_model.get("path")
    )
    expected_model_path = _guard_path(
        expected_model_path, must_exist=True, label="expected model-input manifest"
    )
    _require(
        sha256_file(expected_model_path) == expected_model.get("sha256")
        and locked_model.get("sha256") == expected_model.get("sha256")
        and expected_model_path == locked_model_path,
        "locked outputs and expectations bind different model inputs",
    )
    implementation = _mapping(
        expectation_manifest.get("implementation"), "expectation implementation"
    )
    _require(
        implementation.get("controls_builder_sha256")
        == sha256_file(Path(__file__).resolve())
        and implementation.get("quote_runner_v2_sha256")
        == sha256_file(Path(quote_runner.__file__).resolve()),
        "controls builder or quote runner changed after expectations were sealed",
    )
    suite_binding = _mapping(
        expectation_manifest.get("suite_config"), "expectation suite config"
    )
    suite_path = Path(str(suite_binding.get("path")))
    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path
    suite_path = _guard_path(suite_path, must_exist=True, label="bound suite config")
    suite_value = load_json_strict(suite_path)
    _require(
        SHA256_RE.fullmatch(str(suite_binding.get("canonical_sha256"))) is not None
        and sha256_bytes(canonical_json_bytes(suite_value))
        == suite_binding.get("canonical_sha256"),
        "bound suite config changed after expectations were sealed",
    )
    suite_config, _codebook, actual_codebook_binding = validate_config(
        suite_value, require_frozen_codebook=True
    )
    expected_codebook_binding = _mapping(
        expectation_manifest.get("codebook"), "expectation codebook binding"
    )
    model_codebook_binding = _mapping(
        model_manifest.get("codebook"), "model-input codebook binding"
    )
    _require(
        dict(expected_codebook_binding)
        == dict(model_codebook_binding)
        == dict(suite_config["codebook_binding"])
        == actual_codebook_binding,
        "sealed artifacts do not authenticate the same V2 codebook",
    )

    rows: list[dict[str, Any]] = []
    invalid_outputs = 0
    completion_correct = 0
    completion_exact_correct = 0
    completion_total = 0
    positive_graph_correct = 0
    positive_graph_total = 0
    novelty_correct = 0
    adjudication_correct = 0
    resolver_correct = 0
    candidate_class_correct = {
        "candidate_1_correct": 0,
        "candidate_2_correct": 0,
        "neither_correct": 0,
    }
    per_control: dict[str, bool] = {}
    for expectation in expectations:
        control_id = str(expectation["control_id"])
        pass_name = str(expectation["pass"])
        if pass_name == "resolver":
            resolver_input = _mapping(
                expectation.get("resolver_input"), "resolver control input"
            )
            observed_result = quote_runner.materialize_completion_proposal(
                proposal=resolver_input["proposal"],
                assistant_text=str(resolver_input["assistant_text"]),
            )
            correct = _result_projection(observed_result) == expectation.get(
                "expected_result_projection"
            )
            resolver_correct += int(correct)
            rows.append(
                {
                    "control_id": control_id,
                    "pass": pass_name,
                    "correct": correct,
                    "packet_id_sha256": None,
                }
            )
            per_control[control_id] = correct
            continue

        packet_id = str(expectation["packet_id_sha256"])
        _require(packet_id in outputs, "locked output coverage changed after lock")
        output = outputs[packet_id]
        _require(output.get("pass") == pass_name, "locked output pass changed")
        result = _mapping(output.get("result"), "locked control result")
        if pass_name == "novelty":
            observed_decision = _novelty_decision(result)
            correct = observed_decision == expectation.get("expected_novelty_decision")
            novelty_correct += int(correct)
        else:
            projection = _result_projection(result)
            correct = projection == expectation.get("expected_result_projection")
            expected_proposal = _mapping(
                expectation.get("expected_semantic_proposal"),
                "expected semantic proposal",
            )
            if projection.get("semantic_validation_status") != "valid":
                invalid_outputs += 1
            if pass_name == "completion":
                completion_total += 1
                completion_exact_correct += int(correct)
                category_correct = (
                    projection.get("raw_semantic_decision")
                    == expected_proposal.get("decision")
                )
                completion_correct += int(category_correct)
                if (
                    expected_proposal.get("decision") == "chain"
                    and expectation["expected_result_projection"].get(
                        "materialization_status"
                    )
                    == "resolved_chain"
                ):
                    positive_graph_total += 1
                    positive_graph_correct += int(correct)
            else:
                adjudication_correct += int(correct)
                position_class = str(expectation["expected_position_class"])
                candidate_class_correct[position_class] += int(correct)
        rows.append(
            {
                "control_id": control_id,
                "pass": pass_name,
                "correct": correct,
                "packet_id_sha256": packet_id,
            }
        )
        per_control[control_id] = correct

    marker_pair_invariance = per_control.get("C01") is True and per_control.get("C02") is True
    all_controls_exact = len(per_control) == 56 and all(per_control.values())
    gates = {
        "completion_category_accuracy_1": completion_correct == completion_total == 32,
        "completion_exact_record_accuracy_1": completion_exact_correct
        == completion_total
        == 32,
        "positive_exact_graph_accuracy_1": positive_graph_correct == positive_graph_total == 17,
        "invalid_structured_output_rate_0": invalid_outputs == 0,
        "marker_pair_invariance_1": marker_pair_invariance,
        "unique_tuple_materialization_1": per_control.get("C17") is True,
        "ambiguous_tuple_rejection_1": per_control.get("C18") is True
        and per_control.get("R03") is True,
        "resolver_accuracy_1": resolver_correct == 4,
        "novelty_accuracy_1": novelty_correct == 8,
        "adjudication_accuracy_1": adjudication_correct == 12,
        "candidate_position_balance_4_4_4": candidate_class_correct
        == {
            "candidate_1_correct": 4,
            "candidate_2_correct": 4,
            "neither_correct": 4,
        },
        "all_56_controls_exact_1": all_controls_exact,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "passed" if all(gates.values()) else "failed",
        "scope": {
            "development_controls_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "passing_does_not_establish_target_decoding": True,
        },
        "counts": {
            "completion": completion_total,
            "resolver": 4,
            "novelty": 8,
            "adjudication": 12,
            "invalid_outputs": invalid_outputs,
            "positive_exact_graph": positive_graph_total,
        },
        "correct": {
            "completion_category": completion_correct,
            "completion_exact_record": completion_exact_correct,
            "positive_exact_graph": positive_graph_correct,
            "resolver": resolver_correct,
            "novelty": novelty_correct,
            "adjudication": adjudication_correct,
            "candidate_position_class": candidate_class_correct,
        },
        "gates": gates,
        "rows": rows,
        "inputs": {
            "locked_output_manifest": {
                "path": str(Path(locked_output_manifest_path).resolve()),
                "sha256": sha256_file(Path(locked_output_manifest_path).resolve()),
            },
            "expectation_manifest": {
                "path": str(Path(expectation_manifest_path).resolve()),
                "sha256": sha256_file(Path(expectation_manifest_path).resolve()),
            },
        },
    }
    if report_path is not None:
        _write_json_atomic(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "freeze-helper", help="print the current codebook binding; never edit files"
    )
    subparsers.add_parser("validate", help="validate content with binding optional")
    build = subparsers.add_parser("build", help="build separated sealed artifacts")
    build.add_argument("--output-directory", type=Path, required=True)
    lock = subparsers.add_parser("lock-outputs", help="lock outputs before scoring")
    lock.add_argument("--model-input-manifest", type=Path, required=True)
    lock.add_argument("--output-records", type=Path, required=True)
    lock.add_argument("--output-manifest", type=Path, required=True)
    score = subparsers.add_parser("score", help="score only a prior locked output")
    score.add_argument("--locked-output-manifest", type=Path, required=True)
    score.add_argument("--expectation-manifest", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _guard_path(args.config, must_exist=True, label="control suite config")
    raw_config = dict(_mapping(load_json_strict(config_path), "control suite config"))
    if args.command == "freeze-helper":
        print(canonical_json_text(codebook_freeze_binding(raw_config)))
        return 0
    if args.command == "validate":
        config, _codebook, actual = validate_config(
            raw_config, require_frozen_codebook=False
        )
        frozen = config.get("codebook_binding") == actual
        print(
            canonical_json_text(
                {
                    "status": "passed_content_validation",
                    "codebook_binding_frozen": frozen,
                    "build_enabled": frozen,
                    "counts": config["sealing_contract"]["required_counts"],
                }
            )
        )
        return 0
    if args.command == "build":
        result = build_suite(
            config=raw_config,
            output_directory=args.output_directory,
            config_path=config_path,
        )
    elif args.command == "lock-outputs":
        result = lock_outputs(
            model_input_manifest_path=args.model_input_manifest,
            output_records_path=args.output_records,
            output_manifest_path=args.output_manifest,
        )
    else:
        result = score_locked_outputs(
            locked_output_manifest_path=args.locked_output_manifest,
            expectation_manifest_path=args.expectation_manifest,
            report_path=args.report,
        )
    print(canonical_json_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
