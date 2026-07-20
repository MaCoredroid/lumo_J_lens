#!/usr/bin/env python3
"""Materialize and score frozen synthetic controls for the V4 chain annotator.

The packet manifest never references the expectation sidecar.  The controls
are copied from the already-frozen codebook and therefore measure exact
codebook/interface adherence, not independent semantic generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import swe_task_state_v4_epistemic_chain_annotation_runner as runner  # noqa: E402


SCHEMA_VERSION = 1
FIXTURE_CONTRACT_ID = "visible-epistemic-chain-frozen-codebook-controls-v1"
EXPECTATION_RECORD_KIND = (
    "swe_task_state_v4_epistemic_chain_quality_control_expectation"
)
EXPECTATION_MANIFEST_KIND = (
    "swe_task_state_v4_epistemic_chain_quality_control_expectation_manifest"
)
SCORE_REPORT_KIND = "swe_task_state_v4_epistemic_chain_quality_control_report"


def _expected_semantics(example: Mapping[str, Any]) -> dict[str, Any]:
    has_chain = example.get("has_chain")
    runner._require(isinstance(has_chain, bool), "fixture has_chain must be boolean")
    result: dict[str, Any] = {
        "annotation_status": "available",
        "unknown_reason": None,
        "has_chain": has_chain,
        "evidence_span": None,
        "hypothesis_span": None,
        "action_span": None,
        "evidence_kind": None,
        "belief_edge": None,
        "hypothesis_domain": None,
        "action_intent": None,
        "novelty_status": None,
        "exact_signature": None,
    }
    if not has_chain:
        return result
    text = example.get("assistant_text")
    runner._require(isinstance(text, str), "positive fixture text invalid")
    for slot in ("evidence", "hypothesis", "action"):
        span_text = example.get(f"{slot}_span_text")
        runner._require(
            isinstance(span_text, str)
            and bool(span_text)
            and text.count(span_text) == 1,
            f"positive fixture {slot} span is not unique",
        )
        start = text.index(span_text)
        result[f"{slot}_span"] = {
            "start": start,
            "end": start + len(span_text),
            "text_sha256": runner.sha256_text(span_text),
        }
    for field in (
        "evidence_kind",
        "belief_edge",
        "hypothesis_domain",
        "action_intent",
        "exact_signature",
    ):
        runner._require(isinstance(example.get(field), str), f"fixture {field} invalid")
        result[field] = example[field]
    return result


def _fixture_packet(example: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_id = example.get("id")
    text = example.get("assistant_text")
    runner._require(
        isinstance(fixture_id, str) and bool(fixture_id) and isinstance(text, str),
        "fixture identity or text invalid",
    )
    source_id = runner.sha256_text(f"{FIXTURE_CONTRACT_ID}\0source\0{fixture_id}")
    packet_id = runner.sha256_text(f"{FIXTURE_CONTRACT_ID}\0packet\0{source_id}")
    lane_a = int(runner.sha256_text(f"{packet_id}\0a")[:16], 16) % 8
    lane_b = int(runner.sha256_text(f"{packet_id}\0b")[:16], 16) % 8
    if lane_a == lane_b:
        lane_b = (lane_b + 1) % 8
    packet = {
        "schema_version": SCHEMA_VERSION,
        "kind": runner.COMPLETION_PACKET_KIND,
        "annotation_pass": "completion_chain",
        "packet_id_sha256": packet_id,
        "source_id_sha256": source_id,
        "blind_shards": {"independent_a": lane_a, "independent_b": lane_b},
        "materialized_assistant_text": {
            "text": text,
            "sha256": runner.sha256_text(text),
            "char_start": 0,
            "char_end": len(text),
        },
        "authenticated_boundaries": {
            "fixture_contract_sha256": runner.sha256_text(FIXTURE_CONTRACT_ID),
            "frozen_codebook_example_sha256": runner.sha256_bytes(
                runner.canonical_json_bytes(example)
            ),
        },
        "annotator_visibility": {
            "complete_prefix_text_present": False,
            "assistant_tool_arguments_present": False,
            "tool_results_present": False,
            "repository_or_task_identity_present": False,
            "model_features_present": False,
        },
    }
    runner.validate_packet(packet, annotation_pass="completion_chain")
    expectation = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPECTATION_RECORD_KIND,
        "fixture_id": fixture_id,
        "packet_id_sha256": packet_id,
        "source_id_sha256": source_id,
        "expected_semantics": _expected_semantics(example),
        "codebook_reason": example.get("reason"),
    }
    return packet, expectation


def materialize(
    *, packet_manifest_path: Path, expectation_manifest_path: Path
) -> dict[str, Any]:
    runner._require(
        packet_manifest_path != expectation_manifest_path,
        "packet and expectation manifests must be separate",
    )
    config = runner.validate_config(runner.load_json_strict(runner.CONFIG_PATH))
    _annotation_config, codebook = runner.authenticate_inputs(config)
    examples = [
        *runner._sequence(codebook.get("positive_examples"), "positive examples"),
        *runner._sequence(codebook.get("negative_examples"), "negative examples"),
    ]
    runner._require(len(examples) == 10, "frozen fixture count changed")
    pairs = [_fixture_packet(runner._mapping(item, "fixture")) for item in examples]
    packets = [item[0] for item in pairs]
    expectations = [item[1] for item in pairs]
    packet_jsonl_path = packet_manifest_path.with_suffix(".jsonl")
    expectation_jsonl_path = expectation_manifest_path.with_suffix(".jsonl")
    packet_count, packet_sha = runner._write_jsonl_atomic(packet_jsonl_path, packets)
    packet_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "swe_task_state_v4_epistemic_chain_packet_manifest",
        "status": "frozen_synthetic_quality_controls_no_expectations_in_model_input",
        "annotation_pass": "completion_chain",
        "scope": {
            "development_data_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "private_chain_of_thought_ground_truth_claimed": False,
        },
        "packets": {
            "path": packet_jsonl_path.name,
            "sha256": packet_sha,
            "count": packet_count,
        },
        "inputs": {
            "frozen_codebook": config["inputs"]["annotation_codebook"],
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": runner.sha256_file(Path(__file__).resolve()),
            },
        },
        "control_scope": {
            "source": "all_frozen_codebook_positive_and_negative_examples",
            "positive_count": 3,
            "negative_count": 7,
            "expectation_manifest_or_labels_present": False,
            "measures_interface_and_exact_codebook_adherence_not_independent_generalization": True,
        },
    }
    runner._write_json_atomic(packet_manifest_path, packet_manifest)
    expectation_count, expectation_sha = runner._write_jsonl_atomic(
        expectation_jsonl_path, expectations
    )
    expectation_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXPECTATION_MANIFEST_KIND,
        "status": "locked_before_quality_control_generation",
        "scope": {
            "development_data_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "not_model_input": True,
        },
        "packet_manifest": {
            "path": str(packet_manifest_path.resolve()),
            "sha256": runner.sha256_file(packet_manifest_path),
        },
        "expectations": {
            "path": expectation_jsonl_path.name,
            "sha256": expectation_sha,
            "count": expectation_count,
        },
        "frozen_codebook": config["inputs"]["annotation_codebook"],
    }
    runner._write_json_atomic(expectation_manifest_path, expectation_manifest)
    return {
        "packet_manifest": packet_manifest,
        "expectation_manifest": expectation_manifest,
    }


def _load_expectations(
    path: Path, *, expected_sha256: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if expected_sha256 is not None:
        runner._require(
            runner.sha256_file(path) == expected_sha256,
            "expectation manifest hash changed",
        )
    manifest = dict(runner._mapping(runner.load_json_strict(path), "expectation manifest"))
    runner._require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == EXPECTATION_MANIFEST_KIND
        and manifest.get("status") == "locked_before_quality_control_generation"
        and runner._mapping(manifest.get("scope"), "expectation scope").get(
            "not_model_input"
        )
        is True,
        "expectation manifest identity invalid",
    )
    binding = runner._mapping(manifest.get("expectations"), "expectation binding")
    record_path = path.parent / str(binding.get("path"))
    runner._require(
        runner.sha256_file(record_path) == binding.get("sha256"),
        "expectation records hash changed",
    )
    values: list[dict[str, Any]] = []
    with record_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = dict(runner._mapping(json.loads(line), "expectation record"))
            runner._require(
                value.get("kind") == EXPECTATION_RECORD_KIND,
                "expectation record identity invalid",
            )
            values.append(value)
    runner._require(len(values) == binding.get("count"), "expectation count changed")
    return manifest, values


def score(
    *,
    packet_manifest_path: Path,
    expectation_manifest_path: Path,
    lane_manifest_path: Path,
    output_report_path: Path,
    expected_packet_manifest_sha256: str | None,
    expected_expectation_manifest_sha256: str | None,
    expected_lane_manifest_sha256: str | None,
) -> dict[str, Any]:
    config = runner.validate_config(runner.load_json_strict(runner.CONFIG_PATH))
    annotation_config, _codebook = runner.authenticate_inputs(config)
    packet_manifest, packets, _packet_path = runner.load_packet_manifest(
        packet_manifest_path, expected_sha256=expected_packet_manifest_sha256
    )
    expectation_manifest, expectations = _load_expectations(
        expectation_manifest_path,
        expected_sha256=expected_expectation_manifest_sha256,
    )
    lane_manifest, lane = runner._load_lane(
        lane_manifest_path, expected_sha256=expected_lane_manifest_sha256
    )
    runner._require(
        expectation_manifest["packet_manifest"]["sha256"]
        == runner.sha256_file(packet_manifest_path)
        and lane_manifest.get("role") == "quality_audit"
        and lane_manifest.get("annotation_pass") == "completion_chain"
        and lane_manifest["inputs"]["packet_manifest"]["sha256"]
        == runner.sha256_file(packet_manifest_path),
        "quality-control manifests are not mutually bound",
    )
    packet_by_id = {str(item["packet_id_sha256"]): item for item in packets}
    expected_by_id = {str(item["packet_id_sha256"]): item for item in expectations}
    runner._require(
        set(packet_by_id) == set(expected_by_id) == set(lane),
        "quality-control coverage differs",
    )
    rows: list[dict[str, Any]] = []
    invalid_count = 0
    category_correct = 0
    exact_semantic_correct = 0
    positive_graph_correct = 0
    positive_count = 0
    for packet_id in expected_by_id:
        expected = expected_by_id[packet_id]
        wrapper = lane[packet_id]
        observed = runner._mapping(wrapper.get("annotation_record"), "observed record")
        packet_text = packet_by_id[packet_id]["materialized_assistant_text"]["text"]
        runner.packet_contract.validate_annotation_record(
            observed,
            config=annotation_config,
            stage="completion",
            completion_text=packet_text,
        )
        expected_semantics = runner._mapping(
            expected.get("expected_semantics"), "expected semantics"
        )
        observed_semantics = runner.semantic_projection(observed)
        generation = runner._mapping(wrapper.get("generation"), "generation")
        invalid = generation.get("validation_status") != "valid"
        invalid_count += int(invalid)
        category_ok = (
            observed.get("annotation_status") == "available"
            and observed.get("has_chain") == expected_semantics.get("has_chain")
        )
        semantic_ok = observed_semantics == dict(expected_semantics)
        expected_positive = expected_semantics.get("has_chain") is True
        positive_count += int(expected_positive)
        graph_ok = semantic_ok if expected_positive else None
        category_correct += int(category_ok)
        exact_semantic_correct += int(semantic_ok)
        positive_graph_correct += int(graph_ok is True)
        rows.append(
            {
                "fixture_id": expected["fixture_id"],
                "packet_id_sha256": packet_id,
                "expected_category": "chain" if expected_positive else "no_chain",
                "observed_category": (
                    "unknown"
                    if observed.get("annotation_status") == "unknown"
                    else "chain"
                    if observed.get("has_chain") is True
                    else "no_chain"
                ),
                "category_correct": category_ok,
                "exact_semantics_correct": semantic_ok,
                "positive_exact_graph_correct": graph_ok,
                "generation_valid": not invalid,
                "validation_error": generation.get("validation_error"),
            }
        )
    passed = (
        invalid_count == 0
        and category_correct == len(rows)
        and exact_semantic_correct == len(rows)
        and positive_graph_correct == positive_count
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": SCORE_REPORT_KIND,
        "status": "passed" if passed else "failed",
        "scope": {
            "development_data_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "private_chain_of_thought_ground_truth_claimed": False,
            "measures_interface_and_exact_codebook_adherence_not_independent_generalization": True,
        },
        "counts": {
            "controls": len(rows),
            "positive": positive_count,
            "negative": len(rows) - positive_count,
            "invalid_outputs": invalid_count,
            "category_correct": category_correct,
            "exact_semantics_correct": exact_semantic_correct,
            "positive_exact_graph_correct": positive_graph_correct,
        },
        "metrics": {
            "category_accuracy": category_correct / len(rows),
            "exact_semantic_accuracy": exact_semantic_correct / len(rows),
            "positive_exact_graph_accuracy": positive_graph_correct / positive_count,
        },
        "rows": rows,
        "inputs": {
            "packet_manifest": {
                "path": str(packet_manifest_path.resolve()),
                "sha256": runner.sha256_file(packet_manifest_path),
            },
            "expectation_manifest": {
                "path": str(expectation_manifest_path.resolve()),
                "sha256": runner.sha256_file(expectation_manifest_path),
            },
            "quality_audit_lane_manifest": {
                "path": str(lane_manifest_path.resolve()),
                "sha256": runner.sha256_file(lane_manifest_path),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": runner.sha256_file(Path(__file__).resolve()),
            },
        },
    }
    runner._write_json_atomic(output_report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("materialize")
    make.add_argument("--packet-manifest", type=Path, required=True)
    make.add_argument("--expectation-manifest", type=Path, required=True)
    evaluate = subparsers.add_parser("score")
    evaluate.add_argument("--packet-manifest", type=Path, required=True)
    evaluate.add_argument("--expectation-manifest", type=Path, required=True)
    evaluate.add_argument("--lane-manifest", type=Path, required=True)
    evaluate.add_argument("--output-report", type=Path, required=True)
    evaluate.add_argument("--expected-packet-manifest-sha256")
    evaluate.add_argument("--expected-expectation-manifest-sha256")
    evaluate.add_argument("--expected-lane-manifest-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize(
            packet_manifest_path=args.packet_manifest,
            expectation_manifest_path=args.expectation_manifest,
        )
    else:
        result = score(
            packet_manifest_path=args.packet_manifest,
            expectation_manifest_path=args.expectation_manifest,
            lane_manifest_path=args.lane_manifest,
            output_report_path=args.output_report,
            expected_packet_manifest_sha256=args.expected_packet_manifest_sha256,
            expected_expectation_manifest_sha256=args.expected_expectation_manifest_sha256,
            expected_lane_manifest_sha256=args.expected_lane_manifest_sha256,
        )
    print(runner.canonical_json_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
