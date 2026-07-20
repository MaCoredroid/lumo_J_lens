#!/usr/bin/env python3
"""Pre-label contract and pure primitives for the V4 epistemic-chain decoder.

This module deliberately does not know how to open an annotation sidecar.  It
freezes and checks the decoder geometry before target annotations exist, and
provides label-independent feature, prediction, embedding, retrieval, and
hierarchical-weight primitives for the eventual authenticated runner.

The coarse categorical renderer is not sentence recovery.  Proposition-level
evidence requires the separate frozen-span embedding and training-only
retrieval lane in the bound protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_epistemic_chain_decoder.json"
CONFIG_SHA256 = "2c9a4c16a36ef96c9db32e231aa337947a099f21bb791b2411c261f631d087a1"
FORBIDDEN_PATH_FRAGMENTS = ("reserved", "validation")


class EpistemicDecoderError(ValueError):
    """Raised when a prospective decoder contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EpistemicDecoderError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EpistemicDecoderError(f"cannot load strict JSON: {path}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def lexical_path_preflight(paths: Iterable[Path | None]) -> None:
    """Reject forbidden path text before any filesystem metadata operation."""

    for path in paths:
        if path is None:
            continue
        normalized = os.path.normpath(os.path.abspath(os.fspath(path))).lower()
        if any(
            fragment in component
            for component in Path(normalized).parts
            for fragment in FORBIDDEN_PATH_FRAGMENTS
        ):
            raise EpistemicDecoderError(
                f"forbidden path rejected before filesystem access: {path}"
            )


def resolve_guarded_path(
    path: Path,
    *,
    allowed_root: Path | None = None,
    reject_logical_symlink: bool,
    require_kind: str,
) -> Path:
    """Resolve one path while preserving lexical, symlink, and root boundaries.

    The unresolved spelling is checked before metadata access.  The canonical
    target is checked again before callers open it.  Frozen repository inputs
    additionally reject a final-component symlink and must remain below ROOT;
    hash-bound Hugging Face snapshot files may be symlinks, but their resolved
    targets still pass the forbidden-fragment check before hashing.
    """

    _require(require_kind in {"file", "directory"}, "guarded path kind is invalid")
    lexical_path_preflight([path])
    logical = Path(os.path.abspath(os.fspath(path)))
    if reject_logical_symlink:
        _require(not logical.is_symlink(), f"logical path may not be a symlink: {path}")
    try:
        canonical = logical.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise EpistemicDecoderError(f"cannot resolve guarded path: {path}") from error
    lexical_path_preflight([canonical])
    if allowed_root is not None:
        lexical_path_preflight([allowed_root])
        root = allowed_root.resolve(strict=True)
        lexical_path_preflight([root])
        _require(
            canonical == root or root in canonical.parents,
            f"canonical path escapes allowed root: {path}",
        )
    if require_kind == "file":
        _require(canonical.is_file(), f"guarded path is not a regular file: {path}")
    else:
        _require(canonical.is_dir(), f"guarded path is not a directory: {path}")
    return canonical


def signature_registry(config: Mapping[str, Any]) -> list[str]:
    ontology = _mapping(config.get("ontology"), "ontology")
    signatures = [
        ">".join((evidence, edge, domain, "motivates", action))
        for evidence in ontology["evidence_kind"]
        for edge in ontology["belief_edge"]
        for domain in ontology["hypothesis_domain"]
        for action in ontology["action_intent"]
    ]
    _require(
        len(signatures) == ontology["cartesian_signature_count"] == 252
        and len(signatures) == len(set(signatures)),
        "exact-signature registry is not the frozen 252-way Cartesian product",
    )
    return signatures


def validate_config(value: Any) -> dict[str, Any]:
    config = dict(_mapping(value, "epistemic decoder config"))
    expected_top_level = {
        "schema_version",
        "id",
        "status",
        "scope",
        "path_guard",
        "frozen_inputs",
        "annotation_evidence_contract",
        "ontology",
        "target_projection",
        "support_and_agreement_gate",
        "full_prefix_semantic_baseline",
        "frozen_proposition_embedding_model",
        "proposition_content_lane",
        "causal_prior_chain_control",
        "numeric_feature_blocks",
        "variants",
        "nested_comparisons",
        "split_and_weighting",
        "model",
        "nested_calibration_and_abstention",
        "metrics",
        "full_refit_uncertainty",
        "controls_and_sensitivities",
        "renderer",
        "claim_scope",
        "mandatory_limitations",
    }
    _require(set(config) == expected_top_level, "decoder config schema changed")
    _require(
        config["schema_version"] == 1
        and config["id"]
        == "swe-task-state-v4-visible-novel-epistemic-chain-decoder-v1"
        and config["status"]
        == "prospective_development_protocol_frozen_before_target_annotations",
        "decoder config identity changed",
    )
    _require(
        config["scope"]
        == {
            "development_data_only": True,
            "reserved_validation_closed": True,
            "reserved_validation_accessed": False,
            "target_annotations_available_when_frozen": False,
            "target_results_observed_when_frozen": False,
        }
        and config["path_guard"]
        == {
            "forbidden_path_fragments": ["reserved", "validation"],
            "frozen_input_paths_must_be_relative_to_repository_root": True,
            "frozen_input_logical_files_must_not_be_symlinks": True,
            "frozen_input_canonical_targets_must_remain_within_repository_root": True,
            "every_resolved_path_rechecked_for_forbidden_fragments_before_open": True,
            "model_snapshot_file_symlinks_allowed_only_after_canonical_path_and_hash_check": True,
        },
        "decoder scope or path guard changed",
    )
    for label, record_value in _mapping(
        config["frozen_inputs"], "frozen input registry"
    ).items():
        record = _mapping(record_value, f"frozen input {label}")
        _require(
            set(record) == {"path", "sha256"}
            and isinstance(record["path"], str)
            and bool(record["path"])
            and not Path(record["path"]).is_absolute()
            and ".." not in Path(record["path"]).parts
            and _is_sha256(record["sha256"]),
            f"frozen input binding is invalid: {label}",
        )
    ontology = _mapping(config["ontology"], "ontology")
    _require(
        ontology["evidence_kind"]
        == ["code", "tool_or_test", "spec_contract", "environment"]
        and ontology["belief_edge"] == ["supports", "refutes", "narrows"]
        and ontology["hypothesis_domain"]
        == [
            "source_logic",
            "interface_contract",
            "data_type_shape",
            "environment_dependency",
            "tooling_path",
            "test_fixture",
            "other",
        ]
        and ontology["action_intent"] == ["inspect", "edit", "validate"]
        and ontology["no_chain_class"] == "__no_novel_chain__",
        "ontology changed",
    )
    signature_registry(config)

    evidence = _mapping(
        config["annotation_evidence_contract"], "annotation evidence contract"
    )
    _require(
        evidence["raw_independent_records_retained"] is True
        and evidence["adjudicated_final_records_retained"] is True
        and evidence["all_1549_stable_materialized_rows_finally_available_or_explicit_unknown"]
        is True
        and evidence["fit_before_complete_evidence_and_gate_forbidden"] is True,
        "annotation evidence cannot prove blind agreement and adjudication",
    )
    gate = _mapping(config["support_and_agreement_gate"], "support gate")
    expected_binary_support = {
        "completion_has_chain": {
            "minimum_rows_per_class": 100,
            "minimum_tasks_per_class": 20,
            "minimum_repositories_per_class": 8,
            "minimum_rows_per_class_in_every_outer_training_fold": 1,
        },
        "primary_has_novel_chain": {
            "minimum_rows_per_class": 100,
            "minimum_tasks_per_class": 20,
            "minimum_repositories_per_class": 8,
            "minimum_rows_per_class_in_every_outer_training_fold": 1,
        },
    }
    expected_slot_support = {
        "evidence_kind": {
            "minimum_rows_per_class": 20,
            "minimum_tasks_per_class": 10,
            "minimum_repositories_per_class": 5,
            "minimum_rows_per_class_in_every_outer_training_fold": 1,
        },
        "belief_edge": {
            "minimum_rows_per_class": 40,
            "minimum_tasks_per_class": 20,
            "minimum_repositories_per_class": 8,
            "minimum_rows_per_class_in_every_outer_training_fold": 1,
        },
        "hypothesis_domain": {
            "minimum_rows_per_class": 10,
            "minimum_tasks_per_class": 5,
            "minimum_repositories_per_class": 4,
            "minimum_rows_per_class_in_every_outer_training_fold": 1,
        },
        "action_intent": {
            "minimum_rows_per_class": 20,
            "minimum_tasks_per_class": 10,
            "minimum_repositories_per_class": 5,
            "minimum_rows_per_class_in_every_outer_training_fold": 1,
        },
    }
    _require(
        gate["minimum_novel_positive_chains"] == 100
        and gate["minimum_novel_positive_chains_per_repository"] == 5
        and gate["minimum_each_belief_edge"] == 40
        and gate["minimum_repositories_per_belief_edge"] == 8
        and gate["minimum_completion_exact_graph_agreement"] == 0.75
        and gate["minimum_completion_has_chain_kappa"] == 0.7
        and gate["minimum_novelty_exact_agreement"] == 0.75
        and gate["minimum_novelty_kappa"] == 0.7
        and gate["binary_head_support"] == expected_binary_support
        and gate["conditional_slot_head_support"] == expected_slot_support
        and gate["per_head_failure_actions"]["agreement_failure"]
        == "do_not_fit_any_head"
        and gate["per_head_failure_actions"]["completion_binary_support_failure"]
        == "block_completion_has_chain_head_and_every_full_concept_chain_claim"
        and gate["per_head_failure_actions"]["primary_binary_support_failure"]
        == "block_primary_has_novel_chain_and_all_downstream_heads"
        and gate["per_head_failure_actions"]["belief_edge_support_failure"]
        == "block_original_belief_edge_and_exact_signature_heads_but_do_not_block_predeclared_binary_heads"
        and gate[
            "binary_heads_may_fit_when_their_own_agreement_and_support_gates_pass_even_if_an_edge_or_signature_gate_fails"
        ]
        is True
        and "proposition_content_lane_gate_passed"
        in gate["full_concept_chain_claim_requires"]
        and "predictive_full_concept_chain_gate_passed"
        in gate["full_concept_chain_claim_requires"],
        "support or agreement gate changed",
    )
    semantic = _mapping(
        config["full_prefix_semantic_baseline"], "prefix semantic baseline"
    )
    _require(
        semantic["name"] == "full_prefix_lsa"
        and semantic["source"] == "exact_current_rendered_prompt_text_h_t"
        and semantic["target_completion_or_later_text_forbidden"] is True
        and semantic["complete_prefix_not_suffix_only"] is True
        and semantic["heldout_prefix_text_used_for_fit"] is False
        and semantic["latent_semantic_projection"]["output_width"] == 256,
        "full-prefix semantic baseline changed",
    )
    _require(
        semantic["fold_transform_authentication"]
        == {
            "digest": "sha256_of_canonical_json_component_manifest",
            "binds": [
                "numpy_and_scikit_learn_implementation_versions",
                "all_frozen_vectorizer_parameters",
                "learned_vocabulary_term_to_column_mapping",
                "learned_feature_order",
                "learned_idf_float64_values",
                "all_frozen_svd_parameters",
                "learned_svd_components_float64_values",
                "learned_svd_singular_values_float64_values",
                "learned_svd_explained_variance_float64_values",
                "learned_svd_explained_variance_ratio_float64_values",
            ],
            "single_complete_fold_transform_sha256_required": True,
            "components_only_hash_is_insufficient": True,
        },
        "full-prefix fold-transform authentication contract changed",
    )

    embedder = _mapping(
        config["frozen_proposition_embedding_model"], "proposition embedder"
    )
    _require(
        embedder["repo_id"] == "sentence-transformers/all-MiniLM-L6-v2"
        and embedder["revision"]
        == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        and embedder["embedding_width"] == 384
        and embedder["pooling"]
        == "attention_mask_weighted_mean_of_last_hidden_state"
        and embedder["normalization"] == "row_l2"
        and embedder["embedding_model_never_finetuned"] is True,
        "frozen proposition embedder changed",
    )
    snapshot_files = _mapping(embedder["snapshot_files"], "snapshot files")
    _require(len(snapshot_files) == 9, "frozen embedder file registry changed")
    for name, record_value in snapshot_files.items():
        record = _mapping(record_value, f"embedder file {name}")
        _require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and set(record) == {"sha256", "size_bytes"}
            and _is_sha256(record["sha256"])
            and isinstance(record["size_bytes"], int)
            and not isinstance(record["size_bytes"], bool)
            and record["size_bytes"] > 0,
            f"frozen embedder file binding is invalid: {name}",
        )
    content = _mapping(config["proposition_content_lane"], "content lane")
    claim_gate = _mapping(content["claim_gate"], "content claim gate")
    expected_predictive_deltas = [
        {
            "metric": "factorized_full_signature_nll",
            "improvement_orientation": "reference_minus_candidate",
            "minimum_point_improvement_exclusive": 0.0,
            "minimum_multiplicity_adjusted_paired_interval_lower_bound_exclusive": 0.0,
        },
        {
            "metric": "true_H_cosine",
            "improvement_orientation": "candidate_minus_reference",
            "minimum_point_improvement_exclusive": 0.0,
            "minimum_multiplicity_adjusted_paired_interval_lower_bound_exclusive": 0.0,
        },
        {
            "metric": "training_only_retrieved_H_to_true_H_cosine",
            "improvement_orientation": "candidate_minus_reference",
            "minimum_point_improvement_exclusive": 0.0,
            "minimum_multiplicity_adjusted_paired_interval_lower_bound_exclusive": 0.0,
        },
    ]
    _require(
        content["status"] == "required_for_any_sentence_or_cot_like_content_claim"
        and content["targets"]["primary"]
        == "exact_hypothesis_span_H_text_embedding"
        and content["target_artifact"]["target_embeddings_never_model_inputs"]
        is True
        and claim_gate["coarse_slot_results_alone_never_sufficient"] is True
        and claim_gate["heldout_closed_set_retrieval_alone_never_sufficient"]
        is True
        and claim_gate["training_only_retrieval_and_continuous_cosine_required"]
        is True
        and claim_gate["full_refit_paired_intervals_required"] is True
        and claim_gate[
            "evaluation_completion_without_predictive_improvement_never_sufficient"
        ]
        is True
        and claim_gate[
            "one_predeclared_candidate_must_pass_every_predictive_requirement_without_mixing_candidates"
        ]
        is True
        and claim_gate["predictive_nested_delta_requirements"]
        == expected_predictive_deltas
        and claim_gate["paired_interval_evidence"]
        == {
            "algorithm": "hierarchical_bayesian_cluster_bootstrap_with_full_model_refit",
            "draw_count": 1000,
            "interval_level": 0.95,
            "multiplicity_adjustment_required_across_both_candidates_and_all_required_metrics": True,
            "multiplicity_method": "bonferroni_equal_tailed_percentile",
            "family_size": 6,
            "adjusted_lower_quantile": 0.004166666666666667,
            "adjusted_upper_quantile": 0.9958333333333333,
        }
        and claim_gate["absolute_semantic_quality_floors"]
        == {
            "hierarchical_weighted_true_H_cosine": 0.35,
            "hierarchical_weighted_training_only_retrieved_H_to_true_H_cosine": 0.35,
            "proposition_content_accepted_coverage": 0.2,
        }
        and claim_gate[
            "all_required_categorical_slot_heads_must_pass_for_full_concept_chain_claim"
        ]
        is True
        and claim_gate["literal_or_private_sentence_recovery_claim_forbidden"]
        is True,
        "proposition-content claim boundary changed",
    )

    blocks = dict(_mapping(config["numeric_feature_blocks"], "feature blocks"))
    _require(
        blocks
        == {
            "sequence_logit_j": 256,
            "raw_activation_current": 192,
            "public_j_activation_current": 192,
            "full_prefix_lsa": 256,
            "prior_visible_chain": 24,
        },
        "numeric feature block registry changed",
    )
    variants = config["variants"]
    _require(isinstance(variants, list) and len(variants) == 8, "variant registry changed")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in variants:
        variant = _mapping(raw, "variant")
        _require(
            set(variant) == {"name", "role", "components", "width"}
            and isinstance(variant["name"], str)
            and variant["name"] not in by_name
            and isinstance(variant["components"], list)
            and len(variant["components"]) == len(set(variant["components"]))
            and all(component in blocks for component in variant["components"])
            and variant["width"]
            == sum(blocks[component] for component in variant["components"]),
            "variant is invalid or its width is not additive",
        )
        by_name[str(variant["name"])] = variant
    comparisons = config["nested_comparisons"]
    _require(
        isinstance(comparisons, list) and len(comparisons) == 2,
        "nested comparison registry changed",
    )
    for raw in comparisons:
        comparison = _mapping(raw, "nested comparison")
        candidate = by_name[comparison["candidate"]]
        reference = by_name[comparison["reference"]]
        candidate_components = set(candidate["components"])
        reference_components = set(reference["components"])
        _require(
            reference_components < candidate_components
            and candidate_components - reference_components
            == {comparison["candidate_only_component"]}
            and candidate["width"]
            == reference["width"] + blocks[comparison["candidate_only_component"]],
            "activation comparison is not a strict one-block augmentation",
        )
    split = _mapping(config["split_and_weighting"], "split and weighting")
    _require(
        split["helper_fold_provenance"]
        == {
            "required_fields": [
                "fold_id",
                "training_source_ids",
                "heldout_source_ids",
            ],
            "source_id_format": "lowercase_sha256",
            "training_and_heldout_ids_unique_and_disjoint": True,
            "ridge_training_rows_must_bind_to_training_source_ids": True,
            "ridge_test_rows_must_bind_to_heldout_source_ids": True,
            "training_prototypes_must_bind_to_training_source_ids": True,
            "closed_set_queries_and_candidates_must_bind_to_heldout_source_ids": True,
            "provenance_claims_may_not_be_inferred_from_argument_names": True,
        },
        "helper fold-provenance contract changed",
    )
    uncertainty = _mapping(config["full_refit_uncertainty"], "uncertainty")
    _require(
        uncertainty["draw_count"] == 1000
        and uncertainty["algorithm"]
        == "hierarchical_bayesian_cluster_bootstrap_with_full_model_refit"
        and uncertainty["rows_within_task"]
        == "retain_complete_ordered_trajectory_equal_mass_no_row_resampling"
        and uncertainty["fixed_oof_prediction_bootstrap_for_primary_intervals_forbidden"]
        is True
        and uncertainty["all_draws_required_no_silent_downgrade"] is True,
        "full-refit uncertainty contract changed",
    )
    renderer = _mapping(config["renderer"], "renderer")
    claims = _mapping(config["claim_scope"], "claim scope")
    _require(
        renderer["coarse_renderer_may_be_called_sentence_or_cot_like_content_decoding"]
        is False
        and renderer["nearest_training_proposition_may_be_called_reconstructed_heldout_sentence"]
        is False
        and claims["visible_future_semantic_cot_like_chain_is_distinct_target"]
        is True
        and claims["visible_rationale_word_marker_is_not_substitute_for_chain_target"]
        is True
        and claims[
            "primary_novel_chain_metrics_condition_on_excluding_prefix_exposed_future_positives"
        ]
        is True
        and claims["affect_emotion_confidence_doubt_stress_are_separate_target_lane"]
        is True
        and claims["coarse_ontology_chain_classification_established"] is False
        and claims["proposition_embedding_or_training_only_retrieval_established"]
        is False
        and claims["private_internal_chain_of_thought_decoding_established"]
        is False
        and claims["subjective_emotion_confidence_doubt_or_stress_decoding_established"]
        is False,
        "claim boundary changed",
    )
    return config


def load_frozen_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    canonical = resolve_guarded_path(
        path,
        allowed_root=ROOT,
        reject_logical_symlink=True,
        require_kind="file",
    )
    _require(canonical == CONFIG_PATH, "decoder config path changed")
    _require(sha256_file(canonical) == CONFIG_SHA256, "decoder config hash changed")
    return validate_config(load_json_strict(canonical))


def authenticate_frozen_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Authenticate only the explicitly allowed development input bindings."""

    config = validate_config(config)
    records: dict[str, dict[str, Any]] = {}
    paths = [ROOT / record["path"] for record in config["frozen_inputs"].values()]
    lexical_path_preflight(paths)
    for label, record in config["frozen_inputs"].items():
        path = resolve_guarded_path(
            ROOT / record["path"],
            allowed_root=ROOT,
            reject_logical_symlink=True,
            require_kind="file",
        )
        digest = sha256_file(path)
        _require(digest == record["sha256"], f"frozen input hash changed: {label}")
        records[label] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    return records


def validate_model_snapshot(
    snapshot_path: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate the exact pre-annotation sentence-embedding snapshot."""

    config = validate_config(config)
    snapshot = resolve_guarded_path(
        snapshot_path,
        reject_logical_symlink=True,
        require_kind="directory",
    )
    expected = config["frozen_proposition_embedding_model"]["snapshot_files"]
    observed_names: list[str] = []
    for path in snapshot.rglob("*"):
        lexical_path_preflight([path])
        if path.is_symlink() or path.is_file():
            observed_names.append(str(path.relative_to(snapshot)))
    observed_names.sort()
    _require(observed_names == sorted(expected), "model snapshot file registry changed")
    result: dict[str, dict[str, Any]] = {}
    for name, record in expected.items():
        logical = snapshot / name
        target = resolve_guarded_path(
            logical,
            reject_logical_symlink=False,
            require_kind="file",
        )
        digest = sha256_file(target)
        size = target.stat().st_size
        _require(
            digest == record["sha256"] and size == record["size_bytes"],
            f"model snapshot file binding changed: {name}",
        )
        result[name] = {"sha256": digest, "size_bytes": size}
    return result


def _probability_matrix(
    value: Any, *, width: int, label: str
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(
        matrix.ndim == 2
        and matrix.shape[1] == width
        and len(matrix) > 0
        and bool(np.all(np.isfinite(matrix)))
        and bool(np.all(matrix > 0.0))
        and bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12)),
        f"{label} is not a strictly positive row-normalized probability matrix",
    )
    return matrix


def validate_fold_provenance(value: Any) -> dict[str, Any]:
    """Validate one explicit train/heldout source-ID partition."""

    record = dict(_mapping(value, "fold provenance"))
    _require(
        set(record) == {"fold_id", "training_source_ids", "heldout_source_ids"}
        and isinstance(record["fold_id"], str)
        and bool(record["fold_id"]),
        "fold provenance schema is invalid",
    )
    normalized: dict[str, list[str]] = {}
    for field in ("training_source_ids", "heldout_source_ids"):
        raw = record[field]
        _require(
            isinstance(raw, (list, tuple))
            and bool(raw)
            and all(_is_sha256(source_id) for source_id in raw)
            and len(raw) == len(set(raw)),
            f"fold provenance {field} must contain unique lowercase SHA-256 IDs",
        )
        normalized[field] = list(raw)
    _require(
        set(normalized["training_source_ids"]).isdisjoint(
            normalized["heldout_source_ids"]
        ),
        "fold provenance training and heldout source IDs overlap",
    )
    return {
        "fold_id": record["fold_id"],
        "training_source_ids": normalized["training_source_ids"],
        "heldout_source_ids": normalized["heldout_source_ids"],
    }


def _bound_source_ids(
    value: Sequence[str],
    *,
    expected_length: int,
    allowed_ids: set[str],
    label: str,
) -> list[str]:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == expected_length
        and all(_is_sha256(source_id) for source_id in value)
        and len(value) == len(set(value)),
        f"{label} row binding is invalid",
    )
    result = list(value)
    _require(set(result).issubset(allowed_ids), f"{label} escaped its fold partition")
    return result


def _source_ids_sha256(source_ids: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(source_ids))).hexdigest()


def evaluate_head_support(
    *,
    head_name: str,
    class_support: Mapping[str, Mapping[str, Any]],
    outer_training_fold_class_counts: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen per-class support gate for one model head."""

    config = validate_config(config)
    gate = config["support_and_agreement_gate"]
    if head_name in gate["binary_head_support"]:
        classes = ["no", "yes"]
        threshold = gate["binary_head_support"][head_name]
    else:
        _require(
            head_name in gate["conditional_slot_head_support"],
            f"unknown support-gated head: {head_name}",
        )
        classes = list(config["ontology"][head_name])
        threshold = gate["conditional_slot_head_support"][head_name]
    _require(set(class_support) == set(classes), "head class-support registry changed")
    _require(
        isinstance(outer_training_fold_class_counts, Mapping)
        and bool(outer_training_fold_class_counts),
        "outer training-fold support registry is empty",
    )
    failures: list[str] = []
    for class_name in classes:
        record = _mapping(class_support[class_name], f"{head_name} {class_name} support")
        _require(
            set(record) == {"rows", "tasks", "repositories"}
            and all(
                isinstance(record[field], int)
                and not isinstance(record[field], bool)
                and record[field] >= 0
                for field in ("rows", "tasks", "repositories")
            ),
            f"{head_name} {class_name} support record is invalid",
        )
        for field, threshold_name in (
            ("rows", "minimum_rows_per_class"),
            ("tasks", "minimum_tasks_per_class"),
            ("repositories", "minimum_repositories_per_class"),
        ):
            if record[field] < threshold[threshold_name]:
                failures.append(f"{class_name}:{field}")
    for fold_id, raw_counts in outer_training_fold_class_counts.items():
        _require(isinstance(fold_id, str) and bool(fold_id), "outer fold ID is invalid")
        counts = _mapping(raw_counts, f"outer fold {fold_id} support")
        _require(set(counts) == set(classes), "outer fold class registry changed")
        for class_name in classes:
            count = counts[class_name]
            _require(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                "outer fold class count is invalid",
            )
            if count < threshold[
                "minimum_rows_per_class_in_every_outer_training_fold"
            ]:
                failures.append(f"{fold_id}:{class_name}:training_rows")
    return {
        "head_name": head_name,
        "classes": classes,
        "passed": not failures,
        "failures": failures,
        "thresholds": dict(threshold),
    }


def evaluate_predictive_claim_gate(
    *,
    candidate_name: str,
    absolute_metrics: Mapping[str, Any],
    paired_deltas: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require actual semantic quality and positive full-refit nested deltas.

    Passing this gate remains evidence only for the future-visible semantic
    target.  It can never promote a result to literal sentence recovery or
    private chain-of-thought access.
    """

    config = validate_config(config)
    candidates = {
        comparison["candidate"]: comparison["reference"]
        for comparison in config["nested_comparisons"]
    }
    _require(candidate_name in candidates, "claim candidate is not predeclared")
    reference_name = candidates[candidate_name]
    claim = config["proposition_content_lane"]["claim_gate"]
    requirements = claim["predictive_nested_delta_requirements"]
    expected_metrics = [requirement["metric"] for requirement in requirements]
    _require(
        set(paired_deltas) == set(expected_metrics),
        "predictive paired-delta registry changed",
    )
    floors = claim["absolute_semantic_quality_floors"]
    _require(
        set(absolute_metrics) == set(floors),
        "absolute semantic-quality metric registry changed",
    )
    finite_absolute: dict[str, float] = {}
    failures: list[str] = []
    for metric, floor in floors.items():
        value = absolute_metrics[metric]
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"absolute semantic metric is invalid: {metric}",
        )
        numeric = float(value)
        if metric.endswith("coverage"):
            _require(0.0 <= numeric <= 1.0, f"coverage is out of range: {metric}")
        else:
            _require(-1.0 <= numeric <= 1.0, f"cosine is out of range: {metric}")
        finite_absolute[metric] = numeric
        if numeric < float(floor):
            failures.append(f"absolute_floor:{metric}")
    interval = claim["paired_interval_evidence"]
    normalized_deltas: dict[str, dict[str, Any]] = {}
    for requirement in requirements:
        metric = requirement["metric"]
        record = dict(_mapping(paired_deltas[metric], f"paired delta {metric}"))
        _require(
            set(record)
            == {
                "candidate",
                "reference",
                "improvement_orientation",
                "point",
                "interval_lower",
                "interval_upper",
                "interval_level",
                "algorithm",
                "draw_count",
                "multiplicity_adjusted",
                "multiplicity_method",
                "multiplicity_family_size",
            }
            and record["candidate"] == candidate_name
            and record["reference"] == reference_name
            and record["improvement_orientation"]
            == requirement["improvement_orientation"]
            and record["algorithm"] == interval["algorithm"]
            and record["draw_count"] == interval["draw_count"]
            and record["interval_level"] == interval["interval_level"]
            and record["multiplicity_adjusted"] is True
            and record["multiplicity_method"] == interval["multiplicity_method"]
            and record["multiplicity_family_size"] == interval["family_size"],
            f"paired delta provenance is invalid: {metric}",
        )
        numeric = {}
        for field in ("point", "interval_lower", "interval_upper"):
            value = record[field]
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"paired delta {metric} {field} is invalid",
            )
            numeric[field] = float(value)
        _require(
            numeric["interval_lower"] <= numeric["interval_upper"],
            f"paired delta interval is reversed: {metric}",
        )
        if numeric["point"] <= float(
            requirement["minimum_point_improvement_exclusive"]
        ):
            failures.append(f"point_improvement:{metric}")
        if numeric["interval_lower"] <= float(
            requirement[
                "minimum_multiplicity_adjusted_paired_interval_lower_bound_exclusive"
            ]
        ):
            failures.append(f"paired_interval_lower:{metric}")
        normalized_deltas[metric] = {
            **record,
            **numeric,
        }
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "passed": not failures,
        "failures": failures,
        "absolute_metrics": finite_absolute,
        "paired_deltas": normalized_deltas,
        "private_or_literal_sentence_recovery_established": False,
        "future_visible_semantic_target_only": True,
    }


def factorized_exact_predictions(
    *,
    has_novel_chain_probabilities: Any,
    slot_probabilities: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Render coarse structural predictions and bottleneck confidence.

    This function cannot emit or reconstruct proposition text.  It returns only
    the frozen no-chain token or a four-slot ontology signature.
    """

    config = validate_config(config)
    chain = _probability_matrix(
        has_novel_chain_probabilities,
        width=2,
        label="has-novel-chain probabilities",
    )
    ontology = config["ontology"]
    slot_order = (
        "evidence_kind",
        "belief_edge",
        "hypothesis_domain",
        "action_intent",
    )
    slots: dict[str, np.ndarray] = {}
    for name in slot_order:
        classes = ontology[name]
        _require(name in slot_probabilities, f"slot probability is absent: {name}")
        slots[name] = _probability_matrix(
            slot_probabilities[name], width=len(classes), label=f"{name} probabilities"
        )
        _require(len(slots[name]) == len(chain), "slot probabilities are row-misaligned")
    _require(set(slot_probabilities) == set(slot_order), "slot probability registry changed")

    labels: list[str] = []
    confidence = np.empty(len(chain), dtype=np.float64)
    chain_indices = np.argmax(chain, axis=1)
    slot_indices = {name: np.argmax(slots[name], axis=1) for name in slot_order}
    no_chain = ontology["no_chain_class"]
    for row in range(len(chain)):
        selected_chain_probability = float(chain[row, chain_indices[row]])
        if int(chain_indices[row]) == 0:
            labels.append(no_chain)
            confidence[row] = selected_chain_probability
            continue
        selected = {
            name: ontology[name][int(slot_indices[name][row])] for name in slot_order
        }
        labels.append(
            ">".join(
                (
                    selected["evidence_kind"],
                    selected["belief_edge"],
                    selected["hypothesis_domain"],
                    "motivates",
                    selected["action_intent"],
                )
            )
        )
        confidence[row] = min(
            [selected_chain_probability]
            + [
                float(slots[name][row, slot_indices[name][row]])
                for name in slot_order
            ]
        )
    _require(
        set(labels).issubset({no_chain, *signature_registry(config)})
        and bool(np.all((confidence > 0.0) & (confidence <= 1.0))),
        "factorized prediction escaped the frozen output registry",
    )
    return {
        "labels": labels,
        "confidence": confidence,
        "has_novel_chain_class_index": chain_indices,
        "slot_class_indices": slot_indices,
        "coarse_structure_only_not_sentence_or_cot_like_content": True,
    }


def build_variant_matrices(
    blocks: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Concatenate only frozen numeric blocks; grouping fields are impossible."""

    config = validate_config(config)
    widths = config["numeric_feature_blocks"]
    _require(set(blocks) == set(widths), "feature block registry changed")
    arrays: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for name, width in widths.items():
        matrix = np.asarray(blocks[name], dtype=np.float64)
        _require(
            matrix.ndim == 2
            and matrix.shape[1] == width
            and bool(np.all(np.isfinite(matrix))),
            f"numeric feature block is invalid: {name}",
        )
        if row_count is None:
            row_count = len(matrix)
        _require(len(matrix) == row_count, "numeric feature blocks are row-misaligned")
        arrays[name] = matrix
    _require(row_count is not None and row_count > 0, "feature blocks are empty")
    variants: dict[str, np.ndarray] = {}
    for spec in config["variants"]:
        matrix = np.concatenate([arrays[name] for name in spec["components"]], axis=1)
        _require(
            matrix.shape == (row_count, spec["width"]),
            f"variant shape changed: {spec['name']}",
        )
        variants[spec["name"]] = matrix
    return variants


def hierarchical_bayesian_weights(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> np.ndarray:
    """Draw repo->task weights while retaining every ordered task trajectory."""

    _require(
        bool(rows) and isinstance(seed, int) and not isinstance(seed, bool),
        "hierarchical draw inputs are invalid",
    )
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, row in enumerate(rows):
        repository = row.get("repository")
        task = row.get("task_id_sha256")
        _require(
            isinstance(repository, str)
            and bool(repository)
            and isinstance(task, str)
            and bool(task),
            "hierarchical grouping row is invalid",
        )
        grouped[repository][task].append(position)
    repositories = sorted(grouped)
    _require(len(repositories) >= 2, "hierarchical draw requires two repositories")
    rng = np.random.default_rng(seed)
    repo_draw = rng.exponential(1.0, len(repositories))
    repo_draw /= repo_draw.sum()
    weights = np.zeros(len(rows), dtype=np.float64)
    for repo_position, repository in enumerate(repositories):
        tasks = sorted(grouped[repository])
        task_draw = rng.exponential(1.0, len(tasks))
        task_draw /= task_draw.sum()
        for task_position, task in enumerate(tasks):
            indices = grouped[repository][task]
            mass = repo_draw[repo_position] * task_draw[task_position]
            weights[np.asarray(indices, dtype=np.int64)] = mass / len(indices)
    _require(
        bool(np.all(weights > 0.0))
        and bool(np.all(np.isfinite(weights)))
        and abs(float(weights.sum()) - 1.0) <= 1e-12,
        "hierarchical Bayesian weights are invalid",
    )
    return weights


def full_prefix_lsa_transform_fingerprint(
    *,
    vocabulary: Mapping[str, Any],
    idf: Any,
    svd_components: Any,
    svd_singular_values: Any,
    svd_explained_variance: Any,
    svd_explained_variance_ratio: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every fitted value and frozen parameter needed by the LSA transform."""

    config = validate_config(config)
    _require(isinstance(vocabulary, Mapping) and bool(vocabulary), "LSA vocabulary is empty")
    entries: list[tuple[str, int]] = []
    for term, raw_column in vocabulary.items():
        _require(
            isinstance(term, str)
            and bool(term)
            and isinstance(raw_column, (int, np.integer))
            and not isinstance(raw_column, bool)
            and raw_column >= 0,
            "LSA vocabulary entry is invalid",
        )
        entries.append((term, int(raw_column)))
    entries.sort(key=lambda item: item[1])
    _require(
        [column for _, column in entries] == list(range(len(entries)))
        and len({term for term, _ in entries}) == len(entries),
        "LSA vocabulary columns are not a unique contiguous feature order",
    )
    feature_order = [term for term, _ in entries]
    mapping_records = [
        {"term": term, "column": column} for term, column in entries
    ]
    width = int(
        config["full_prefix_semantic_baseline"]["latent_semantic_projection"]
        ["n_components"]
    )
    feature_count = len(entries)
    arrays = {
        "idf": np.asarray(idf, dtype="<f8"),
        "svd_components": np.asarray(svd_components, dtype="<f8"),
        "svd_singular_values": np.asarray(svd_singular_values, dtype="<f8"),
        "svd_explained_variance": np.asarray(
            svd_explained_variance, dtype="<f8"
        ),
        "svd_explained_variance_ratio": np.asarray(
            svd_explained_variance_ratio, dtype="<f8"
        ),
    }
    _require(
        arrays["idf"].shape == (feature_count,)
        and arrays["svd_components"].shape == (width, feature_count)
        and arrays["svd_singular_values"].shape == (width,)
        and arrays["svd_explained_variance"].shape == (width,)
        and arrays["svd_explained_variance_ratio"].shape == (width,)
        and all(bool(np.all(np.isfinite(array))) for array in arrays.values())
        and bool(np.all(arrays["idf"] > 0.0))
        and bool(np.all(arrays["svd_singular_values"] >= 0.0))
        and bool(np.all(arrays["svd_explained_variance"] >= 0.0))
        and bool(np.all(arrays["svd_explained_variance_ratio"] >= 0.0)),
        "LSA fitted transform state is invalid",
    )
    contract = config["full_prefix_semantic_baseline"]
    try:
        import sklearn
    except ImportError as error:
        raise EpistemicDecoderError(
            "scikit-learn is required for the LSA transform fingerprint"
        ) from error
    manifest = {
        "schema_version": 1,
        "implementation_versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "vectorizer_parameters": dict(contract["vectorizer"]),
        "vocabulary_term_to_column_sha256": hashlib.sha256(
            canonical_json_bytes(mapping_records)
        ).hexdigest(),
        "feature_order_sha256": hashlib.sha256(
            canonical_json_bytes(feature_order)
        ).hexdigest(),
        "idf_sha256": sha256_array(arrays["idf"]),
        "svd_parameters": dict(contract["latent_semantic_projection"]),
        "svd_components_sha256": sha256_array(arrays["svd_components"]),
        "svd_singular_values_sha256": sha256_array(
            arrays["svd_singular_values"]
        ),
        "svd_explained_variance_sha256": sha256_array(
            arrays["svd_explained_variance"]
        ),
        "svd_explained_variance_ratio_sha256": sha256_array(
            arrays["svd_explained_variance_ratio"]
        ),
        "feature_count": feature_count,
        "component_count": width,
    }
    return {
        "fold_transform_sha256": hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest(),
        "component_manifest": manifest,
    }


def fit_full_prefix_lsa(
    *,
    training_texts: Sequence[str],
    heldout_texts: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the frozen full-prefix semantic transform on training text only."""

    config = validate_config(config)
    _require(
        len(training_texts) > 1
        and len(heldout_texts) > 0
        and all(isinstance(text, str) for text in (*training_texts, *heldout_texts)),
        "prefix semantic text inputs are invalid",
    )
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as error:
        raise EpistemicDecoderError("scikit-learn is required for full-prefix LSA") from error
    contract = config["full_prefix_semantic_baseline"]
    vector = contract["vectorizer"]
    projector = contract["latent_semantic_projection"]
    vectorizer = TfidfVectorizer(
        input=vector["input"],
        encoding=vector["encoding"],
        decode_error=vector["decode_error"],
        analyzer=vector["analyzer"],
        ngram_range=tuple(vector["ngram_range"]),
        lowercase=vector["lowercase"],
        strip_accents=vector["strip_accents"],
        preprocessor=vector["preprocessor"],
        tokenizer=vector["tokenizer"],
        stop_words=vector["stop_words"],
        token_pattern=vector["token_pattern"],
        min_df=vector["min_df"],
        max_df=vector["max_df"],
        max_features=vector["max_features"],
        vocabulary=vector["vocabulary"],
        binary=vector["binary"],
        sublinear_tf=vector["sublinear_tf"],
        norm=vector["norm"],
        use_idf=vector["use_idf"],
        smooth_idf=vector["smooth_idf"],
        dtype=np.float64,
    )
    training_tfidf = vectorizer.fit_transform(training_texts)
    heldout_tfidf = vectorizer.transform(heldout_texts)
    width = int(projector["n_components"])
    _require(
        training_tfidf.shape[0] > width
        and training_tfidf.shape[1] > width,
        "training fold has insufficient rank for frozen 256-wide LSA",
    )
    svd = TruncatedSVD(
        n_components=width,
        algorithm=projector["algorithm"],
        n_iter=projector["n_iter"],
        n_oversamples=projector["n_oversamples"],
        power_iteration_normalizer=projector["power_iteration_normalizer"],
        random_state=projector["random_state"],
        tol=projector["tol"],
    )
    training = np.asarray(svd.fit_transform(training_tfidf), dtype=np.float64)
    heldout = np.asarray(svd.transform(heldout_tfidf), dtype=np.float64)
    _require(
        training.shape == (len(training_texts), width)
        and heldout.shape == (len(heldout_texts), width)
        and bool(np.all(np.isfinite(training)))
        and bool(np.all(np.isfinite(heldout))),
        "prefix LSA transform is invalid",
    )
    fingerprint = full_prefix_lsa_transform_fingerprint(
        vocabulary=vectorizer.vocabulary_,
        idf=vectorizer.idf_,
        svd_components=svd.components_,
        svd_singular_values=svd.singular_values_,
        svd_explained_variance=svd.explained_variance_,
        svd_explained_variance_ratio=svd.explained_variance_ratio_,
        config=config,
    )
    diagnostics = {
        "training_row_count": len(training_texts),
        "heldout_row_count": len(heldout_texts),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "components_sha256": fingerprint["component_manifest"][
            "svd_components_sha256"
        ],
        "fold_transform_sha256": fingerprint["fold_transform_sha256"],
        "fold_transform_component_manifest": fingerprint["component_manifest"],
        "heldout_text_used_for_fit": False,
    }
    return training, heldout, diagnostics


def fit_predict_weighted_proposition_ridge(
    *,
    X_train: Any,
    target_embeddings_train: Any,
    training_weights: Any,
    X_test: Any,
    X_train_source_ids: Sequence[str],
    target_source_ids_train: Sequence[str],
    X_test_source_ids: Sequence[str],
    fold_provenance: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the predeclared multioutput ridge and normalize each span block."""

    config = validate_config(config)
    X = np.asarray(X_train, dtype=np.float64)
    targets = np.asarray(target_embeddings_train, dtype=np.float64)
    weights = np.asarray(training_weights, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)
    provenance = validate_fold_provenance(fold_provenance)
    embedding_width = int(
        config["frozen_proposition_embedding_model"]["embedding_width"]
    )
    _require(
        X.ndim == targets.ndim == test.ndim == 2
        and weights.ndim == 1
        and len(X) == len(targets) == len(weights) > 0
        and X.shape[1] == test.shape[1]
        and targets.shape[1] in {embedding_width, 3 * embedding_width}
        and bool(np.all(np.isfinite(X)))
        and bool(np.all(np.isfinite(targets)))
        and bool(np.all(np.isfinite(test)))
        and bool(np.all(np.isfinite(weights)))
        and bool(np.all(weights > 0.0))
        and abs(float(weights.sum()) - 1.0) <= 1e-12,
        "weighted proposition ridge inputs are invalid",
    )
    allowed_training = set(provenance["training_source_ids"])
    allowed_heldout = set(provenance["heldout_source_ids"])
    feature_training_ids = _bound_source_ids(
        X_train_source_ids,
        expected_length=len(X),
        allowed_ids=allowed_training,
        label="ridge training feature",
    )
    target_training_ids = _bound_source_ids(
        target_source_ids_train,
        expected_length=len(targets),
        allowed_ids=allowed_training,
        label="ridge training target",
    )
    test_ids = _bound_source_ids(
        X_test_source_ids,
        expected_length=len(test),
        allowed_ids=allowed_heldout,
        label="ridge heldout feature",
    )
    _require(
        feature_training_ids == target_training_ids,
        "ridge training features and target embeddings are row-misaligned",
    )
    _require(
        set(feature_training_ids).isdisjoint(test_ids),
        "ridge fit and test source IDs overlap",
    )
    mean = weights @ X
    centered = X - mean
    variance = np.maximum(weights @ (centered * centered), 0.0)
    scale = np.sqrt(variance)
    constant = scale <= np.finfo(np.float64).eps
    scale[constant] = 1.0
    standardized_train = centered / scale
    standardized_test = (test - mean) / scale
    try:
        from sklearn.linear_model import Ridge
    except ImportError as error:
        raise EpistemicDecoderError(
            "scikit-learn is required for proposition ridge"
        ) from error
    contract = config["proposition_content_lane"]["regressor"]
    model = Ridge(
        alpha=float(contract["alpha"]),
        fit_intercept=bool(contract["fit_intercept"]),
        solver=str(contract["solver"]),
    )
    model.fit(
        standardized_train,
        targets,
        sample_weight=weights * len(weights),
    )
    predicted = np.asarray(model.predict(standardized_test), dtype=np.float64)
    _require(
        predicted.shape == (len(test), targets.shape[1])
        and bool(np.all(np.isfinite(predicted))),
        "weighted proposition ridge prediction is invalid",
    )
    blocks = []
    for start in range(0, predicted.shape[1], embedding_width):
        blocks.append(
            _l2_normalize(
                predicted[:, start : start + embedding_width],
                label=f"predicted proposition block {start // embedding_width}",
            )
        )
    normalized = np.concatenate(blocks, axis=1)
    return normalized, {
        "alpha": float(contract["alpha"]),
        "solver": str(contract["solver"]),
        "training_row_count": len(X),
        "test_row_count": len(test),
        "target_width": targets.shape[1],
        "span_block_count": len(blocks),
        "constant_feature_count": int(constant.sum()),
        "standardizer_mean_sha256": sha256_array(mean.astype("<f8")),
        "standardizer_scale_sha256": sha256_array(scale.astype("<f8")),
        "coefficient_sha256": sha256_array(
            np.asarray(model.coef_, dtype="<f8")
        ),
        "intercept_sha256": sha256_array(
            np.atleast_1d(np.asarray(model.intercept_, dtype="<f8"))
        ),
        "heldout_targets_used_for_fit": False,
        "fold_id": provenance["fold_id"],
        "training_source_ids_sha256": _source_ids_sha256(feature_training_ids),
        "heldout_source_ids_sha256": _source_ids_sha256(test_ids),
        "fold_provenance_enforced": True,
    }


def _l2_normalize(value: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(
        matrix.ndim == 2
        and len(matrix) > 0
        and matrix.shape[1] > 0
        and bool(np.all(np.isfinite(matrix))),
        f"{label} embedding matrix is invalid",
    )
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    _require(bool(np.all(norm > 0.0)), f"{label} contains a zero embedding")
    normalized = matrix / norm
    _require(
        bool(np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-12)),
        f"{label} normalization failed",
    )
    return normalized


def training_prototype_retrieval(
    *,
    predicted_embeddings: Any,
    training_embeddings: Any,
    candidate_keys: Sequence[str],
    predicted_source_ids: Sequence[str],
    candidate_source_ids: Sequence[str],
    fold_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Retrieve a fold-authenticated training candidate for heldout queries."""

    predicted = _l2_normalize(predicted_embeddings, label="predicted")
    candidates = _l2_normalize(training_embeddings, label="training candidate")
    provenance = validate_fold_provenance(fold_provenance)
    _require(
        predicted.shape[1] == candidates.shape[1]
        and len(candidate_keys) == len(candidates)
        and len(candidate_keys) == len(set(candidate_keys))
        and all(isinstance(key, str) and bool(key) for key in candidate_keys),
        "training prototype registry is invalid",
    )
    query_ids = _bound_source_ids(
        predicted_source_ids,
        expected_length=len(predicted),
        allowed_ids=set(provenance["heldout_source_ids"]),
        label="training-retrieval heldout query",
    )
    candidate_ids = _bound_source_ids(
        candidate_source_ids,
        expected_length=len(candidates),
        allowed_ids=set(provenance["training_source_ids"]),
        label="training-retrieval candidate",
    )
    _require(
        set(query_ids).isdisjoint(candidate_ids),
        "training retrieval query and candidate source IDs overlap",
    )
    order = np.asarray(sorted(range(len(candidate_keys)), key=lambda i: candidate_keys[i]))
    ordered_candidates = candidates[order]
    similarities = predicted @ ordered_candidates.T
    selected_ordered = np.argmax(similarities, axis=1)
    selected = order[selected_ordered]
    selected_cosine = similarities[np.arange(len(predicted)), selected_ordered]
    return {
        "candidate_indices": selected.astype(np.int64),
        "candidate_keys": [candidate_keys[int(index)] for index in selected],
        "cosine": selected_cosine.astype(np.float64),
        "candidate_pool_scope": "training_only",
        "fold_id": provenance["fold_id"],
        "query_source_ids_sha256": _source_ids_sha256(query_ids),
        "candidate_source_ids_sha256": _source_ids_sha256(candidate_ids),
        "fold_provenance_enforced": True,
    }


def heldout_closed_set_ranks(
    *,
    predicted_embeddings: Any,
    candidate_embeddings: Any,
    candidate_keys: Sequence[str],
    true_keys: Sequence[str],
    predicted_source_ids: Sequence[str],
    candidate_source_ids: Sequence[str],
    fold_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a declared future-derived heldout candidate audit without fitting."""

    predicted = _l2_normalize(predicted_embeddings, label="predicted")
    candidates = _l2_normalize(candidate_embeddings, label="heldout candidate")
    provenance = validate_fold_provenance(fold_provenance)
    _require(
        predicted.shape[1] == candidates.shape[1]
        and len(candidate_keys) == len(candidates)
        and len(true_keys) == len(predicted)
        and len(candidate_keys) == len(set(candidate_keys))
        and all(isinstance(key, str) and bool(key) for key in candidate_keys)
        and all(isinstance(key, str) and bool(key) for key in true_keys)
        and set(true_keys).issubset(set(candidate_keys)),
        "heldout closed-set candidate registry is invalid",
    )
    heldout_ids = set(provenance["heldout_source_ids"])
    query_ids = _bound_source_ids(
        predicted_source_ids,
        expected_length=len(predicted),
        allowed_ids=heldout_ids,
        label="closed-set heldout query",
    )
    candidate_ids = _bound_source_ids(
        candidate_source_ids,
        expected_length=len(candidates),
        allowed_ids=heldout_ids,
        label="closed-set heldout candidate",
    )
    similarities = predicted @ candidates.T
    ranks = np.empty(len(predicted), dtype=np.int64)
    for row, true_key in enumerate(true_keys):
        ordered = sorted(
            range(len(candidate_keys)),
            key=lambda index: (-float(similarities[row, index]), candidate_keys[index]),
        )
        ranks[row] = 1 + next(
            position
            for position, candidate_index in enumerate(ordered)
            if candidate_keys[candidate_index] == true_key
        )
    return {
        "ranks": ranks,
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "candidate_pool_is_future_derived": True,
        "open_ended_sentence_generation": False,
        "fold_id": provenance["fold_id"],
        "query_source_ids_sha256": _source_ids_sha256(query_ids),
        "candidate_source_ids_sha256": _source_ids_sha256(candidate_ids),
        "fold_provenance_enforced": True,
    }


def embed_span_texts(
    texts: Sequence[str], *, snapshot_path: Path, config: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed exact span text with the pre-annotation, hash-bound encoder."""

    config = validate_config(config)
    validate_model_snapshot(snapshot_path, config)
    _require(
        bool(texts) and all(isinstance(text, str) and bool(text) for text in texts),
        "span texts must be nonempty strings",
    )
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise EpistemicDecoderError(
            "torch and transformers are required for span embedding"
        ) from error
    contract = config["frozen_proposition_embedding_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path, local_files_only=True
    )
    model = AutoModel.from_pretrained(snapshot_path, local_files_only=True)
    model.eval()
    encoded = tokenizer(
        list(texts),
        add_special_tokens=contract["tokenization"]["add_special_tokens"],
        padding=contract["tokenization"]["padding"],
        truncation=contract["tokenization"]["truncation"],
        max_length=contract["tokenization"]["max_length"],
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.shape).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    embeddings = normalized.detach().cpu().numpy().astype("<f4", copy=False)
    _require(
        embeddings.shape == (len(texts), contract["embedding_width"])
        and bool(np.all(np.isfinite(embeddings)))
        and bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)),
        "frozen span embeddings are invalid",
    )
    untruncated = tokenizer(
        list(texts), add_special_tokens=True, padding=False, truncation=False
    )["input_ids"]
    token_counts = [len(tokens) for tokens in untruncated]
    return embeddings, {
        "embedding_width": embeddings.shape[1],
        "token_counts": token_counts,
        "truncated": [count > contract["tokenization"]["max_length"] for count in token_counts],
        "embedding_sha256": sha256_array(embeddings),
        "model_finetuned": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--authenticate-frozen-inputs", action="store_true")
    parser.add_argument("--model-snapshot", type=Path)
    args = parser.parse_args(argv)

    # This is intentionally the first filesystem-sensitive operation.
    lexical_path_preflight([args.config, args.model_snapshot])
    config = load_frozen_config(args.config)
    if args.authenticate_frozen_inputs:
        authenticate_frozen_inputs(config)
    if args.model_snapshot is not None:
        validate_model_snapshot(args.model_snapshot, config)
    print(
        json.dumps(
            {
                "status": "passed_pre_label_contract_only",
                "config_sha256": CONFIG_SHA256,
                "signature_count": len(signature_registry(config)),
                "target_annotations_opened": False,
                "coarse_slots_are_sentence_recovery": False,
                "proposition_content_lane_required": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "CONFIG_SHA256",
    "EpistemicDecoderError",
    "authenticate_frozen_inputs",
    "build_variant_matrices",
    "embed_span_texts",
    "evaluate_head_support",
    "evaluate_predictive_claim_gate",
    "factorized_exact_predictions",
    "fit_full_prefix_lsa",
    "fit_predict_weighted_proposition_ridge",
    "full_prefix_lsa_transform_fingerprint",
    "heldout_closed_set_ranks",
    "hierarchical_bayesian_weights",
    "load_frozen_config",
    "resolve_guarded_path",
    "signature_registry",
    "training_prototype_retrieval",
    "validate_config",
    "validate_fold_provenance",
    "validate_model_snapshot",
]
