#!/usr/bin/env python3
"""Hash-bound V3 replay extraction for V4 causal sequence features.

V4 keeps V3's exact prospective same-request target, eligibility, provenance,
and prompt/report alignment.  The only extraction change is that the original
96-wide ordinary-logit and public-J vectors are retained alongside V3's exact
14-wide pre-current-action history.  After all stable, history-complete rows
have been collected, the V4 feature builder is called exactly once.  Unknown
action rows therefore update temporal sensor state, while numerically unstable
rows never reach that state machine.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V3_ANALYZER_PATH = ROOT / "scripts/analyze_swe_task_state_v3.py"
V3_ANALYZER_SHA256 = (
    "53c7d41688f6c5ab21f7ad029d343af06e9b13c777fd2e5517ff8d5254ad9e6c"
)
V3_MODULE_NAME = "scripts.analyze_swe_task_state_v3"
RAW_SENSOR_WIDTH = 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_v3(
    *, expected_sha256: str = V3_ANALYZER_SHA256
) -> Any:
    """Load V3 under its canonical module name only after exact byte checks."""

    if not V3_ANALYZER_PATH.is_file() or V3_ANALYZER_PATH.is_symlink():
        raise ValueError(f"frozen V3 analyzer is not a regular file: {V3_ANALYZER_PATH}")
    observed = sha256_file(V3_ANALYZER_PATH)
    if observed != expected_sha256:
        raise ValueError("frozen V3 analyzer SHA-256 changed")

    existing = sys.modules.get(V3_MODULE_NAME)
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_path != V3_ANALYZER_PATH.resolve():
            raise ValueError("canonical V3 module name resolves to an unexpected path")
        if sha256_file(existing_path) != expected_sha256:
            raise ValueError("loaded V3 analyzer SHA-256 changed")
        return existing

    specification = importlib.util.spec_from_file_location(
        V3_MODULE_NAME, V3_ANALYZER_PATH
    )
    if specification is None or specification.loader is None:
        raise ValueError("could not construct the frozen V3 analyzer import")
    module = importlib.util.module_from_spec(specification)
    sys.modules[V3_MODULE_NAME] = module
    try:
        specification.loader.exec_module(module)
        if sha256_file(V3_ANALYZER_PATH) != expected_sha256:
            raise ValueError("frozen V3 analyzer changed while it was imported")
    except BaseException:
        if sys.modules.get(V3_MODULE_NAME) is module:
            sys.modules.pop(V3_MODULE_NAME, None)
        raise
    return module


# Load the canonical dependency before importing the feature module, whose
# compact summaries deliberately delegate to this exact V3 implementation.
V3 = _load_pinned_v3()

try:
    from scripts import swe_task_state_v4_features as V4_FEATURES
except ModuleNotFoundError as error:  # Support direct execution from scripts/.
    if error.name != "scripts":
        raise
    import swe_task_state_v4_features as V4_FEATURES  # type: ignore[no-redef]


def _validate_feature_dependency() -> None:
    dependency = getattr(V4_FEATURES, "_V3", None)
    dependency_path = Path(str(getattr(dependency, "__file__", ""))).resolve()
    if dependency_path != V3_ANALYZER_PATH.resolve():
        raise ValueError("V4 feature compact summaries do not resolve to frozen V3")
    if sha256_file(dependency_path) != V3_ANALYZER_SHA256:
        raise ValueError("V4 feature compact-summary V3 dependency changed")


_validate_feature_dependency()


def _extract_aligned_stable_rows(
    prompts_for_context: Sequence[Mapping[str, Any]],
    aligned_pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    prompt_count: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract V3-identical eligible rows while retaining raw sensor vectors."""

    # These V3 helpers are deliberately separate from feature construction.
    auxiliary_by_id = V3.auxiliary_diagnostic_labels(prompts_for_context)
    history_by_id, history_coverage = V3.causal_history_features(
        prompts_for_context
    )
    raw_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    stable_count = 0
    processed_count = 0
    known_source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for prompt, experiment in aligned_pairs:
        processed_count += 1
        prompt_id = str(prompt["id"])
        V3.require(
            experiment.get("prompt") == prompt.get("text")
            and experiment.get("prompt_token_ids") == prompt.get("token_ids")
            and experiment.get("metadata") == prompt.get("metadata"),
            f"{prompt_id} report payload is not bound to supplied prompt",
        )
        prompt_token_ids = V3.sequence(
            prompt.get("token_ids"), f"{prompt_id} token IDs"
        )
        V3.require(bool(prompt_token_ids), f"{prompt_id} token IDs are empty")
        expected_position = len(prompt_token_ids) - 1
        V3.require(
            experiment.get("capture_positions_resolved") == [expected_position],
            f"{prompt_id} was not captured only at the final prompt token",
        )
        scored = V3.mapping(
            experiment.get("scored_vocabulary"),
            f"{prompt_id} scored vocabulary",
        )
        V3.require(
            scored.get("token_ids") == prompt.get("score_token_ids"),
            f"{prompt_id} scored vocabulary differs from prompt contract",
        )
        stable, stability_reasons = V3.HISTORICAL_V1._numerically_stable(
            experiment, protocol["eligibility"]
        )
        if not stable:
            exclusion_counts["numerically_unstable"] += 1
            exclusions.append(
                {
                    "row_id": prompt_id,
                    "reason": "numerically_unstable",
                    "details": stability_reasons,
                }
            )
            continue
        stable_count += 1
        history = history_by_id.get(prompt_id)
        if history is None:
            exclusion_counts["causal_history_unavailable"] += 1
            exclusions.append(
                {
                    "row_id": prompt_id,
                    "reason": "causal_history_unavailable",
                    "details": ["complete consecutive probe bundle required"],
                }
            )
            continue

        ordinary = V3.HISTORICAL_V1._layer_class_features(
            experiment,
            layers=protocol["layers"],
            class_ids=protocol["source_class_ids"],
            token_ids_by_class=protocol["token_ids_by_class"],
            method="ordinary_logit",
            expected_token_position=expected_position,
        )
        public = V3.HISTORICAL_V1._layer_class_features(
            experiment,
            layers=protocol["layers"],
            class_ids=protocol["source_class_ids"],
            token_ids_by_class=protocol["token_ids_by_class"],
            method="public_jacobian",
            expected_token_position=expected_position,
        )
        V3.require(
            len(ordinary) == len(public) == RAW_SENSOR_WIDTH,
            f"{prompt_id} current score width changed",
        )
        source_action, action_status = V3._source_action(prompt)
        label = V3.COLLAPSE[source_action] if source_action is not None else None
        if source_action is None:
            known_source_counts["unknown"] += 1
            target_counts["unknown_metric_ineligible"] += 1
        else:
            known_source_counts[source_action] += 1
            target_counts[str(label)] += 1

        metadata = V3.mapping(prompt.get("metadata"), f"{prompt_id} metadata")
        task = V3.mapping(metadata.get("task"), f"{prompt_id} task")
        selection = V3.mapping(
            metadata.get("selection"), f"{prompt_id} selection"
        )
        cohort = metadata.get("cohort")
        cohort_id = (
            str(cohort["id"])
            if isinstance(cohort, dict) and isinstance(cohort.get("id"), str)
            else "unspecified"
        )
        request_index = V3.integer(
            selection.get("task_request_index"),
            "task request index",
            minimum=1,
        )
        raw_rows.append(
            {
                "row_id": prompt_id,
                "task_id": V3.nonempty_string(task.get("instance_id"), "task ID"),
                "repo": V3.nonempty_string(task.get("repo"), "task repository"),
                "cohort_id": cohort_id,
                "task_request_index": request_index,
                "checkpoint_ordinal": selection.get("checkpoint_ordinal"),
                "source_action_label_status": action_status,
                "source_action_class_id": source_action,
                "label_status": (
                    "available" if label is not None else "unknown_current_action"
                ),
                "label": label,
                "metric_evaluable": label is not None,
                "auxiliary_diagnostics": auxiliary_by_id[prompt_id],
                "history": list(history),
                "ordinary_logit": list(ordinary),
                "public_jacobian": list(public),
            }
        )

    V3.require(
        processed_count == prompt_count,
        "aligned prompt/report stream count differs from declared prompt count",
    )
    # This is intentionally one all-row pass.  Do not prefilter unknown actions.
    rows = V4_FEATURES.build_feature_rows(raw_rows)
    V3.require(len(rows) == len(raw_rows), "V4 feature row count changed")
    known_count = sum(row["metric_evaluable"] for row in rows)
    return {
        "rows": rows,
        "eligibility": {
            "all_replayed_prompt_count": prompt_count,
            "numerically_stable_prompt_count": stable_count,
            "stable_feature_complete_prediction_count": len(rows),
            "known_current_action_prediction_count": known_count,
            "unknown_current_action_prediction_count": len(rows) - known_count,
            "numerical_stability_fraction": (
                stable_count / prompt_count if prompt_count else 0.0
            ),
            "stable_feature_complete_prediction_fraction": (
                len(rows) / stable_count if stable_count else 0.0
            ),
            "stable_feature_complete_prediction_fraction_numerator": len(rows),
            "stable_feature_complete_prediction_fraction_denominator": stable_count,
            "known_current_action_fraction_of_predictions": (
                known_count / len(rows) if rows else 0.0
            ),
            "predictions_emitted_for_unknown_current_actions": True,
            "current_action_used_only_after_causal_history_was_computed": True,
            "source_action_support": dict(sorted(known_source_counts.items())),
            "target_support": dict(sorted(target_counts.items())),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "exclusions": exclusions,
            "causal_history": history_coverage,
        },
    }


def extract_stable_rows(
    prompt_bundle_value: Any,
    report_value: Any,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """In-memory V4 reference extraction with exact V3 validation semantics."""

    prompts = V3.sequence(prompt_bundle_value, "prompt bundle")
    report = V3.mapping(report_value, "public report")
    V3.HISTORICAL_V1._validate_report_provenance(
        report, protocol=protocol["report_helper_protocol"]
    )
    experiments = V3.sequence(report.get("experiments"), "report experiments")
    V3.require(len(prompts) == len(experiments), "prompt/report row counts differ")
    prompt_ids = [V3.nonempty_string(row.get("id"), "prompt ID") for row in prompts]
    experiment_ids = [
        V3.nonempty_string(row.get("id"), "experiment ID") for row in experiments
    ]
    V3.require(len(prompt_ids) == len(set(prompt_ids)), "prompt IDs are duplicated")
    V3.require(
        len(experiment_ids) == len(set(experiment_ids)),
        "report experiment IDs are duplicated",
    )
    V3.require(prompt_ids == experiment_ids, "prompt/report IDs or order differ")
    return _extract_aligned_stable_rows(
        prompts,
        zip(prompts, experiments, strict=True),
        prompt_count=len(prompts),
        protocol=protocol,
    )


def extract_stable_rows_streaming(
    prompts_path: Path,
    report_path: Path,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Bounded-memory V4 extraction with V3's lockstep streaming primitives."""

    prompt_context = V3._stream_prompt_context(prompts_path)
    expected_prompt_ids = [str(prompt["id"]) for prompt in prompt_context]
    report_metadata: dict[str, Any] = {}
    prompt_iterator = iter(
        V3._stream_json_array_objects(prompts_path, label="prompt bundle")
    )
    experiment_iterator = iter(
        V3._stream_report_experiments(report_path, report_metadata)
    )

    def aligned_pairs() -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        sentinel = object()
        seen_prompt_ids: set[str] = set()
        seen_experiment_ids: set[str] = set()
        index = 0
        try:
            while True:
                prompt = next(prompt_iterator, sentinel)
                experiment = next(experiment_iterator, sentinel)
                if prompt is sentinel and experiment is sentinel:
                    break
                V3.require(
                    prompt is not sentinel,
                    "public report contains trailing experiment rows",
                )
                V3.require(
                    experiment is not sentinel,
                    "prompt bundle contains trailing prompt rows",
                )
                prompt_row = V3.mapping(prompt, "streamed prompt row")
                experiment_row = V3.mapping(experiment, "streamed experiment row")
                prompt_id = V3.nonempty_string(prompt_row.get("id"), "prompt ID")
                experiment_id = V3.nonempty_string(
                    experiment_row.get("id"), "experiment ID"
                )
                V3.require(
                    prompt_id not in seen_prompt_ids, "prompt IDs are duplicated"
                )
                V3.require(
                    experiment_id not in seen_experiment_ids,
                    "report experiment IDs are duplicated",
                )
                seen_prompt_ids.add(prompt_id)
                seen_experiment_ids.add(experiment_id)
                V3.require(
                    index < len(expected_prompt_ids)
                    and prompt_id == expected_prompt_ids[index],
                    "prompt order changed between bounded-memory passes",
                )
                V3.require(
                    prompt_id == experiment_id,
                    "prompt/report IDs or order differ",
                )
                index += 1
                yield prompt_row, experiment_row
            V3.require(
                index == len(expected_prompt_ids),
                "prompt stream count changed between bounded-memory passes",
            )
        finally:
            for iterator in (prompt_iterator, experiment_iterator):
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    result = _extract_aligned_stable_rows(
        prompt_context,
        aligned_pairs(),
        prompt_count=len(prompt_context),
        protocol=protocol,
    )
    V3.HISTORICAL_V1._validate_report_provenance(
        report_metadata, protocol=protocol["report_helper_protocol"]
    )
    return result


# Explicit aliases make call sites self-documenting without a second path.
extract_v4_stable_rows = extract_stable_rows
extract_v4_stable_rows_streaming = extract_stable_rows_streaming


__all__ = [
    "RAW_SENSOR_WIDTH",
    "V3",
    "V3_ANALYZER_PATH",
    "V3_ANALYZER_SHA256",
    "extract_stable_rows",
    "extract_stable_rows_streaming",
    "extract_v4_stable_rows",
    "extract_v4_stable_rows_streaming",
    "sha256_file",
]
