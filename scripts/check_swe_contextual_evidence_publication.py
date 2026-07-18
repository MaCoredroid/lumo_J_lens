#!/usr/bin/env python3
"""Fail-closed checks for the compact SWE contextual-evidence publication."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "validation/jlens-swe-contextual-evidence-2026-07-18/publication"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_MANIFEST_SHA256 = "6dd84b8c585ca7182c6474fe177ae5c0f1fa6bd76c4658b27d1a0bd96d3d73ce"
EXPECTED_FILES = {
    "analysis.json",
    "cards.json",
    "checksums.sha256",
    "materialization-manifest.json",
    "run_manifest.json",
}
EXPECTED_INCLUDED = {
    "analysis": (
        "analysis.json",
        539_989,
        "7ff997dc97c4fdc8d4b66f78ee795cd4a762645852fb977e473646a5a38f511b",
    ),
    "cards": (
        "cards.json",
        37_973,
        "f9aedb3f8052f47be9b9841b601bc96485a8d56e368ee66ad17a1957f7125cdf",
    ),
    "materialization_manifest": (
        "materialization-manifest.json",
        331_506,
        "9f7f0fdd4c90c596147b78cd17d9eb1469922dd6324103f058a1d42986fda01c",
    ),
}
EXPECTED_SOURCE_INPUTS = {
    "analyzer": (
        "scripts/analyze_swe_contextual_evidence.py",
        86_320,
        "fee3fa80f922b4465cad6c9ba62a9905fa73d00bd29d84e90aa05e7559c9bc8e",
    ),
    "analyzer_test": (
        "tests/test_analyze_swe_contextual_evidence.py",
        24_753,
        "7a63d3257693c90e112ecb488931eea06de93803c16cb4d2141815eb55111e40",
    ),
    "materializer": (
        "scripts/materialize_swe_contextual_evidence.py",
        42_302,
        "49199e77b504723d0a7db0268c3acf0f421114165ee3e5d50f674435e8eda4a1",
    ),
    "materializer_test": (
        "tests/test_materialize_swe_contextual_evidence.py",
        14_984,
        "0637773bf97e716007ad40cecb7fbc3ab99cc0687b5dc9fecfdd3e1c7efe3e3a",
    ),
    "protocol": (
        "configs/swe_contextual_evidence_protocol.json",
        48_643,
        "d2e32b4aa027ed387f1c8105046b2a102c3cfbe51d7879dbe86ebd804b58160a",
    ),
}
EXPECTED_REPORTS = {
    "public": {
        "bytes": 46_514_668,
        "sha256": "3f166a2c6a84054323c423c121662022694e6154dd2ec991d6b185155ce1018f",
        "lens_sha256": "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
        "n_prompts": 1000,
    },
    "nf4": {
        "bytes": 46_519_992,
        "sha256": "30dc0d7fcaa4951c9c5f2bcb2785b380664e1381fb767aaf73f93b8d042e05ed",
        "lens_sha256": "54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f",
        "n_prompts": 10,
    },
    "native": {
        "bytes": 46_520_445,
        "sha256": "79ff0cce0c2d51d7091e3e0bb5b3b6ea70d9885170b07251403c13fc92f8f71c",
        "lens_sha256": "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057",
        "n_prompts": 10,
    },
}
EXPECTED_STABLE = {
    "ordinary_logit": (0.004653859996493898, 0.14419388886662954, 0.2232638888888889, 0.0),
    "public_jacobian": (0.3218676724682748, 0.17747469583804398, 0.3052624458874459, 0.125),
    "nf4_jacobian": (0.40177136380155787, 0.1754706831978655, 0.30511363636363636, 0.125),
    "native_jacobian": (0.4471906875037318, 0.18403223417434988, 0.31553030303030305, 0.125),
}
SUPPORTED_CARD_IDS = {
    "django-10914-mode",
    "pytest-10356-getattr",
    "sphinx-8269-masking",
}


class PublicationError(ValueError):
    """Raised when published evidence is missing, stale, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}",
    )


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_file(path: Path, label: str) -> Any:
    try:
        return json.loads(
            path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PublicationError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"cannot parse {label}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be nonempty text")
    require("\\" not in value and "\x00" not in value, f"unsafe {label}")
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in ("", ".", "..") for part in path.parts),
        f"non-canonical {label}: {value!r}",
    )
    return value


def regular_file(root: Path, relative: str, label: str) -> Path:
    relative = canonical_relative(relative, label)
    root = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        require(not current.is_symlink(), f"{label} traverses a symlink")
    try:
        mode = current.lstat().st_mode
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise PublicationError(f"missing {label}: {relative}: {error}") from error
    require(resolved.is_relative_to(root), f"{label} escapes its root")
    require(stat.S_ISREG(mode), f"{label} is not a regular file")
    return current


def require_digest(value: Any, label: str) -> str:
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
        f"{label} changed: expected {expected!r}, got {actual!r}",
    )


def metric(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return mapping(mapping(summary.get("metrics"), "metrics").get(name), name)


def interval(record: Mapping[str, Any], label: str) -> tuple[float, float]:
    bootstrap = mapping(record.get("bootstrap"), f"{label} bootstrap")
    bounds = mapping(bootstrap.get("confidence_interval"), f"{label} interval")
    lower, upper = bounds.get("lower"), bounds.get("upper")
    require(
        isinstance(lower, (int, float))
        and isinstance(upper, (int, float))
        and math.isfinite(float(lower))
        and math.isfinite(float(upper))
        and lower <= upper,
        f"{label} interval is invalid",
    )
    return float(lower), float(upper)


def verify_manifest(value: Any) -> dict[str, Path]:
    manifest = mapping(value, "publication manifest")
    exact_keys(
        manifest,
        {
            "schema_version",
            "kind",
            "status",
            "created_from_completed_run_at",
            "study",
            "claim_boundary",
            "included_artifacts",
            "omitted_artifacts",
            "source_inputs",
        },
        "publication manifest",
    )
    require(
        manifest["schema_version"] == 1
        and manifest["kind"] == "swe_contextual_evidence_compact_publication"
        and manifest["status"] == "complete",
        "publication identity/status changed",
    )
    study = mapping(manifest["study"], "study")
    require(
        study
        == {
            "analysis_version": "paired-evidence-update-development-v1",
            "fixed_layers": list(range(24, 48)),
            "primary_control_eligible_tasks": 8,
            "primary_repositories": 7,
            "prompt_count": 24,
            "scored_token_count": 90,
            "task_count": 12,
        },
        "study contract changed",
    )
    claims = mapping(manifest["claim_boundary"], "claim boundary")
    require(
        claims.get("classification_is_not_operational_usefulness") is True
        and claims.get("independent_raw_reanalysis_requires_omitted_artifacts") is True
        and claims.get("omitted_hashes_are_identity_bindings_not_content_availability") is True
        and claims.get("raw_prompts_and_reports_committed") is False,
        "publication claim boundary weakened",
    )
    require(
        len(sequence(claims.get("compact_bundle_supports"), "supported claims")) == 3
        and len(sequence(claims.get("not_available_in_this_bundle"), "unavailable evidence")) == 4,
        "publication availability disclosure changed",
    )

    included = mapping(manifest["included_artifacts"], "included artifacts")
    exact_keys(included, set(EXPECTED_INCLUDED), "included artifacts")
    paths: dict[str, Path] = {}
    for key, (relative, byte_count, digest) in EXPECTED_INCLUDED.items():
        record = mapping(included[key], f"included {key}")
        require(
            record == {"path": relative, "bytes": byte_count, "sha256": digest},
            f"included {key} record changed",
        )
        path = regular_file(BUNDLE, relative, f"included {key}")
        require(path.stat().st_size == byte_count, f"included {key} byte count mismatch")
        require(sha256_file(path) == digest, f"included {key} digest mismatch")
        paths[key] = path

    sources = mapping(manifest["source_inputs"], "source inputs")
    exact_keys(sources, set(EXPECTED_SOURCE_INPUTS), "source inputs")
    for key, (relative, byte_count, digest) in EXPECTED_SOURCE_INPUTS.items():
        record = mapping(sources[key], f"source {key}")
        require(
            record == {"path": relative, "bytes": byte_count, "sha256": digest},
            f"source {key} record changed",
        )
        path = regular_file(ROOT, relative, f"source {key}")
        require(
            path.stat().st_size == byte_count and sha256_file(path) == digest,
            f"source {key} content is stale",
        )

    omitted = mapping(manifest["omitted_artifacts"], "omitted artifacts")
    exact_keys(omitted, {"prompt_bundle", "reports"}, "omitted artifacts")
    prompt = mapping(omitted["prompt_bundle"], "omitted prompt bundle")
    require(
        prompt
        == {
            "bytes": 8_520_573,
            "committed": False,
            "count": 24,
            "path": ".cache/swe_contextual_evidence/prompts.json",
            "sha256": "0662a327c4d13e4f359935e5766df53803939917095d00f1d5122865b425497d",
        },
        "omitted prompt binding changed",
    )
    reports = mapping(omitted["reports"], "omitted reports")
    exact_keys(reports, set(EXPECTED_REPORTS), "omitted reports")
    for name, expected in EXPECTED_REPORTS.items():
        report = mapping(reports[name], f"omitted report {name}")
        require(
            report.get("committed") is False
            and report.get("status") == "failed"
            and report.get("schema_version") == 3
            and report.get("experiment_count") == 24
            and report.get("bytes") == expected["bytes"]
            and report.get("sha256") == expected["sha256"],
            f"omitted report {name} identity changed",
        )
        canonical_relative(report.get("path"), f"omitted report {name} path")
        lens = mapping(report.get("lens"), f"omitted report {name} lens")
        require(
            lens.get("sha256") == expected["lens_sha256"]
            and lens.get("n_prompts") == expected["n_prompts"],
            f"omitted report {name} lens pin changed",
        )
        runtime = mapping(report.get("runtime"), f"omitted report {name} runtime")
        require(
            runtime.get("mtp_enabled") is False
            and runtime.get("language_model_only") is True
            and runtime.get("enforce_eager") is True
            and runtime.get("max_model_len") == 65_536
            and runtime.get("max_num_batched_tokens") == 4_096
            and runtime.get("capture_adapter") == "vLLM apply_model forward hooks",
            f"omitted report {name} runtime contract changed",
        )
    return paths


def verify_materialization(value: Any, manifest: Mapping[str, Any]) -> None:
    materialization = mapping(value, "materialization manifest")
    require(
        materialization.get("schema_version") == 1
        and materialization.get("kind") == "swe_contextual_evidence_materialization"
        and materialization.get("analysis_version") == "paired-evidence-update-development-v1"
        and materialization.get("lens_outputs_used_for_boundary_or_labels") is False,
        "materialization identity changed",
    )
    require(
        mapping(materialization.get("protocol"), "materialization protocol").get("sha256")
        == EXPECTED_SOURCE_INPUTS["protocol"][2],
        "materialization protocol binding changed",
    )
    prompt = mapping(materialization.get("prompt_bundle"), "materialization prompt")
    omitted_prompt = mapping(
        mapping(manifest.get("omitted_artifacts"), "omitted artifacts").get("prompt_bundle"),
        "omitted prompt",
    )
    require(
        prompt.get("sha256") == omitted_prompt.get("sha256")
        and prompt.get("bytes") == omitted_prompt.get("bytes")
        and prompt.get("count") == 24,
        "materialization prompt binding changed",
    )
    vocabulary = mapping(materialization.get("score_vocabulary"), "score vocabulary")
    require(
        vocabulary.get("token_count") == 90
        and len(sequence(vocabulary.get("token_ids"), "score token IDs")) == 90,
        "score vocabulary changed",
    )
    tasks = sequence(materialization.get("tasks"), "materialization tasks")
    require(
        materialization.get("task_count") == 12
        and materialization.get("prompt_count") == 24
        and len(tasks) == 12
        and all(len(sequence(task.get("prompts"), "task prompts")) == 2 for task in tasks),
        "materialization task/prompt grid changed",
    )


def verify_analysis(value: Any, manifest: Mapping[str, Any]) -> None:
    analysis = mapping(value, "analysis")
    require(
        analysis.get("schema_version") == 1
        and analysis.get("kind") == "swe_contextual_evidence_update_analysis"
        and analysis.get("task_count") == 12
        and analysis.get("repository_count") == 9,
        "analysis identity/counts changed",
    )
    require(
        analysis.get("protocol_sha256") == EXPECTED_SOURCE_INPUTS["protocol"][2]
        and analysis.get("manifest_sha256") == EXPECTED_INCLUDED["materialization_manifest"][2]
        and analysis.get("prompt_bundle_sha256")
        == mapping(mapping(manifest["omitted_artifacts"], "omitted")["prompt_bundle"], "prompt")[
            "sha256"
        ],
        "analysis input binding changed",
    )
    require(
        analysis.get("report_status") == {"public": "failed", "nf4": "failed", "native": "failed"},
        "analysis report status changed",
    )
    pairing = mapping(analysis.get("report_pairing"), "report pairing")
    require(
        pairing.get("report_labels") == ["public", "nf4", "native"]
        and pairing.get("prompt_count") == 24
        and all(
            pairing.get(field) is True
            for field in (
                "exact_prompt_pairing",
                "residual_capture_manifests_equal",
                "ordinary_logit_readouts_equal",
                "numerical_diagnostics_equal",
            )
        ),
        "analysis report pairing changed",
    )
    inputs = mapping(analysis.get("inputs"), "analysis inputs")
    input_reports = mapping(inputs.get("reports"), "analysis report inputs")
    for name, expected in EXPECTED_REPORTS.items():
        require(
            mapping(input_reports.get(name), f"analysis {name} report").get("sha256")
            == expected["sha256"],
            f"analysis {name} report binding changed",
        )

    profiles = mapping(analysis.get("profiles"), "analysis profiles")
    exact_keys(profiles, {"primary_stable", "legacy_strict"}, "analysis profiles")
    stable = mapping(profiles["primary_stable"], "stable profile")
    methods = mapping(stable.get("methods"), "stable methods")
    exact_keys(methods, set(EXPECTED_STABLE), "stable methods")
    names = (
        "target_vs_foils_update",
        "context_selectivity",
        "own_target_reciprocal_rank",
        "own_target_top1",
    )
    for method_name, expected_values in EXPECTED_STABLE.items():
        summary = mapping(methods[method_name], method_name)
        support = mapping(summary.get("support"), f"{method_name} support")
        require(
            support.get("task_count") == 8
            and support.get("repository_count") == 7
            and support.get("numerically_excluded_task_ids") == [],
            f"{method_name} stable support changed",
        )
        for metric_name, expected in zip(names, expected_values, strict=True):
            require_close(metric(summary, metric_name).get("estimate"), expected, f"{method_name} {metric_name}")

    decision = mapping(stable.get("public_usefulness_decision"), "stable decision")
    require(
        decision.get("classification") == "frozen_directional_point_rule_pass"
        and decision.get("classification_is_not_operational_usefulness") is True
        and decision.get("passed") is True
        and decision.get("failed_criteria") == [],
        "stable directional decision changed",
    )
    uncertainty = mapping(decision.get("uncertainty_diagnostic"), "uncertainty diagnostic")
    require(
        uncertainty.get("all_four_confidence_interval_lowers_positive") is False
        and all(value is False for value in mapping(uncertainty.get("criteria"), "uncertainty criteria").values()),
        "uncertainty boundary changed",
    )
    comparisons = mapping(stable.get("paired_comparisons"), "paired comparisons")
    for comparison_name in (
        "public_jacobian_minus_ordinary_logit",
        "nf4_jacobian_minus_ordinary_logit",
        "native_jacobian_minus_ordinary_logit",
        "public_jacobian_minus_nf4_jacobian",
        "public_jacobian_minus_native_jacobian",
    ):
        comparison = mapping(comparisons.get(comparison_name), comparison_name)
        for metric_name in names:
            lower, upper = interval(metric(comparison, metric_name), f"{comparison_name} {metric_name}")
            require(lower <= 0.0 <= upper, f"{comparison_name} {metric_name} no longer spans zero")

    copy = mapping(stable.get("copy_retrieval_diagnostic"), "copy diagnostic")
    require(copy.get("passed") is False and copy.get("task_count") == 8, "copy diagnostic changed")
    for metric_name in (
        "mrr_minus_copy_frequency",
        "top1_minus_copy_frequency",
        "mrr_minus_copy_recency",
        "top1_minus_copy_recency",
    ):
        record = metric(copy, metric_name)
        lower, upper = interval(record, f"copy {metric_name}")
        require(record.get("estimate", 0.0) < 0.0 and upper < 0.0, f"copy {metric_name} no longer fails")
    baselines = mapping(analysis.get("copy_baselines"), "copy baselines")
    baseline_summary = mapping(baselines.get("task_equal_summary"), "copy baseline summary")
    require_close(
        mapping(
            mapping(baseline_summary["copy_frequency"], "frequency").get(
                "own_target_reciprocal_rank"
            ),
            "copy frequency MRR",
        ).get("estimate"),
        0.6813186813186813,
        "copy frequency MRR",
    )
    require_close(
        mapping(
            mapping(baseline_summary["copy_recency"], "recency").get(
                "own_target_reciprocal_rank"
            ),
            "copy recency MRR",
        ).get("estimate"),
        0.6256944444444444,
        "copy recency MRR",
    )
    require(
        analysis.get("cards_summary") == {
            "supported_lens_why_count": 3,
            "withheld_lens_why_count": 9,
        },
        "analysis card summary changed",
    )
    limitations = sequence(analysis.get("limitations"), "analysis limitations")
    require(
        len(limitations) == 3 and any("chain-of-thought" in item for item in limitations),
        "analysis claim limitations changed",
    )


def verify_cards(value: Any) -> None:
    cards_doc = mapping(value, "cards")
    require(
        cards_doc.get("schema_version") == 1
        and cards_doc.get("kind") == "swe_contextual_evidence_cards",
        "cards identity changed",
    )
    cards = sequence(cards_doc.get("cards"), "cards")
    require(len(cards) == 12, "card count changed")
    supported = {
        card.get("task_id")
        for card in cards
        if mapping(card.get("lens_why_guard"), "lens WHY guard").get("status") == "supported"
    }
    require(supported == SUPPORTED_CARD_IDS, "supported lens WHY card set changed")
    for card in cards:
        fields = mapping(card.get("fields"), "card fields")
        exact_keys(fields, {"WHY", "WHERE", "EVIDENCE", "NEXT"}, "card fields")
        guard = mapping(card.get("lens_why_guard"), "lens WHY guard")
        why = mapping(fields["WHY"], "WHY field")
        lens = mapping(why.get("lens"), "WHY lens field")
        if card.get("task_id") in SUPPORTED_CARD_IDS:
            require(
                guard.get("status") == "supported"
                and lens.get("status") == "supported_contextual_evidence_summary"
                and isinstance(lens.get("claim"), str),
                "supported WHY card was weakened",
            )
        else:
            require(
                guard.get("status") == "withheld"
                and lens.get("status") == "withheld"
                and lens.get("claim") is None,
                "withheld WHY card exposes a claim",
            )
        for field in ("WHERE", "EVIDENCE", "NEXT"):
            record = mapping(fields[field], f"{field} field")
            observed = mapping(record.get("observed"), f"{field} observed")
            inferred = mapping(record.get("lens"), f"{field} lens")
            require(
                observed.get("lens_inference") is False
                and observed.get("status") == "observed_from_bound_qwen_code_trajectory"
                and inferred.get("claim") is None,
                f"{field} observation/lens boundary changed",
            )


def verify_checksums() -> None:
    checksum_path = regular_file(BUNDLE, "checksums.sha256", "checksum ledger")
    expected = [
        f"{digest}  {relative}\n"
        for relative, _, digest in EXPECTED_INCLUDED.values()
    ]
    manifest_digest = sha256_file(BUNDLE / "run_manifest.json")
    require(manifest_digest == RUN_MANIFEST_SHA256, "run manifest digest changed")
    expected.append(f"{manifest_digest}  run_manifest.json\n")
    require(checksum_path.read_text(encoding="ascii") == "".join(expected), "checksum ledger changed")


def verify_publication() -> dict[str, Any]:
    require(BUNDLE.is_dir() and not BUNDLE.is_symlink(), "publication directory is unsafe")
    actual_files = {path.name for path in BUNDLE.iterdir() if path.is_file()}
    require(actual_files == EXPECTED_FILES, "publication file set changed")
    require(not any(path.is_dir() or path.is_symlink() for path in BUNDLE.iterdir()), "publication has unsafe entries")
    manifest = mapping(strict_json_file(BUNDLE / "run_manifest.json", "run manifest"), "run manifest")
    paths = verify_manifest(manifest)
    materialization = strict_json_file(paths["materialization_manifest"], "materialization")
    analysis = strict_json_file(paths["analysis"], "analysis")
    cards = strict_json_file(paths["cards"], "cards")
    verify_materialization(materialization, manifest)
    verify_analysis(analysis, manifest)
    verify_cards(cards)
    verify_checksums()
    return {
        "task_count": analysis["task_count"],
        "primary_task_count": manifest["study"]["primary_control_eligible_tasks"],
        "supported_card_count": analysis["cards_summary"]["supported_lens_why_count"],
        "classification": analysis["profiles"]["primary_stable"]["public_usefulness_decision"]["classification"],
        "operational_usefulness": False,
    }


def main() -> int:
    result = verify_publication()
    print(
        "Contextual-evidence publication checks passed: "
        f"{result['primary_task_count']} primary tasks, "
        f"{result['supported_card_count']} guarded WHY cards, "
        "not operationally useful."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
