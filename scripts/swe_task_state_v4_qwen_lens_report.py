#!/usr/bin/env python3
"""P3: the Qwen-only J-lens report.

Synthesizes, from Qwen-only materialized artifacts and free labels, the three
things the lens delivers:
  1. action gauge   — per-source next-action predictive performance (the
     inspect/edit/check forecast over Qwen internals), from
     observable-action-phase-v1.json;
  2. faithfulness   — surface-vs-internal divergence flags lens error
     (source_disagreement, cohort-scale, permutation-tested), from
     swe_task_state_v4_source_divergence;
  3. free CoT       — the self-labeled epistemic timeline from the Qwen
     trajectory (swe_task_state_v4_trajectory_cot_reader).

No GPT-OSS / Mistral / annotation anywhere. Read-only over materialized artifacts;
the report is a deliverable, never a predictor input.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import swe_task_state_v4_source_divergence as divergence  # noqa: E402
from scripts import swe_task_state_v4_trajectory_cot_reader as cot  # noqa: E402
from scripts import (  # noqa: E402
    swe_task_state_v4_cot_concept_faithfulness as cot_faithfulness,
)

ACTION_PHASE_ARTIFACT = divergence.ACTION_PHASE_ARTIFACT
REPORT_PATH = ROOT / "artifacts/jlens-qwen-only-report-v1.json"

# Sources reported for the action gauge (Qwen internals + baseline).
_GAUGE_VARIANTS = ("history_only", "sequence_logit", "sequence_j", "sequence_logit_j")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metric_value(metric: Any) -> float | None:
    if isinstance(metric, dict) and metric.get("status", "").startswith("available"):
        return metric.get("value")
    return None


def _action_gauge(artifact_path: Path) -> dict[str, Any]:
    outer = json.loads(Path(artifact_path).read_text())["evaluation"][
        "outer_evaluation"
    ]
    variants = outer["variants"]
    gauge: dict[str, Any] = {}
    for name in _GAUGE_VARIANTS:
        metrics = variants[name].get("metrics", {})
        gauge[name] = {
            "weighted_accuracy": _metric_value(metrics.get("weighted_accuracy")),
            "weighted_auprc": _metric_value(metrics.get("weighted_auprc")),
            "weighted_auroc": _metric_value(metrics.get("weighted_auroc")),
        }
    return {
        "classes": list(outer["classes"]),
        "row_count": outer["row_count"],
        "target": "prospective same-request next observable action",
        "primary_forecast": "sequence_logit_j",
        "per_source": gauge,
    }


def build_report(*, permutations: int = 2000) -> dict[str, Any]:
    sources = divergence.load_action_phase_sources(ACTION_PHASE_ARTIFACT)
    reliability = divergence.evaluate_divergence_flags_error(
        sources, permutations=permutations
    )
    report: dict[str, Any] = {
        "kind": "jlens_qwen_only_report_v1",
        "scope": {
            "subject_model_family": "qwen",
            "external_annotators_used": False,
            "gpt_oss_or_mistral_used": False,
            "human_labels_used": False,
            "labels_are_free_from_trajectory": True,
        },
        "action_gauge": _action_gauge(ACTION_PHASE_ARTIFACT),
        "lens_reliability_flag": {
            "metric": "source_disagreement = normalized JSD(sequence_logit, sequence_j)",
            "claim": "high probe disagreement flags lens error (pooled forecast mispredicts the agent's next action)",
            "interpretation": (
                "an uncertainty / trust gauge, NOT a CoT-faithfulness measure. Both "
                "sequence_logit and sequence_j are INTERNAL activation probes (ordinary "
                "logit vs Jacobian), so this is probe-vs-probe disagreement, not "
                "stated-reasoning vs actual-computation. Disagreement-predicts-error is "
                "a generic ensemble-uncertainty effect. Real CoT-faithfulness is a "
                "separate target: CoT-event <-> internal-concept agreement (in progress)."
            ),
            **reliability,
        },
        "provenance": {
            "action_phase_artifact_sha256": _sha256_file(ACTION_PHASE_ARTIFACT),
        },
        "limitations": [
            "action gauge + reliability flag are cohort-scale (1570 rows); the free-CoT "
            "timeline below is a single demonstrative task.",
            "the reliability flag is NOT a faithfulness measure (see its interpretation); "
            "CoT-faithfulness is a separate, harder target being built.",
            "the reliability effect is small in absolute JSD (probes usually agree) but "
            "statistically significant; it is a flag, not a calibrated error rate.",
        ],
    }

    if (
        cot_faithfulness.CONCEPT_CHAIN_ARTIFACT.exists()
        and cot.DEFAULT_TRAJECTORY.exists()
    ):
        faith = cot_faithfulness.score_faithfulness()
        report["cot_faithfulness"] = {
            "definition": "does the internal concept-chain readout encode the concept "
            "the model's own CoT claims at that boundary (top-1 agreement)",
            "distinct_from": "the lens_reliability_flag above (which is probe-vs-probe, not faithfulness)",
            "result": (
                "WEAK/PARTIAL: internal top-1 matches the CoT-implied concept "
                f"{faith['faithfulness_top1_agreement_all']:.2f} of the time (public_j) / "
                f"{faith['faithfulness_top1_agreement_native_j_all']:.2f} (native_j) "
                f"across {faith['n_mapped_events_aligned']} events; "
                f"{faith['faithfulness_top1_agreement_strict']:.2f} on strict-fidelity "
                f"boundaries. Free CoT events agree with human labels "
                f"{faith['free_event_vs_human_label_agreement']:.2f}, but the internal "
                "readout tracks neither reliably. Root cause: public_j collapses onto "
                f"focused_validation on "
                f"{faith['focused_validation_bias']['public_j_top1_is_focused_validation']:.1f} "
                "of boundaries (a degenerate bias)."
            ),
            **faith,
        }

    if cot.DEFAULT_TRAJECTORY.exists():
        ctx = cot.read_free_reasoning_context()
        report["free_cot_timeline_demo"] = {
            "trajectory": str(cot.DEFAULT_TRAJECTORY.name),
            "n_turns": ctx["n_turns"],
            "n_boundaries": ctx["n_boundaries"],
            "epistemic_chain": [
                {"turn": t["turn"], "stage": t["stage"], "events": t["semantic_events"]}
                for t in ctx["turns"]
            ],
        }
    else:
        report["free_cot_timeline_demo"] = {"status": "trajectory_not_present"}
    return report


def main(argv: Any = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    report = build_report(permutations=args.permutations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    gauge = report["action_gauge"]["per_source"]
    flag = report["lens_reliability_flag"]
    print(f"Qwen-only J-lens report -> {args.output}")
    print(
        "  action gauge (weighted acc): "
        + ", ".join(f"{k}={v['weighted_accuracy']:.3f}" for k, v in gauge.items())
    )
    print(
        f"  reliability flag (NOT faithfulness): probe disagreement flags error"
        f" AUC={flag['error_detection_auc']:.3f} p={flag['permutation_p_value']:.2g}"
        f" (error {flag['divergence_mean_on_error']:.4f} vs correct"
        f" {flag['divergence_mean_on_correct']:.4f})"
    )
    if "cot_faithfulness" in report:
        cf = report["cot_faithfulness"]
        print(
            f"  CoT faithfulness (real): internal concept matches CoT claim"
            f" {cf['faithfulness_top1_agreement_all']:.2f} top-1"
            f" ({cf['n_mapped_events_aligned']} events, single task) — WEAK/PARTIAL"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
