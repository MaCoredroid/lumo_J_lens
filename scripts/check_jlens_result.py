#!/usr/bin/env python3
"""Validate the committed Qwen3.6 NVFP4 Jacobian Lens certificate."""

from __future__ import annotations

import json
from pathlib import Path

from download_jlens import (
    LENS_FILENAME,
    LENS_REVISION,
    LENS_SHA256,
    LENS_SIZE,
)
from run_jlens_nvfp4 import (
    FINAL_LOGIT_MAX_ABS_TOLERANCE,
    FINAL_LOGIT_RMS_TOLERANCE,
    FINAL_NORM_MAX_ABS_TOLERANCE,
    FINAL_NORM_RMS_TOLERANCE,
    FINAL_TOPK_PARITY_K,
    MODEL_CONFIG_SHA256,
    MODEL_INDEX_SHA256,
    MODEL_REPO,
    MODEL_REVISION,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "validation" / "jlens-nvfp4-2026-07-16.json"
HISTORICAL_RESULT_SCHEMA_VERSION = 2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    result = json.loads(RESULT.read_text())
    require(
        result["schema_version"] == HISTORICAL_RESULT_SCHEMA_VERSION,
        "schema version mismatch",
    )
    require(result["status"] == "passed", "result status is not passed")
    require(all(result["assertions"].values()), "one or more assertions failed")

    model = result["model"]
    require(model["repo_id"] == MODEL_REPO, "model repository mismatch")
    require(model["revision"] == MODEL_REVISION, "model revision mismatch")
    require(model["config_sha256"] == MODEL_CONFIG_SHA256, "model config hash mismatch")
    require(model["index_sha256"] == MODEL_INDEX_SHA256, "model index hash mismatch")
    require(model["quant_method"] == "modelopt", "quantization method mismatch")
    require(model["quant_algo"] == "MIXED_PRECISION", "quantization algorithm mismatch")
    require(model["model_info"]["layer_count"] == 64, "model layer count mismatch")
    require(model["model_info"]["hidden_size"] == 5120, "model width mismatch")

    lens = result["lens"]
    require(lens["revision"] == LENS_REVISION, "lens revision mismatch")
    require(lens["filename"] == LENS_FILENAME, "lens filename mismatch")
    require(lens["sha256"] == LENS_SHA256, "lens SHA-256 mismatch")
    require(lens["size_bytes"] == LENS_SIZE, "lens size mismatch")
    require(lens["n_prompts"] == 1000, "lens prompt count mismatch")
    require(lens["d_model"] == 5120, "lens width mismatch")
    require(lens["source_layers"] == list(range(63)), "lens source layers mismatch")

    require(result["runtime"]["mtp_enabled"] is False, "diagnostic unexpectedly used MTP")
    require(result["runtime"]["enforce_eager"] is True, "diagnostic was not eager")
    experiments = result["experiments"]
    require(
        [item["id"] for item in experiments]
        == ["currency_boot", "capital_france"],
        "prompt set mismatch",
    )
    derived_top1_matches = []
    derived_reconstruction_matches = []
    for experiment in experiments:
        experiment_id = experiment["id"]
        final_position = experiment["final_validation_position"]
        final_index = experiment["capture_positions_resolved"].index(final_position)
        generated_token_id = experiment["generated_token_id"]
        reconstructed = experiment["final_model_readout"][final_index]
        captured = experiment["captured_final_model_readout"][final_index]
        derived_top1 = (
            reconstructed["token_ids"][0] == generated_token_id
            and captured["token_ids"][0] == generated_token_id
        )
        derived_top1_matches.append(derived_top1)
        require(derived_top1, f"{experiment_id}: derived final top-1 parity failed")
        require(
            experiment["final_layer_top1_matches_greedy"] == derived_top1,
            f"{experiment_id}: stored final parity flag is inconsistent",
        )

        norm = experiment["final_norm_reconstruction"]
        require(
            norm["max_abs_tolerance"] == FINAL_NORM_MAX_ABS_TOLERANCE,
            f"{experiment_id}: norm max tolerance mismatch",
        )
        require(
            norm["rms_tolerance"] == FINAL_NORM_RMS_TOLERANCE,
            f"{experiment_id}: norm RMS tolerance mismatch",
        )
        derived_norm = (
            norm["max_abs_error"] <= FINAL_NORM_MAX_ABS_TOLERANCE
            and norm["rms_error"] <= FINAL_NORM_RMS_TOLERANCE
        )
        require(
            norm["within_tolerance"] == derived_norm and derived_norm,
            f"{experiment_id}: final norm reconstruction failed",
        )

        logits = experiment["final_logits_reconstruction"]
        require(
            logits["max_abs_tolerance"] == FINAL_LOGIT_MAX_ABS_TOLERANCE,
            f"{experiment_id}: logit max tolerance mismatch",
        )
        require(
            logits["rms_tolerance"] == FINAL_LOGIT_RMS_TOLERANCE,
            f"{experiment_id}: logit RMS tolerance mismatch",
        )
        require(
            logits["top_k_prefix"] == FINAL_TOPK_PARITY_K,
            f"{experiment_id}: top-k parity width mismatch",
        )
        derived_prefix_match = (
            reconstructed["token_ids"][:FINAL_TOPK_PARITY_K]
            == captured["token_ids"][:FINAL_TOPK_PARITY_K]
        )
        derived_logits = (
            logits["max_abs_error"] <= FINAL_LOGIT_MAX_ABS_TOLERANCE
            and logits["rms_error"] <= FINAL_LOGIT_RMS_TOLERANCE
            and derived_prefix_match
        )
        require(
            logits["top_k_prefix_token_ids_match"] == derived_prefix_match,
            f"{experiment_id}: stored top-k parity flag is inconsistent",
        )
        require(
            logits["within_tolerance"] == derived_logits and derived_logits,
            f"{experiment_id}: final logits reconstruction failed",
        )
        derived_reconstruction_matches.append(derived_norm and derived_logits)

        require(
            [row["layer"] for row in experiment["layers"]] == list(range(63)),
            f"{experiment_id}: layer grid mismatch",
        )
        require(
            len(experiment["positions_resolved"]) == 1,
            f"{experiment_id}: position count mismatch",
        )
        for row in experiment["layers"]:
            require(
                len(row["positions"]) == 1,
                f"{experiment_id}: missing position at layer {row['layer']}",
            )
            for key in ("logit_lens", "jacobian_lens"):
                readout = row["positions"][0][key]
                require(
                    len(readout["token_ids"]) == 10,
                    f"{experiment_id}: top-k ID count mismatch",
                )
                require(
                    len(readout["tokens"]) == 10,
                    f"{experiment_id}: top-k token count mismatch",
                )
                require(readout["token_ids"][0] >= 0, f"{experiment_id}: invalid top token ID")

    by_id = {experiment["id"]: experiment for experiment in experiments}
    currency = {
        row["layer"]: row["positions"][0]["jacobian_lens"]["tokens"][0]
        for row in by_id["currency_boot"]["layers"]
    }
    france = {
        row["layer"]: row["positions"][0]["jacobian_lens"]["tokens"][0]
        for row in by_id["capital_france"]["layers"]
    }
    require(
        currency[40] == " Italy" and currency[48] == " Italy",
        "currency country transition mismatch",
    )
    require(
        currency[58] == " euro" and currency[60] == " euro",
        "currency answer transition mismatch",
    )
    require(
        all(france[layer] == " Paris" for layer in (56, 58, 60, 62)),
        "France answer transition mismatch",
    )

    require(
        result["assertions"]["all_final_layer_top1_match_greedy"]
        == all(derived_top1_matches),
        "aggregate top-1 assertion mismatch",
    )
    require(
        result["assertions"][
            "all_final_adapter_reconstructions_within_tolerance"
        ]
        == all(derived_reconstruction_matches),
        "aggregate reconstruction assertion mismatch",
    )

    print(
        "Jacobian Lens result integrity passed: pins, 2 prompts, 63 layers, "
        "vector/logit parity, semantic transitions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
