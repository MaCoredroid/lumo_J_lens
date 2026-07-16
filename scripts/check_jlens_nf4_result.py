#!/usr/bin/env python3
"""Validate the committed NF4 fit, evaluation, and paired NVFP4 evidence.

This checker intentionally needs neither a lens checkpoint nor a GPU. It binds
the small verification record to the committed provenance JSON, checks the
numeric/evaluation reports, and derives the paired-run failure state from the
recorded readouts and reconstruction errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"

D_MODEL = 5120
SOURCE_LAYERS = list(range(63))
POSITIONS = [16, 32, 64, 96]
CAPTURE_POSITIONS = [16, 32, 64, 96, 127]
PROMPT_ROWS = [3, 18, 42, 49]
PROMPT_TOKEN_COUNT = 128
TOP_K = 10

FIT_MODEL = "Qwen/Qwen3.6-27B"
FIT_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
FIT_ARTIFACT_SHA256 = "54d95f9626d8d120d56c161cfc8943ec76fd77172a9c0c54d5d913a5a639424f"
FIT_ARTIFACT_SIZE = 6606048039
FIT_PROVENANCE_SHA256 = "08dfe022cc74bab226308dcd77a3901ec18806f8832dc8dec46524374fa3b1e7"
FIT_PROVENANCE_SIZE = 1261860
FIT_CONTRACT_SHA256 = "2a720e5193f0e6cc733521392ee1ce3d38f8afa8425ff954f14cc1d25dd5553d"
NF4_WEIGHTS_SHA256 = "964aef016bf13e0c68b5322eada1af8036ee56e7b57dfadce09557c9253be0d9"
NF4_MODULE_COUNT = 496

PUBLIC_REPO = "neuronpedia/jacobian-lens"
PUBLIC_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
PUBLIC_SHA256 = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
PUBLIC_SIZE = 3303032772

NVFP4_MODEL = "nvidia/Qwen3.6-27B-NVFP4"
NVFP4_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
NVFP4_CONFIG_SHA256 = "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
NVFP4_INDEX_SHA256 = "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"

EVAL_PROMPTS_SHA256 = "cd0fe64e800c7b937fcd891196eed6d7c30a8ff1246b9555dc2962bf61c9a56b"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

EXPECTED_CROSS_MODEL = {
    "local_nf4_lens_on_nvfp4": {
        "mean_layer_top1_agreement": 0.7698412698412699,
        "mean_layer_top5_overlap": 0.816468253968254,
        "mean_layer_target_rank_spearman": 0.9740662565168904,
    },
    "public_lens_on_nvfp4": {
        "mean_layer_top1_agreement": 0.7559523809523809,
        "mean_layer_top5_overlap": 0.8184523809523809,
        "mean_layer_target_rank_spearman": 0.9799860002527503,
    },
}


@dataclass(frozen=True)
class Evidence:
    verification: dict[str, Any]
    provenance: dict[str, Any]
    geometry: dict[str, Any]
    evaluation: dict[str, Any]
    local_nvfp4: dict[str, Any]
    public_nvfp4: dict[str, Any]
    provenance_sha256: str
    provenance_size_bytes: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name}: root must be an object")
    return value


def load_evidence(validation_dir: Path = VALIDATION) -> Evidence:
    provenance_path = validation_dir / "jlens-nf4-fit-provenance-2026-07-16.json"
    provenance_bytes = provenance_path.read_bytes()
    return Evidence(
        verification=_load_json(
            validation_dir / "jlens-nf4-artifact-verification-2026-07-16.json"
        ),
        provenance=json.loads(provenance_bytes),
        geometry=_load_json(
            validation_dir / "jlens-nf4-vs-public-2026-07-16.json"
        ),
        evaluation=_load_json(validation_dir / "jlens-nf4-eval-2026-07-16.json"),
        local_nvfp4=_load_json(
            validation_dir / "jlens-nf4-on-nvfp4-2026-07-16.json"
        ),
        public_nvfp4=_load_json(
            validation_dir / "jlens-public-on-nvfp4-heldout-2026-07-16.json"
        ),
        provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
        provenance_size_bytes=len(provenance_bytes),
    )


def _finite_numbers(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(value), f"{label}: non-finite numeric value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_numbers(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_numbers(item, f"{label}.{key}")


def _close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{label}: expected {expected}, got {actual}",
    )


def _check_sha256(value: Any, label: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label}: invalid SHA-256",
    )


def _check_local_verification(evidence: Evidence) -> None:
    record = evidence.verification
    expected = {
        "kind": "local_fit",
        "sha256": FIT_ARTIFACT_SHA256,
        "size_bytes": FIT_ARTIFACT_SIZE,
        "provenance_sha256": FIT_PROVENANCE_SHA256,
        "provenance_size_bytes": FIT_PROVENANCE_SIZE,
        "contract_sha256": FIT_CONTRACT_SHA256,
        "d_model": D_MODEL,
        "n_prompts": 10,
        "source_layers": SOURCE_LAYERS,
        "target_layer": 63,
        "tensor_dtype": "torch.float32",
        "tensor_shape": [D_MODEL, D_MODEL],
        "finite_checked": True,
        "fit_estimator": "anthropic-future-summed",
        "fit_model": FIT_MODEL,
        "fit_model_revision": FIT_MODEL_REVISION,
        "fit_quantization": "bitsandbytes-nf4-double-quant-bfloat16",
        "checkpoint_keys": ["J", "d_model", "metadata", "n_prompts", "source_layers"],
    }
    for key, value in expected.items():
        require(record.get(key) == value, f"fit verification {key} mismatch")
    require(
        Path(record["path"]).name == "Qwen3.6-27B-jlens-nf4-n10-fp32.pt",
        "fit artifact filename mismatch",
    )
    require(
        Path(record["provenance_path"]).name
        == "Qwen3.6-27B-jlens-nf4-n10-fp32.provenance.json",
        "fit provenance filename mismatch",
    )
    require(
        evidence.provenance_sha256 == record["provenance_sha256"],
        "committed fit provenance SHA-256 mismatch",
    )
    require(
        evidence.provenance_size_bytes == record["provenance_size_bytes"],
        "committed fit provenance size mismatch",
    )


def _check_provenance(evidence: Evidence) -> None:
    provenance = evidence.provenance
    verification = evidence.verification
    require(provenance.get("schema_version") == 1, "fit provenance schema mismatch")
    require(provenance.get("complete") is True, "fit provenance is incomplete")
    require(provenance.get("status") == "completed", "fit provenance status mismatch")
    require(
        provenance.get("contract_sha256") == verification["contract_sha256"],
        "fit contract hash mismatch",
    )
    contract = provenance["contract"]
    require(contract["model_id"] == verification["fit_model"], "fit model mismatch")
    require(
        contract["model_revision"] == verification["fit_model_revision"],
        "fit model revision mismatch",
    )
    require(
        contract["quantization"]
        == {
            "blocksize": 64,
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "method": "bitsandbytes",
            "nested_blocksize": 256,
            "storage_dtype": "uint8",
            "type": "nf4",
        },
        "fit quantization contract mismatch",
    )
    estimator = contract["estimator"]
    require(estimator["name"] == "anthropic_future_summed_vjp", "fit estimator mismatch")
    require(estimator["source_layers"] == SOURCE_LAYERS, "fit estimator layers mismatch")
    require(estimator["target_layer"] == 63, "fit target layer mismatch")
    require(estimator["row_limit"] == D_MODEL, "fit row limit mismatch")
    require(estimator["max_seq_len"] == PROMPT_TOKEN_COUNT, "fit sequence length mismatch")
    require(
        provenance["source"]["identity"] == contract["source_identity"],
        "fit source identity mismatch",
    )
    require(
        provenance["source"]["script_sha256"] == contract["script_sha256"],
        "fit script identity mismatch",
    )

    prompts = provenance["prompts"]
    require(
        prompts["dataset"]
        == {
            "config": "wikitext-103-raw-v1",
            "repo": "Salesforce/wikitext",
            "revision": DATASET_REVISION,
            "split": "train",
        },
        "fit prompt dataset mismatch",
    )
    require(
        prompts["input_manifest_sha256"] == contract["prompt_manifest_sha256"],
        "fit prompt manifest hash mismatch",
    )
    require(
        prompts["frozen_prompt_sha256"] == contract["prompts_sha256"],
        "frozen fit prompt hash mismatch",
    )
    prompt_records = prompts["prompts"]
    require(len(prompt_records) == verification["n_prompts"], "fit prompt count mismatch")
    require(
        len({item["id"] for item in prompt_records}) == len(prompt_records),
        "fit prompt IDs are not unique",
    )
    for item in prompt_records:
        require(item["token_count"] == PROMPT_TOKEN_COUNT, "fit prompt length mismatch")
        require(len(item["token_ids"]) == PROMPT_TOKEN_COUNT, "fit prompt token mismatch")
        _check_sha256(item["text_sha256"], f"fit prompt {item['id']}")

    weights = provenance["model"]["quantized_weights"]
    require(weights["aggregate_sha256"] == NF4_WEIGHTS_SHA256, "fit NF4 hash mismatch")
    require(weights["module_count"] == NF4_MODULE_COUNT, "fit NF4 module count mismatch")
    require(len(weights["modules"]) == NF4_MODULE_COUNT, "fit NF4 module list mismatch")

    result = provenance["result"]
    require(result["sha256"] == verification["sha256"], "fit result hash mismatch")
    require(result["d_model"] == D_MODEL, "fit result width mismatch")
    require(result["n_prompts"] == verification["n_prompts"], "fit result prompt mismatch")
    require(result["storage_dtype"] == "float32", "fit result dtype mismatch")
    require(Path(result["path"]).name == Path(verification["path"]).name, "fit result path mismatch")
    require([item["layer"] for item in result["layers"]] == SOURCE_LAYERS, "fit result layers mismatch")
    for item in result["layers"]:
        label = f"fit result layer {item['layer']}"
        require(item["shape"] == [D_MODEL, D_MODEL], f"{label}: shape mismatch")
        require(item["dtype"] == "float32", f"{label}: dtype mismatch")
        require(item["finite_count"] == D_MODEL * D_MODEL, f"{label}: finite count mismatch")
        _check_sha256(item["sha256"], label)
        for key, value in item["published"].items():
            require(item.get(key) == value, f"{label}: published {key} mismatch")
    _finite_numbers(provenance, "fit provenance")


def _check_public_artifact(record: dict[str, Any]) -> None:
    expected = {
        "kind": "pinned_public",
        "repo_id": PUBLIC_REPO,
        "revision": PUBLIC_REVISION,
        "sha256": PUBLIC_SHA256,
        "size_bytes": PUBLIC_SIZE,
        "d_model": D_MODEL,
        "n_prompts": 1000,
        "source_layers": SOURCE_LAYERS,
        "tensor_dtype": "torch.float16",
        "tensor_shape": [D_MODEL, D_MODEL],
    }
    for key, value in expected.items():
        require(record.get(key) == value, f"public lens {key} mismatch")


def _check_geometry(evidence: Evidence) -> None:
    geometry = evidence.geometry
    require(geometry["schema_version"] == 1, "geometry schema mismatch")
    require(
        geometry["scope"]
        == "numeric matrix comparison only; no held-out token or logit evaluation",
        "geometry scope mismatch",
    )
    require(
        geometry["artifacts"] == evidence.evaluation["artifacts"],
        "geometry/evaluation artifact summaries differ",
    )
    require(
        geometry["artifacts"]["local"] == evidence.verification,
        "geometry local artifact summary mismatch",
    )
    _check_public_artifact(geometry["artifacts"]["public"])
    configuration = geometry["configuration"]
    require(configuration["d_model"] == D_MODEL, "geometry width mismatch")
    require(configuration["source_layers"] == SOURCE_LAYERS, "geometry layers mismatch")
    require(configuration["relative_difference_reference"] == "public lens", "geometry reference mismatch")
    layers = geometry["layers"]
    require([item["layer"] for item in layers] == SOURCE_LAYERS, "geometry layer grid mismatch")
    aggregate = geometry["aggregate"]
    require(aggregate["layer_count"] == len(SOURCE_LAYERS), "geometry layer count mismatch")
    require(aggregate["all_rows_cosine"]["count"] == D_MODEL * len(SOURCE_LAYERS), "geometry row count mismatch")
    local_norm = math.sqrt(sum(item["local"]["frobenius_norm"] ** 2 for item in layers))
    public_norm = math.sqrt(sum(item["public"]["frobenius_norm"] ** 2 for item in layers))
    difference_norm = math.sqrt(sum(item["difference_frobenius_norm"] ** 2 for item in layers))
    inner_product = sum(
        item["frobenius_cosine"]
        * item["local"]["frobenius_norm"]
        * item["public"]["frobenius_norm"]
        for item in layers
    )
    _close(aggregate["local_total_frobenius_norm"], local_norm, "geometry local norm")
    _close(aggregate["public_total_frobenius_norm"], public_norm, "geometry public norm")
    _close(aggregate["difference_total_frobenius_norm"], difference_norm, "geometry difference norm")
    _close(aggregate["global_frobenius_cosine"], inner_product / (local_norm * public_norm), "geometry global cosine")
    _close(aggregate["global_relative_frobenius_difference"], difference_norm / public_norm, "geometry relative difference")
    for item in layers:
        layer = item["layer"]
        require(item["row_wise_cosine"]["count"] == D_MODEL, f"geometry layer {layer}: row count mismatch")
        _close(
            item["relative_frobenius_difference"],
            item["difference_frobenius_norm"] / item["public"]["frobenius_norm"],
            f"geometry layer {layer}: relative difference",
        )
        require(-1.0 <= item["frobenius_cosine"] <= 1.0, f"geometry layer {layer}: invalid cosine")
    _finite_numbers(geometry, "geometry")


def _average_ranks(values: Sequence[int]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _spearman(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    require(len(left) == len(right) and len(left) >= 2, "invalid Spearman inputs")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    coefficient = None if denominator == 0 else numerator / denominator
    return {
        "observations": len(left),
        "defined": coefficient is not None,
        "coefficient": coefficient,
    }


def _method_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(record["target_rank"]) for record in records]
    top1 = sum(rank == 1 for rank in ranks)
    top5 = sum(rank <= 5 for rank in ranks)
    return {
        "observations": len(ranks),
        "mean_target_rank": sum(ranks) / len(ranks),
        "target_top1_count": top1,
        "target_top1_rate": top1 / len(ranks),
        "target_top5_count": top5,
        "target_top5_rate": top5 / len(ranks),
    }


def _comparison_metrics(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    require(len(left) == len(right) and len(left) >= 2, "invalid comparison inputs")
    top1 = 0
    top5_exact = 0
    top5_overlap = 0
    left_ranks = []
    right_ranks = []
    for left_record, right_record in zip(left, right, strict=True):
        require(left_record["target_token_id"] == right_record["target_token_id"], "comparison target mismatch")
        left_tokens = left_record["token_ids"]
        right_tokens = right_record["token_ids"]
        left_top5 = set(left_tokens[:5])
        right_top5 = set(right_tokens[:5])
        top1 += left_tokens[0] == right_tokens[0]
        top5_exact += left_top5 == right_top5
        top5_overlap += len(left_top5 & right_top5)
        left_ranks.append(int(left_record["target_rank"]))
        right_ranks.append(int(right_record["target_rank"]))
    observations = len(left)
    return {
        "observations": observations,
        "top1_agreement_count": top1,
        "top1_agreement_rate": top1 / observations,
        "top5_exact_set_agreement_count": top5_exact,
        "top5_exact_set_agreement_rate": top5_exact / observations,
        "top5_overlap_count": top5_overlap,
        "top5_overlap_mean_fraction": top5_overlap / (5 * observations),
        "spearman_target_rank": _spearman(left_ranks, right_ranks),
    }


def _check_readout(record: dict[str, Any], target_id: int, label: str) -> None:
    require(record["target_token_id"] == target_id, f"{label}: target ID mismatch")
    for key in ("token_ids", "tokens", "scores"):
        require(len(record[key]) == TOP_K, f"{label}: {key} count mismatch")
    require(record["target_rank"] >= 1, f"{label}: invalid target rank")


def _derive_eval_summary(experiments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for layer in SOURCE_LAYERS:
        methods: dict[str, list[dict[str, Any]]] = {
            "logit_lens": [],
            "local_jacobian_lens": [],
            "public_jacobian_lens": [],
        }
        for experiment in experiments:
            layer_record = experiment["layers"][layer]
            for position, target in zip(layer_record["positions"], experiment["targets"], strict=True):
                require(position["token_position"] == target["token_position"], "evaluation target position mismatch")
                methods["logit_lens"].append(position["vanilla_logit_lens"])
                methods["local_jacobian_lens"].append(position["local_jacobian_lens"])
                methods["public_jacobian_lens"].append(position["public_jacobian_lens"])
        summaries.append(
            {
                "layer": layer,
                "methods": {
                    name: _method_metrics(records) for name, records in methods.items()
                },
                "comparisons": {
                    "local_vs_public": _comparison_metrics(
                        methods["local_jacobian_lens"], methods["public_jacobian_lens"]
                    ),
                    "logit_vs_public": _comparison_metrics(
                        methods["logit_lens"], methods["public_jacobian_lens"]
                    ),
                },
            }
        )
    return summaries


def _check_evaluation(evidence: Evidence) -> None:
    evaluation = evidence.evaluation
    require(evaluation["schema_version"] == 1, "evaluation schema mismatch")
    require(evaluation["status"] == "completed", "evaluation status mismatch")
    require(
        evaluation["scope"] == "held-out NF4 readout evaluation; no fitting or NVFP4 equivalence claim",
        "evaluation scope mismatch",
    )
    configuration = evaluation["configuration"]
    require(configuration["offline"] is True, "evaluation was not offline")
    require(configuration["deterministic_algorithms"] is True, "evaluation was not deterministic")
    require(configuration["layers"] == SOURCE_LAYERS, "evaluation layers mismatch")
    require(configuration["positions"] == POSITIONS, "evaluation positions mismatch")
    require(configuration["top_k"] == TOP_K, "evaluation top-k mismatch")
    require(configuration["target_semantics"] == "teacher_forced_next_token", "evaluation target semantics mismatch")

    model = evaluation["model"]
    require(model["repo_id"] == FIT_MODEL, "evaluation model mismatch")
    require(model["revision"] == FIT_MODEL_REVISION, "evaluation model revision mismatch")
    require(model["nf4_linear_count"] == NF4_MODULE_COUNT, "evaluation NF4 count mismatch")
    require(
        model["source_artifacts"] == evidence.provenance["contract"]["model_artifacts"],
        "evaluation model artifacts differ from fit",
    )
    weights = model["nf4_weights"]
    require(weights["module_count"] == NF4_MODULE_COUNT, "evaluation NF4 module mismatch")
    require(weights["aggregate_sha256"] == NF4_WEIGHTS_SHA256, "evaluation NF4 hash mismatch")
    require(weights["fit_aggregate_sha256"] == NF4_WEIGHTS_SHA256, "evaluation/fit NF4 hash mismatch")
    require(weights["matches_fit"] is True, "evaluation NF4 weights do not match fit")

    prompts = evaluation["prompts"]
    require(prompts["count"] == len(PROMPT_ROWS), "evaluation prompt count mismatch")
    require(prompts["token_count"] == PROMPT_TOKEN_COUNT, "evaluation prompt length mismatch")
    require(prompts["sha256"] == EVAL_PROMPTS_SHA256, "evaluation prompt hash mismatch")
    require(
        prompts["dataset"]
        == {
            "config": "wikitext-103-raw-v1",
            "repo": "Salesforce/wikitext",
            "revision": DATASET_REVISION,
            "split": "validation",
        },
        "evaluation dataset mismatch",
    )

    experiments = evaluation["experiments"]
    expected_ids = [f"validation-row-{row}" for row in PROMPT_ROWS]
    require([item["id"] for item in experiments] == expected_ids, "evaluation prompt IDs mismatch")
    require([item["dataset_row_index"] for item in experiments] == PROMPT_ROWS, "evaluation prompt rows mismatch")
    for experiment in experiments:
        label = experiment["id"]
        require(experiment["dataset_split"] == "validation", f"{label}: split mismatch")
        require(experiment["positions_requested"] == POSITIONS, f"{label}: requested positions mismatch")
        require(experiment["positions_resolved"] == POSITIONS, f"{label}: resolved positions mismatch")
        require(len(experiment["token_ids"]) == PROMPT_TOKEN_COUNT, f"{label}: token IDs mismatch")
        require(len(experiment["tokens"]) == PROMPT_TOKEN_COUNT, f"{label}: tokens mismatch")
        require(
            hashlib.sha256(experiment["text"].encode("utf-8")).hexdigest()
            == experiment["text_sha256"],
            f"{label}: text hash mismatch",
        )
        require([item["token_position"] for item in experiment["targets"]] == POSITIONS, f"{label}: target positions mismatch")
        for target in experiment["targets"]:
            position = target["token_position"]
            target_id = experiment["token_ids"][position + 1]
            require(target["source"] == "teacher_forced_next_token", f"{label}: target source mismatch")
            require(target["target_token_id"] == target_id, f"{label}: target token mismatch")
            _check_readout(target["final_model"], target_id, f"{label}: final model")
        require([item["layer"] for item in experiment["layers"]] == SOURCE_LAYERS, f"{label}: layer grid mismatch")
        for layer in experiment["layers"]:
            require([item["token_position"] for item in layer["positions"]] == POSITIONS, f"{label}: layer {layer['layer']} positions mismatch")
            for position_record, target in zip(layer["positions"], experiment["targets"], strict=True):
                for method in (
                    "vanilla_logit_lens",
                    "local_jacobian_lens",
                    "public_jacobian_lens",
                ):
                    _check_readout(
                        position_record[method],
                        target["target_token_id"],
                        f"{label}: layer {layer['layer']} {method}",
                    )
    require(evaluation["summary"] == _derive_eval_summary(experiments), "evaluation summary is inconsistent with readouts")
    _finite_numbers(evaluation, "evaluation")


def _strip_application(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "application"}


def _check_nvfp4_identity(evidence: Evidence) -> None:
    local = evidence.local_nvfp4
    public = evidence.public_nvfp4
    require(local["schema_version"] == public["schema_version"] == 2, "paired schema mismatch")
    require(local["model"] == public["model"], "paired NVFP4 models differ")
    model = local["model"]
    expected_model = {
        "repo_id": NVFP4_MODEL,
        "revision": NVFP4_REVISION,
        "config_sha256": NVFP4_CONFIG_SHA256,
        "index_sha256": NVFP4_INDEX_SHA256,
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
    }
    for key, value in expected_model.items():
        require(model.get(key) == value, f"NVFP4 model {key} mismatch")
    require(model["model_info"]["hidden_size"] == D_MODEL, "NVFP4 width mismatch")
    require(model["model_info"]["layer_count"] == 64, "NVFP4 layer count mismatch")

    require(_strip_application(local["lens"]) == evidence.verification, "local NVFP4 lens identity mismatch")
    expected_public = dict(evidence.geometry["artifacts"]["public"])
    expected_public.pop("kind")
    require(_strip_application(public["lens"]) == expected_public, "public NVFP4 lens identity mismatch")

    local_runtime = dict(local["runtime"])
    public_runtime = dict(public["runtime"])
    require(local_runtime.pop("model_load_seconds") > 0, "local NVFP4 load time invalid")
    require(public_runtime.pop("model_load_seconds") > 0, "public NVFP4 load time invalid")
    require(local_runtime == public_runtime, "paired NVFP4 runtime contracts differ")
    expected_runtime = {
        "capture_adapter": "vLLM apply_model forward hooks",
        "enforce_eager": True,
        "language_model_only": True,
        "max_model_len": 256,
        "mtp_enabled": False,
        "readout_dtype": "torch.bfloat16",
        "transport_dtype": "torch.float32",
    }
    for key, value in expected_runtime.items():
        require(local_runtime.get(key) == value, f"NVFP4 runtime {key} mismatch")


def _derived_reconstruction(experiment: dict[str, Any], label: str) -> tuple[bool, bool]:
    norm = experiment["final_norm_reconstruction"]
    logits = experiment["final_logits_reconstruction"]
    norm_max_pass = norm["max_abs_error"] <= norm["max_abs_tolerance"]
    norm_rms_pass = norm["rms_error"] <= norm["rms_tolerance"]
    derived_norm = norm_max_pass and norm_rms_pass
    require(norm["within_tolerance"] == derived_norm, f"{label}: norm tolerance flag mismatch")

    prefix = logits["top_k_prefix"]
    derived_prefix = all(
        reconstructed["token_ids"][:prefix] == captured["token_ids"][:prefix]
        for reconstructed, captured in zip(
            experiment["final_model_readout"],
            experiment["captured_final_model_readout"],
            strict=True,
        )
    )
    require(logits["top_k_prefix_token_ids_match"] == derived_prefix, f"{label}: top-k prefix flag mismatch")
    logit_max_pass = logits["max_abs_error"] <= logits["max_abs_tolerance"]
    logit_rms_pass = logits["rms_error"] <= logits["rms_tolerance"]
    derived_logits = logit_max_pass and logit_rms_pass and derived_prefix
    require(logits["within_tolerance"] == derived_logits, f"{label}: logit tolerance flag mismatch")
    require(norm_rms_pass and logit_rms_pass, f"{label}: RMS reconstruction check failed")
    reconstruction_passes = derived_norm and derived_logits
    max_abs_failed = not norm_max_pass or not logit_max_pass
    require(reconstruction_passes or max_abs_failed, f"{label}: failure is not accompanied by a max-abs violation")
    return reconstruction_passes, max_abs_failed


def _paired_experiment_check(
    evaluation: dict[str, Any],
    local: dict[str, Any],
    public: dict[str, Any],
) -> tuple[bool, bool]:
    label = local["id"]
    require(local["id"] == public["id"] == f"wikitext-{evaluation['id']}", f"{label}: paired ID mismatch")
    identity_fields = (
        "prompt",
        "prompt_token_ids",
        "prompt_tokens",
        "positions_requested",
        "positions_resolved",
        "capture_positions_resolved",
        "final_validation_position",
        "position_tokens",
        "generated_token_id",
        "generated_token",
        "generated_text",
        "final_model_readout",
        "captured_final_model_readout",
    )
    for key in identity_fields:
        require(local[key] == public[key], f"{label}: paired {key} mismatch")
    require(local["prompt_token_ids"] == evaluation["token_ids"], f"{label}: NF4/NVFP4 token IDs mismatch")
    require(local["prompt_tokens"] == evaluation["tokens"], f"{label}: NF4/NVFP4 tokens mismatch")
    require(local["prompt"] == "".join(local["prompt_tokens"]), f"{label}: prompt decode mismatch")
    require(evaluation["text"].startswith(local["prompt"]), f"{label}: prompt text mismatch")
    require(local["positions_requested"] == POSITIONS, f"{label}: requested positions mismatch")
    require(local["positions_resolved"] == POSITIONS, f"{label}: resolved positions mismatch")
    require(local["capture_positions_resolved"] == CAPTURE_POSITIONS, f"{label}: capture positions mismatch")
    require(local["final_validation_position"] == CAPTURE_POSITIONS[-1], f"{label}: final position mismatch")
    require(
        local["position_tokens"] == [local["prompt_tokens"][position] for position in POSITIONS],
        f"{label}: position tokens mismatch",
    )

    require([item["layer"] for item in local["layers"]] == SOURCE_LAYERS, f"{label}: local layer grid mismatch")
    require([item["layer"] for item in public["layers"]] == SOURCE_LAYERS, f"{label}: public layer grid mismatch")
    for local_layer, public_layer in zip(local["layers"], public["layers"], strict=True):
        require(local_layer["layer_type"] == public_layer["layer_type"], f"{label}: layer type mismatch")
        for local_position, public_position in zip(local_layer["positions"], public_layer["positions"], strict=True):
            require(
                (local_position["capture_index"], local_position["token_position"])
                == (public_position["capture_index"], public_position["token_position"]),
                f"{label}: paired layer position mismatch",
            )
            require(local_position["logit_lens"] == public_position["logit_lens"], f"{label}: paired logit lens mismatch")
            require(local_position["token_position"] in POSITIONS, f"{label}: unexpected layer position")
            for position in (local_position, public_position):
                for method in ("logit_lens", "jacobian_lens"):
                    require(len(position[method]["token_ids"]) == TOP_K, f"{label}: readout top-k mismatch")

    final_index = local["capture_positions_resolved"].index(local["final_validation_position"])
    generated = local["generated_token_id"]
    derived_top1 = (
        local["final_model_readout"][final_index]["token_ids"][0] == generated
        and local["captured_final_model_readout"][final_index]["token_ids"][0] == generated
    )
    require(derived_top1, f"{label}: final greedy top-1 mismatch")
    require(local["final_layer_top1_matches_greedy"] == derived_top1, f"{label}: local top-1 flag mismatch")
    require(public["final_layer_top1_matches_greedy"] == derived_top1, f"{label}: public top-1 flag mismatch")
    require(local["final_norm_reconstruction"] == public["final_norm_reconstruction"], f"{label}: paired norm diagnostics differ")
    require(local["final_logits_reconstruction"] == public["final_logits_reconstruction"], f"{label}: paired logit diagnostics differ")
    reconstruction_passes, max_abs_failed = _derived_reconstruction(local, label)
    return reconstruction_passes, max_abs_failed


def _cross_model_metrics(
    evaluation: dict[str, Any], report: dict[str, Any], eval_method: str
) -> dict[str, float]:
    layer_top1 = []
    layer_top5 = []
    layer_spearman = []
    for layer in SOURCE_LAYERS:
        nf4_records = []
        nvfp4_records = []
        for eval_experiment, nvfp4_experiment in zip(
            evaluation["experiments"], report["experiments"], strict=True
        ):
            nf4_records.extend(
                position[eval_method]
                for position in eval_experiment["layers"][layer]["positions"]
            )
            nvfp4_records.extend(
                position["jacobian_lens"]
                for position in nvfp4_experiment["layers"][layer]["positions"]
            )
        metrics = _comparison_metrics(nf4_records, nvfp4_records)
        layer_top1.append(metrics["top1_agreement_rate"])
        layer_top5.append(metrics["top5_overlap_mean_fraction"])
        require(metrics["spearman_target_rank"]["defined"], f"layer {layer}: undefined cross-model Spearman")
        layer_spearman.append(metrics["spearman_target_rank"]["coefficient"])
    return {
        "mean_layer_top1_agreement": sum(layer_top1) / len(layer_top1),
        "mean_layer_top5_overlap": sum(layer_top5) / len(layer_top5),
        "mean_layer_target_rank_spearman": sum(layer_spearman) / len(layer_spearman),
    }


def _check_paired_nvfp4(evidence: Evidence) -> dict[str, dict[str, float]]:
    _check_nvfp4_identity(evidence)
    local = evidence.local_nvfp4
    public = evidence.public_nvfp4
    expected_assertions = {
        "all_final_adapter_reconstructions_within_tolerance": False,
        "all_final_layer_top1_match_greedy": True,
        "lens_hash_matches": True,
        "lens_metadata_matches": True,
        "model_architecture_matches": True,
    }
    require(local["assertions"] == public["assertions"] == expected_assertions, "paired assertions mismatch")
    require(local["status"] == public["status"] == "failed", "paired status mismatch")
    require(
        [item["id"] for item in local["experiments"]]
        == [f"wikitext-validation-row-{row}" for row in PROMPT_ROWS],
        "local NVFP4 prompt IDs mismatch",
    )
    require(
        [item["id"] for item in public["experiments"]]
        == [f"wikitext-validation-row-{row}" for row in PROMPT_ROWS],
        "public NVFP4 prompt IDs mismatch",
    )

    reconstruction_by_id = {}
    max_abs_by_id = {}
    for eval_experiment, local_experiment, public_experiment in zip(
        evidence.evaluation["experiments"],
        local["experiments"],
        public["experiments"],
        strict=True,
    ):
        reconstruction, max_abs_failed = _paired_experiment_check(
            eval_experiment, local_experiment, public_experiment
        )
        reconstruction_by_id[local_experiment["id"]] = reconstruction
        max_abs_by_id[local_experiment["id"]] = max_abs_failed
    require(
        {key for key, value in reconstruction_by_id.items() if value}
        == {"wikitext-validation-row-42"},
        "unexpected per-prompt reconstruction pass pattern",
    )
    require(
        {key for key, value in max_abs_by_id.items() if value}
        == {
            "wikitext-validation-row-3",
            "wikitext-validation-row-18",
            "wikitext-validation-row-49",
        },
        "unexpected per-prompt max-abs failure pattern",
    )
    require(
        local["experiments"][1]["final_logits_reconstruction"][
            "top_k_prefix_token_ids_match"
        ]
        is False,
        "row 18 top-k prefix diagnostic mismatch",
    )
    require(
        not all(reconstruction_by_id.values()),
        "paired reconstruction assertion should fail",
    )
    _finite_numbers(local, "local NVFP4 report")
    _finite_numbers(public, "public NVFP4 report")

    cross_model = {
        "local_nf4_lens_on_nvfp4": _cross_model_metrics(
            evidence.evaluation, local, "local_jacobian_lens"
        ),
        "public_lens_on_nvfp4": _cross_model_metrics(
            evidence.evaluation, public, "public_jacobian_lens"
        ),
    }
    for comparison, expected_metrics in EXPECTED_CROSS_MODEL.items():
        for metric, expected in expected_metrics.items():
            _close(cross_model[comparison][metric], expected, f"{comparison} {metric}")
    return cross_model


def validate_evidence(evidence: Evidence) -> dict[str, dict[str, float]]:
    _check_local_verification(evidence)
    _check_provenance(evidence)
    _check_geometry(evidence)
    _check_evaluation(evidence)
    return _check_paired_nvfp4(evidence)


def main() -> int:
    cross_model = validate_evidence(load_evidence())
    print(
        "NF4 Jacobian Lens evidence integrity passed: provenance, geometry, "
        "held-out evaluation, paired NVFP4 diagnostics"
    )
    print(json.dumps({"cross_model": cross_model}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
