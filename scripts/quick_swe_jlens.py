#!/usr/bin/env python3
"""Replay selected agent completions once and emit a compact J-lens timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
DEFAULT_SOURCE_REPORT = (
    ROOT / "validation/jlens-swe-qwen-code-native-nvfp4-ste-2026-07-17.json"
)
DEFAULT_OUTPUT_DIR = ROOT / ".cache/swe_jlens_quick"
DEFAULT_LAYERS = (24, 31, 32, 39, 40, 62)
NATIVE_LENS_PATH = ROOT / ".cache/Qwen3.6-27B-jlens-nvfp4-ste-n10-fp32.pt"
NATIVE_LENS_SHA256 = (
    "82be61c805d127427b37b2b4715885b756c2ca7af96291578fa4da9cd783e057"
)
NATIVE_PROVENANCE_PATH = ROOT / ".cache/nvfp4_ste_fit/final-mean/metadata.json"
NATIVE_STATE_PATH = ROOT / ".cache/nvfp4_ste_fit/state.json"
NATIVE_STATE_SHA256 = (
    "f5ee70cfda416327be6b2583a67f5662cbe4036dbc68ce4ba470884383bfbcf6"
)


class QuickReplayError(RuntimeError):
    """Raised when quick-replay input or output violates its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise QuickReplayError(f"{label} must be a JSON object")
    return value


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must not contain duplicates")
    if any(layer < 0 or layer > 62 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be in 0..62")
    return tuple(sorted(layers))


def parse_request_selection(
    value: str, available: Sequence[int]
) -> tuple[int, ...]:
    """Parse ``all`` or a comma-separated list of indices and ranges."""

    available_order = tuple(available)
    available_set = set(available_order)
    if not available_order or len(available_set) != len(available_order):
        raise QuickReplayError("available request indices must be unique and nonempty")
    if value.strip().lower() == "all":
        return available_order

    selected: list[int] = []
    try:
        for raw_part in value.split(","):
            part = raw_part.strip()
            if not part:
                raise ValueError
            if "-" in part:
                start_text, stop_text = part.split("-", 1)
                start, stop = int(start_text), int(stop_text)
                if start > stop:
                    raise ValueError
                selected.extend(range(start, stop + 1))
            else:
                selected.append(int(part))
    except ValueError as exc:
        raise QuickReplayError(
            "--requests must be 'all' or indices/ranges such as 1,3-5,9"
        ) from exc
    if not selected or any(index <= 0 for index in selected):
        raise QuickReplayError("request indices must be positive")
    if len(set(selected)) != len(selected):
        raise QuickReplayError("request selection contains duplicates")
    missing = [index for index in selected if index not in available_set]
    if missing:
        raise QuickReplayError(
            f"source report does not contain selected requests: {missing}; "
            f"available={list(available_order)}"
        )
    return tuple(selected)


def request_index(experiment: Mapping[str, Any]) -> int:
    metadata = require_mapping(experiment.get("metadata"), "experiment metadata")
    stage = metadata.get("stage")
    candidate = stage.get("request_index") if isinstance(stage, dict) else None
    if candidate is None:
        candidate = metadata.get("request_index")
    if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
        raise QuickReplayError("experiment metadata has no positive request index")
    return candidate


def extract_request_prompts(
    report: Mapping[str, Any], selection: str
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Extract exact token inputs for selected model completions in one agent loop."""

    model = require_mapping(report.get("model"), "source report model")
    if model.get("repo_id") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise QuickReplayError("source report is not the pinned Qwen3.6 NVFP4 model")
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise QuickReplayError("source report experiments must be a nonempty list")

    by_request: dict[int, Mapping[str, Any]] = {}
    for ordinal, raw_experiment in enumerate(experiments, 1):
        experiment = require_mapping(raw_experiment, f"source experiment {ordinal}")
        index = request_index(experiment)
        if index in by_request:
            raise QuickReplayError(f"duplicate source request index {index}")
        by_request[index] = experiment
    available = tuple(sorted(by_request))
    selected = parse_request_selection(selection, available)

    prompts: list[dict[str, Any]] = []
    for index in selected:
        experiment = by_request[index]
        token_ids = experiment.get("prompt_token_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise QuickReplayError(f"request {index} has invalid prompt_token_ids")
        if len(token_ids) + 1 > 16384:
            raise QuickReplayError(
                f"request {index} needs {len(token_ids) + 1} model slots; quick "
                "runtime is pinned to 16384"
            )
        prompts.append(
            {
                "id": str(experiment.get("id", f"request-{index:02d}")),
                "token_ids": list(token_ids),
                "metadata": dict(
                    require_mapping(experiment.get("metadata"), "source metadata")
                ),
            }
        )
    return prompts, selected


def native_lens_args() -> list[str]:
    return [
        "--lens-kind",
        "nvfp4-ste",
        "--lens-path",
        str(NATIVE_LENS_PATH),
        "--lens-sha256",
        NATIVE_LENS_SHA256,
        "--lens-provenance",
        str(NATIVE_PROVENANCE_PATH),
        "--lens-state",
        str(NATIVE_STATE_PATH),
        "--lens-state-sha256",
        NATIVE_STATE_SHA256,
    ]


def build_runner_command(
    *, lens_kind: str, prompt_path: Path, report_path: Path, layers: Sequence[int]
) -> list[str]:
    command = [str(ROOT / "scripts/run_jlens_nvfp4.sh")]
    if lens_kind == "native":
        command.extend(native_lens_args())
    elif lens_kind == "public":
        command.extend(("--lens-kind", "public"))
    else:
        raise QuickReplayError(f"unsupported lens kind: {lens_kind}")
    command.extend(
        (
            "--prompts-file",
            str(prompt_path),
            "--layers",
            ",".join(str(layer) for layer in layers),
            "--positions=-1",
            "--top-k",
            "10",
            "--max-model-len",
            "16384",
            "--max-num-batched-tokens",
            "4096",
            "--mamba-block-size",
            "1024",
            "--enable-prefix-caching",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--stream-final-only",
            "--gpu-memory-utilization",
            "0.78",
            "--output",
            str(report_path),
        )
    )
    return command


def top_tokens(readout: Mapping[str, Any]) -> list[str]:
    tokens = readout.get("tokens")
    if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
        raise QuickReplayError("runner readout is missing decoded top tokens")
    return list(tokens)


def summarize_experiment(experiment: Mapping[str, Any]) -> dict[str, Any]:
    metadata = require_mapping(experiment.get("metadata"), "runner metadata")
    stage = metadata.get("stage")
    stage_name = stage.get("name") if isinstance(stage, dict) else None
    sampled_next = metadata.get("sampled_next")
    sampled_token = (
        sampled_next.get("first_token_text") if isinstance(sampled_next, dict) else None
    )
    layers = experiment.get("layers")
    if not isinstance(layers, list) or not layers:
        raise QuickReplayError("runner experiment has no layer readouts")
    layer_rows = []
    for raw_layer in layers:
        layer = require_mapping(raw_layer, "runner layer")
        positions = layer.get("positions")
        if not isinstance(positions, list) or len(positions) != 1:
            raise QuickReplayError("quick layers must contain one final-boundary position")
        position = require_mapping(positions[0], "runner position")
        jacobian = require_mapping(position.get("jacobian_lens"), "Jacobian readout")
        logit = require_mapping(position.get("logit_lens"), "logit readout")
        layer_rows.append(
            {
                "layer": layer.get("layer"),
                "layer_type": layer.get("layer_type"),
                "jacobian_top_tokens": top_tokens(jacobian),
                "jacobian_target_rank": jacobian.get("target_rank"),
                "logit_lens_top_tokens": top_tokens(logit),
                "logit_target_rank": logit.get("target_rank"),
            }
        )
    norm = require_mapping(
        experiment.get("final_norm_reconstruction"), "final norm reconstruction"
    )
    logits = require_mapping(
        experiment.get("final_logits_reconstruction"), "final logits reconstruction"
    )
    return {
        "request_index": request_index(experiment),
        "stage_name": stage_name,
        "completion_boundary": experiment.get("id"),
        "prompt_tokens": len(experiment.get("prompt_token_ids", [])),
        "generated_first_token": experiment.get("generated_token"),
        "original_sampled_first_token": sampled_token,
        "strict_adapter": {
            "final_norm_within_tolerance": norm.get("within_tolerance"),
            "final_logits_within_tolerance": logits.get("within_tolerance"),
            "final_top1_matches_greedy": experiment.get(
                "final_layer_top1_matches_greedy"
            ),
        },
        "layers": layer_rows,
    }


def summarize_report(
    report: Mapping[str, Any],
    *,
    prompts: Sequence[Mapping[str, Any]],
    selected_requests: Sequence[int],
    source_report_path: Path,
    source_report_sha256: str,
    prompt_path: Path,
    prompt_sha256: str,
) -> dict[str, Any]:
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != len(prompts):
        raise QuickReplayError("runner report prompt count does not match quick input")
    expected_ids = [prompt["id"] for prompt in prompts]
    actual_ids = [experiment.get("id") for experiment in experiments]
    if actual_ids != expected_ids:
        raise QuickReplayError("runner report prompt order/IDs do not match quick input")
    for prompt, experiment in zip(prompts, experiments, strict=True):
        if experiment.get("prompt_token_ids") != prompt["token_ids"]:
            raise QuickReplayError(f"runner token IDs differ for {prompt['id']}")

    timeline = [summarize_experiment(require_mapping(item, "runner experiment")) for item in experiments]
    if tuple(row["request_index"] for row in timeline) != tuple(selected_requests):
        raise QuickReplayError("runner request order does not match selection")
    status = report.get("status")
    if status not in ("passed", "failed"):
        raise QuickReplayError(f"unexpected runner status: {status!r}")
    lens = require_mapping(report.get("lens"), "runner lens")
    lens_fit_n = lens.get("n_prompts")
    if isinstance(lens_fit_n, bool) or not isinstance(lens_fit_n, int) or lens_fit_n <= 0:
        raise QuickReplayError("runner lens has no positive fit prompt count")
    return {
        "schema_version": 1,
        "kind": "quick_swe_agent_completion_jlens_timeline",
        "sample_sizes": {
            "task_count": 1,
            "task_request_count": len(timeline),
            "lens_fit_prompt_count": lens_fit_n,
            "explanation": (
                "task_request_count is model completions in the agent loop; "
                "lens_fit_prompt_count is the separate background corpus used once "
                "to estimate the lens"
            ),
        },
        "runner_status": status,
        "lens": {
            "kind": lens.get("kind", "public_pretrained"),
            "sha256": lens.get("sha256"),
            "application": lens.get("application"),
        },
        "elapsed_seconds": report.get("elapsed_seconds"),
        "runtime": report.get("runtime"),
        "timeline": timeline,
        "source": {
            "report_path": str(source_report_path),
            "report_sha256": source_report_sha256,
            "prompt_bundle_path": str(prompt_path),
            "prompt_bundle_sha256": prompt_sha256,
        },
        "interpretation": (
            "Each row is a next-token distribution at one frozen model-completion "
            "boundary. It is sparse task coverage, not a hidden chain-of-thought "
            "transcript or a causal explanation."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        default="all",
        help="request indices/ranges such as 1,3-5,9, or all (default: all)",
    )
    parser.add_argument(
        "--lens",
        choices=("public", "native"),
        default="public",
        help="reuse the public n=1000 lens by default; native needs local fit files",
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=DEFAULT_LAYERS,
        help="source layers (default: 24,31,32,39,40,62)",
    )
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write exact inputs and print the GPU command without loading artifacts",
    )
    parser.add_argument(
        "--require-strict",
        action="store_true",
        help="return nonzero when replay completes with a failed adapter certificate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_report_path = args.source_report.expanduser().resolve(strict=True)
    source_report_sha256 = sha256_file(source_report_path)
    source_value = json.loads(source_report_path.read_text(encoding="utf-8"))
    prompts, selected = extract_request_prompts(
        require_mapping(source_value, "source report"), args.requests
    )

    output_dir = args.output_dir.expanduser().resolve()
    selection_label = "-".join(str(index) for index in selected)
    stem = f"requests-{selection_label}-{args.lens}"
    prompt_path = output_dir / f"{stem}-prompts.json"
    report_path = output_dir / f"{stem}-report.json"
    summary_path = output_dir / f"{stem}-timeline.json"
    atomic_write_json(prompt_path, prompts)
    prompt_sha256 = sha256_file(prompt_path)
    command = build_runner_command(
        lens_kind=args.lens,
        prompt_path=prompt_path,
        report_path=report_path,
        layers=args.layers,
    )
    if args.dry_run:
        print(shlex.join(command))
        print(f"wrote exact {len(prompts)}-request input: {prompt_path}")
        return 0

    report_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)
    # The runner also prints its complete multi-megabyte JSON report. The same
    # report is already written to report_path, so keep quick replay output terse.
    completed = subprocess.run(command, check=False, stdout=subprocess.DEVNULL)
    if completed.returncode not in (0, 1):
        raise QuickReplayError(
            f"runner failed before a valid report (exit {completed.returncode})"
        )
    if not report_path.is_file():
        raise QuickReplayError("runner did not write the requested report")
    report_value = json.loads(report_path.read_text(encoding="utf-8"))
    report = require_mapping(report_value, "runner report")
    expected_code = 0 if report.get("status") == "passed" else 1
    if completed.returncode != expected_code:
        raise QuickReplayError(
            "runner exit code disagrees with report status: "
            f"exit={completed.returncode}, status={report.get('status')!r}"
        )
    summary = summarize_report(
        report,
        prompts=prompts,
        selected_requests=selected,
        source_report_path=source_report_path,
        source_report_sha256=source_report_sha256,
        prompt_path=prompt_path,
        prompt_sha256=prompt_sha256,
    )
    atomic_write_json(summary_path, summary)
    print(f"wrote raw report: {report_path}")
    print(f"wrote compact {len(prompts)}-request timeline: {summary_path}")
    if report.get("status") == "failed":
        print("NOTE: replay completed, but its strict adapter certificate failed")
    return completed.returncode if args.require_strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
