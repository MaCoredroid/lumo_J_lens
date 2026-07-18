#!/usr/bin/env python3
"""Tests for the one-load quick SWE J-lens timeline."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quick_swe_jlens", ROOT / "scripts" / "quick_swe_jlens.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source_experiment(index: int, token_ids: list[int]) -> dict[str, object]:
    return {
        "id": f"request-{index:02d}",
        "prompt_token_ids": token_ids,
        "metadata": {
            "stage": {"request_index": index, "name": f"stage-{index}"},
            "sampled_next": {"first_token_text": f"sampled-{index}"},
        },
    }


def source_report() -> dict[str, object]:
    return {
        "model": {
            "repo_id": MODULE.MODEL_REPO,
            "revision": MODULE.MODEL_REVISION,
        },
        "experiments": [
            source_experiment(1, [10, 11]),
            source_experiment(2, [20, 21, 22]),
            source_experiment(3, [30, 31, 32, 33]),
        ],
    }


def runner_experiment(prompt: dict[str, object]) -> dict[str, object]:
    index = prompt["metadata"]["stage"]["request_index"]
    readout = {
        "tokens": ["alpha", "beta"],
        "target_rank": index,
    }
    return {
        "id": prompt["id"],
        "prompt_token_ids": prompt["token_ids"],
        "metadata": prompt["metadata"],
        "generated_token": f"greedy-{index}",
        "final_layer_top1_matches_greedy": True,
        "final_norm_reconstruction": {"within_tolerance": True},
        "final_logits_reconstruction": {"within_tolerance": index != 2},
        "layers": [
            {
                "layer": 31,
                "layer_type": "full_attention",
                "positions": [
                    {
                        "jacobian_lens": dict(readout),
                        "logit_lens": dict(readout),
                    }
                ],
            }
        ],
    }


class QuickSweJlensTest(unittest.TestCase):
    def test_request_selection_supports_all_lists_and_ranges(self) -> None:
        available = (1, 2, 3, 5, 9)
        self.assertEqual(
            MODULE.parse_request_selection("all", available), available
        )
        self.assertEqual(
            MODULE.parse_request_selection("1-3,5,9", available), available
        )
        with self.assertRaisesRegex(MODULE.QuickReplayError, "duplicates"):
            MODULE.parse_request_selection("1-2,2", available)
        with self.assertRaisesRegex(MODULE.QuickReplayError, "does not contain"):
            MODULE.parse_request_selection("4", available)

    def test_extracts_multiple_exact_prompts_in_requested_order(self) -> None:
        prompts, selected = MODULE.extract_request_prompts(source_report(), "3,1")
        self.assertEqual(selected, (3, 1))
        self.assertEqual([prompt["id"] for prompt in prompts], ["request-03", "request-01"])
        self.assertEqual(prompts[0]["token_ids"], [30, 31, 32, 33])
        self.assertEqual(prompts[0]["metadata"]["stage"]["name"], "stage-3")

    def test_tracked_report_exposes_nine_agent_completions(self) -> None:
        report = json.loads(MODULE.DEFAULT_SOURCE_REPORT.read_text(encoding="utf-8"))
        prompts, selected = MODULE.extract_request_prompts(report, "all")
        self.assertEqual(selected, tuple(range(1, 10)))
        self.assertEqual(
            [len(prompt["token_ids"]) for prompt in prompts],
            [11861, 12148, 12743, 13629, 13883, 14522, 15073, 15327, 15678],
        )

    def test_public_command_uses_one_pinned_long_context_invocation(self) -> None:
        command = MODULE.build_runner_command(
            lens_kind="public",
            prompt_path=Path("prompts.json"),
            report_path=Path("report.json"),
            layers=(24, 31, 32),
        )
        self.assertEqual(command.count("--prompts-file"), 1)
        self.assertIn("--lens-kind", command)
        self.assertEqual(command[command.index("--lens-kind") + 1], "public")
        self.assertEqual(command[command.index("--layers") + 1], "24,31,32")
        self.assertIn("--stream-final-only", command)
        self.assertIn("--enable-prefix-caching", command)
        self.assertEqual(
            command[command.index("--max-num-batched-tokens") + 1], "4096"
        )

    def test_summary_separates_task_request_n_from_lens_fit_n(self) -> None:
        prompts, selected = MODULE.extract_request_prompts(source_report(), "1-2")
        report = {
            "status": "failed",
            "elapsed_seconds": 12.5,
            "lens": {
                "n_prompts": 1000,
                "sha256": "a" * 64,
                "application": "public control",
            },
            "runtime": {"mtp_enabled": False},
            "experiments": [runner_experiment(prompt) for prompt in prompts],
        }
        summary = MODULE.summarize_report(
            report,
            prompts=prompts,
            selected_requests=selected,
            source_report_path=Path("source.json"),
            source_report_sha256="b" * 64,
            prompt_path=Path("prompts.json"),
            prompt_sha256="c" * 64,
        )
        self.assertEqual(summary["sample_sizes"]["task_request_count"], 2)
        self.assertEqual(summary["sample_sizes"]["lens_fit_prompt_count"], 1000)
        self.assertEqual(summary["elapsed_seconds"], 12.5)
        self.assertEqual(
            [row["request_index"] for row in summary["timeline"]], [1, 2]
        )
        self.assertEqual(summary["timeline"][1]["stage_name"], "stage-2")
        self.assertEqual(
            summary["timeline"][0]["original_sampled_first_token"], "sampled-1"
        )

    def test_dry_run_needs_no_model_or_lens_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(source_report()), encoding="utf-8")
            output = root / "output"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = MODULE.main(
                    [
                        "--source-report",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--requests",
                        "1,3",
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn("--lens-kind public", stdout.getvalue())
            prompt_files = list(output.glob("*-prompts.json"))
            self.assertEqual(len(prompt_files), 1)
            bundle = json.loads(prompt_files[0].read_text(encoding="ascii"))
            self.assertEqual([item["id"] for item in bundle], ["request-01", "request-03"])

    def test_run_removes_stale_outputs_before_starting_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(source_report()), encoding="utf-8")
            output = root / "output"
            output.mkdir()
            stem = "requests-1-public"
            report = output / f"{stem}-report.json"
            timeline = output / f"{stem}-timeline.json"
            report.write_text("stale", encoding="ascii")
            timeline.write_text("stale", encoding="ascii")
            completed = MODULE.subprocess.CompletedProcess([], 1)
            with mock.patch.object(
                MODULE.subprocess, "run", return_value=completed
            ) as runner:
                with self.assertRaisesRegex(
                    MODULE.QuickReplayError, "runner did not write"
                ):
                    MODULE.main(
                        [
                            "--source-report",
                            str(source),
                            "--output-dir",
                            str(output),
                            "--requests",
                            "1",
                        ]
                    )
            self.assertFalse(report.exists())
            self.assertFalse(timeline.exists())
            self.assertEqual(runner.call_args.kwargs["stdout"], MODULE.subprocess.DEVNULL)

    def test_cli_defaults_to_public_all_requests(self) -> None:
        args = MODULE.parse_args([])
        self.assertEqual(args.lens, "public")
        self.assertEqual(args.requests, "all")
        self.assertEqual(args.layers, MODULE.DEFAULT_LAYERS)


if __name__ == "__main__":
    unittest.main()
