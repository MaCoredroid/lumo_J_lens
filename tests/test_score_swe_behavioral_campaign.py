import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_swe_behavioral_campaign.sh"
SWE_PYTHON = ROOT / ".venv-swe" / "bin" / "python"
MODEL_NAME = "qwen3.6-27b-nvfp4-mtp::qwen-code-0.19.4"


class BehavioralCampaignScorerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary_directory.name)
        self.run_root = self.temp / "run"
        self.run_root.mkdir()
        self.instance_ids = [f"repo{i}__repo{i}-{1000 + i}" for i in range(10)]
        self.config_path = self.temp / "campaign.json"
        self.config = {
            "schema_version": 1,
            "kind": "swe_verified_behavioral_trajectory_campaign",
            "dataset": {
                "repo_id": "princeton-nlp/SWE-bench_Verified",
                "revision": "a" * 40,
            },
            "selection": {"lens_outputs_used": False},
            "generation": {
                "served_model": "qwen3.6-27b-nvfp4",
                "qwen_code_version": "0.19.4",
            },
            "instance_ids": self.instance_ids,
        }
        self._write_json(self.config_path, self.config)
        self._write_json(
            self.run_root / "dataset.json",
            [{"instance_id": instance_id} for instance_id in self.instance_ids],
        )
        self.image_records = []
        self.docker_inspections = {}
        for index, instance_id in enumerate(self.instance_ids):
            tag = (
                "swebench/sweb.eval.x86_64."
                + instance_id.replace("__", "_1776_")
                + ":latest"
            )
            image_id = "sha256:" + f"{index + 1:064x}"
            digest = tag.removesuffix(":latest") + "@sha256:" + f"{100 + index:064x}"
            record = {
                "instance_id": instance_id,
                "tag": tag,
                "image_id": image_id,
                "repo_digests": [digest],
            }
            self.image_records.append(record)
            self.docker_inspections[tag] = {
                "Architecture": "amd64",
                "Id": image_id,
                "RepoDigests": [digest],
            }
        self.image_manifest_path = self.run_root / "image_manifest.json"
        self._write_image_manifest()

        predictions_path = self.run_root / "generation" / "verified" / "predictions.jsonl"
        predictions_path.parent.mkdir(parents=True)
        source_rows = []
        for index, instance_id in enumerate(self.instance_ids):
            if index == 2:
                continue
            source_rows.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": MODEL_NAME,
                    "model_patch": "   " if index == 1 else f"patch-{index}",
                }
            )
        predictions_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in source_rows
            ),
            encoding="ascii",
        )

        self.fake_bin = self.temp / "bin"
        self.fake_bin.mkdir()
        self.docker_map_path = self.temp / "docker-images.json"
        self._write_json(self.docker_map_path, self.docker_inspections)
        self._write_executable(
            self.fake_bin / "docker",
            f"""#!{sys.executable}
import json, os, sys
images = json.load(open(os.environ["FAKE_DOCKER_IMAGES"]))
if sys.argv[1:3] != ["image", "inspect"] or sys.argv[3] not in images:
    raise SystemExit(2)
print(json.dumps([images[sys.argv[3]]]))
""",
        )
        self._write_executable(
            self.fake_bin / "curl",
            "#!/usr/bin/env bash\nexit 1\n",
        )
        self.fake_modules = self.temp / "modules"
        (self.fake_modules / "swebench" / "harness").mkdir(parents=True)
        (self.fake_modules / "swebench" / "__init__.py").write_text("", encoding="ascii")
        (self.fake_modules / "swebench" / "harness" / "__init__.py").write_text(
            "", encoding="ascii"
        )
        (self.fake_modules / "swebench" / "harness" / "run_evaluation.py").write_text(
            self._fake_harness_source(), encoding="ascii"
        )
        self.invocations_path = self.temp / "harness-invocations.jsonl"

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_json(path: Path, value) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="ascii")
        path.chmod(0o755)

    def _write_image_manifest(self, *, config_hash: str | None = None) -> None:
        config_bytes = self.config_path.read_bytes()
        self._write_json(
            self.image_manifest_path,
            {
                "schema_version": 1,
                "kind": "swe_verified_behavioral_campaign_image_manifest",
                "campaign_config_sha256": config_hash
                or hashlib.sha256(config_bytes).hexdigest(),
                "dataset": self.config["dataset"],
                "images": self.image_records,
            },
        )

    @staticmethod
    def _fake_harness_source() -> str:
        return r'''import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name")
parser.add_argument("--split")
parser.add_argument("--predictions_path")
parser.add_argument("--instance_ids", nargs="+")
parser.add_argument("--run_id")
parser.add_argument("--namespace")
parser.add_argument("--cache_level")
parser.add_argument("--clean")
parser.add_argument("--max_workers")
parser.add_argument("--timeout")
parser.add_argument("--report_dir")
args = parser.parse_args()
with open(os.environ["FAKE_HARNESS_INVOCATIONS"], "a", encoding="ascii") as handle:
    handle.write(json.dumps(vars(args), sort_keys=True) + "\n")
mode = os.environ.get("FAKE_HARNESS_MODE", "success")
if mode == "exit_7":
    print("mock harness stdout before failure")
    print("mock harness stderr before failure", file=__import__("sys").stderr)
    raise SystemExit(7)

rows = [json.loads(line) for line in open(args.predictions_path) if line.strip()]
predictions = {row["instance_id"]: row for row in rows}
empty = sorted(
    instance_id
    for instance_id in args.instance_ids
    if predictions[instance_id]["model_patch"] == ""
)
completed = []
resolved = []
unresolved = []
model_name = rows[0]["model_name_or_path"]
for index, instance_id in enumerate(args.instance_ids):
    if instance_id in empty:
        continue
    is_resolved = index % 2 == 0
    completed.append(instance_id)
    (resolved if is_resolved else unresolved).append(instance_id)
    report_dir = (
        Path.cwd()
        / "logs"
        / "run_evaluation"
        / args.run_id
        / model_name.replace("/", "__")
        / instance_id
    )
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps({instance_id: {"resolved": is_resolved}}), encoding="ascii"
    )

submitted = sorted(args.instance_ids)
if mode == "bad_coverage":
    submitted = submitted[:-1]
aggregate = {
    "schema_version": 2,
    "total_instances": len(args.instance_ids),
    "submitted_instances": len(submitted),
    "completed_instances": len(completed),
    "resolved_instances": len(resolved),
    "unresolved_instances": len(unresolved),
    "empty_patch_instances": len(empty),
    "error_instances": 0,
    "completed_ids": sorted(completed),
    "incomplete_ids": [],
    "empty_patch_ids": empty,
    "submitted_ids": submitted,
    "resolved_ids": sorted(resolved),
    "unresolved_ids": sorted(unresolved),
    "error_ids": [],
}
# SWE-bench 4.1 creates --report_dir but writes the aggregate in cwd.
output = Path.cwd()
output.mkdir(parents=True, exist_ok=True)
(output / f"{model_name}.{args.run_id}.json").write_text(
    json.dumps(aggregate), encoding="ascii"
)
print("mock harness completed")
'''

    def _environment(self, *, mode: str = "success") -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CONFIG": str(self.config_path),
                "RUN_NAME": "focused_behavioral_test",
                "RUN_ROOT": str(self.run_root),
                "SWE_PYTHON": str(SWE_PYTHON),
                "ENDPOINT": "http://127.0.0.1:1/v1",
                "PATH": str(self.fake_bin) + os.pathsep + environment["PATH"],
                "PYTHONPATH": str(self.fake_modules),
                "FAKE_DOCKER_IMAGES": str(self.docker_map_path),
                "FAKE_HARNESS_INVOCATIONS": str(self.invocations_path),
                "FAKE_HARNESS_MODE": mode,
            }
        )
        return environment

    def _run(self, *, mode: str = "success") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=self._environment(mode=mode),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_end_to_end_normalizes_missing_and_empty_predictions(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        normalized_path = self.run_root / "official_score" / "official_outcomes.json"
        normalized = json.loads(normalized_path.read_text(encoding="ascii"))
        self.assertEqual(normalized["instance_ids"], self.instance_ids)
        self.assertEqual(normalized["counts"]["empty"], 2)
        self.assertEqual(
            normalized["inputs"]["empty_prediction_ids"],
            [self.instance_ids[1], self.instance_ids[2]],
        )
        self.assertEqual(
            normalized["inputs"]["missing_prediction_ids"], [self.instance_ids[2]]
        )
        self.assertEqual(
            {row["instance_id"] for row in normalized["outcomes"]},
            set(self.instance_ids),
        )

        attempt_dir = Path(normalized["attempt_path"])
        completed_rows = [
            json.loads(line)
            for line in (attempt_dir / "predictions.complete.jsonl")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        self.assertEqual([row["instance_id"] for row in completed_rows], self.instance_ids)
        self.assertEqual(completed_rows[1]["model_patch"], "")
        self.assertEqual(completed_rows[2]["model_patch"], "")
        self.assertEqual((attempt_dir / "harness.exit_status").read_text(), "0\n")
        invocation = json.loads(self.invocations_path.read_text().splitlines()[0])
        self.assertEqual(invocation["max_workers"], "1")
        self.assertEqual(invocation["dataset_name"], str(self.run_root / "dataset.json"))
        self.assertEqual(invocation["instance_ids"], self.instance_ids)

    def test_rerun_uses_a_fresh_attempt_and_archives_previous_result(self):
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        first_outcome = json.loads(
            (self.run_root / "official_score" / "official_outcomes.json").read_text()
        )
        second = self._run()
        self.assertEqual(second.returncode, 0, second.stderr)
        second_outcome = json.loads(
            (self.run_root / "official_score" / "official_outcomes.json").read_text()
        )
        self.assertNotEqual(first_outcome["attempt_path"], second_outcome["attempt_path"])
        self.assertEqual(first_outcome["run_id"], second_outcome["run_id"])
        self.assertEqual(len(self.invocations_path.read_text().splitlines()), 2)
        archives = list((self.run_root / "official_score" / "archive").glob("*.json"))
        self.assertEqual(len(archives), 1)

    def test_config_hash_mismatch_fails_before_harness(self):
        self._write_image_manifest(config_hash="0" * 64)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preflight failed", result.stderr)
        self.assertFalse(self.invocations_path.exists())
        self.assertFalse(
            (self.run_root / "official_score" / "official_outcomes.json").exists()
        )

    def test_aggregate_coverage_mismatch_fails_closed(self):
        result = self._run(mode="bad_coverage")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("score validation failed", result.stderr)
        self.assertFalse(
            (self.run_root / "official_score" / "official_outcomes.json").exists()
        )
        attempts = list(
            (self.run_root / "official_score" / "attempts").glob("*/*")
        )
        self.assertEqual(len(attempts), 1)
        self.assertEqual((attempts[0] / "harness.exit_status").read_text(), "0\n")
        self.assertNotEqual((attempts[0] / "validation.exit_status").read_text(), "0\n")

    def test_harness_failure_preserves_stdout_stderr_and_status(self):
        result = self._run(mode="exit_7")
        self.assertNotEqual(result.returncode, 0)
        attempts = list(
            (self.run_root / "official_score" / "attempts").glob("*/*")
        )
        self.assertEqual(len(attempts), 1)
        attempt = attempts[0]
        self.assertEqual((attempt / "harness.exit_status").read_text(), "7\n")
        self.assertIn("stdout before failure", (attempt / "harness.stdout.log").read_text())
        self.assertIn("stderr before failure", (attempt / "harness.stderr.log").read_text())
        self.assertFalse(
            (self.run_root / "official_score" / "official_outcomes.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
