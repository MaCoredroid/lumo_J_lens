from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_swe_task_state_v3_development_cohort",
    ROOT / "scripts/check_swe_task_state_v3_development_cohort.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def hash_order(seed: str, values: list[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hashlib.sha256(
                    seed.encode("utf-8") + b"\x00" + value.encode("ascii")
                ).digest(),
                value,
            ),
        )
    )


def write_bundle_fixture(
    directory: Path, declaration: MODULE.DeclarationBundle
) -> tuple[Path, Path, list[dict[str, object]], Path]:
    cohort = declaration.cohort
    cohort_sha256 = MODULE.sha256_file(declaration.cohort_path)
    rows = cohort["cohorts"]
    instance_ids = [*declaration.campaign_ids[0], *declaration.campaign_ids[1]]
    runs_root = directory / "runs"
    runs_root.mkdir()
    image_manifest_hashes: list[str] = []
    runner_hashes: dict[str, str] = {}
    for cohort_index, row in enumerate(rows):
        run_root = runs_root / row["run_label"]
        run_root.mkdir()
        dataset_path = run_root / "dataset.json"
        dataset_path.write_text("[]\n", encoding="ascii")
        registry_path = ROOT / row["image_registry"]["path"]
        registry = json.loads(registry_path.read_bytes())
        image_rows = []
        for instance_id in declaration.campaign_ids[cohort_index]:
            pin = registry["images"][instance_id]["x86_64"]
            image_rows.append(
                {
                    "instance_id": instance_id,
                    "tag": "swebench/sweb.eval.x86_64."
                    + instance_id.replace("__", "_1776_")
                    + ":latest",
                    "image_id": pin["image_id"],
                    "repo_digests": [pin["reference"]],
                }
            )
            metadata_path = (
                run_root
                / "generation/verified/per_task"
                / instance_id
                / "runner_metadata.json"
            )
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "dataset_name": str(dataset_path.resolve()),
                        "image": pin["reference"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="ascii",
            )
            runner_hashes[instance_id] = MODULE.sha256_file(metadata_path)
        image_manifest = {
            "schema_version": 1,
            "kind": "swe_verified_behavioral_campaign_image_manifest",
            "campaign_config_path": str(declaration.campaign_paths[cohort_index]),
            "campaign_config_sha256": row["campaign_sha256"],
            "image_config_path": str(registry_path.resolve()),
            "image_config_sha256": row["image_registry"]["sha256"],
            "selection_proof_path": str(declaration.proof_path),
            "selection_proof_sha256": MODULE.sha256_file(declaration.proof_path),
            "dataset": MODULE.EXPECTED_CAMPAIGN_DATASET,
            "images": image_rows,
        }
        manifest_path = run_root / "image_manifest.json"
        manifest_path.write_text(
            json.dumps(image_manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        image_manifest_hashes.append(MODULE.sha256_file(manifest_path))
    cohort_bindings = [
        {
            "id": row["id"],
            "index": index,
            "campaign_sha256": row["campaign_sha256"],
            "cohort_manifest_sha256": cohort_sha256,
            "source_task_instance_ids": row["instance_ids"],
            "source_task_count": 30,
            "task_offset": index * 30,
            "global_request_offset": index * 30,
            "source_global_request_count": 30,
            "source_prompt_count": 30,
            "source_run_label": row["run_label"],
            "source_image_manifest_sha256": image_manifest_hashes[index],
        }
        for index, row in enumerate(rows)
    ]
    prompts: list[dict[str, object]] = []
    for index, instance_id in enumerate(instance_ids):
        cohort_index = 0 if index < 30 else 1
        source_ids = list(declaration.campaign_ids[cohort_index])
        prompt: dict[str, object] = {
            "id": f"v3-fixture-{index + 1}",
            "metadata": {
                "campaign_sha256": rows[cohort_index]["campaign_sha256"],
                "action_protocol_sha256": cohort["pins"]["action_protocol_sha256"],
                "chat_template_sha256": cohort["pins"]["chat_template_sha256"],
                "cohort": cohort_bindings[cohort_index],
                "task": {
                    "instance_id": instance_id,
                    "selection_index": index,
                    "source_selection_index": index - 30 * cohort_index,
                    "request_count": 1,
                    "probeable_request_count": 1,
                    "probeable_request_indices": [1],
                },
                "selection": {
                    "task_request_index": 1,
                    "global_request_index": index + 1,
                    "source_global_request_index": index - 30 * cohort_index + 1,
                    "probeable_request_indices": [1],
                    "candidate_count": 1,
                    "checkpoint_count": 1,
                    "max_checkpoints": None,
                },
                "provenance": {
                    "combination": {
                        "cohort_manifest_sha256": cohort_sha256,
                        "combined_global_request_index": index + 1,
                        "source_campaign_global_request_index": index - 30 * cohort_index + 1,
                        "source_image_manifest_sha256": image_manifest_hashes[
                            cohort_index
                        ],
                    },
                    "runner_metadata_path": (
                        "generation/verified/per_task/"
                        + instance_id
                        + "/runner_metadata.json"
                    ),
                    "runner_metadata_sha256": runner_hashes[instance_id],
                },
            },
        }
        prompt["metadata"]["provenance"]["prompt_record_payload_sha256"] = (
            MODULE._prompt_payload_sha256(prompt)
        )
        prompts.append(prompt)
    prompts_path = directory / "prompts.json"
    prompts_path.write_text(
        json.dumps(prompts, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    campaign_hashes = [row["campaign_sha256"] for row in rows]
    summary = {
        "schema_version": 1,
        "kind": "swe_verified_behavioral_probe_combination",
        "cohort_manifest_sha256": cohort_sha256,
        "source_campaign_sha256s": campaign_hashes,
        "campaign_sha256s": campaign_hashes,
        "action_protocol_sha256": cohort["pins"]["action_protocol_sha256"],
        "chat_template_sha256": cohort["pins"]["chat_template_sha256"],
        "cohort_count": 2,
        "task_count": 60,
        "global_request_count": 60,
        "prompt_count": 60,
        "prompt_bundle_sha256": MODULE.sha256_file(prompts_path),
        "cohorts": cohort_bindings,
        "prompts": [
            {
                "id": prompt["id"],
                "cohort_id": prompt["metadata"]["cohort"]["id"],
                "instance_id": prompt["metadata"]["task"]["instance_id"],
                "global_request_index": prompt["metadata"]["selection"][
                    "global_request_index"
                ],
                "prompt_record_payload_sha256": prompt["metadata"]["provenance"][
                    "prompt_record_payload_sha256"
                ],
            }
            for prompt in prompts
        ],
        "task_audits": [
            {
                "instance_id": instance_id,
                "selection_index": index,
                "cohort_id": rows[0 if index < 30 else 1]["id"],
                "campaign_sha256": rows[0 if index < 30 else 1][
                    "campaign_sha256"
                ],
                "request_count": 1,
                "probeable_request_indices": [1],
                "selected_request_indices": [1],
                "selected_checkpoint_count": 1,
                "global_request_start": index + 1,
                "global_request_end": index + 1,
                "source_global_request_start": index % 30 + 1,
                "source_global_request_end": index % 30 + 1,
                "source_selection_index": index % 30,
            }
            for index, instance_id in enumerate(instance_ids)
        ],
    }
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return prompts_path, summary_path, prompts, runs_root


class V3DevelopmentCohortCheckerTests(unittest.TestCase):
    def test_replay_runner_local_import_closure_is_source_frozen(self) -> None:
        pending = ["scripts/run_jlens_nvfp4.py"]
        local_closure: set[str] = set()
        while pending:
            logical_path = pending.pop()
            if logical_path in local_closure:
                continue
            local_closure.add(logical_path)
            tree = ast.parse((ROOT / logical_path).read_bytes(), filename=logical_path)
            imported_modules = {
                node.module.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.level == 0
                and isinstance(node.module, str)
            }
            imported_modules.update(
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            for module_name in imported_modules:
                candidate = ROOT / "scripts" / f"{module_name}.py"
                if candidate.is_file():
                    pending.append(candidate.relative_to(ROOT).as_posix())

        self.assertEqual(
            local_closure,
            {
                "scripts/run_jlens_nvfp4.py",
                "scripts/download_jlens.py",
                "scripts/verify_nvfp4_ste_artifact.py",
                "scripts/modelopt_checkpoint.py",
                "scripts/nvfp4_ste.py",
            },
        )
        self.assertLessEqual(local_closure, set(MODULE.SOURCE_FREEZE_PATHS))

    def test_analyzer_dynamic_inputs_and_transitive_requirements_are_source_frozen(
        self,
    ) -> None:
        analyzer_inputs = {
            "scripts/analyze_swe_task_state_interpreter.py",
            "scripts/analyze_swe_binary_phase_v2.py",
            "scripts/swe_task_state_readout.py",
            "scripts/check_swe_task_state_validation_cohort.py",
            "configs/swe_task_state_interpreter_protocol.json",
            "configs/swe_binary_phase_interpreter_v2.json",
            "configs/swe_stage_action_probes.json",
            "configs/swe_behavioral_readout_protocol.json",
        }
        self.assertLessEqual(analyzer_inputs, set(MODULE.SOURCE_FREEZE_PATHS))

        pending = [ROOT / "requirements-v3-state-interpreter.txt"]
        requirement_closure: set[str] = set()
        while pending:
            requirement_path = pending.pop()
            logical_path = requirement_path.relative_to(ROOT).as_posix()
            if logical_path in requirement_closure:
                continue
            requirement_closure.add(logical_path)
            for raw_line in requirement_path.read_text(encoding="utf-8").splitlines():
                fields = raw_line.split("#", 1)[0].strip().split()
                if len(fields) == 2 and fields[0] in {"-r", "-c"}:
                    pending.append((requirement_path.parent / fields[1]).resolve())

        self.assertEqual(
            requirement_closure,
            {"requirements-v3-state-interpreter.txt", "requirements-readout-v2.txt"},
        )
        self.assertLessEqual(requirement_closure, set(MODULE.SOURCE_FREEZE_PATHS))

    def test_certified_campaign_cannot_source_ignored_dotenv(self) -> None:
        campaign = (ROOT / "scripts/run_swe_state_interpreter_v3_campaign.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '[[ ! -e "$ROOT/.env" && ! -L "$ROOT/.env" ]]', campaign
        )
        self.assertIn("export LUMO_V3_CERTIFIED_NO_DOTENV=1", campaign)
        self.assertNotIn('source "$ROOT/.env"', campaign)

        for logical_path in (
            "scripts/start_server.sh",
            "scripts/serve_qwen36_27b_nvfp4_mtp.sh",
            "scripts/stop_server.sh",
        ):
            helper = (ROOT / logical_path).read_text(encoding="utf-8")
            self.assertIn("LUMO_V3_CERTIFIED_NO_DOTENV", helper)
            self.assertIn('source "$ROOT/.env"', helper)
            self.assertIn(logical_path, MODULE.SOURCE_FREEZE_PATHS)

    def test_checked_in_declaration_binds_exact_n60_and_two_image_registries(self) -> None:
        declaration = MODULE.validate_declaration()

        self.assertEqual(declaration.cohort["kind"], MODULE.COHORT_KIND)
        self.assertEqual([len(ids) for ids in declaration.campaign_ids], [30, 30])
        self.assertFalse(set(declaration.campaign_ids[0]) & set(declaration.campaign_ids[1]))
        self.assertEqual(
            [row["image_registry"]["sha256"] for row in declaration.cohort["cohorts"]],
            list(MODULE.EXPECTED_IMAGE_SHA256S),
        )

    def test_reproduces_hash_rank_quota_and_odd_quota_balancing(self) -> None:
        seed = "synthetic-v3"
        grouped = {
            "alpha/repo": [f"alpha__repo-{index}" for index in range(1, 4)],
            "beta/repo": [f"beta__repo-{index}" for index in range(1, 4)],
            "gamma/repo": [f"gamma__repo-{index}" for index in range(1, 3)],
        }
        rows = [
            (instance_id, repository)
            for repository, values in grouped.items()
            for instance_id in values
        ]

        result = MODULE.reproduce_selection(
            rows,
            prior_ids=frozenset(),
            seed=seed,
            target_count=8,
            repository_minimum=2,
            base_quota=2,
        )

        ordered = {
            repository: hash_order(seed, values)
            for repository, values in grouped.items()
        }
        self.assertEqual(
            result.campaign_a,
            (
                ordered["alpha/repo"][0],
                ordered["alpha/repo"][2],
                ordered["beta/repo"][1],
                ordered["gamma/repo"][0],
            ),
        )
        self.assertEqual(
            result.campaign_b,
            (
                ordered["alpha/repo"][1],
                ordered["beta/repo"][0],
                ordered["beta/repo"][2],
                ordered["gamma/repo"][1],
            ),
        )

    def test_historical_scan_uses_only_supplied_frozen_blob_payloads(self) -> None:
        official = frozenset(
            {"alpha__repo-1", "alpha__repo-2", "beta__repo-1"}
        )
        result = MODULE.scan_historical_config_payloads(
            [
                ("configs/a.json", "1" * 40, b'"alpha__repo-1"'),
                (
                    "configs/b.json",
                    "2" * 40,
                    b'"alpha__repo-1" "beta__repo-1" "not__official-9"',
                ),
            ],
            official_ids=official,
        )

        self.assertEqual(result.prior_ids, {"alpha__repo-1", "beta__repo-1"})
        self.assertEqual(
            [row["match_count"] for row in result.matched_sources], [1, 2]
        )
        self.assertNotIn("alpha__repo-2", result.prior_ids)

    def test_campaign_selection_evidence_tamper_fails_closed(self) -> None:
        declaration = MODULE.validate_declaration()
        campaign = copy.deepcopy(declaration.campaigns[0])
        campaign["selection"]["lens_outputs_used"] = True

        with self.assertRaisesRegex(
            MODULE.CohortValidationError, "forbidden non-identity evidence"
        ):
            MODULE.validate_campaign_declaration(
                campaign,
                row=declaration.cohort["cohorts"][0],
                campaign_index=0,
            )

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version": 1, "schema_version": 1}\n')
            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "duplicate JSON key"
            ):
                MODULE.strict_json_file(path, "duplicate fixture")

    def test_streaming_json_rejects_duplicate_nonfinite_and_trailing_data(self) -> None:
        invalid = {
            "duplicate": '[{"key": 1, "key": 2}]',
            "nonfinite": "[NaN]",
            "trailing": "[] []",
        }
        with tempfile.TemporaryDirectory() as directory:
            for label, payload in invalid.items():
                path = Path(directory) / f"{label}.json"
                path.write_text(payload, encoding="ascii")
                with self.subTest(label=label), self.assertRaises(
                    MODULE.CohortValidationError
                ):
                    list(MODULE.iter_strict_json_array(path, label))

    def test_declaration_rejects_a_byte_copy_at_an_unfrozen_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            copied = Path(directory) / "cohort-copy.json"
            copied.write_bytes(MODULE.DEFAULT_COHORT.read_bytes())
            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "exact checked-in V3 N=60 manifest"
            ):
                MODULE.validate_declaration(copied)

    def test_materialized_bundle_authenticates_all_sixty_tasks(self) -> None:
        declaration = MODULE.validate_declaration()
        with tempfile.TemporaryDirectory() as directory:
            prompts_path, summary_path, _prompts, runs_root = write_bundle_fixture(
                Path(directory), declaration
            )

            with mock.patch.multiple(
                MODULE,
                V3_RUNS_ROOT=runs_root,
                V3_OUTPUT_ROOT=Path(directory),
            ):
                result = MODULE.validate_materialized_bundle(
                    declaration,
                    prompts_path=prompts_path,
                    summary_path=summary_path,
                )

        self.assertEqual(result["cohort_count"], 2)
        self.assertEqual(result["task_count"], 60)
        self.assertEqual(result["prompt_count"], 60)

    def test_run_image_provenance_rejects_runner_image_drift(self) -> None:
        declaration = MODULE.validate_declaration()
        with tempfile.TemporaryDirectory() as directory:
            _prompts_path, _summary_path, _prompts, runs_root = write_bundle_fixture(
                Path(directory), declaration
            )
            instance_id = declaration.campaign_ids[0][0]
            metadata_path = (
                runs_root
                / declaration.cohort["cohorts"][0]["run_label"]
                / "generation/verified/per_task"
                / instance_id
                / "runner_metadata.json"
            )
            metadata = json.loads(metadata_path.read_bytes())
            metadata["image"] = "swebench/sweb.eval.x86_64.forged@sha256:" + "0" * 64
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with mock.patch.object(MODULE, "V3_RUNS_ROOT", runs_root):
                with self.assertRaisesRegex(
                    MODULE.CohortValidationError,
                    "runner metadata image/dataset binding changed",
                ):
                    MODULE.validate_run_image_provenance(declaration)

    def test_materialized_bundle_rejects_prompt_and_summary_symlinks(self) -> None:
        declaration = MODULE.validate_declaration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path, summary_path, _prompts, runs_root = write_bundle_fixture(
                root, declaration
            )
            prompts_link = root / "prompts-link.json"
            prompts_link.symlink_to(prompts_path.name)
            summary_link = root / "summary-link.json"
            summary_link.symlink_to(summary_path.name)
            with mock.patch.multiple(
                MODULE,
                V3_RUNS_ROOT=runs_root,
                V3_OUTPUT_ROOT=root,
            ):
                for candidate_prompts, candidate_summary in (
                    (prompts_link, summary_path),
                    (prompts_path, summary_link),
                ):
                    with self.subTest(
                        prompts=candidate_prompts.name,
                        summary=candidate_summary.name,
                    ), self.assertRaisesRegex(
                        MODULE.CohortValidationError,
                        "non-symlink",
                    ):
                        MODULE.validate_materialized_bundle(
                            declaration,
                            prompts_path=candidate_prompts,
                            summary_path=candidate_summary,
                        )

    def test_materialized_bundle_rejects_missing_frozen_task(self) -> None:
        declaration = MODULE.validate_declaration()
        with tempfile.TemporaryDirectory() as directory:
            prompts_path, summary_path, prompts, runs_root = write_bundle_fixture(
                Path(directory), declaration
            )
            prompts_path.write_text(
                json.dumps(prompts[:-1], indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "task order/coverage changed"
            ):
                with mock.patch.multiple(
                    MODULE,
                    V3_RUNS_ROOT=runs_root,
                    V3_OUTPUT_ROOT=Path(directory),
                ):
                    MODULE.validate_materialized_bundle(
                        declaration,
                        prompts_path=prompts_path,
                        summary_path=summary_path,
                    )

    def test_receipt_git_freeze_requires_clean_exact_child_receipt_only_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*arguments: str) -> str:
                result = subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                return result.stdout.strip()

            subprocess.run(["git", "init", "-q", str(root)], check=True)
            git("config", "user.name", "Receipt Test")
            git("config", "user.email", "receipt@example.invalid")
            (root / "source.txt").write_text("frozen\n", encoding="ascii")
            git("add", "source.txt")
            git("commit", "-q", "-m", "source freeze")
            source_commit = git("rev-parse", "HEAD")
            receipt_path = root / MODULE.EXPECTED_MATERIALIZATION_RECEIPT_PATH
            receipt_path.parent.mkdir(parents=True)
            receipt = {"source_freeze_git_commit": source_commit}
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="ascii"
            )
            git("add", MODULE.EXPECTED_MATERIALIZATION_RECEIPT_PATH)
            git("commit", "-q", "-m", "materialization data freeze")
            data_commit = git("rev-parse", "HEAD")
            with mock.patch.object(MODULE, "ROOT", root):
                self.assertEqual(
                    MODULE._validate_receipt_git_freeze(receipt_path, receipt),
                    data_commit,
                )
                receipt_path.write_text("{}\n", encoding="ascii")
                with self.assertRaisesRegex(
                    MODULE.CohortValidationError, "working tree|index"
                ):
                    MODULE._validate_receipt_git_freeze(receipt_path, receipt)

    def test_run_source_inventory_is_exhaustive_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_root = root / "runs"
            rows = [
                {"id": "development_a", "run_label": "run-a"},
                {"id": "development_b", "run_label": "run-b"},
            ]
            campaign_ids = (("repo__task-1",), ("repo__task-2",))
            for row, instance_ids in zip(rows, campaign_ids, strict=True):
                run = runs_root / row["run_label"]
                (run / "proxy_dumps").mkdir(parents=True)
                (run / "dataset.json").write_text("{}\n", encoding="ascii")
                (run / "image_manifest.json").write_text("{}\n", encoding="ascii")
                (run / "proxy_dumps/usage.jsonl").write_text("{}\n", encoding="ascii")
                (run / "proxy_dumps/chat_0001.json").write_text("{}\n", encoding="ascii")
                (run / "official_score").mkdir()
                (run / "official_score/official_outcomes.json").write_text(
                    "{}\n", encoding="ascii"
                )
                for instance_id in instance_ids:
                    task = run / "generation/verified/per_task" / instance_id
                    task.mkdir(parents=True)
                    for name in ("runner_metadata.json", "qwen_trace.json", "patch.diff"):
                        (task / name).write_text("{}\n", encoding="ascii")
            declaration = types.SimpleNamespace(
                cohort={"cohorts": rows}, campaign_ids=campaign_ids
            )
            with mock.patch.multiple(MODULE, ROOT=root, V3_RUNS_ROOT=runs_root):
                records, count = MODULE._run_source_inventory(declaration)
                self.assertEqual(len(records), 2)
                self.assertEqual(count, sum(record["file_count"] for record in records))
                self.assertTrue(
                    any(
                        value["path"].endswith("official_outcomes.json")
                        for record in records
                        for value in record["files"]
                    )
                )
                target = root / "target"
                target.write_text("x", encoding="ascii")
                (runs_root / "run-a" / "unsafe-link").symlink_to(target)
                with self.assertRaisesRegex(MODULE.CohortValidationError, "symlink"):
                    MODULE._run_source_inventory(declaration)

    def test_materialized_bundle_rejects_prompt_payload_tamper(self) -> None:
        declaration = MODULE.validate_declaration()
        with tempfile.TemporaryDirectory() as directory:
            prompts_path, summary_path, prompts, runs_root = write_bundle_fixture(
                Path(directory), declaration
            )
            prompts[0]["tampered"] = True
            prompts_path.write_text(
                json.dumps(prompts, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "payload hash differs"
            ):
                with mock.patch.multiple(
                    MODULE,
                    V3_RUNS_ROOT=runs_root,
                    V3_OUTPUT_ROOT=Path(directory),
                ):
                    MODULE.validate_materialized_bundle(
                        declaration,
                        prompts_path=prompts_path,
                        summary_path=summary_path,
                    )

    def test_self_consistent_prompt_provenance_forgery_is_rejected(self) -> None:
        declaration = MODULE.validate_declaration()
        with tempfile.TemporaryDirectory() as directory:
            prompts_path, summary_path, prompts, runs_root = write_bundle_fixture(
                Path(directory), declaration
            )
            prompts[0]["metadata"]["campaign_sha256"] = "0" * 64
            prompts[0]["metadata"]["provenance"][
                "prompt_record_payload_sha256"
            ] = MODULE._prompt_payload_sha256(prompts[0])
            prompts_path.write_text(
                json.dumps(prompts, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            summary = json.loads(summary_path.read_bytes())
            summary["prompts"][0]["prompt_record_payload_sha256"] = prompts[0][
                "metadata"
            ]["provenance"]["prompt_record_payload_sha256"]
            summary["prompt_bundle_sha256"] = MODULE.sha256_file(prompts_path)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "top-level provenance binding differs"
            ):
                with mock.patch.multiple(
                    MODULE,
                    V3_RUNS_ROOT=runs_root,
                    V3_OUTPUT_ROOT=Path(directory),
                ):
                    MODULE.validate_materialized_bundle(
                        declaration,
                        prompts_path=prompts_path,
                        summary_path=summary_path,
                    )


if __name__ == "__main__":
    unittest.main()
