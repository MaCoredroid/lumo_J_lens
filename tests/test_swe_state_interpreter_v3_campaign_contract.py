#!/usr/bin/env python3
"""Focused tests for the V3 campaign launcher and preflight contract."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from typing import Any
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "swe_state_interpreter_v3_campaign_contract",
    ROOT / "scripts/swe_state_interpreter_v3_campaign_contract.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixtures(count: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    instance_ids = [f"owner{index}__repo-{1000 + index}" for index in range(count)]
    local: dict[str, Any] = {}
    pins: dict[str, Any] = {}
    for index, instance_id in enumerate(instance_ids, 1):
        digest = f"{index:064x}"
        tag = MODULE.docker_tag(instance_id)
        reference = f"{tag.removesuffix(':latest')}@sha256:{digest}"
        image_id = f"sha256:{digest}"
        pins[instance_id] = {
            "x86_64": {"reference": reference, "image_id": image_id}
        }
        local[tag] = [
            {
                "Architecture": "amd64",
                "Id": image_id,
                "RepoDigests": [reference],
            }
        ]
    campaign = {
        "schema_version": 1,
        "kind": MODULE.CAMPAIGN_KIND,
        "dataset": copy.deepcopy(MODULE.EXPECTED_DATASET),
        "selection": {
            "lens_outputs_used": False,
            "official_outcomes_used": False,
            "rule": "synthetic lens-blind selection",
        },
        "generation": copy.deepcopy(MODULE.EXPECTED_GENERATION),
        "instance_ids": instance_ids,
    }
    return campaign, {"schema_version": 1, "images": pins}, local


def authorize_proof_images(proof: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key in ("a", "b"):
        path = f"configs/swe_task_state_v3_development_{key}_image_digests.json"
        proof["campaigns"][key]["image_registry"] = {
            "generation_authorized": True,
            "path": path,
            "sha256": "f" * 64,
            "status": "frozen_before_generation",
        }
        paths[key] = path
    return paths


class V3CampaignContractTests(unittest.TestCase):
    def test_exact_thirty_task_campaign_is_supported(self) -> None:
        campaign, registry, local = fixtures(30)
        instance_ids, images = MODULE.validate_campaign_contract(
            campaign, registry, image_inspector=local.__getitem__
        )
        self.assertEqual(instance_ids, campaign["instance_ids"])
        self.assertEqual(len(images), 30)
        self.assertEqual(
            [row["instance_id"] for row in images], campaign["instance_ids"]
        )

    def test_other_campaign_sizes_are_rejected(self) -> None:
        for count in (1, 10, 29, 31):
            campaign, registry, local = fixtures(count)
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, "exactly 30"
            ):
                MODULE.validate_campaign_contract(
                    campaign, registry, image_inspector=local.__getitem__
                )

    def test_image_registry_coverage_must_exactly_match_campaign(self) -> None:
        campaign, registry, local = fixtures(30)
        missing = copy.deepcopy(registry)
        missing["images"].pop(campaign["instance_ids"][0])
        extra = copy.deepcopy(registry)
        extra["images"]["extra__repo-9999"] = copy.deepcopy(
            next(iter(extra["images"].values()))
        )
        for value in (missing, extra):
            with self.subTest(keys=sorted(value["images"])), self.assertRaisesRegex(
                ValueError, "coverage must exactly equal"
            ):
                MODULE.validate_campaign_contract(
                    campaign, value, image_inspector=local.__getitem__
                )

    def test_local_architecture_image_id_and_repo_digest_are_all_required(self) -> None:
        campaign, registry, local = fixtures(30)
        first_id = campaign["instance_ids"][0]
        first_tag = MODULE.docker_tag(first_id)
        mutations = {
            "architecture": {"Architecture": "arm64"},
            "image ID": {"Id": "sha256:" + "f" * 64},
            "RepoDigest": {"RepoDigests": []},
        }
        for label, mutation in mutations.items():
            changed = copy.deepcopy(local)
            changed[first_tag][0].update(mutation)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, label):
                MODULE.validate_campaign_contract(
                    campaign, registry, image_inspector=changed.__getitem__
                )

    def test_repo_digest_must_name_the_exact_campaign_tag(self) -> None:
        campaign, registry, local = fixtures(30)
        first_id = campaign["instance_ids"][0]
        pin = registry["images"][first_id]["x86_64"]
        wrong_reference = pin["reference"].replace("owner0_1776_repo", "other_1776_repo")
        pin["reference"] = wrong_reference
        local[MODULE.docker_tag(first_id)][0]["RepoDigests"] = [wrong_reference]
        with self.assertRaisesRegex(ValueError, "does not name the campaign image"):
            MODULE.validate_campaign_contract(
                campaign, registry, image_inspector=local.__getitem__
            )

    def test_frozen_dataset_generation_and_selection_contracts_fail_closed(self) -> None:
        campaign, registry, local = fixtures(30)
        mutations = (
            ("dataset", "revision", "0" * 40),
            ("generation", "max_session_turns", 49),
            ("selection", "lens_outputs_used", True),
            ("selection", "official_outcomes_used", True),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(campaign)
            changed[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                MODULE.validate_campaign_contract(
                    changed, registry, image_inspector=local.__getitem__
                )

    def test_protected_reserved_tasks_are_rejected_before_image_inspection(self) -> None:
        campaign, registry, local = fixtures(30)
        protected = frozenset({campaign["instance_ids"][3]})
        with self.assertRaisesRegex(ValueError, "overlaps protected prior/reserved tasks"):
            MODULE.validate_campaign_contract(
                campaign,
                registry,
                image_inspector=local.__getitem__,
                forbidden_instance_ids=protected,
            )

    def test_checked_in_reserved_protection_set_is_hash_bound(self) -> None:
        proof = json.loads(
            (
                ROOT
                / "validation/swe-task-state-v3-development-cohort-selection.json"
            ).read_bytes()
        )
        protected = MODULE.protected_reserved_instance_ids(
            types.SimpleNamespace(proof=proof)
        )
        self.assertEqual(len(protected), 154)
        self.assertIn("sympy__sympy-13757", protected)

    def test_only_exact_frozen_campaign_paths_and_hashes_are_authorized(self) -> None:
        proof = json.loads(
            (
                ROOT
                / "validation/swe-task-state-v3-development-cohort-selection.json"
            ).read_bytes()
        )
        for logical, expected_sha in MODULE.EXPECTED_CAMPAIGNS.items():
            campaign_path = ROOT / logical
            campaign = json.loads(campaign_path.read_bytes())
            observed_sha = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
            self.assertEqual(observed_sha, expected_sha)
            authorized = copy.deepcopy(proof)
            image_paths = authorize_proof_images(authorized)
            campaign_key = "a" if "_a_campaign" in logical else "b"
            self.assertEqual(
                MODULE.validate_selection_binding(
                    authorized,
                    campaign_logical_path=logical,
                    campaign_sha256=observed_sha,
                    instance_ids=campaign["instance_ids"],
                    run_label=MODULE.EXPECTED_RUN_LABELS[logical],
                    image_config_logical_path=image_paths[campaign_key],
                    image_config_sha256="f" * 64,
                ),
                "f" * 64,
            )

            with self.assertRaisesRegex(ValueError, "exact frozen V3"):
                MODULE.validate_selection_binding(
                    authorized,
                    campaign_logical_path=logical,
                    campaign_sha256="0" * 64,
                    instance_ids=campaign["instance_ids"],
                    run_label=MODULE.EXPECTED_RUN_LABELS[logical],
                    image_config_logical_path=image_paths[campaign_key],
                    image_config_sha256="f" * 64,
                )

    def test_pending_or_mismatched_image_binding_cannot_authorize_generation(self) -> None:
        proof_path = (
            ROOT / "validation/swe-task-state-v3-development-cohort-selection.json"
        )
        proof = json.loads(proof_path.read_bytes())
        logical = next(iter(MODULE.EXPECTED_CAMPAIGNS))
        campaign_path = ROOT / logical
        campaign = json.loads(campaign_path.read_bytes())
        campaign_sha = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
        arguments = {
            "campaign_logical_path": logical,
            "campaign_sha256": campaign_sha,
            "instance_ids": campaign["instance_ids"],
            "run_label": MODULE.EXPECTED_RUN_LABELS[logical],
            "image_config_logical_path": "configs/final-images.json",
            "image_config_sha256": "f" * 64,
        }
        with self.assertRaisesRegex(
            ValueError,
            "no per-campaign image registry|does not authorize generation|differs from the finalized",
        ):
            MODULE.validate_selection_binding(proof, **arguments)

        authorized = copy.deepcopy(proof)
        image_paths = authorize_proof_images(authorized)
        arguments["image_config_logical_path"] = image_paths["a"]
        authorized["campaigns"]["a"]["image_registry"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "differs from the finalized"):
            MODULE.validate_selection_binding(authorized, **arguments)

        wrong_run = dict(arguments)
        wrong_run["run_label"] = "swe_task_state_validation_a_20260718"
        with self.assertRaisesRegex(ValueError, "RUN_ROOT label differs"):
            MODULE.validate_selection_binding(authorized, **wrong_run)

    def test_run_root_is_confined_to_a_dedicated_direct_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "v3-development"
            candidate = allowed / "campaign-a"
            resolved, label = MODULE.validate_run_root(
                candidate, allowed_root=allowed
            )
            self.assertEqual(resolved, candidate)
            self.assertEqual(label, "campaign-a")

            for invalid in (root / "reserved", allowed, allowed / "nested" / "run"):
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    MODULE.validate_run_root(invalid, allowed_root=allowed)

            allowed.mkdir()
            target = allowed / "real"
            target.mkdir()
            symlink = allowed / "campaign-link"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "new and absent"):
                MODULE.validate_run_root(symlink, allowed_root=allowed)

    def test_run_root_is_securely_created_once_and_symlink_namespace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = root / "runs" / "v3"
            namespace.parent.mkdir()
            candidate = namespace / "campaign-a"
            created = MODULE.securely_create_run_root(candidate, namespace=namespace)
            self.assertEqual(created, candidate)
            self.assertTrue(created.is_dir())
            self.assertEqual(list(created.iterdir()), [])
            with self.assertRaisesRegex(ValueError, "new and absent"):
                MODULE.securely_create_run_root(candidate, namespace=namespace)

            link_parent = root / "linked-runs"
            target_parent = root / "real-runs"
            target_parent.mkdir()
            link_parent.symlink_to(target_parent, target_is_directory=True)
            linked_namespace = link_parent / "v3"
            with self.assertRaisesRegex(ValueError, "unsafe"):
                MODULE.securely_create_run_root(
                    linked_namespace / "campaign-b", namespace=linked_namespace
                )

    def test_pinned_declaration_failure_happens_before_any_run_write(self) -> None:
        checker = types.SimpleNamespace(
            V3_RUNS_ROOT=MODULE.V3_RUNS_ROOT,
            validate_declaration=mock.Mock(side_effect=ValueError("declaration failed"))
        )
        with (
            mock.patch.object(MODULE, "_load_pinned_checker", return_value=checker),
            mock.patch.object(MODULE, "securely_create_run_root") as create,
            self.assertRaisesRegex(ValueError, "declaration failed"),
        ):
            MODULE.prepare_campaign(
                config_path=Path("ignored"),
                image_config_path=Path("ignored"),
                selection_proof_path=Path("ignored"),
                run_root=Path("ignored"),
            )
        create.assert_not_called()

    def test_launcher_has_no_historical_or_reserved_defaults(self) -> None:
        launcher = (
            ROOT / "scripts/run_swe_state_interpreter_v3_campaign.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CONFIG=${CONFIG:-}", launcher)
        self.assertIn("IMAGE_CONFIG=${IMAGE_CONFIG:-}", launcher)
        self.assertIn("SELECTION_PROOF=${SELECTION_PROOF:-}", launcher)
        self.assertIn("RUN_ROOT=${RUN_ROOT:-}", launcher)
        self.assertIn("PROXY_PORT=${PROXY_PORT:-}", launcher)
        self.assertIn(
            "runs/swe_state_interpreter_v3_development", launcher
        )
        self.assertEqual(
            MODULE.EXPECTED_SELECTION_PROOF_PATH,
            "validation/swe-task-state-v3-development-cohort-selection.json",
        )
        self.assertNotIn("run_swe_behavioral_campaign.sh", launcher)
        self.assertNotIn("swe_task_state_validation_a_20260718", launcher)
        self.assertNotIn("swe_task_state_validation_b_20260718", launcher)
        self.assertNotIn("LATEST_RUN", launcher)
        self.assertNotIn("using compatible endpoint", launcher)
        self.assertNotIn("check_endpoint.py", launcher)
        self.assertIn('"$ROOT/scripts/start_server.sh"', launcher)

    def test_launcher_freezes_generation_and_capture_settings(self) -> None:
        launcher = (
            ROOT / "scripts/run_swe_state_interpreter_v3_campaign.sh"
        ).read_text(encoding="utf-8")
        required_fragments = (
            "MAX_MODEL_LEN=65536",
            "MODEL_REVISION=${MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}",
            "QUANTIZATION=${QUANTIZATION:-modelopt_fp4}",
            "ATTENTION_BACKEND=${ATTENTION_BACKEND:-TRITON_ATTN}",
            "NUM_SPEC_TOKENS=1",
            "KV_CACHE_DTYPE=fp8_e4m3",
            "LUMO_PROXY_FORCE_TEMPERATURE=1.0",
            "LUMO_PROXY_FORCE_TOP_P=0.95",
            "LUMO_PROXY_FORCE_TOP_K=20",
            "LUMO_PROXY_FORCE_SEED=880001234",
            "--agent-wall-s 900",
            "--qwen-max-wall 840s",
            "--max-session-turns 50",
            "--proxy-context-limit 65536",
            "--allow-empty-predictions",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, launcher)


if __name__ == "__main__":
    unittest.main()
