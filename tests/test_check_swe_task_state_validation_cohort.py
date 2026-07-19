from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/check_swe_task_state_validation_cohort.py"
SPEC = importlib.util.spec_from_file_location(
    "check_swe_task_state_validation_cohort", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
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


def write_bundle_fixture(directory: Path):
    instance_ids = ["alpha__project-1", "beta__project-2"]
    campaign_hashes = ["1" * 64, "2" * 64]
    cohort_ids = ["validation_a", "validation_b"]
    cohort = {
        "cohorts": [
            {
                "id": cohort_ids[index],
                "campaign_sha256": campaign_hashes[index],
                "instance_ids": [instance_id],
            }
            for index, instance_id in enumerate(instance_ids)
        ],
        "instance_ids": instance_ids,
        "pins": {
            "action_protocol_sha256": "3" * 64,
            "chat_template_sha256": "4" * 64,
        },
    }
    campaigns = [{"instance_ids": [instance_id]} for instance_id in instance_ids]
    cohort_path = directory / "cohort.json"
    cohort_path.write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    cohort_sha256 = MODULE.sha256_file(cohort_path)
    prompts = []
    for index, instance_id in enumerate(instance_ids):
        prompt = {
            "id": f"prompt-{index}",
            "metadata": {
                "cohort": {
                    "id": cohort_ids[index],
                    "index": index,
                    "campaign_sha256": campaign_hashes[index],
                    "cohort_manifest_sha256": cohort_sha256,
                    "source_task_instance_ids": [instance_id],
                },
                "task": {
                    "instance_id": instance_id,
                    "probeable_request_count": 1,
                    "probeable_request_indices": [1],
                },
                "selection": {
                    "task_request_index": 1,
                    "global_request_index": index + 1,
                    "probeable_request_indices": [1],
                    "candidate_count": 1,
                    "checkpoint_count": 1,
                    "max_checkpoints": None,
                },
                "provenance": {},
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
    summary = {
        "schema_version": 1,
        "kind": "swe_verified_behavioral_probe_combination",
        "cohort_manifest_sha256": cohort_sha256,
        "source_campaign_sha256s": campaign_hashes,
        "campaign_sha256s": campaign_hashes,
        "action_protocol_sha256": "3" * 64,
        "chat_template_sha256": "4" * 64,
        "cohort_count": 2,
        "task_count": 2,
        "global_request_count": 2,
        "prompt_count": 2,
        "prompt_bundle_sha256": MODULE.sha256_file(prompts_path),
        "cohorts": [
            {
                "id": cohort_ids[index],
                "index": index,
                "campaign_sha256": campaign_hashes[index],
                "cohort_manifest_sha256": cohort_sha256,
                "source_task_instance_ids": [instance_id],
                "source_task_count": 1,
            }
            for index, instance_id in enumerate(instance_ids)
        ],
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
                "cohort_id": cohort_ids[index],
                "campaign_sha256": campaign_hashes[index],
                "request_count": 1,
                "probeable_request_indices": [1],
                "selected_request_indices": [1],
                "selected_checkpoint_count": 1,
            }
            for index, instance_id in enumerate(instance_ids)
        ],
    }
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return cohort, campaigns, cohort_path, prompts_path, summary_path, prompts


class TaskStateCohortCheckerTests(unittest.TestCase):
    def test_repository_bindings_match_existing_combined_materializer_schema(self) -> None:
        cohort, image_config, campaigns = MODULE.load_and_validate_bindings(
            ROOT / "configs/swe_task_state_validation_cohort.json",
            ROOT / "configs/swe_task_state_validation_image_digests.json",
        )

        self.assertEqual(cohort["kind"], MODULE.COHORT_KIND)
        self.assertEqual(len(cohort["cohorts"]), 2)
        self.assertEqual([len(value["instance_ids"]) for value in campaigns], [10, 10])
        self.assertEqual(set(image_config["images"]), set(cohort["instance_ids"]))
        for row in cohort["cohorts"]:
            self.assertIn("campaign_path", row)
            self.assertIn("campaign_sha256", row)
            self.assertIn("run_label", row)
            self.assertIn("instance_ids", row)

    def test_reproduces_hash_order_quota_allocation_and_balanced_partition(self) -> None:
        seed = "unit-selection"
        by_repository = {
            "alpha": [f"alpha__project-{index}" for index in range(1, 4)],
            "beta": [f"beta__project-{index}" for index in range(1, 4)],
            "gamma": [f"gamma__project-{index}" for index in range(1, 3)],
        }
        all_ids = frozenset(
            value for values in by_repository.values() for value in values
        )
        selection = {
            "seed_text": seed,
            "candidate_count_after_exclusion": 8,
            "candidate_instance_ids_sha256": MODULE.sha256_sorted_strings(
                tuple(all_ids)
            ),
            "repository_minimum_candidates": 2,
            "repository_count": 3,
            "eligible_instance_ids_sha256": MODULE.sha256_sorted_strings(
                tuple(all_ids)
            ),
            "base_quota_per_repository": 2,
            "selected_task_count": 8,
        }

        result = MODULE.reproduce_selection(
            selection,
            official_ids=all_ids,
            cached_amd64_ids=all_ids,
            prior_used_ids=frozenset(),
        )

        ordered = {
            repository: hash_order(seed, values)
            for repository, values in by_repository.items()
        }
        self.assertEqual(result.quotas, {"alpha": 3, "beta": 3, "gamma": 2})
        self.assertEqual(result.ordered_by_repository, ordered)
        self.assertEqual(
            result.batches[0],
            (
                ordered["alpha"][0],
                ordered["alpha"][2],
                ordered["beta"][0],
                ordered["gamma"][0],
            ),
        )
        self.assertEqual(
            result.batches[1],
            (
                ordered["alpha"][1],
                ordered["beta"][1],
                ordered["beta"][2],
                ordered["gamma"][1],
            ),
        )

    def test_candidate_count_drift_fails_closed(self) -> None:
        ids = frozenset(
            {
                "alpha__project-1",
                "alpha__project-2",
                "beta__project-1",
                "beta__project-2",
            }
        )
        selection = {
            "seed_text": "unit-selection",
            "candidate_count_after_exclusion": 5,
            "repository_minimum_candidates": 2,
            "repository_count": 2,
            "base_quota_per_repository": 2,
            "selected_task_count": 4,
        }
        with self.assertRaisesRegex(
            MODULE.CohortValidationError, "candidate count.*changed"
        ):
            MODULE.reproduce_selection(
                selection,
                official_ids=ids,
                cached_amd64_ids=ids,
                prior_used_ids=frozenset(),
            )

    def test_same_count_candidate_identity_drift_fails_closed(self) -> None:
        frozen = frozenset(
            {
                "alpha__project-1",
                "alpha__project-2",
                "beta__project-1",
                "beta__project-2",
            }
        )
        changed = frozenset(
            {
                "alpha__project-1",
                "alpha__project-9",
                "beta__project-1",
                "beta__project-2",
            }
        )
        selection = {
            "seed_text": "unit-selection",
            "candidate_count_after_exclusion": 4,
            "candidate_instance_ids_sha256": MODULE.sha256_sorted_strings(
                tuple(frozen)
            ),
            "repository_minimum_candidates": 2,
            "repository_count": 2,
            "eligible_instance_ids_sha256": MODULE.sha256_sorted_strings(
                tuple(frozen)
            ),
            "base_quota_per_repository": 2,
            "selected_task_count": 4,
        }
        with self.assertRaisesRegex(
            MODULE.CohortValidationError, "candidate instance-set identity changed"
        ):
            MODULE.reproduce_selection(
                selection,
                official_ids=changed,
                cached_amd64_ids=changed,
                prior_used_ids=frozenset(),
            )

    def test_eligible_instance_set_identity_drift_fails_closed(self) -> None:
        ids = frozenset(
            {
                "alpha__project-1",
                "alpha__project-2",
                "beta__project-1",
            }
        )
        selection = {
            "seed_text": "unit-selection",
            "candidate_count_after_exclusion": 3,
            "candidate_instance_ids_sha256": MODULE.sha256_sorted_strings(
                tuple(ids)
            ),
            "repository_minimum_candidates": 2,
            "repository_count": 1,
            "eligible_instance_ids_sha256": MODULE.sha256_sorted_strings(
                tuple(ids)
            ),
            "base_quota_per_repository": 2,
            "selected_task_count": 2,
        }
        with self.assertRaisesRegex(
            MODULE.CohortValidationError, "eligible instance-set identity changed"
        ):
            MODULE.reproduce_selection(
                selection,
                official_ids=ids,
                cached_amd64_ids=ids,
                prior_used_ids=frozenset(),
            )

    def test_image_pin_validation_checks_local_bytes_and_digest_proof(self) -> None:
        instance_id = "alpha__project-1"
        tag = MODULE.canonical_tag(instance_id)
        image_id = f"sha256:{'1' * 64}"
        reference = f"{tag.removesuffix(':latest')}@sha256:{'1' * 64}"
        image_config = {
            "images": {
                instance_id: {
                    "x86_64": {"reference": reference, "image_id": image_id}
                }
            }
        }
        inspected = {
            tag: {
                "Architecture": "amd64",
                "Id": image_id,
                "RepoDigests": [reference],
            }
        }

        MODULE.validate_image_pins(
            image_config,
            selected_ids=[instance_id],
            cached_tags={instance_id: tag},
            inspected=inspected,
        )
        changed = copy.deepcopy(inspected)
        changed[tag]["Id"] = f"sha256:{'2' * 64}"
        with self.assertRaisesRegex(MODULE.CohortValidationError, "bytes differ"):
            MODULE.validate_image_pins(
                image_config,
                selected_ids=[instance_id],
                cached_tags={instance_id: tag},
                inspected=changed,
            )

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version": 1, "schema_version": 1}\n')
            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "duplicate JSON key"
            ):
                MODULE.strict_json_file(path, "duplicate fixture")

    def test_materialized_bundle_is_bound_to_every_frozen_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_bundle_fixture(Path(directory))
            cohort, campaigns, cohort_path, prompts_path, summary_path, _ = fixture

            result = MODULE.validate_materialized_bundle(
                cohort,
                campaigns,
                cohort_path=cohort_path,
                prompts_path=prompts_path,
                summary_path=summary_path,
            )

            self.assertEqual(result["task_count"], 2)
            self.assertEqual(result["prompt_count"], 2)

    def test_materialized_bundle_rejects_missing_task_even_with_valid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_bundle_fixture(Path(directory))
            cohort, campaigns, cohort_path, prompts_path, summary_path, prompts = fixture
            prompts_path.write_text(
                json.dumps(prompts[:1], indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )

            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "materialized task order changed"
            ):
                MODULE.validate_materialized_bundle(
                    cohort,
                    campaigns,
                    cohort_path=cohort_path,
                    prompts_path=prompts_path,
                    summary_path=summary_path,
                )

    def test_materialized_bundle_rejects_self_consistent_prompt_summary_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_bundle_fixture(Path(directory))
            cohort, campaigns, cohort_path, prompts_path, summary_path, _ = fixture
            summary = json.loads(summary_path.read_bytes())
            summary["prompts"] = list(reversed(summary["prompts"]))
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )

            with self.assertRaisesRegex(
                MODULE.CohortValidationError, "summary prompt binding changed"
            ):
                MODULE.validate_materialized_bundle(
                    cohort,
                    campaigns,
                    cohort_path=cohort_path,
                    prompts_path=prompts_path,
                    summary_path=summary_path,
                )


if __name__ == "__main__":
    unittest.main()
