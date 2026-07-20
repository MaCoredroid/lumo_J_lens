#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from scripts import analyze_swe_task_state_v4_transport_power as transport


class V4TransportPowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_config = json.loads(transport.DEFAULT_CONFIG.read_text())
        cls.raw_source = json.loads(
            transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT.read_text()
        )
        cls.config = transport.validate_transport_config(cls.raw_config)
        cls.source = transport.validate_source_feasibility_artifact(
            cls.raw_source, cls.config
        )
        cls.binding_hashes = transport.verify_bound_files()
        cls.profiles, cls.profile_summary = (
            transport.load_frozen_task_profiles()
        )

    def test_frozen_config_and_all_artifact_bindings_are_exact(self) -> None:
        self.assertEqual(
            transport._sha256_file(transport.DEFAULT_CONFIG),
            transport.TRANSPORT_CONFIG_SHA256,
        )
        self.assertEqual(
            transport._sha256_file(transport.DEFAULT_DESIGN_ARTIFACT),
            transport.DESIGN_ARTIFACT_SHA256,
        )
        self.assertEqual(
            transport._sha256_file(
                transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT
            ),
            transport.SOURCE_FEASIBILITY_ARTIFACT_SHA256,
        )
        self.assertEqual(
            transport._sha256_file(transport.SOURCE_FEASIBILITY_CONFIG),
            transport.SOURCE_FEASIBILITY_CONFIG_SHA256,
        )
        self.assertEqual(
            transport._sha256_file(transport.SOURCE_FEASIBILITY_ANALYZER),
            transport.SOURCE_FEASIBILITY_ANALYZER_SHA256,
        )
        self.assertEqual(
            transport._sha256_file(transport.PAIRED_POWER_ANALYZER),
            transport.PAIRED_POWER_ANALYZER_SHA256,
        )
        self.assertEqual(
            transport._sha256_file(transport.PAIRED_POWER_CONFIG),
            transport.PAIRED_POWER_CONFIG_SHA256,
        )
        self.assertEqual(transport.np.__version__, "2.5.1")
        self.assertEqual(transport.scipy.__version__, "1.18.0")
        self.assertEqual(
            self.source["source_binding"]["config_sha256"],
            transport.SOURCE_FEASIBILITY_CONFIG_SHA256,
        )
        self.assertEqual(
            self.source["source_binding"]["analyzer_sha256"],
            transport.SOURCE_FEASIBILITY_ANALYZER_SHA256,
        )

    def test_direct_script_context_can_import_bound_power_module(self) -> None:
        script = Path(transport.__file__).resolve()
        code = (
            "import runpy; "
            f"ns=runpy.run_path({str(script)!r}, run_name='transport_preflight'); "
            "module=ns['load_bound_power_module'](); "
            "print(module.__name__)"
        )
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.stdout.strip(),
            "scripts.analyze_swe_task_state_v4_power",
        )

    def test_config_rejects_power_policy_and_binding_corruption(self) -> None:
        cases = [
            (
                ("resampling_contract", "seed_sequence_entropy"),
                1,
                "resampling identity",
            ),
            (
                (
                    "transport_scenarios",
                    "primary_unseen_repository_coupling",
                ),
                "independent",
                "scenario identity",
            ),
            (
                ("monte_carlo_decision", "minimum_acceptable_joint_power"),
                0.5,
                "Monte Carlo decision",
            ),
            (
                ("decision_rule", "pilot_authorized"),
                True,
                "decision rule",
            ),
            (
                ("data_policy", "reserved_validation_allowed"),
                True,
                "data policy",
            ),
            (
                (
                    "bindings",
                    "paired_power_implementation",
                    "analyzer_sha256",
                ),
                "0" * 64,
                "implementation binding",
            ),
        ]
        for keys, replacement, message in cases:
            with self.subTest(keys=keys):
                tampered = copy.deepcopy(self.raw_config)
                target = tampered
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = replacement
                with self.assertRaisesRegex(ValueError, message):
                    transport.validate_transport_config(tampered)

    def test_source_artifact_rejects_count_hash_and_policy_corruption(
        self,
    ) -> None:
        tampered = copy.deepcopy(self.raw_source)
        tampered["complement"]["repository_counts"]["pallets/flask"] = 11
        with self.assertRaisesRegex(ValueError, "complement identity or counts"):
            transport.validate_source_feasibility_artifact(
                tampered, self.config
            )

        tampered = copy.deepcopy(self.raw_source)
        tampered["source_binding"]["analyzer_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "config/analyzer binding"):
            transport.validate_source_feasibility_artifact(
                tampered, self.config
            )

        tampered = copy.deepcopy(self.raw_source)
        tampered["fresh_cohort_selection_authorized"] = True
        with self.assertRaisesRegex(ValueError, "policy flag"):
            transport.validate_source_feasibility_artifact(
                tampered, self.config
            )

    def test_exact_hash_check_rejects_corrupt_or_symlinked_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.json"
            path.write_bytes(b'{"safe":true}\n')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                transport._require_exact_file_hash(path, expected, "input"),
                expected,
            )
            path.write_bytes(b'{"safe":false}\n')
            with self.assertRaisesRegex(ValueError, "frozen hash"):
                transport._require_exact_file_hash(path, expected, "input")

            target = root / "target.json"
            target.write_bytes(b"{}\n")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                transport._require_exact_file_hash(
                    link,
                    hashlib.sha256(target.read_bytes()).hexdigest(),
                    "input",
                )

    def test_supply_math_is_exact_and_fail_closed(self) -> None:
        result = transport.derive_supply_feasibility(
            transport.OFFICIAL_SUPPLY,
            transport.PILOT_ALLOCATION,
            transport.CONFIRMATION_ALLOCATION,
        )
        self.assertEqual(result["pilot_task_count"], 15)
        self.assertEqual(result["fresh_confirmation_task_count"], 230)
        self.assertEqual(result["combined_task_count"], 245)
        self.assertEqual(
            result["official_only_feasible_confirmation_task_count"], 215
        )
        self.assertEqual(
            result["qualified_shortfall"],
            {"mwaskom/seaborn": 10, "pallets/flask": 5},
        )
        self.assertEqual(
            result["official_only_feasible_confirmation"]["mwaskom/seaborn"],
            10,
        )
        self.assertEqual(
            result["official_only_feasible_confirmation"]["pallets/flask"],
            5,
        )
        self.assertFalse(result["qualified_source_supply_passes"])

    def test_seedsequence_streams_are_separated_and_reproducible(self) -> None:
        first, first_manifest = transport._spawn_generators(20260729)
        second, second_manifest = transport._spawn_generators(20260729)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(
            [item["spawn_key"] for item in first_manifest["child_streams_in_order"]],
            [[0], [1], [2], [3], [4], [5]],
        )
        first_values = {
            name: generator.random(8)
            for name, generator in first.items()
        }
        second_values = {
            name: generator.random(8)
            for name, generator in second.items()
        }
        for name in transport.CHILD_STREAMS:
            np.testing.assert_array_equal(first_values[name], second_values[name])
        for left, right in zip(
            transport.CHILD_STREAMS,
            transport.CHILD_STREAMS[1:],
        ):
            self.assertFalse(np.array_equal(first_values[left], first_values[right]))

    @staticmethod
    def _uniform_profiles() -> OrderedDict[str, np.ndarray]:
        return OrderedDict(
            (
                repository,
                np.asarray(
                    [[-0.01, -0.005], [-0.02, -0.007]],
                    dtype=np.float64,
                ),
            )
            for repository in transport.KNOWN_REPOSITORIES
        )

    def test_small_simulation_is_deterministic_and_keeps_stream_contract(
        self,
    ) -> None:
        profiles = self._uniform_profiles()
        original = {key: value.copy() for key, value in profiles.items()}
        arguments = {
            "seed": 1234,
            "outer_cohorts": 20,
            "inner_draws": 40,
            "batch_size": 5,
            "upper_quantile": 0.975,
            "one_sided_alpha": 0.0125,
        }
        first = transport.simulate_transport_power(
            profiles, transport.CONFIRMATION_ALLOCATION, **arguments
        )
        second = transport.simulate_transport_power(
            profiles, transport.CONFIRMATION_ALLOCATION, **arguments
        )
        self.assertEqual(first, second)
        self.assertEqual(
            list(first["primary_scenarios"]),
            list(transport.PRIMARY_SCENARIOS),
        )
        self.assertEqual(
            list(first["independent_unseen_repository_sensitivities"]),
            list(transport.SENSITIVITY_SCENARIOS),
        )
        for result in (
            *first["primary_scenarios"].values(),
            *first["independent_unseen_repository_sensitivities"].values(),
        ):
            self.assertEqual(
                result["joint_gate"]["passing_cohort_count"], 20
            )
            self.assertEqual(result["joint_gate"]["power"], 1.0)
        for key in profiles:
            np.testing.assert_array_equal(profiles[key], original[key])

    def test_nonuniform_golden_distinguishes_transport_coupling(self) -> None:
        profiles: OrderedDict[str, np.ndarray] = OrderedDict()
        for index, repository in enumerate(transport.KNOWN_REPOSITORIES):
            base = -0.024 + 0.004 * index
            profiles[repository] = np.asarray(
                [
                    [base, base * 0.7],
                    [base + 0.002, base * 0.7 + 0.001],
                ],
                dtype=np.float64,
            )
        result = transport.simulate_transport_power(
            profiles,
            transport.CONFIRMATION_ALLOCATION,
            seed=4321,
            outer_cohorts=200,
            inner_draws=100,
            batch_size=10,
            upper_quantile=0.975,
            one_sided_alpha=0.0125,
        )
        coupled = result["primary_scenarios"]
        independent = result[
            "independent_unseen_repository_sensitivities"
        ]
        self.assertEqual(
            coupled["django_domain_proxy"]["joint_gate"][
                "passing_cohort_count"
            ],
            180,
        )
        self.assertEqual(
            independent[
                "django_domain_proxy_independent_unseen_repositories"
            ]["joint_gate"]["passing_cohort_count"],
            182,
        )
        self.assertEqual(
            coupled["exchangeable_repository_task_proxy"]["joint_gate"][
                "passing_cohort_count"
            ],
            69,
        )
        self.assertEqual(
            independent[
                "exchangeable_repository_task_proxy_"
                "independent_unseen_repositories"
            ]["joint_gate"]["passing_cohort_count"],
            51,
        )

    def test_clopper_pearson_and_strict_joint_gate_are_exact(self) -> None:
        self.assertAlmostEqual(
            transport.clopper_pearson_lower(8174, 10000, 0.0125),
            0.808574,
            places=6,
        )
        self.assertAlmostEqual(
            transport.clopper_pearson_lower(1498, 10000, 0.0125),
            0.141882,
            places=6,
        )
        self.assertEqual(
            transport.clopper_pearson_lower(0, 10000, 0.0125), 0.0
        )
        upper = np.asarray(
            [
                [-0.1, -0.2],
                [-0.1, 0.0],
                [0.0, -0.2],
                [0.1, 0.2],
            ],
            dtype=np.float64,
        )
        summary = transport._gate_summary(upper, one_sided_alpha=0.0125)
        self.assertEqual(summary["nll_gate"]["passing_cohort_count"], 2)
        self.assertEqual(summary["brier_gate"]["passing_cohort_count"], 2)
        self.assertEqual(summary["joint_gate"]["passing_cohort_count"], 1)
        self.assertEqual(summary["joint_gate"]["power"], 0.25)

    def test_structural_p90_arithmetic_fails_brier_permanently(self) -> None:
        result = transport.derive_structural_p90(
            self.profile_summary, self.config
        )
        self.assertAlmostEqual(
            result["infinite_task_equal_twelve_repository_point"][
                "multiclass_negative_log_likelihood"
            ],
            -0.00476882900345764,
            places=15,
        )
        self.assertAlmostEqual(
            result["infinite_task_equal_twelve_repository_point"][
                "multiclass_brier"
            ],
            -0.0013123805451938037,
            places=15,
        )
        self.assertLess(
            result["recovered_repository_gamma_upper_0_975"][
                "multiclass_negative_log_likelihood"
            ],
            0.0,
        )
        self.assertGreater(
            result["recovered_repository_gamma_upper_0_975"][
                "multiclass_brier"
            ],
            0.0,
        )
        self.assertFalse(result["strict_brier_gate_passes"])
        self.assertFalse(result["joint_gate_passes"])
        self.assertFalse(result["more_within_repository_tasks_can_repair"])

    def test_small_report_is_aggregate_only_and_all_flags_fail_closed(
        self,
    ) -> None:
        report = transport.build_transport_report(
            self.config,
            self.source,
            self.profiles,
            self.profile_summary,
            binding_hashes=self.binding_hashes,
            analyzer_sha256=transport._sha256_file(Path(transport.__file__)),
            outer_cohorts=20,
            inner_draws=40,
            batch_size=5,
        )
        self.assertEqual(report["decision"], "NO_GO")
        self.assertEqual(
            report["fail_closed_decision_labels"],
            [
                "NO_GO_RETAINED_TRANSPORT_ENVELOPE",
                "NO_GO_QUALIFIED_SOURCE_SHORTFALL",
            ],
        )
        self.assertFalse(report["production_resampling_contract_executed"])
        for key in (
            "pilot_authorized",
            "confirmation_authorized",
            "fresh_cohort_selection_or_generation_authorized",
            "selection_performed",
            "selection_authorized",
            "generation_performed",
            "generation_authorized",
            "reserved_membership_accessed",
            "reserved_validation_accessed",
            "reserved_validation_allowed",
            "task_text_prompt_completion_patch_or_trajectory_fields_consumed",
            "raw_task_ids_emitted",
            "confirmatory_interpretation",
            "operational_reliability_claim",
            "independent_v4_development_result",
        ):
            self.assertIs(report[key], False, key)
        serialized = json.dumps(report, sort_keys=True)
        for forbidden_key in (
            '"task_id"',
            '"instance_id"',
            '"predictions"',
            '"candidate_rows"',
            '"reference_rows"',
        ):
            self.assertNotIn(forbidden_key, serialized)

    def test_paths_reject_symlinks_forbidden_tokens_and_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            safe = root / "transport.json"
            validated = transport.validate_cli_paths(
                config_path=transport.DEFAULT_CONFIG,
                design_artifact_path=transport.DEFAULT_DESIGN_ARTIFACT,
                source_feasibility_artifact_path=(
                    transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT
                ),
                output_path=safe,
                output_root=root,
            )
            self.assertEqual(validated["output"], safe)

            with self.assertRaisesRegex(ValueError, "reserved or validation"):
                transport.validate_cli_paths(
                    config_path=transport.DEFAULT_CONFIG,
                    design_artifact_path=transport.DEFAULT_DESIGN_ARTIFACT,
                    source_feasibility_artifact_path=(
                        transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT
                    ),
                    output_path=root / "reserved.json",
                    output_root=root,
                )

            with self.assertRaisesRegex(ValueError, "reserved or validation"):
                transport.validate_cli_paths(
                    config_path=transport.DEFAULT_CONFIG,
                    design_artifact_path=transport.DEFAULT_DESIGN_ARTIFACT,
                    source_feasibility_artifact_path=(
                        transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT
                    ),
                    output_path=root / "my_reserved_result.json",
                    output_root=root,
                )

            nested = root / "nested"
            nested.mkdir()
            with self.assertRaisesRegex(ValueError, "directly under"):
                transport.validate_cli_paths(
                    config_path=transport.DEFAULT_CONFIG,
                    design_artifact_path=transport.DEFAULT_DESIGN_ARTIFACT,
                    source_feasibility_artifact_path=(
                        transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT
                    ),
                    output_path=nested / "transport.json",
                    output_root=root,
                )

            link = root / "config.json"
            link.symlink_to(transport.DEFAULT_CONFIG)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                transport.validate_cli_paths(
                    config_path=link,
                    design_artifact_path=transport.DEFAULT_DESIGN_ARTIFACT,
                    source_feasibility_artifact_path=(
                        transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT
                    ),
                    output_path=safe,
                    output_root=root,
                )

            safe.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "already exists"):
                transport.validate_cli_paths(
                    config_path=transport.DEFAULT_CONFIG,
                    design_artifact_path=transport.DEFAULT_DESIGN_ARTIFACT,
                    source_feasibility_artifact_path=(
                        transport.DEFAULT_SOURCE_FEASIBILITY_ARTIFACT
                    ),
                    output_path=safe,
                    output_root=root,
                )

    def test_o_excl_writer_never_clobbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            transport._write_json_no_clobber(path, {"decision": "NO_GO"})
            original = path.read_bytes()
            self.assertEqual(json.loads(original), {"decision": "NO_GO"})
            with self.assertRaises(FileExistsError):
                transport._write_json_no_clobber(path, {"decision": "GO"})
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_profile_and_allocation_shapes_fail_closed(self) -> None:
        profiles = self._uniform_profiles()
        reversed_profiles = OrderedDict(reversed(list(profiles.items())))
        with self.assertRaisesRegex(ValueError, "repository order"):
            transport.simulate_transport_power(
                reversed_profiles,
                transport.CONFIRMATION_ALLOCATION,
                seed=1,
                outer_cohorts=2,
                inner_draws=2,
                batch_size=1,
                upper_quantile=0.975,
                one_sided_alpha=0.0125,
            )
        reversed_allocation = OrderedDict(
            reversed(list(transport.CONFIRMATION_ALLOCATION.items()))
        )
        with self.assertRaisesRegex(ValueError, "allocation repository order"):
            transport.simulate_transport_power(
                profiles,
                reversed_allocation,
                seed=1,
                outer_cohorts=2,
                inner_draws=2,
                batch_size=1,
                upper_quantile=0.975,
                one_sided_alpha=0.0125,
            )

if __name__ == "__main__":
    unittest.main()
