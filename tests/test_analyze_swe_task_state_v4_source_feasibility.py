from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import analyze_swe_task_state_v4_source_feasibility as SOURCE


def _synthetic_rows() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    full = [
        ("alpha__alpha-1", "org/alpha"),
        ("alpha__alpha-2", "org/alpha"),
        ("beta__beta-3", "org/beta"),
        ("beta__beta-4", "org/beta"),
    ]
    verified = [
        ("alpha__alpha-1", "org/alpha"),
        ("beta__beta-3", "org/beta"),
    ]
    return full, verified


def _mini_expected_config(summary: dict[str, object]) -> dict[str, object]:
    complement = summary["complement"]
    assert isinstance(complement, dict)
    return {
        "sources": {
            "full_test": summary["full_test"],
            "verified": summary["verified"],
        },
        "derivation": {
            "complement_instance_count": complement["instance_count"],
            "complement_instance_ids_set_sha256": complement[
                "instance_ids_set_sha256"
            ],
            "complement_repository_counts": complement["repository_counts"],
        },
    }


class SourceFeasibilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = SOURCE.validate_source_config(
            SOURCE._read_json(SOURCE.DEFAULT_CONFIG, "checked-in test config")
        )

    def test_checked_in_config_hash_and_exact_source_bindings(self) -> None:
        self.assertEqual(SOURCE.sha256_file(SOURCE.DEFAULT_CONFIG), SOURCE.CONFIG_SHA256)
        self.assertEqual(
            self.config["sources"]["full_test"]["instance_ids_set_sha256"],
            "8348567d58d34d7749213678ac7b3e08cc21c14839262839db45c8c8f4aa4369",
        )
        self.assertEqual(
            self.config["sources"]["verified"]["instance_ids_set_sha256"],
            "33e18be7a9bd9f674790b63ed4d0b3fb17c176994802e3062b7d5a430a4e7d16",
        )
        self.assertEqual(
            self.config["sources"]["verified"]["file_logical_path"],
            "data/test-00000-of-00001.parquet",
        )
        self.assertEqual(
            self.config["sources"]["verified"]["file_format"], "parquet"
        )
        self.assertEqual(
            self.config["sources"]["verified"]["file_sha256"],
            "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd",
        )
        self.assertEqual(
            self.config["sources"]["verified"]["file_size_bytes"], 2_096_679
        )
        self.assertEqual(
            self.config["derivation"]["complement_instance_ids_set_sha256"],
            "953b83337651cfa8e68f812f30e3ba1394a8a08e1f66980680832a1d6bd02861",
        )
        self.assertEqual(
            sum(self.config["derivation"]["complement_repository_counts"].values()),
            1794,
        )

    def test_derivation_returns_aggregate_only_exact_complement(self) -> None:
        full, verified = _synthetic_rows()
        summary = SOURCE.summarize_identity_sources(full, verified)
        validated = SOURCE.validate_identity_summary(
            summary, _mini_expected_config(summary)
        )
        self.assertEqual(validated["full_test"]["instance_count"], 4)
        self.assertEqual(validated["verified"]["instance_count"], 2)
        self.assertEqual(validated["complement"]["instance_count"], 2)
        self.assertEqual(
            validated["complement"]["repository_counts"],
            {"org/alpha": 1, "org/beta": 1},
        )
        serialized = json.dumps(validated, sort_keys=True)
        for instance_id, _repository in full:
            self.assertNotIn(instance_id, serialized)

    def test_duplicate_identity_fails_closed(self) -> None:
        full, verified = _synthetic_rows()
        full.append(full[0])
        with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "repeats"):
            SOURCE.summarize_identity_sources(full, verified)

    def test_verified_non_subset_fails_closed(self) -> None:
        full, verified = _synthetic_rows()
        verified.append(("gamma__gamma-5", "org/gamma"))
        with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "not an exact subset"):
            SOURCE.summarize_identity_sources(full, verified)

    def test_verified_repository_mismatch_fails_closed(self) -> None:
        full, verified = _synthetic_rows()
        verified[0] = (verified[0][0], "other/alpha")
        with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "repository identity"):
            SOURCE.summarize_identity_sources(full, verified)

    def test_wrong_expected_identity_hash_fails_closed(self) -> None:
        full, verified = _synthetic_rows()
        summary = SOURCE.summarize_identity_sources(full, verified)
        expected = _mini_expected_config(summary)
        expected["derivation"]["complement_instance_ids_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "aggregate changed"):
            SOURCE.validate_identity_summary(summary, expected)

    def test_file_hash_validation_fails_before_projection(self) -> None:
        with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "full-test source bytes"):
            SOURCE.validate_input_file_hashes(
                self.config,
                full_file_sha256="0" * 64,
                verified_file_sha256=self.config["sources"]["verified"][
                    "file_sha256"
                ],
                verified_file_size_bytes=self.config["sources"]["verified"][
                    "file_size_bytes"
                ],
            )

    def test_verified_file_size_validation_fails_before_projection(self) -> None:
        with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "byte size"):
            SOURCE.validate_input_file_hashes(
                self.config,
                full_file_sha256=self.config["sources"]["full_test"][
                    "file_sha256"
                ],
                verified_file_sha256=self.config["sources"]["verified"][
                    "file_sha256"
                ],
                verified_file_size_bytes=2_096_678,
            )

    def test_loader_receives_exact_identity_projection(self) -> None:
        calls: list[tuple[Path, str, tuple[str, str]]] = []

        def loader(
            path: Path, file_format: str, columns: tuple[str, str]
        ) -> list[tuple[str, str]]:
            calls.append((path, file_format, columns))
            return [("alpha__alpha-1", "org/alpha")]

        path = Path("opaque.parquet")
        rows = SOURCE.load_identity_rows(
            path, file_format="parquet", loader=loader
        )
        self.assertEqual(rows, [("alpha__alpha-1", "org/alpha")])
        self.assertEqual(
            calls,
            [(path, "parquet", ("instance_id", "repo"))],
        )

    def test_pyarrow_projector_uses_exact_columns_for_both_sources(self) -> None:
        class FakeColumn:
            def __init__(self, values: list[str]) -> None:
                self.values = values

            def to_pylist(self) -> list[str]:
                return list(self.values)

        class FakeTable:
            column_names = ["instance_id", "repo"]

            def __init__(self, instance_id: str, repository: str) -> None:
                self.columns = {
                    "instance_id": FakeColumn([instance_id]),
                    "repo": FakeColumn([repository]),
                }

            def column(self, name: str) -> FakeColumn:
                return self.columns[name]

        read_table = mock.Mock(
            side_effect=[
                FakeTable("alpha__alpha-1", "org/alpha"),
                FakeTable("beta__beta-2", "org/beta"),
            ]
        )
        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.read_table = read_table
        pyarrow.parquet = parquet
        with mock.patch.dict(
            sys.modules,
            {"pyarrow": pyarrow, "pyarrow.parquet": parquet},
        ):
            full_rows = SOURCE._project_identity_columns_pyarrow(
                Path("full.parquet"), "parquet", SOURCE.IDENTITY_COLUMNS
            )
            verified_rows = SOURCE._project_identity_columns_pyarrow(
                Path("verified.parquet"), "parquet", SOURCE.IDENTITY_COLUMNS
            )

        self.assertEqual(full_rows, [("alpha__alpha-1", "org/alpha")])
        self.assertEqual(verified_rows, [("beta__beta-2", "org/beta")])
        self.assertEqual(
            read_table.call_args_list,
            [
                mock.call(
                    "full.parquet",
                    columns=["instance_id", "repo"],
                    memory_map=True,
                    use_threads=False,
                ),
                mock.call(
                    "verified.parquet",
                    columns=["instance_id", "repo"],
                    memory_map=True,
                    use_threads=False,
                ),
            ],
        )

    def test_report_is_aggregate_only_and_all_authorizations_are_false(self) -> None:
        full, verified = _synthetic_rows()
        summary = SOURCE.summarize_identity_sources(full, verified)
        report = SOURCE.build_source_feasibility_report(
            self.config,
            summary,
            config_sha256=SOURCE.CONFIG_SHA256,
            full_file_sha256=self.config["sources"]["full_test"]["file_sha256"],
            verified_file_sha256=self.config["sources"]["verified"]["file_sha256"],
            verified_file_size_bytes=self.config["sources"]["verified"][
                "file_size_bytes"
            ],
            analyzer_sha256="1" * 64,
        )
        false_fields = (
            "fresh_cohort_selection_performed",
            "fresh_cohort_selection_authorized",
            "generation_performed",
            "generation_authorized",
            "task_payload_fields_read",
            "raw_instance_ids_emitted",
            "reserved_membership_accessed",
            "reserved_validation_data_accessed",
            "reserved_validation_accessed",
            "reserved_validation_allowed",
            "confirmatory_interpretation",
            "operational_reliability_claim",
            "independent_v4_development_result",
            "power_analysis_performed",
        )
        for field in false_fields:
            self.assertIs(report[field], False, field)
        serialized = json.dumps(report, sort_keys=True)
        for instance_id, _repository in full:
            self.assertNotIn(instance_id, serialized)

    def test_cli_paths_accept_only_regular_direct_safe_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            full = root / "full.parquet"
            verified = root / "verified.parquet"
            output_root = root / "out"
            output = output_root / "result.json"
            config.write_text("{}", encoding="utf-8")
            full.write_bytes(b"full")
            verified.write_bytes(b"verified")
            output_root.mkdir()
            paths = SOURCE.validate_cli_paths(
                config_path=config,
                full_test_path=full,
                verified_path=verified,
                output_path=output,
                canonical_config=config,
                output_root=output_root,
            )
            self.assertEqual(paths["full_test"], full)
            self.assertEqual(paths["verified"], verified)

    def test_cli_paths_reject_forbidden_tokens_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            full = root / "full.parquet"
            verified = root / "verified.parquet"
            output_root = root / "out"
            config.write_text("{}", encoding="utf-8")
            full.write_bytes(b"full")
            verified.write_bytes(b"verified")
            output_root.mkdir()

            forbidden_dir = root / "reserved_inputs"
            forbidden_dir.mkdir()
            forbidden = forbidden_dir / "full.parquet"
            forbidden.write_bytes(b"full")
            with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "forbidden"):
                SOURCE.validate_cli_paths(
                    config_path=config,
                    full_test_path=forbidden,
                    verified_path=verified,
                    output_path=output_root / "result.json",
                    canonical_config=config,
                    output_root=output_root,
                )

            symlink = root / "linked.parquet"
            try:
                symlink.symlink_to(full)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "non-symlink"):
                SOURCE.validate_cli_paths(
                    config_path=config,
                    full_test_path=symlink,
                    verified_path=verified,
                    output_path=output_root / "result.json",
                    canonical_config=config,
                    output_root=output_root,
                )

    def test_no_clobber_writer_preserves_first_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            SOURCE.write_json_no_clobber(output, {"first": True})
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                SOURCE.write_json_no_clobber(output, {"second": True})
            self.assertEqual(output.read_bytes(), before)

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "duplicate"):
                SOURCE._read_json(path, "duplicate fixture")

    def test_default_projection_rejects_unknown_format_without_importing_data(self) -> None:
        with self.assertRaisesRegex(SOURCE.SourceFeasibilityError, "unsupported"):
            SOURCE._project_identity_columns_pyarrow(
                Path("not-opened.data"), "unknown", SOURCE.IDENTITY_COLUMNS
            )


if __name__ == "__main__":
    unittest.main()
