from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_visible_baselines.py"
CONFIG_PATH = ROOT / "configs" / "swe_task_state_v4_visible_baselines.json"

spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_visible_baselines", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def alignment_index(rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": "swe_task_state_v4_label_free_alignment_index",
        "status": "passed",
        "scope": "grouping_order_and_stability_only_no_labels",
        "config": {"path": "config.json", "sha256": "a" * 64},
        "implementation": {"path": "implementation.py", "sha256": "b" * 64},
        "sources": [],
        "eligibility_source": {},
        "row_count": len(rows),
        "stable_row_count": sum(row["stable_feature_eligible"] for row in rows),
        "feature_use": {
            "allowed": [
                "task-local ordering for causal temporal transforms",
                "repository and task grouping for held-out splits and weights",
                "stable eligibility filtering",
            ],
            "forbidden": [
                "hashing or one-hot encoding IDs as model features",
                "repository or request index as semantic model features",
            ],
        },
        "rows": rows,
    }


def alignment_row(
    index: int,
    row_id: str,
    *,
    task_id: str = "task-a",
    request_index: int,
    stable: bool,
) -> dict:
    return {
        "global_index": index,
        "source_id_sha256": module.sha256_text(row_id),
        "task_id_sha256": module.sha256_text(task_id),
        "repository": "owner/repo",
        "request_index": request_index,
        "stable_feature_eligible": stable,
    }


def extracted_row(row_id: str, request_index: int, *, offset: float = 0.0) -> dict:
    return {
        "row_id": row_id,
        "task_id": "task-a",
        "repo": "owner/repo",
        "task_request_index": request_index,
        # These target/status fields are present in the frozen extractor output
        # and must not affect any numeric baseline tensor.
        "label": "inspect",
        "source_action_class_id": "inspect",
        "metric_evaluable": True,
        "features": {
            name: (np.arange(width, dtype=np.float64) + offset).tolist()
            for name, width in module.VARIANT_WIDTHS.items()
        },
    }


class VisibleBaselineTests(unittest.TestCase):
    def config(self):
        self.assertEqual(module.sha256_file(CONFIG_PATH), module.CONFIG_SHA256)
        return module.validate_config(module.load_json_strict(CONFIG_PATH))

    def test_config_freezes_exact_inputs_numeric_variants_and_false_claims(self):
        config = self.config()
        self.assertEqual(
            config["inputs"]["development_prompts"]["sha256"],
            "17f664b3029220458ff62b8e80a90ec5e796f0372217f02b727d6589df38d3d0",
        )
        self.assertEqual(
            config["inputs"]["development_public_report"]["sha256"],
            "7c943132163749f69bd35e4fa2e52bcfee2318fe349fa77603324a37ffaabe46",
        )
        self.assertEqual(
            config["inputs"]["label_free_alignment_index"]["sha256"],
            "c0d9e4bd6ad8f6962cf58776441ec4ef042eebbbae8a9959b4149e605c9ae38e",
        )
        self.assertEqual(config["campaign"]["stable_row_count"], 1606)
        self.assertEqual(
            config["campaign"]["output_tensor_keys"], list(module.OUTPUT_KEYS)
        )
        self.assertEqual(
            {name: value["width"] for name, value in config["variants"].items()},
            module.VARIANT_WIDTHS,
        )
        self.assertTrue(all(value is False for value in config["claim_scope"].values()))
        self.assertFalse(config["feature_boundary"]["label_sidecar_accepted"])

        changed = copy.deepcopy(config)
        changed["campaign"]["stable_row_count"] -= 1
        with self.assertRaises(module.BaselineError):
            module.validate_config(changed)

    def test_extraction_protocol_view_never_calls_broad_v3_validator(self):
        protocol = module.load_json_strict(
            ROOT / "configs" / "swe_task_state_interpreter_v3.json"
        )
        actions = module.load_json_strict(
            ROOT / "configs" / "swe_task_state_v3_action_probes.json"
        )
        with mock.patch.object(
            module.EXTRACT.V3,
            "validate_protocol",
            side_effect=AssertionError("broad validator touched closed tree"),
        ):
            derived = module.build_extraction_protocol(protocol, actions)
        self.assertEqual(derived["layers"], list(range(24, 48)))
        self.assertEqual(
            list(derived["token_ids_by_class"]),
            ["inspect", "edit", "validate", "finalize"],
        )
        self.assertEqual(derived["eligibility"], {"stable_rms": 0.02, "stable_max": 0.125})

    def test_alignment_validation_is_strict_and_label_free(self):
        rows = [
            alignment_row(0, "p1", request_index=1, stable=True),
            alignment_row(1, "p2", request_index=2, stable=False),
            alignment_row(2, "p3", request_index=3, stable=True),
        ]
        index = alignment_index(rows)
        observed = module.validate_alignment_index(
            index, expected_total_count=3, expected_stable_count=2
        )
        self.assertEqual(observed, rows)

        changed = copy.deepcopy(index)
        changed["rows"][0]["label"] = "inspect"
        with self.assertRaisesRegex(module.BaselineError, "schema changed"):
            module.validate_alignment_index(
                changed, expected_total_count=3, expected_stable_count=2
            )
        changed = copy.deepcopy(index)
        changed["rows"][2]["request_index"] = 2
        with self.assertRaisesRegex(module.BaselineError, "task order"):
            module.validate_alignment_index(
                changed, expected_total_count=3, expected_stable_count=2
            )

    def test_numeric_tensors_align_to_stable_global_indices_and_ignore_targets(self):
        alignments = [
            alignment_row(0, "p1", request_index=1, stable=True),
            alignment_row(1, "p2", request_index=2, stable=False),
            alignment_row(2, "p3", request_index=3, stable=True),
        ]
        rows = [extracted_row("p1", 1), extracted_row("p3", 3, offset=1000.0)]
        tensors = module.build_baseline_tensors(
            extracted_rows=rows,
            alignment_rows=alignments,
            expected_total_count=3,
            expected_stable_count=2,
        )
        self.assertEqual(tuple(tensors), module.OUTPUT_KEYS)
        np.testing.assert_array_equal(tensors["global_index"], [0, 2])
        for name, width in module.VARIANT_WIDTHS.items():
            self.assertEqual(tensors[name].shape, (2, width))
            self.assertEqual(tensors[name].dtype, np.dtype("<f8"))
        self.assertEqual(tensors["global_index"].dtype, np.dtype("<i8"))
        self.assertTrue(all(values.dtype.kind in {"i", "f"} for values in tensors.values()))

        changed_targets = copy.deepcopy(rows)
        changed_targets[0]["label"] = "check_or_finish"
        changed_targets[0]["source_action_class_id"] = "finalize"
        changed_targets[0]["metric_evaluable"] = False
        repeated = module.build_baseline_tensors(
            extracted_rows=changed_targets,
            alignment_rows=alignments,
            expected_total_count=3,
            expected_stable_count=2,
        )
        for name in module.OUTPUT_KEYS:
            np.testing.assert_array_equal(tensors[name], repeated[name])

    def test_shape_finiteness_and_identity_mismatches_fail_closed(self):
        alignments = [alignment_row(0, "p1", request_index=1, stable=True)]
        row = extracted_row("p1", 1)
        changed = copy.deepcopy(row)
        changed["features"]["history_only"] = [0.0] * 13
        with self.assertRaisesRegex(module.BaselineError, "width or finiteness"):
            module.build_baseline_tensors(
                extracted_rows=[changed],
                alignment_rows=alignments,
                expected_total_count=1,
                expected_stable_count=1,
            )
        changed = copy.deepcopy(row)
        changed["features"]["sequence_j"][0] = float("nan")
        with self.assertRaisesRegex(module.BaselineError, "width or finiteness"):
            module.build_baseline_tensors(
                extracted_rows=[changed],
                alignment_rows=alignments,
                expected_total_count=1,
                expected_stable_count=1,
            )
        changed = copy.deepcopy(row)
        changed["row_id"] = "different"
        with self.assertRaisesRegex(module.BaselineError, "identity differs"):
            module.build_baseline_tensors(
                extracted_rows=[changed],
                alignment_rows=alignments,
                expected_total_count=1,
                expected_stable_count=1,
            )

    def test_extraction_coverage_binds_stable_and_unstable_order(self):
        alignments = [
            alignment_row(0, "p1", request_index=1, stable=True),
            alignment_row(1, "p2", request_index=2, stable=False),
            alignment_row(2, "p3", request_index=3, stable=True),
        ]
        extraction = {
            "rows": [extracted_row("p1", 1), extracted_row("p3", 3)],
            "eligibility": {
                "all_replayed_prompt_count": 3,
                "numerically_stable_prompt_count": 2,
                "stable_feature_complete_prediction_count": 2,
                "exclusion_counts": {"numerically_unstable": 1},
                "exclusions": [
                    {"row_id": "p2", "reason": "numerically_unstable", "details": []}
                ],
            },
        }
        rows, _ = module.validate_extraction_coverage(
            extraction,
            alignment_rows=alignments,
            expected_total_count=3,
            expected_stable_count=2,
            expected_unstable_count=1,
        )
        self.assertEqual([row["row_id"] for row in rows], ["p1", "p3"])
        changed = copy.deepcopy(extraction)
        changed["eligibility"]["exclusions"][0]["row_id"] = "wrong"
        with self.assertRaisesRegex(module.BaselineError, "differs from alignment"):
            module.validate_extraction_coverage(
                changed,
                alignment_rows=alignments,
                expected_total_count=3,
                expected_stable_count=2,
                expected_unstable_count=1,
            )

    def test_numeric_npz_and_manifest_writers_are_strict_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "features.npz"
            manifest_path = root / "features.json"
            arrays = {
                "global_index": np.asarray([0], dtype="<i8"),
                "history_only": np.zeros((1, 14), dtype="<f8"),
            }
            module._write_npz_no_clobber(data_path, arrays)
            with np.load(data_path, allow_pickle=False) as loaded:
                self.assertEqual(set(loaded.files), set(arrays))
                self.assertTrue(all(loaded[name].dtype.kind in {"i", "f"} for name in loaded.files))
            with self.assertRaisesRegex(module.BaselineError, "overwrite"):
                module._write_npz_no_clobber(data_path, arrays)

            module._write_json_no_clobber(manifest_path, {"status": "passed"})
            self.assertEqual(json.loads(manifest_path.read_text()), {"status": "passed"})
            with self.assertRaisesRegex(module.BaselineError, "overwrite"):
                module._write_json_no_clobber(manifest_path, {"status": "passed"})

    def test_forbidden_path_is_rejected_before_filesystem_access(self):
        args = module.build_parser().parse_args(
            [
                "--output-data",
                "/tmp/validation/features.npz",
                "--output-manifest",
                "/tmp/features.json",
            ]
        )
        with mock.patch.object(
            module, "canonical_path_preflight", side_effect=AssertionError("resolve touched")
        ), mock.patch.object(
            module, "_require_regular_file", side_effect=AssertionError("file touched")
        ), mock.patch.object(
            module, "sha256_file", side_effect=AssertionError("hash touched")
        ):
            with self.assertRaisesRegex(module.BaselineError, "forbidden path"):
                module.run(args)


if __name__ == "__main__":
    unittest.main()
