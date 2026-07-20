from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_observable_decoder.py"

spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_observable_decoder", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def small_rows() -> list[dict[str, object]]:
    rows = []
    labels_by_repo = {
        "repo-a": ["yes", "yes", "yes", "yes"],
        "repo-b": ["no", "no", "no", "no"],
        "repo-c": ["no", "no", "yes", "yes"],
    }
    global_index = 0
    for repository, labels in labels_by_repo.items():
        for request_index, _label in enumerate(labels, start=1):
            rows.append(
                {
                    "global_index": global_index,
                    "source_id_sha256": f"{global_index:064x}",
                    "task_id_sha256": f"{repository}-{request_index // 3}",
                    "repository": repository,
                    "request_index": request_index,
                }
            )
            global_index += 1
    return rows


class ObservableDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(module.load_json(module.CONFIG_PATH))

    def test_config_freezes_surface_marker_horizon_and_limitations(self) -> None:
        target = self.config["target_policy"]
        self.assertEqual(target["primary_target"], module.PRIMARY_TARGET)
        self.assertEqual(target["primary_horizon"], module.PRIMARY_HORIZON)
        self.assertIn("surface_language", target["primary_semantics"])
        self.assertTrue(target["private_or_hidden_semantics_forbidden"])
        limitations = " ".join(self.config["mandatory_limitations"]).lower()
        self.assertIn("not a diagnosis", limitations)
        self.assertIn("private chain-of-thought", limitations)
        self.assertIn("no emotion label", limitations)
        variants = {item["name"]: item for item in self.config["variants"]}
        self.assertEqual(
            variants["sequence_logit_j_plus_raw_activation_current"]["components"],
            ["sequence_logit_j", "raw_activation_current"],
        )
        self.assertEqual(
            variants["sequence_logit_j_plus_public_j_activation_current"]["width"],
            448,
        )
        self.assertIn(
            "sequence_logit_j",
            self.config["paired_comparisons"]["reference_baselines"],
        )
        self.assertEqual(self.config["status_scope"], module.STATUS_SCOPE)
        self.assertEqual(
            self.config["analysis_chronology"], module.ANALYSIS_CHRONOLOGY
        )
        self.assertTrue(
            self.config["analysis_chronology"][
                "nested_augmented_variants_added_after_initial_oof_results_were_observed"
            ]
        )
        self.assertFalse(
            self.config["analysis_chronology"]["nested_comparisons_preregistered"]
        )
        self.assertFalse(
            self.config["claim_scope"]["cot_or_cot_like_decoding_established"]
        )
        self.assertFalse(
            self.config["claim_scope"][
                "semantic_sentence_or_chain_decoding_established"
            ]
        )
        self.assertIn("not preregistered", limitations)
        self.assertIn("execution and contract-check completion only", limitations)

    def test_label_free_seams_accept_no_label_argument(self) -> None:
        for function in (
            module.assemble_label_free_feature_arrays,
            module.sanitize_visible_baseline_rows,
            module.build_variant_matrices,
        ):
            parameters = inspect.signature(function).parameters
            self.assertNotIn("labels", parameters)
            self.assertNotIn("targets", parameters)
            self.assertNotIn("outcomes", parameters)
        fold_parameters = inspect.signature(module._fit_predict_fold).parameters
        self.assertIn("y_train", fold_parameters)
        self.assertNotIn("y_test", fold_parameters)
        self.assertNotIn("heldout_labels", fold_parameters)

    def test_feature_array_allowlist_rejects_forbidden_field_first(self) -> None:
        with self.assertRaisesRegex(module.DecoderError, "forbidden field"):
            module.validate_feature_arrays(
                {"label": np.zeros(1606)}, alignment_rows=[]
            )

    def test_grouping_metadata_cannot_become_variant_component(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["variants"] = [
            {
                "name": "bad",
                "role": "activation_candidate",
                "components": ["repository"],
                "width": 1,
            }
        ]
        with self.assertRaisesRegex(module.DecoderError, "forbidden component"):
            module.build_variant_matrices({"repository": np.zeros(1606)}, changed)

    def test_caller_array_bundle_writer_is_not_public(self) -> None:
        self.assertNotIn("write_feature_bundle", module.__all__)
        self.assertNotIn("make_feature_manifest", module.__all__)
        self.assertFalse(hasattr(module, "write_feature_bundle"))
        self.assertFalse(hasattr(module, "make_feature_manifest"))

    def test_hierarchical_weights_equalize_repository_then_task_then_row(self) -> None:
        rows = [
            {"repository": "a", "task_id_sha256": "a1"},
            {"repository": "a", "task_id_sha256": "a1"},
            {"repository": "a", "task_id_sha256": "a2"},
            {"repository": "b", "task_id_sha256": "b1"},
            {"repository": "b", "task_id_sha256": "b1"},
            {"repository": "b", "task_id_sha256": "b1"},
        ]
        weights = module.hierarchical_row_weights(rows)
        np.testing.assert_allclose(weights, [0.125, 0.125, 0.25, 1 / 6, 1 / 6, 1 / 6])
        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3:].sum()), 0.5)

    def test_standardizer_uses_only_supplied_training_rows_and_weights(self) -> None:
        training = np.asarray([[0.0, 5.0], [2.0, 5.0]])
        mean, scale, constant = module.fit_weighted_standardizer(
            training, np.asarray([0.25, 0.75])
        )
        np.testing.assert_allclose(mean, [1.5, 5.0])
        np.testing.assert_allclose(scale, [np.sqrt(0.75), 1.0])
        np.testing.assert_array_equal(constant, [False, True])

    def test_loro_has_identical_folds_and_explicit_single_class_fold(self) -> None:
        rows = small_rows()
        labels = [
            "yes",
            "yes",
            "yes",
            "yes",
            "no",
            "no",
            "no",
            "no",
            "no",
            "no",
            "yes",
            "yes",
        ]
        signal = np.asarray(
            [[float(label == "yes"), index / 10.0] for index, label in enumerate(labels)]
        )
        matrices = {"reference": signal, "candidate": signal.copy()}
        specs = [
            {
                "name": "reference",
                "role": "ordinary_action_word_probe_sequence_baseline",
                "components": ["reference"],
                "width": 2,
            },
            {
                "name": "candidate",
                "role": "activation_candidate",
                "components": ["candidate"],
                "width": 2,
            },
        ]
        result = module.repository_heldout_oof(
            X_by_variant=matrices,
            rows=rows,
            labels=labels,
            classes=["no", "yes"],
            variant_specs=specs,
            model_contract=self.config["model"],
        )
        self.assertTrue(result["same_outer_folds_for_every_variant"])
        for variant in result["variants"].values():
            folds = {fold["heldout_repository"]: fold for fold in variant["folds"]}
            self.assertEqual(folds["repo-a"]["fit_status"], "fit_all_contract_classes")
            # repo-c carries both classes, so holding it out leaves repo-a=yes
            # and repo-b=no. Every fold in this fixture remains fit-capable.
            self.assertTrue(
                all(fold["heldout_repository_absent_from_training"] for fold in folds.values())
            )
            self.assertTrue(
                all(not fold["heldout_labels_used_for_fit_or_standardization"] for fold in folds.values())
            )
            self.assertTrue(
                all(fold["per_fold_balanced_or_auc_metrics_reported"] for fold in folds.values())
            )
            self.assertTrue(
                all("heldout_metrics" in fold for fold in folds.values())
            )
        np.testing.assert_array_equal(
            result["_probabilities"]["reference"],
            result["_probabilities"]["candidate"],
        )

        # A separate fixture makes the repo-a fold single-class in training.
        single_fold_labels = ["yes"] * 4 + ["no"] * 8
        single_result = module.repository_heldout_oof(
            X_by_variant={"reference": signal},
            rows=rows,
            labels=single_fold_labels,
            classes=["no", "yes"],
            variant_specs=specs[:1],
            model_contract=self.config["model"],
        )
        folds = {
            fold["heldout_repository"]: fold
            for fold in single_result["variants"]["reference"]["folds"]
        }
        self.assertEqual(folds["repo-a"]["fit_status"], "single_class_train_constant")
        self.assertFalse(folds["repo-a"]["model_fitted"])
        heldout_probabilities = np.asarray(
            single_result["variants"]["reference"]["oof_probabilities"]
        )[:4]
        self.assertTrue(np.all(heldout_probabilities[:, 0] > 0.999999))
        self.assertTrue(np.all(heldout_probabilities[:, 1] > 0.0))

    def test_single_class_oof_auc_metrics_are_explicitly_undefined(self) -> None:
        metrics = module.weighted_metrics(
            y=np.zeros(3, dtype=np.int64),
            probabilities=np.asarray(
                [[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]], dtype=np.float64
            ),
            weights=np.full(3, 1 / 3, dtype=np.float64),
            classes=["no", "yes"],
        )
        self.assertIsNone(metrics["weighted_auroc"]["value"])
        self.assertIsNone(metrics["weighted_auprc"]["value"])
        self.assertEqual(
            metrics["weighted_auroc"]["status"], "undefined_single_class_oof"
        )

    def test_paired_point_deltas_use_identical_rows_and_make_no_claim(self) -> None:
        probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        labels = np.asarray([0, 1], dtype=np.int64)
        weights = np.asarray([0.5, 0.5], dtype=np.float64)
        metrics = module.weighted_metrics(
            y=labels,
            probabilities=probabilities,
            weights=weights,
            classes=["no", "yes"],
        )
        paired = module.paired_oof_comparison(
            candidate_probabilities=probabilities,
            reference_probabilities=probabilities.copy(),
            y=labels,
            weights=weights,
            candidate_metrics=metrics,
            reference_metrics=metrics,
        )
        self.assertTrue(paired["identical_rows_labels_weights_and_folds"])
        self.assertEqual(
            paired["paired_weighted_mean_row_deltas"]["negative_log_likelihood"],
            0.0,
        )
        self.assertFalse(paired["point_deltas_establish_incremental_value"])

    def test_reserved_paths_are_rejected_lexically(self) -> None:
        with self.assertRaisesRegex(module.DecoderError, "before filesystem access"):
            module.frozen_lexical_path_preflight(
                [Path("/tmp/reserved-observable-decoder/input.json")]
            )

    def test_upstream_dtype_spelling_is_strictly_canonicalized(self) -> None:
        records = {
            "global_index": {
                "shape": [1606],
                "dtype": "little-endian-int64",
                "logical_sha256": "1" * 64,
            },
            "history_only": {
                "shape": [1606, 14],
                "dtype": "little-endian-float64",
                "logical_sha256": "2" * 64,
            },
        }
        normalized = module._canonicalize_upstream_array_records(
            records,
            names=["global_index", "history_only"],
            label="fixture arrays",
        )
        self.assertEqual(normalized["global_index"]["dtype"], "<i8")
        self.assertEqual(normalized["history_only"]["dtype"], "<f8")

        changed = copy.deepcopy(records)
        changed["history_only"]["dtype"] = "float64"
        with self.assertRaisesRegex(module.DecoderError, "contract changed"):
            module._canonicalize_upstream_array_records(
                changed,
                names=["global_index", "history_only"],
                label="fixture arrays",
            )

    def test_precommit_reauthentication_rejects_same_size_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authenticated = root / "authenticated.bin"
            authenticated.write_bytes(b"first-generation")
            record = module._artifact_byte_record(
                authenticated, label="authenticated fixture"
            )
            authenticated.write_bytes(b"mutated-generatn")
            self.assertEqual(authenticated.stat().st_size, record["size_bytes"])
            with self.assertRaisesRegex(
                module.DecoderError, "changed before report commit"
            ):
                module._assert_byte_records_unchanged(
                    {"authenticated_fixture": (authenticated, record)}
                )

    def test_before_publish_mutation_leaves_no_report_or_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authenticated = root / "decoder.py"
            authenticated.write_bytes(b"startup decoder bytes")
            record = module._artifact_byte_record(
                authenticated, label="decoder implementation"
            )
            output = root / "report.json"

            def mutate_and_verify() -> None:
                authenticated.write_bytes(b"mutated decoder bytes")
                module._assert_byte_records_unchanged(
                    {"decoder_implementation": (authenticated, record)}
                )

            with self.assertRaisesRegex(
                module.DecoderError, "changed before report commit"
            ):
                module._write_json_no_clobber(
                    output,
                    {"status": "passed"},
                    before_publish=mutate_and_verify,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".report.json.tmp-*")), [])

    def test_atomic_publish_race_preserves_competitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"

            def competing_link(
                _source: Path, destination: Path, *, follow_symlinks: bool
            ) -> None:
                self.assertFalse(follow_symlinks)
                Path(destination).write_bytes(b"competitor\n")
                raise FileExistsError(destination)

            with mock.patch.object(module.os, "link", side_effect=competing_link):
                with self.assertRaisesRegex(module.DecoderError, "overwrite"):
                    module._write_json_no_clobber(output, {"ours": True})
            self.assertEqual(output.read_bytes(), b"competitor\n")

    def test_dangling_output_symlink_is_rejected_without_creating_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "unintended-target.json"
            output = root / "report.json"
            output.symlink_to(target)
            with self.assertRaisesRegex(module.DecoderError, "overwrite"):
                module._write_json_no_clobber(output, {"ours": True})
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
