from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import swe_task_state_v4_epistemic_chain_decoder as DECODER


ROOT = Path(__file__).resolve().parents[1]
MODEL_SNAPSHOT = Path(
    "/home/mark/.cache/huggingface/hub/"
    "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)


def _source_id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fold_provenance(
    training_source_ids: list[str], heldout_source_ids: list[str]
) -> dict[str, object]:
    return {
        "fold_id": "outer-repository-fixture",
        "training_source_ids": training_source_ids,
        "heldout_source_ids": heldout_source_ids,
    }


class EpistemicChainDecoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DECODER.load_frozen_config()

    def test_config_is_hash_bound_and_has_no_target_results(self) -> None:
        digest = hashlib.sha256(DECODER.CONFIG_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, DECODER.CONFIG_SHA256)
        self.assertFalse(
            self.config["scope"]["target_annotations_available_when_frozen"]
        )
        self.assertFalse(self.config["scope"]["target_results_observed_when_frozen"])
        self.assertEqual(
            self.config["frozen_inputs"]["current_observable_artifact_index"][
                "sha256"
            ],
            "68034c0089dfedb17b804f069b728acea9d9d554369b5b7cc86b34a67dff8f9f",
        )

    def test_signature_registry_is_exact_frozen_cartesian_product(self) -> None:
        signatures = DECODER.signature_registry(self.config)
        self.assertEqual(len(signatures), 252)
        self.assertEqual(len(signatures), len(set(signatures)))
        self.assertEqual(
            signatures[0], "code>supports>source_logic>motivates>inspect"
        )
        self.assertEqual(
            signatures[-1], "environment>narrows>other>motivates>validate"
        )

    def test_coarse_slots_cannot_be_promoted_to_sentence_content(self) -> None:
        broken = copy.deepcopy(self.config)
        broken["renderer"][
            "coarse_renderer_may_be_called_sentence_or_cot_like_content_decoding"
        ] = True
        with self.assertRaisesRegex(
            DECODER.EpistemicDecoderError, "claim boundary"
        ):
            DECODER.validate_config(broken)

    def test_sparse_edge_does_not_erase_binary_head_contract(self) -> None:
        gate = self.config["support_and_agreement_gate"]
        self.assertTrue(
            gate[
                "binary_heads_may_fit_when_their_own_agreement_and_support_gates_pass_even_if_an_edge_or_signature_gate_fails"
            ]
        )
        self.assertIn(
            "do_not_block_predeclared_binary_heads",
            gate["per_head_failure_actions"]["belief_edge_support_failure"],
        )
        self.assertIn(
            "proposition_content_lane_gate_passed",
            gate["full_concept_chain_claim_requires"],
        )

    def test_factorized_prediction_is_structural_only_with_bottleneck_confidence(self) -> None:
        chain = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        slots = {
            "evidence_kind": np.asarray(
                [[0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]
            ),
            "belief_edge": np.asarray([[0.6, 0.2, 0.2], [0.7, 0.2, 0.1]]),
            "hypothesis_domain": np.asarray(
                [
                    [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.05, 0.65, 0.1, 0.05, 0.05, 0.05, 0.05],
                ]
            ),
            "action_intent": np.asarray([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]),
        }
        result = DECODER.factorized_exact_predictions(
            has_novel_chain_probabilities=chain,
            slot_probabilities=slots,
            config=self.config,
        )
        self.assertEqual(result["labels"][0], "__no_novel_chain__")
        self.assertEqual(
            result["labels"][1],
            "code>supports>interface_contract>motivates>edit",
        )
        np.testing.assert_allclose(result["confidence"], [0.8, 0.6])
        self.assertTrue(
            result["coarse_structure_only_not_sentence_or_cot_like_content"]
        )

    def test_variant_builder_proves_strict_one_block_augmentations(self) -> None:
        row_count = 3
        blocks = {
            name: np.full((row_count, width), index + 1.0)
            for index, (name, width) in enumerate(
                self.config["numeric_feature_blocks"].items()
            )
        }
        variants = DECODER.build_variant_matrices(blocks, self.config)
        self.assertEqual(variants["strong_visible_reference"].shape, (3, 536))
        self.assertEqual(
            variants["strong_visible_reference_plus_raw_activation_current"].shape,
            (3, 728),
        )
        reference = variants["strong_visible_reference"]
        candidate = variants[
            "strong_visible_reference_plus_public_j_activation_current"
        ]
        np.testing.assert_array_equal(candidate[:, :536], reference)

    def test_hierarchical_bayesian_draw_is_deterministic_and_keeps_rows(self) -> None:
        rows = [
            {"repository": "a", "task_id_sha256": "a1"},
            {"repository": "a", "task_id_sha256": "a1"},
            {"repository": "a", "task_id_sha256": "a2"},
            {"repository": "b", "task_id_sha256": "b1"},
            {"repository": "b", "task_id_sha256": "b1"},
        ]
        first = DECODER.hierarchical_bayesian_weights(rows, seed=7)
        second = DECODER.hierarchical_bayesian_weights(rows, seed=7)
        other = DECODER.hierarchical_bayesian_weights(rows, seed=8)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertAlmostEqual(float(first.sum()), 1.0)
        self.assertTrue(np.all(first > 0.0))
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[3], first[4])

    def test_training_prototype_retrieval_uses_deterministic_key_tie_break(self) -> None:
        predicted = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        candidates = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        training_ids = [_source_id(f"train-{index}") for index in range(3)]
        heldout_ids = [_source_id(f"heldout-{index}") for index in range(2)]
        result = DECODER.training_prototype_retrieval(
            predicted_embeddings=predicted,
            training_embeddings=candidates,
            candidate_keys=["b", "a", "c"],
            predicted_source_ids=heldout_ids,
            candidate_source_ids=training_ids,
            fold_provenance=_fold_provenance(training_ids, heldout_ids),
        )
        self.assertEqual(result["candidate_keys"], ["a", "c"])
        self.assertEqual(result["candidate_pool_scope"], "training_only")
        self.assertTrue(result["fold_provenance_enforced"])
        np.testing.assert_allclose(result["cosine"], [1.0, 1.0])

    def test_weighted_proposition_ridge_is_fixed_and_normalizes_span_blocks(self) -> None:
        rng = np.random.default_rng(3)
        X_train = rng.normal(size=(20, 5))
        target = rng.normal(size=(20, 384))
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        weights = np.full(20, 1.0 / 20.0)
        training_ids = [_source_id(f"ridge-train-{index}") for index in range(20)]
        heldout_ids = [_source_id(f"ridge-heldout-{index}") for index in range(3)]
        predicted, diagnostics = DECODER.fit_predict_weighted_proposition_ridge(
            X_train=X_train,
            target_embeddings_train=target,
            training_weights=weights,
            X_test=X_train[:3],
            X_train_source_ids=training_ids,
            target_source_ids_train=training_ids,
            X_test_source_ids=heldout_ids,
            fold_provenance=_fold_provenance(training_ids, heldout_ids),
            config=self.config,
        )
        self.assertEqual(predicted.shape, (3, 384))
        np.testing.assert_allclose(
            np.linalg.norm(predicted, axis=1), np.ones(3), atol=1e-12
        )
        self.assertEqual(diagnostics["alpha"], 100.0)
        self.assertEqual(diagnostics["span_block_count"], 1)
        self.assertFalse(diagnostics["heldout_targets_used_for_fit"])
        self.assertTrue(diagnostics["fold_provenance_enforced"])

    def test_closed_set_ranks_are_explicitly_not_open_ended_generation(self) -> None:
        predicted = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        candidates = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        training_ids = [_source_id("closed-training")]
        heldout_ids = [_source_id(f"closed-heldout-{index}") for index in range(3)]
        result = DECODER.heldout_closed_set_ranks(
            predicted_embeddings=predicted,
            candidate_embeddings=candidates,
            candidate_keys=["x", "y", "z"],
            true_keys=["x", "z"],
            predicted_source_ids=heldout_ids[:2],
            candidate_source_ids=heldout_ids,
            fold_provenance=_fold_provenance(training_ids, heldout_ids),
        )
        np.testing.assert_array_equal(result["ranks"], [1, 2])
        self.assertEqual(result["recall_at_1"], 0.5)
        self.assertEqual(result["mean_reciprocal_rank"], 0.75)
        self.assertTrue(result["candidate_pool_is_future_derived"])
        self.assertFalse(result["open_ended_sentence_generation"])
        self.assertTrue(result["fold_provenance_enforced"])

    def test_fold_provenance_rejects_overlap_and_cross_partition_candidates(self) -> None:
        training_id = _source_id("partition-training")
        heldout_id = _source_id("partition-heldout")
        with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "overlap"):
            DECODER.validate_fold_provenance(
                _fold_provenance([training_id], [training_id])
            )
        provenance = _fold_provenance([training_id], [heldout_id])
        with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "escaped"):
            DECODER.training_prototype_retrieval(
                predicted_embeddings=np.asarray([[1.0, 0.0]]),
                training_embeddings=np.asarray([[1.0, 0.0]]),
                candidate_keys=["candidate"],
                predicted_source_ids=[heldout_id],
                candidate_source_ids=[heldout_id],
                fold_provenance=provenance,
            )
        with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "escaped"):
            DECODER.heldout_closed_set_ranks(
                predicted_embeddings=np.asarray([[1.0, 0.0]]),
                candidate_embeddings=np.asarray([[1.0, 0.0]]),
                candidate_keys=["candidate"],
                true_keys=["candidate"],
                predicted_source_ids=[heldout_id],
                candidate_source_ids=[training_id],
                fold_provenance=provenance,
            )

    def test_ridge_rejects_feature_target_source_order_mismatch(self) -> None:
        rng = np.random.default_rng(41)
        training_ids = [_source_id("ridge-order-a"), _source_id("ridge-order-b")]
        heldout_ids = [_source_id("ridge-order-heldout")]
        with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "row-misaligned"):
            DECODER.fit_predict_weighted_proposition_ridge(
                X_train=rng.normal(size=(2, 3)),
                target_embeddings_train=rng.normal(size=(2, 384)),
                training_weights=np.asarray([0.5, 0.5]),
                X_test=rng.normal(size=(1, 3)),
                X_train_source_ids=training_ids,
                target_source_ids_train=list(reversed(training_ids)),
                X_test_source_ids=heldout_ids,
                fold_provenance=_fold_provenance(training_ids, heldout_ids),
                config=self.config,
            )

    def test_guarded_path_rejects_logical_symlink_and_canonical_root_escape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jlens-path-guard-") as temporary:
            base = Path(temporary)
            allowed = base / "allowed"
            allowed.mkdir()
            regular = allowed / "regular.json"
            regular.write_text("{}", encoding="utf-8")
            logical_link = allowed / "logical-link.json"
            logical_link.symlink_to(regular)
            with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "symlink"):
                DECODER.resolve_guarded_path(
                    logical_link,
                    allowed_root=allowed,
                    reject_logical_symlink=True,
                    require_kind="file",
                )

            outside = base / "outside"
            outside.mkdir()
            escaped_file = outside / "escaped.json"
            escaped_file.write_text("{}", encoding="utf-8")
            redirected_parent = allowed / "redirected-parent"
            redirected_parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "escapes"):
                DECODER.resolve_guarded_path(
                    redirected_parent / "escaped.json",
                    allowed_root=allowed,
                    reject_logical_symlink=False,
                    require_kind="file",
                )

    def test_every_head_has_concrete_class_and_outer_fold_support(self) -> None:
        slot_classes = self.config["ontology"]["evidence_kind"]
        support = {
            class_name: {"rows": 20, "tasks": 10, "repositories": 5}
            for class_name in slot_classes
        }
        folds = {
            "outer-a": {class_name: 1 for class_name in slot_classes},
            "outer-b": {class_name: 2 for class_name in slot_classes},
        }
        passed = DECODER.evaluate_head_support(
            head_name="evidence_kind",
            class_support=support,
            outer_training_fold_class_counts=folds,
            config=self.config,
        )
        self.assertTrue(passed["passed"])
        broken_support = copy.deepcopy(support)
        broken_support[slot_classes[0]]["rows"] = 19
        failed = DECODER.evaluate_head_support(
            head_name="evidence_kind",
            class_support=broken_support,
            outer_training_fold_class_counts=folds,
            config=self.config,
        )
        self.assertFalse(failed["passed"])
        self.assertIn(f"{slot_classes[0]}:rows", failed["failures"])

    def test_predictive_claim_gate_requires_quality_and_positive_interval_bounds(self) -> None:
        candidate = "strong_visible_reference_plus_raw_activation_current"
        reference = "strong_visible_reference"
        requirements = self.config["proposition_content_lane"]["claim_gate"][
            "predictive_nested_delta_requirements"
        ]
        deltas = {
            requirement["metric"]: {
                "candidate": candidate,
                "reference": reference,
                "improvement_orientation": requirement["improvement_orientation"],
                "point": 0.05,
                "interval_lower": 0.01,
                "interval_upper": 0.09,
                "interval_level": 0.95,
                "algorithm": "hierarchical_bayesian_cluster_bootstrap_with_full_model_refit",
                "draw_count": 1000,
                "multiplicity_adjusted": True,
                "multiplicity_method": "bonferroni_equal_tailed_percentile",
                "multiplicity_family_size": 6,
            }
            for requirement in requirements
        }
        absolute = {
            "hierarchical_weighted_true_H_cosine": 0.4,
            "hierarchical_weighted_training_only_retrieved_H_to_true_H_cosine": 0.4,
            "proposition_content_accepted_coverage": 0.25,
        }
        passed = DECODER.evaluate_predictive_claim_gate(
            candidate_name=candidate,
            absolute_metrics=absolute,
            paired_deltas=deltas,
            config=self.config,
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(passed["private_or_literal_sentence_recovery_established"])

        crossing = copy.deepcopy(deltas)
        crossing["true_H_cosine"]["interval_lower"] = 0.0
        failed_interval = DECODER.evaluate_predictive_claim_gate(
            candidate_name=candidate,
            absolute_metrics=absolute,
            paired_deltas=crossing,
            config=self.config,
        )
        self.assertFalse(failed_interval["passed"])
        self.assertIn(
            "paired_interval_lower:true_H_cosine", failed_interval["failures"]
        )

        below_floor = dict(absolute)
        below_floor["hierarchical_weighted_true_H_cosine"] = 0.349
        failed_floor = DECODER.evaluate_predictive_claim_gate(
            candidate_name=candidate,
            absolute_metrics=below_floor,
            paired_deltas=deltas,
            config=self.config,
        )
        self.assertFalse(failed_floor["passed"])

        mixed = copy.deepcopy(deltas)
        mixed["true_H_cosine"][
            "candidate"
        ] = "strong_visible_reference_plus_public_j_activation_current"
        with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "provenance"):
            DECODER.evaluate_predictive_claim_gate(
                candidate_name=candidate,
                absolute_metrics=absolute,
                paired_deltas=mixed,
                config=self.config,
            )

    def test_config_rejects_erased_support_or_predictive_gates(self) -> None:
        broken_support = copy.deepcopy(self.config)
        broken_support["support_and_agreement_gate"]["binary_head_support"][
            "completion_has_chain"
        ]["minimum_rows_per_class"] = 0
        with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "support"):
            DECODER.validate_config(broken_support)

        broken_claim = copy.deepcopy(self.config)
        broken_claim["proposition_content_lane"]["claim_gate"][
            "absolute_semantic_quality_floors"
        ]["hierarchical_weighted_true_H_cosine"] = 0.0
        with self.assertRaisesRegex(DECODER.EpistemicDecoderError, "claim"):
            DECODER.validate_config(broken_claim)

    def test_full_prefix_lsa_fits_training_text_only(self) -> None:
        # Repeated token pairs survive min_df=2 and create more than 256 columns.
        training = [
            " ".join(
                [
                    "shared semantic prefix",
                    f"token{index % 180}",
                    f"bridge{(index + 1) % 180}",
                    f"token{(index + 2) % 180}",
                ]
            )
            for index in range(300)
        ]
        heldout = ["heldout_only_term shared semantic prefix"]
        train_matrix, heldout_matrix, diagnostics = DECODER.fit_full_prefix_lsa(
            training_texts=training,
            heldout_texts=heldout,
            config=self.config,
        )
        self.assertEqual(train_matrix.shape, (300, 256))
        self.assertEqual(heldout_matrix.shape, (1, 256))
        self.assertFalse(diagnostics["heldout_text_used_for_fit"])
        self.assertGreater(diagnostics["vocabulary_size"], 256)
        self.assertEqual(len(diagnostics["fold_transform_sha256"]), 64)
        self.assertIn(
            "idf_sha256", diagnostics["fold_transform_component_manifest"]
        )
        self.assertIn(
            "feature_order_sha256",
            diagnostics["fold_transform_component_manifest"],
        )

    def test_lsa_transform_hash_binds_vocabulary_and_idf_with_fixed_components(self) -> None:
        component_count = self.config["full_prefix_semantic_baseline"][
            "latent_semantic_projection"
        ]["n_components"]
        feature_count = component_count + 1
        vocabulary = {
            f"feature-{index}": index for index in range(feature_count)
        }
        idf = np.linspace(1.0, 2.0, feature_count, dtype=np.float64)
        components = np.zeros(
            (component_count, feature_count), dtype=np.float64
        )
        singular_values = np.ones(component_count, dtype=np.float64)
        explained_variance = np.ones(component_count, dtype=np.float64)
        explained_variance_ratio = np.full(
            component_count, 1.0 / component_count, dtype=np.float64
        )

        def fingerprint(
            bound_vocabulary: dict[str, int], bound_idf: np.ndarray
        ) -> dict[str, object]:
            return DECODER.full_prefix_lsa_transform_fingerprint(
                vocabulary=bound_vocabulary,
                idf=bound_idf,
                svd_components=components,
                svd_singular_values=singular_values,
                svd_explained_variance=explained_variance,
                svd_explained_variance_ratio=explained_variance_ratio,
                config=self.config,
            )

        baseline = fingerprint(vocabulary, idf)
        changed_vocabulary = dict(vocabulary)
        changed_vocabulary["feature-0"] = 1
        changed_vocabulary["feature-1"] = 0
        vocabulary_changed = fingerprint(changed_vocabulary, idf)
        changed_idf = idf.copy()
        changed_idf[0] += 0.25
        idf_changed = fingerprint(vocabulary, changed_idf)

        self.assertEqual(
            baseline["component_manifest"]["svd_components_sha256"],
            vocabulary_changed["component_manifest"]["svd_components_sha256"],
        )
        self.assertEqual(
            baseline["component_manifest"]["svd_components_sha256"],
            idf_changed["component_manifest"]["svd_components_sha256"],
        )
        self.assertNotEqual(
            baseline["fold_transform_sha256"],
            vocabulary_changed["fold_transform_sha256"],
        )
        self.assertNotEqual(
            baseline["fold_transform_sha256"], idf_changed["fold_transform_sha256"]
        )

    def test_primary_novel_chain_scope_is_explicitly_conditional(self) -> None:
        self.assertTrue(
            self.config["claim_scope"][
                "primary_novel_chain_metrics_condition_on_excluding_prefix_exposed_future_positives"
            ]
        )
        self.assertTrue(
            any(
                "not all-boundary operational performance" in limitation
                for limitation in self.config["mandatory_limitations"]
            )
        )

    def test_hash_bound_embedding_snapshot_is_present_and_authenticates(self) -> None:
        if not MODEL_SNAPSHOT.exists():
            self.skipTest("pre-annotation MiniLM snapshot is not present in this environment")
        records = DECODER.validate_model_snapshot(MODEL_SNAPSHOT, self.config)
        self.assertEqual(len(records), 9)
        self.assertEqual(records["model.safetensors"]["size_bytes"], 90868376)


if __name__ == "__main__":
    unittest.main()
