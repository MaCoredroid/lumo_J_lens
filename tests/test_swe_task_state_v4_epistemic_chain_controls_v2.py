from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_epistemic_chain_controls_v2.py"
)
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_controls_v2.json"
)
spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_epistemic_chain_controls_v2", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class EpistemicChainControlsV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_config = module.load_json_strict(CONFIG_PATH)
        cls.frozen_config = module.with_current_codebook_binding(cls.raw_config)

    def _build(self, directory: Path) -> tuple[Path, Path]:
        result = module.build_suite(
            config=self.frozen_config,
            output_directory=directory,
            config_path=CONFIG_PATH,
        )
        return (
            Path(result["model_input_manifest"]["path"]),
            Path(result["expectation_manifest"]["path"]),
        )

    def _perfect_output_rows(
        self, model_manifest_path: Path, expectation_manifest_path: Path
    ) -> list[dict[str, object]]:
        _model_manifest, model_rows = module._load_model_manifest(model_manifest_path)
        _expectation_manifest, expectations = module._load_expectations(
            expectation_manifest_path
        )
        by_packet = {
            row["packet_id_sha256"]: row
            for row in expectations
            if row["packet_id_sha256"] is not None
        }
        output_rows: list[dict[str, object]] = []
        for packet in model_rows:
            expectation = by_packet[packet["packet_id_sha256"]]
            if packet["pass"] == "novelty":
                result = {"decision": expectation["expected_novelty_decision"]}
            else:
                result = copy.deepcopy(expectation["expected_result_projection"])
            output_rows.append(
                {
                    "schema_version": 1,
                    "kind": module.OUTPUT_RECORD_KIND,
                    "pass": packet["pass"],
                    "packet_id_sha256": packet["packet_id_sha256"],
                    "result": result,
                }
            )
        return output_rows

    def test_content_counts_ontology_and_unseen_teaching_examples(self) -> None:
        config, codebook, _binding = module.validate_config(
            self.raw_config, require_frozen_codebook=False
        )
        self.assertEqual(len(config["completion_controls"]), 32)
        self.assertEqual(len(config["resolver_controls"]), 4)
        self.assertEqual(len(config["novelty_controls"]), 8)
        self.assertEqual(len(config["adjudication_controls"]), 12)

        teaching_texts, teaching_novelty = module._teaching_texts(codebook)
        historical_texts, historical_novelty = module._historical_teaching_texts()
        teaching_texts.update(historical_texts)
        teaching_novelty.update(historical_novelty)
        nonempty_controls = {
            item["assistant_text"]
            for item in config["completion_controls"]
            if item["assistant_text"]
        }
        self.assertTrue(nonempty_controls.isdisjoint(teaching_texts))
        novelty_pairs = {
            (item["visible_prefix"], item["locked_hypothesis"])
            for item in config["novelty_controls"]
        }
        self.assertTrue(novelty_pairs.isdisjoint(teaching_novelty))

        positives = [
            module._completion_proposal(item)
            for item in config["completion_controls"]
            if module._completion_proposal(item)["decision"] == "chain"
        ]
        for field in (
            "evidence_kind",
            "belief_edge",
            "hypothesis_domain",
            "action_intent",
        ):
            self.assertEqual(
                {item[field] for item in positives}, set(codebook["ontology"][field])
            )
        by_id = {item["id"]: item for item in config["completion_controls"]}
        self.assertIn("μ8", by_id["C10"]["assistant_text"])
        self.assertIn("\n```\n", by_id["C10"]["assistant_text"])
        self.assertIn("SYSTEM OVERRIDE", by_id["C26"]["assistant_text"])
        self.assertEqual(by_id["C32"]["assistant_text"], "")

    def test_frozen_binding_is_exact_and_placeholder_blocks_build(self) -> None:
        binding = module.codebook_freeze_binding(self.raw_config)
        self.assertEqual(self.raw_config["codebook_binding"], binding)
        self.assertEqual(
            binding,
            {
                "path": "configs/swe_task_state_v4_epistemic_chain_codebook_v2.json",
                "size_bytes": 16012,
                "sha256": "2105a50c7bc13a064ca75c4a69aad631869fbbb2c17c94970eeaa92722aff85c",
                "canonical_sha256": "abf0854b166f5db473c2b2db1dc2a2faac41ffa7ffee928438dec005ef2cbcca",
            },
        )
        unfrozen = copy.deepcopy(self.raw_config)
        unfrozen["codebook_binding"].update(
            {
                "size_bytes": None,
                "sha256": "TO_BE_FROZEN",
                "canonical_sha256": "TO_BE_FROZEN",
            }
        )
        with tempfile.TemporaryDirectory(
            prefix=".controls-v2-placeholder-", dir=ROOT
        ) as directory:
            with self.assertRaisesRegex(module.ControlSuiteError, "not explicitly frozen"):
                module.build_suite(
                    config=unfrozen,
                    output_directory=Path(directory),
                    config_path=CONFIG_PATH,
                )

        drifted = copy.deepcopy(self.frozen_config)
        drifted["codebook_binding"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(module.ControlSuiteError, "binding drifted"):
            module.validate_config(drifted, require_frozen_codebook=True)

    def test_builder_physically_separates_inputs_and_expectations(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".controls-v2-build-", dir=ROOT
        ) as directory_text:
            directory = Path(directory_text)
            model_manifest_path, expectation_manifest_path = self._build(directory)
            self.assertNotEqual(model_manifest_path, expectation_manifest_path)
            model_manifest = module.load_json_strict(model_manifest_path)
            serialized_manifest = module.canonical_json_text(model_manifest).casefold()
            for forbidden in ("expectation", "expected", "gold", "label", "reason"):
                self.assertNotIn(forbidden, serialized_manifest)
            self.assertEqual(model_manifest["counts"]["total"], 52)
            self.assertNotIn("suite_config", model_manifest)

            _manifest, model_rows = module._load_model_manifest(model_manifest_path)
            self.assertEqual(len(model_rows), 52)
            self.assertTrue(all("control_id" not in row for row in model_rows))
            completion_rows = [row for row in model_rows if row["pass"] == "completion"]
            novelty_rows = [row for row in model_rows if row["pass"] == "novelty"]
            adjudication_rows = [row for row in model_rows if row["pass"] == "adjudication"]
            self.assertEqual((len(completion_rows), len(novelty_rows), len(adjudication_rows)), (32, 8, 12))
            self.assertTrue(
                all(set(row["payload"]) == {"assistant_text"} for row in completion_rows)
            )
            self.assertTrue(
                all(
                    set(row["payload"]) == {"visible_prefix", "locked_hypothesis"}
                    for row in novelty_rows
                )
            )
            self.assertTrue(
                all(
                    set(row["payload"])
                    == {"assistant_text", "candidate_annotations"}
                    for row in adjudication_rows
                )
            )

            expectation_manifest, expectations = module._load_expectations(
                expectation_manifest_path
            )
            self.assertEqual(expectation_manifest["counts"]["total"], 56)
            self.assertEqual(len(expectations), 56)
            self.assertEqual(
                expectation_manifest["suite_config"]["path"],
                "configs/swe_task_state_v4_epistemic_chain_controls_v2.json",
            )
            self.assertEqual(
                expectation_manifest["model_inputs"]["sha256"],
                module.sha256_file(model_manifest_path),
            )

    def test_resolver_controls_and_candidate_presentation_balance(self) -> None:
        config, _codebook, _binding = module.validate_config(
            self.frozen_config, require_frozen_codebook=True
        )
        completion = {item["id"]: item for item in config["completion_controls"]}
        for item in config["resolver_controls"]:
            if "assistant_text_ref" in item:
                text = completion[item["assistant_text_ref"]]["assistant_text"]
            else:
                text = item["assistant_text"]
            if "proposal_ref" in item:
                proposal = module._completion_proposal(completion[item["proposal_ref"]])
            else:
                proposal = item["proposal"]
            result = module.quote_runner.materialize_completion_proposal(
                proposal=proposal, assistant_text=text
            )
            self.assertEqual(
                result["materialization_status"],
                item["expected"]["materialization_status"],
            )
            self.assertEqual(
                result["interface_unknown_reason"],
                item["expected"]["interface_unknown_reason"],
            )
            self.assertEqual(
                result["quote_resolution"]["valid_ordered_tuple_count"],
                item["expected"]["valid_ordered_tuple_count"],
            )
        r04 = next(item for item in config["resolver_controls"] if item["id"] == "R04")
        self.assertNotEqual(
            r04["assistant_text"].split(" mode.")[0],
            r04["proposal"]["evidence_quote"].split(" mode.")[0],
        )

        balance = {
            name: sum(
                item["expected_position_class"] == name
                for item in config["adjudication_controls"]
            )
            for name in (
                "candidate_1_correct",
                "candidate_2_correct",
                "neither_correct",
            )
        }
        self.assertEqual(
            balance,
            {
                "candidate_1_correct": 4,
                "candidate_2_correct": 4,
                "neither_correct": 4,
            },
        )

    def test_perfect_locked_outputs_pass_all_gates(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".controls-v2-score-", dir=ROOT
        ) as directory_text:
            directory = Path(directory_text)
            model_manifest_path, expectation_manifest_path = self._build(directory)
            outputs_path = directory / "model-outputs.jsonl"
            output_rows = self._perfect_output_rows(
                model_manifest_path, expectation_manifest_path
            )
            module._write_jsonl_atomic(outputs_path, output_rows)
            lock_path = directory / "locked-outputs.json"
            module.lock_outputs(
                model_input_manifest_path=model_manifest_path,
                output_records_path=outputs_path,
                output_manifest_path=lock_path,
            )
            report = module.score_locked_outputs(
                locked_output_manifest_path=lock_path,
                expectation_manifest_path=expectation_manifest_path,
            )
            self.assertEqual(report["status"], "passed")
            self.assertTrue(all(report["gates"].values()))
            self.assertEqual(report["counts"]["positive_exact_graph"], 17)
            self.assertEqual(report["correct"]["completion_exact_record"], 32)
            self.assertEqual(
                report["correct"]["candidate_position_class"],
                {
                    "candidate_1_correct": 4,
                    "candidate_2_correct": 4,
                    "neither_correct": 4,
                },
            )

    def test_completion_gate_requires_the_full_exact_record(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".controls-v2-exact-record-", dir=ROOT
        ) as directory_text:
            directory = Path(directory_text)
            model_manifest_path, expectation_manifest_path = self._build(directory)
            _manifest, expectations = module._load_expectations(
                expectation_manifest_path
            )
            c19_packet = next(
                row["packet_id_sha256"]
                for row in expectations
                if row["control_id"] == "C19"
            )
            output_rows = self._perfect_output_rows(
                model_manifest_path, expectation_manifest_path
            )
            c19_output = next(
                row for row in output_rows if row["packet_id_sha256"] == c19_packet
            )
            c19_output["result"]["interface_unknown_reason"] = "wrong_but_category_same"
            outputs_path = directory / "model-outputs.jsonl"
            module._write_jsonl_atomic(outputs_path, output_rows)
            lock_path = directory / "locked-outputs.json"
            module.lock_outputs(
                model_input_manifest_path=model_manifest_path,
                output_records_path=outputs_path,
                output_manifest_path=lock_path,
            )
            report = module.score_locked_outputs(
                locked_output_manifest_path=lock_path,
                expectation_manifest_path=expectation_manifest_path,
            )
            self.assertEqual(report["correct"]["completion_category"], 32)
            self.assertEqual(report["correct"]["completion_exact_record"], 31)
            self.assertFalse(report["gates"]["completion_exact_record_accuracy_1"])
            self.assertFalse(report["gates"]["all_56_controls_exact_1"])
            self.assertEqual(report["status"], "failed")

    def test_scoring_rejects_unlocked_or_tampered_outputs(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".controls-v2-lock-", dir=ROOT
        ) as directory_text:
            directory = Path(directory_text)
            model_manifest_path, expectation_manifest_path = self._build(directory)
            outputs_path = directory / "model-outputs.jsonl"
            module._write_jsonl_atomic(
                outputs_path,
                self._perfect_output_rows(model_manifest_path, expectation_manifest_path),
            )
            fake_lock_path = directory / "not-locked.json"
            module._write_json_atomic(
                fake_lock_path,
                {
                    "schema_version": 1,
                    "kind": module.LOCKED_OUTPUT_MANIFEST_KIND,
                    "status": "draft",
                },
            )
            with self.assertRaisesRegex(module.ControlSuiteError, "not locked"):
                module.score_locked_outputs(
                    locked_output_manifest_path=fake_lock_path,
                    expectation_manifest_path=expectation_manifest_path,
                )

            lock_path = directory / "locked.json"
            module.lock_outputs(
                model_input_manifest_path=model_manifest_path,
                output_records_path=outputs_path,
                output_manifest_path=lock_path,
            )
            with outputs_path.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(module.ControlSuiteError, "hash changed"):
                module.score_locked_outputs(
                    locked_output_manifest_path=lock_path,
                    expectation_manifest_path=expectation_manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
