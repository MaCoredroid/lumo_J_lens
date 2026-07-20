from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts import swe_task_state_v4_epistemic_chain_annotation_runner_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_annotation_runner_v2.json"
)
CONFIG_SHA256 = "a22604d78938917972dc23c517c942fb9022e867e17ce0b5a060189dd0dd1b4d"


class QuoteFirstRunnerV2ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_is_frozen_and_runner_accepts_it(self) -> None:
        self.assertEqual(runner.legacy.sha256_file(CONFIG_PATH), CONFIG_SHA256)
        runner.validate_v2_config(self.config)
        codebook_path = ROOT / self.config["inputs"]["annotation_codebook"]["path"]
        runner.authenticate_passed_contracts(
            config=self.config,
            config_path=CONFIG_PATH,
            codebook=json.loads(codebook_path.read_text(encoding="utf-8")),
            codebook_path=codebook_path,
        )

    def test_all_bound_repository_files_authenticate(self) -> None:
        bindings = [
            self.config["implementation"]["quote_runner"],
            self.config["implementation"]["quote_runner_tests"],
            self.config["inputs"]["annotation_codebook"],
            self.config["inputs"]["sealed_control_model_inputs"],
            self.config["inputs"]["sealed_control_expectations"],
        ]
        for binding in bindings:
            path = ROOT / binding["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(runner.legacy.sha256_file(path), binding["sha256"])

    def test_roster_is_exactly_three_distinct_authenticated_families(self) -> None:
        roles = self.config["roles"]
        self.assertEqual(
            runner.validate_distinct_base_model_lineages(roles),
            {
                "independent_a": "qwen3.6",
                "independent_b": "gpt-oss",
                "adjudicator": "mistral-small-3.1",
            },
        )
        self.assertEqual(
            {role: spec["snapshot_tree_sha256"] for role, spec in roles.items()},
            {
                "independent_a": "9e81d31df546344ad68696c3cfd6cadce4ad6d3952710a3dc2021c1c2d42414d",
                "independent_b": "60b907629e6da0ac8239a1103c366894c97bd7054f899c4e6e1f6833c84bad18",
                "adjudicator": "16bed3903625052c1d2871a8135254acbe731169082f76d910b4b1ddfdf04241",
            },
        )
        for role, spec in roles.items():
            self.assertEqual(
                spec["vllm_engine_kwargs"],
                runner.expected_role_vllm_engine_kwargs(spec),
                role,
            )

    def test_generation_and_claim_boundaries_are_fail_closed(self) -> None:
        generation = self.config["generation"]
        self.assertEqual(
            generation["structured_outputs_config"],
            runner.STRUCTURED_OUTPUTS_ENGINE_CONFIG,
        )
        self.assertEqual(
            generation["prompt_token_accounting"],
            "vllm_request_output.prompt_token_ids",
        )
        self.assertEqual(generation["max_model_len"], 90112)
        self.assertTrue(generation["no_input_truncation"])
        self.assertIsNone(generation["cuda_home_override"])
        self.assertEqual(generation["vllm_use_flashinfer_sampler"], "0")
        self.assertTrue(
            self.config["readiness_gate"][
                "target_packet_selection_or_generation_forbidden_until_controls_pass"
            ]
        )
        claims = self.config["claim_scope"]
        self.assertFalse(claims["private_chain_of_thought_recovery_established"])
        self.assertFalse(claims["latent_sentence_or_concept_chain_recovery_established"])
        self.assertFalse(
            claims["emotion_affect_confidence_doubt_or_stress_recovery_established"]
        )

    def test_sealed_model_manifest_remains_expectation_free(self) -> None:
        model_path = ROOT / self.config["inputs"]["sealed_control_model_inputs"]["path"]
        expectation_path = (
            ROOT / self.config["inputs"]["sealed_control_expectations"]["path"]
        )
        self.assertNotEqual(model_path.resolve(), expectation_path.resolve())
        text = model_path.read_text(encoding="utf-8").casefold()
        for forbidden in ("expectation", "expected", "gold", "label", "reason"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
