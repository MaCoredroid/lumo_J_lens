from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts import swe_task_state_v4_epistemic_chain_control_executor_v2 as executor


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "swe_task_state_v4_epistemic_chain_control_executor_v2.json"
)
CONFIG_SHA256 = "de07fb91c37734b0651f31403734f4111d169a10ca2807a9eecfaefbf981d7ca"


class SemanticControlExecutorV2FrozenConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_config_is_exact_freeze_helper_output(self) -> None:
        self.assertEqual(executor.sha256_file(CONFIG_PATH), CONFIG_SHA256)
        executor.validate_execution_config(self.config)
        expected = executor.freeze_execution_config(
            runtime_config_path=(
                ROOT / self.config["bindings"]["runtime_config"]["path"]
            ),
            model_input_manifest_path=(
                ROOT / self.config["bindings"]["sealed_model_inputs"]["path"]
            ),
        )
        self.assertEqual(
            executor.canonical_json_bytes(self.config),
            executor.canonical_json_bytes(expected),
        )

    def test_generation_context_authenticates_without_expectation_access(self) -> None:
        context = executor._authenticate_execution_context(
            execution_config_path=CONFIG_PATH,
            expected_execution_config_sha256=CONFIG_SHA256,
        )
        self.assertEqual(len(context["model_input_rows"]), 52)
        self.assertEqual(
            context["runtime_config"]["roles"]["independent_a"][
                "base_model_lineage"
            ],
            "qwen3.6",
        )
        self.assertEqual(
            context["runtime_config"]["roles"]["independent_b"][
                "base_model_lineage"
            ],
            "gpt-oss",
        )
        self.assertEqual(
            context["runtime_config"]["roles"]["adjudicator"][
                "base_model_lineage"
            ],
            "mistral-small-3.1",
        )

    def test_all_implementation_and_decoder_bindings_authenticate(self) -> None:
        bindings = self.config["bindings"]
        for binding in bindings["implementations"].values():
            self.assertEqual(
                executor.sha256_file(ROOT / binding["path"]), binding["sha256"]
            )
        decoder = bindings["decoder_v2_addendum"]
        decoder_path = ROOT / decoder["path"]
        decoder_value = json.loads(decoder_path.read_text(encoding="utf-8"))
        self.assertEqual(executor.sha256_file(decoder_path), decoder["sha256"])
        self.assertEqual(
            executor.sha256_bytes(executor.canonical_json_bytes(decoder_value)),
            decoder["canonical_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
