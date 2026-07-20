from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "swe_task_state_v4_counterfactual_pilot.py"
SPEC = importlib.util.spec_from_file_location(
    "swe_task_state_v4_counterfactual_pilot", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def factorial_block(source_id: str = "a" * 64) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    order = 0
    for evidence in module.EVIDENCE_LEVELS:
        for pressure in module.PRESSURE_LEVELS:
            for replica in (0, 1):
                rows.append(
                    {
                        "source_id_sha256": source_id,
                        "condition_id_sha256": f"{order + 1:064x}",
                        "condition_order_within_boundary": order,
                        "evidence_level": evidence,
                        "pressure_level": pressure,
                        "paraphrase_replica": replica,
                    }
                )
                order += 1
    return rows


class CounterfactualPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(module.load_json(module.CONFIG_PATH))
        cls.tokenizer = module.load_tokenizer(cls.config)

    def test_config_is_exact_and_forbids_subjective_claims(self) -> None:
        self.assertEqual(module.sha256_file(module.CONFIG_PATH), module.CONFIG_SHA256)
        claims = self.config["claim_scope"]
        self.assertFalse(claims["factorial_effect_established_by_one_boundary"])
        self.assertFalse(claims["private_chain_of_thought_reconstructed"])
        self.assertFalse(claims["subjective_confidence_or_doubt_inferred"])
        self.assertFalse(claims["experienced_stress_inferred"])
        self.assertFalse(claims["experienced_emotion_inferred"])

    def test_templates_are_exactly_matched_without_padding(self) -> None:
        rendering = self.config["rendering"]
        self.assertFalse(rendering["semantic_padding_tokens_used"])
        for replica in ("0", "1"):
            evidence_counts = {
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in rendering["evidence_text"][replica].values()
            }
            pressure_counts = {
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in rendering["pressure_text"][replica].values()
            }
            self.assertEqual(evidence_counts, {19})
            self.assertEqual(pressure_counts, {26} if replica == "0" else {23})

    def test_rendered_capture_rows_are_blinded_and_only_declared_ranges_differ(self) -> None:
        suffix = self.config["rendering"]["generation_suffix"]
        source_text = "<|im_start|>user\nTest\n<|im_end|>\n" + suffix
        source = {
            "id": "fixture",
            "text": source_text,
            "token_ids": self.tokenizer.encode(source_text, add_special_tokens=False),
        }
        capture, key_rows, summary = module.render_block(
            config=self.config,
            tokenizer=self.tokenizer,
            source=source,
            block=factorial_block(),
        )
        self.assertEqual(len(capture), 12)
        self.assertEqual(len(key_rows), 12)
        self.assertTrue(all(set(row) == {"id", "token_ids"} for row in capture))
        self.assertEqual(summary["evidence_segment_token_count_by_replica"], {"0": 19, "1": 19})
        self.assertEqual(summary["pressure_segment_token_count_by_replica"], {"0": 26, "1": 23})
        self.assertEqual(len({row["prompt_id"] for row in key_rows}), 12)
        self.assertNotIn("emotion", capture[0])
        self.assertNotIn("evidence_level", capture[0])

    def test_pairwise_validator_rejects_a_difference_outside_manipulation_ranges(self) -> None:
        suffix = self.config["rendering"]["generation_suffix"]
        source_text = "<|im_start|>user\nTest\n<|im_end|>\n" + suffix
        source = {
            "id": "fixture",
            "text": source_text,
            "token_ids": self.tokenizer.encode(source_text, add_special_tokens=False),
        }
        capture, key_rows, _summary = module.render_block(
            config=self.config,
            tokenizer=self.tokenizer,
            source=source,
            block=factorial_block(),
        )
        by_prompt = {row["id"]: list(row["token_ids"]) for row in capture}
        victim = str(key_rows[1]["prompt_id"])
        by_prompt[victim][0] += 1
        with self.assertRaisesRegex(module.PilotError, "escapes declared"):
            module.validate_pairwise_matching(key_rows, by_prompt)

    def test_forbidden_paths_fail_before_access(self) -> None:
        with self.assertRaisesRegex(module.PilotError, "before filesystem access"):
            module.lexical_path_preflight((Path("/tmp/reserved-pilot/input.json"),))


if __name__ == "__main__":
    unittest.main()
