import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_endpoint import validate_models_payload  # noqa: E402


class EndpointContractTests(unittest.TestCase):
    def test_exact_contract_passes(self):
        payload = {"data": [{"id": "qwen", "max_model_len": 32768}]}
        self.assertTrue(validate_models_payload(payload, "qwen", 32768)[0])

    def test_wrong_model_fails(self):
        payload = {"data": [{"id": "other", "max_model_len": 32768}]}
        self.assertFalse(validate_models_payload(payload, "qwen", 32768)[0])

    def test_wrong_context_fails(self):
        payload = {"data": [{"id": "qwen", "max_model_len": 8192}]}
        self.assertFalse(validate_models_payload(payload, "qwen", 32768)[0])

    def test_malformed_payload_fails(self):
        self.assertFalse(validate_models_payload({"data": "wrong"}, "qwen", 32768)[0])
