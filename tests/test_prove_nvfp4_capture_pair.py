from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prove_nvfp4_capture_pair.py"
SPEC = importlib.util.spec_from_file_location("prove_nvfp4_capture_pair", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

try:
    import torch
except ModuleNotFoundError:
    torch = None


def descriptor(content_hash: str) -> dict[str, object]:
    return {
        "shape": [2],
        "stride": [1],
        "dtype": "torch.int32",
        "numel": 2,
        "logical_bytes": 8,
        "logical_row_major_sha256": content_hash,
    }


class ManifestComparisonTests(unittest.TestCase):
    def test_exact_shared_and_replay_manifests_pass(self) -> None:
        shared = descriptor("a" * 64)
        replay = descriptor("b" * 64)
        baseline = {
            "tensors": {
                "gdn.layer0.q": shared,
                "replay.weight": replay,
            }
        }
        observer = {
            "tensors": {
                "gdn.layer0.q": dict(shared),
                "h0_post_block": descriptor("c" * 64),
                "replay.weight": dict(replay),
            }
        }

        internals, parameters = MODULE._compare_manifests(
            baseline,
            observer,
            expected_shared_non_replay=1,
            expected_observer_only_non_replay=1,
            expected_replay_parameters=1,
        )

        self.assertTrue(internals["all_shared_bit_exact"])
        self.assertEqual(internals["shared_tensor_count"], 1)
        self.assertTrue(parameters["all_content_hashes_equal"])

    def test_content_hash_mismatch_fails_bit_exact_gate(self) -> None:
        baseline = {"tensors": {"attention.layer3.q": descriptor("a" * 64)}}
        observer = {"tensors": {"attention.layer3.q": descriptor("b" * 64)}}
        with self.assertRaisesRegex(MODULE.VerificationError, "not bit-exact"):
            MODULE._compare_manifests(
                baseline,
                observer,
                expected_shared_non_replay=1,
                expected_observer_only_non_replay=0,
                expected_replay_parameters=0,
            )

    def test_categories_distinguish_observer_outputs(self) -> None:
        self.assertEqual(MODULE._category("linear.layers.1.mlp.down.output"), "compiled_linear_output")
        self.assertEqual(MODULE._category("h1_post_block"), "post_block_h")
        self.assertEqual(MODULE._category("layers.1.mlp.swiglu_output"), "swiglu_output")


class MetadataContractTests(unittest.TestCase):
    def metadata_pair(self) -> tuple[dict[str, object], dict[str, object]]:
        generation = {
            "prompt_token_ids": [1],
            "generated_token_id": 2,
            "generated_text": "x",
            "finish_reason": "length",
            "top_logprobs": [
                {
                    "token_id": 2,
                    "logprob": -0.25,
                    "rank": 1,
                    "decoded_token": "x",
                }
            ],
        }
        common = {
            "schema_version": 1,
            "model": {"repo_id": "model", "revision": "rev"},
            "prompt": {"text": "p", "token_ids": [1], "token_count": 1},
        }
        baseline = {
            **common,
            "runtime": {
                "mode": "compiled",
                "target_profile": "all",
                "target_layers": list(range(64)),
                "mtp_enabled": False,
                "language_model_only": True,
                "compiled_observer": False,
            },
            "authority": {"compiled_endpoint": "uninstrumented"},
            "authoritative_compiled_generation": generation,
            "instrumented_generation": generation,
            "capture": {},
        }
        observer = {
            **common,
            "status": "captured",
            "runtime": {
                "mode": "compiled-observer",
                "target_profile": "all",
                "target_layers": list(range(64)),
                "mtp_enabled": False,
                "language_model_only": True,
                "compiled_observer": True,
            },
            "authority": {
                "detailed_tensor_capture_is_unmodified_serving_graph": False
            },
            "instrumented_generation": generation,
            "capture": {"missing_required": [], "truncated": []},
        }
        return baseline, observer

    def test_full_generation_record_equality_is_required(self) -> None:
        baseline, observer = self.metadata_pair()
        result = MODULE._validate_json_contract(baseline, observer)
        self.assertEqual(result["record"]["generated_token_id"], 2)

        observer["instrumented_generation"] = {
            **observer["instrumented_generation"],
            "finish_reason": "stop",
        }
        with self.assertRaisesRegex(MODULE.VerificationError, "full generation"):
            MODULE._validate_json_contract(baseline, observer)

    def test_required_missing_observer_tensor_fails(self) -> None:
        baseline, observer = self.metadata_pair()
        observer["capture"]["missing_required"] = ["h0_post_block"]
        with self.assertRaisesRegex(MODULE.VerificationError, "required missing"):
            MODULE._validate_json_contract(baseline, observer)


@unittest.skipIf(torch is None, "Torch is required for PT integration")
class TinyPairIntegrationTests(unittest.TestCase):
    def tensor_summary(self, tensor) -> dict[str, object]:
        record = MODULE._tensor_descriptor(tensor, chunk_bytes=1024)
        return {
            "shape": record["shape"],
            "dtype": record["dtype"],
            "sha256": record["logical_row_major_sha256"],
        }

    def write_pair(self, root: Path) -> tuple[Path, Path, Path, Path]:
        generation = {
            "prompt_token_ids": [1],
            "generated_token_id": 2,
            "generated_text": "x",
            "finish_reason": "length",
            "top_logprobs": [
                {
                    "token_id": 2,
                    "logprob": -0.25,
                    "rank": 1,
                    "decoded_token": "x",
                }
            ],
        }
        shared = {
            "gdn.layer0.q": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "attention.layer3.v": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        }
        replay = {
            "replay.weight": torch.tensor([5, 6], dtype=torch.int32),
            "replay.scale": torch.tensor(0.5, dtype=torch.float32),
        }
        observer_only = {
            "h0_post_block": torch.tensor([7.0, 8.0], dtype=torch.bfloat16)
        }
        common_payload = {
            "schema_version": 1,
            "model_revision": "rev",
            "prompt": "p",
            "prompt_token_ids": [1],
            "target_profile": "all",
            "target_layers": list(range(64)),
        }
        baseline_pt = root / "baseline.pt"
        observer_pt = root / "observer.pt"
        torch.save(
            {**common_payload, "mode": "compiled", "tensors": {**shared, **replay}},
            baseline_pt,
        )
        torch.save(
            {
                **common_payload,
                "mode": "compiled-observer",
                "tensors": {**shared, **observer_only, **replay},
            },
            observer_pt,
        )

        provenance = {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "role": "test",
            }
            for name, tensor in replay.items()
        }
        common_json = {
            "schema_version": 1,
            "model": {"repo_id": "model", "revision": "rev"},
            "prompt": {"text": "p", "token_ids": [1], "token_count": 1},
        }
        baseline_json_value = {
            **common_json,
            "status": "incomplete",
            "runtime": {
                "mode": "compiled",
                "target_profile": "all",
                "target_layers": list(range(64)),
                "mtp_enabled": False,
                "language_model_only": True,
                "compiled_observer": False,
            },
            "authority": {"compiled_endpoint": "uninstrumented"},
            "authoritative_compiled_generation": generation,
            "instrumented_generation": generation,
            "capture": {
                "tensor_summaries": {
                    name: self.tensor_summary(tensor) for name, tensor in shared.items()
                },
                "replay_parameter_provenance": provenance,
            },
        }
        observer_json_value = {
            **common_json,
            "status": "captured",
            "runtime": {
                "mode": "compiled-observer",
                "target_profile": "all",
                "target_layers": list(range(64)),
                "mtp_enabled": False,
                "language_model_only": True,
                "compiled_observer": True,
            },
            "authority": {
                "detailed_tensor_capture_is_unmodified_serving_graph": False
            },
            "instrumented_generation": generation,
            "capture": {
                "missing": [],
                "missing_required": [],
                "truncated": [],
                "compile_visible_observer_tensors": list(observer_only),
                "tensor_summaries": {
                    name: self.tensor_summary(tensor)
                    for name, tensor in {**shared, **observer_only}.items()
                },
                "replay_parameter_provenance": provenance,
            },
        }
        baseline_json = root / "baseline.json"
        observer_json = root / "observer.json"
        baseline_json.write_text(json.dumps(baseline_json_value))
        observer_json.write_text(json.dumps(observer_json_value))
        return baseline_json, baseline_pt, observer_json, observer_pt

    def test_tiny_pair_is_bound_and_proved_in_sequential_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.write_pair(Path(temp))
            proof = MODULE._build_proof(
                baseline_json_path=paths[0],
                baseline_tensors_path=paths[1],
                observer_json_path=paths[2],
                observer_tensors_path=paths[3],
                expected_shared_non_replay=2,
                expected_observer_only_non_replay=1,
                expected_replay_parameters=2,
                chunk_mib=1,
            )

        self.assertEqual(proof["status"], "passed")
        self.assertTrue(proof["generation_record_parity"]["exact"])
        self.assertEqual(proof["shared_internal_tensor_parity"]["shared_tensor_count"], 2)
        self.assertEqual(proof["replay_parameter_parity"]["parameter_count"], 2)
        self.assertEqual(
            proof["memory_bounded_verification"]["maximum_concurrent_tensor_payloads"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
