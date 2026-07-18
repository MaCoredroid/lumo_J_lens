#!/usr/bin/env python3
"""Focused tests for the SWE multistage publication verifier."""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/check_swe_multistage_publication.py"
SPEC = importlib.util.spec_from_file_location("check_swe_multistage_publication", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicationVerifierTests(unittest.TestCase):
    def test_real_publication_bundle_passes(self) -> None:
        result = MODULE.verify_publication()
        self.assertEqual(result["evidence_rows"], 45)
        self.assertEqual(result["legacy_artifacts"], 8)
        self.assertEqual(result["prompts"], 8)

    def test_sidecar_is_exact_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "manifest.json"
            target.write_bytes(b"{}\n")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            sidecar = root / "manifest.sha256"
            sidecar.write_text(f"{digest}  manifest.json\n", encoding="ascii")
            MODULE.verify_sidecar(sidecar, "manifest.json", digest)
            sidecar.write_text(f"{digest} manifest.json\n", encoding="ascii")
            with self.assertRaisesRegex(MODULE.PublicationError, "sidecar grammar"):
                MODULE.verify_sidecar(sidecar, "manifest.json", digest)

    def test_duplicate_json_keys_and_nonfinite_numbers_fail(self) -> None:
        for value in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.subTest(value=value):
                with self.assertRaises(MODULE.PublicationError):
                    MODULE.strict_json_bytes(value, "test")

    def test_canonical_paths_reject_escape_and_backslashes(self) -> None:
        for value in ("../escape", "/absolute", "a/./b", "a//b", "a\\b"):
            with self.subTest(value=value):
                with self.assertRaises(MODULE.PublicationError):
                    MODULE.canonical_relative(value, "test path")

    def test_gzip_rejects_trailing_member_or_garbage(self) -> None:
        value = gzip.compress(b'{}\n', mtime=0)
        self.assertEqual(MODULE.decompress_one_gzip(value, "test"), b'{}\n')
        for changed in (value + b"garbage", value + value, value[:-2]):
            with self.subTest(length=len(changed)):
                with self.assertRaises(MODULE.PublicationError):
                    MODULE.decompress_one_gzip(changed, "test")

    def test_ledger_rejects_duplicate_and_unsafe_rows(self) -> None:
        digest = "a" * 64
        good = [f"{digest}  validation/file-{index}\n" for index in range(45)]
        self.assertEqual(len(MODULE.parse_ledger("".join(good).encode("ascii"))), 45)
        duplicate = good[:-1] + [good[0]]
        with self.assertRaisesRegex(MODULE.PublicationError, "duplicate"):
            MODULE.parse_ledger("".join(duplicate).encode("ascii"))
        unsafe = list(good)
        unsafe[0] = f"{digest}  ../escape\n"
        with self.assertRaisesRegex(MODULE.PublicationError, "unsafe"):
            MODULE.parse_ledger("".join(unsafe).encode("ascii"))

    def test_semantic_contract_rejects_hidden_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [f"swe-s{index}-000-django__django-13297" for index in range(8)]
            stages = [f"S{index}" for index in range(8)]
            summary = {
                "prompt_count": 8,
                "hidden_prompt_count": 0,
                "explicit_control_prompt_count": 8,
                "prompts": [
                    {"id": prompt_id, "stage_id": stage}
                    for prompt_id, stage in zip(ids, stages, strict=True)
                ],
            }
            prompts = [
                {
                    "id": ids[index],
                    "text": f"prompt {index}",
                    "token_ids": [index],
                    "score_token_ids": [100 + index],
                    "metadata": {
                        "analysis_role": "explicit_contaminated_control",
                        "stage": {"id": stages[index]},
                        "visibility_audit": {
                            "records": [{"subject": "target", "exposed": True}]
                        },
                    }
                }
                for index in range(8)
            ]
            action_protocol = {"test": True}

            def write(name: str, value: object) -> Path:
                path = root / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                return path

            paths = {
                "prompts_summary": write("prompts_summary", summary),
                "prompts": write("prompts", prompts),
                "action_protocol": write("action_protocol", action_protocol),
            }
            source_sha = MODULE.sha256_file(paths["prompts"])
            protocol_sha = MODULE.sha256_file(paths["action_protocol"])
            action_prompts = []
            for prompt in prompts:
                action = copy.deepcopy(prompt)
                action["metadata"]["stage_action_probe"] = {
                    "source_prompt_bundle_sha256": source_sha,
                    "action_protocol_sha256": protocol_sha,
                    "exact_prompt_text_preserved": True,
                    "exact_prompt_token_ids_preserved": True,
                }
                action_prompts.append(action)
            paths["action_prompts"] = write("action_prompts", action_prompts)
            action_sha = MODULE.sha256_file(paths["action_prompts"])
            action_summary = {
                "schema_version": 1,
                "kind": "swe_verified_stage_action_probe_materialization",
                "source_prompt_bundle_sha256": source_sha,
                "prompt_bundle_sha256": action_sha,
                "action_protocol_sha256": protocol_sha,
                "prompt_count": 8,
                "available_action_label_count": 8,
                "missing_action_label_count": 0,
                "action_class_counts": {
                    "inspect": 5,
                    "edit": 1,
                    "validate": 1,
                    "finalize": 1,
                },
                "prompts": [
                    {"id": prompt_id, "stage_id": stage}
                    for prompt_id, stage in zip(ids, stages, strict=True)
                ],
            }
            paths["action_prompts_summary"] = write(
                "action_prompts_summary", action_summary
            )
            analysis = {
                "gold_probe_status": "no_hidden_gold_eligible_prompts",
                "source_prompt_bundle_sha256": source_sha,
                "augmented_prompt_bundle_sha256": action_sha,
                "action_protocol_sha256": protocol_sha,
                "interpretation_contract": {
                    "hidden_gold_eligible_prompt_count": 0,
                    "explicit_control_prompt_count": 8,
                },
                "prompt_details": [
                    {"id": prompt_id, "stage_id": stage}
                    for prompt_id, stage in zip(ids, stages, strict=True)
                ],
            }
            paths["analysis"] = write("analysis", analysis)
            for name in ("public_report", "nf4_report", "native_report"):
                lens_expected = MODULE.EXPECTED_REPORT_LENSES[name]
                report = {
                    "schema_version": 3,
                    "status": "failed",
                    "model": {
                        **MODULE.EXPECTED_RUNTIME["model"],
                        "quant_method": "modelopt",
                        "quant_algo": "MIXED_PRECISION",
                    },
                    "runtime": {
                        "enforce_eager": True,
                        "language_model_only": True,
                        "mtp_enabled": False,
                        "capture_adapter": "vLLM apply_model forward hooks",
                        "enable_prefix_caching": True,
                        "gpu_memory_utilization": 0.78,
                        "kv_cache_dtype": "fp8_e4m3",
                        "kv_offloading_backend": "native",
                        "kv_offloading_size": 8.0,
                        "mamba_block_size": 1024,
                        "max_model_len": 49152,
                        "max_num_batched_tokens": 4096,
                        "stream_final_only": True,
                        "transport_dtype": "torch.float32",
                        "readout_dtype": "torch.bfloat16",
                    },
                    "lens": {
                        **lens_expected,
                        "d_model": 5120,
                        "source_layers": list(range(63)),
                        "tensor_shape": [5120, 5120],
                    },
                    "experiments": [
                        {
                            "id": action["id"],
                            "prompt": action["text"],
                            "prompt_token_ids": action["token_ids"],
                            "metadata": action["metadata"],
                            "scored_vocabulary": {
                                "token_ids": action["score_token_ids"]
                            },
                        }
                        for action in action_prompts
                    ],
                }
                paths[name] = write(name, report)
            MODULE.verify_semantic_contract(paths)

            action_prompts[0]["metadata"]["analysis_role"] = "oracle_hidden"
            paths["action_prompts"].write_text(
                json.dumps(action_prompts), encoding="utf-8"
            )
            with self.assertRaisesRegex(MODULE.PublicationError, "inherited source"):
                MODULE.verify_semantic_contract(paths)
            action_prompts[0]["metadata"][
                "analysis_role"
            ] = "explicit_contaminated_control"
            paths["action_prompts"].write_text(
                json.dumps(action_prompts), encoding="utf-8"
            )

            public = json.loads(paths["public_report"].read_text(encoding="utf-8"))
            public["runtime"]["mtp_enabled"] = True
            paths["public_report"].write_text(json.dumps(public), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PublicationError, "runtime identity"):
                MODULE.verify_semantic_contract(paths)
            public["runtime"]["mtp_enabled"] = False
            paths["public_report"].write_text(json.dumps(public), encoding="utf-8")

            prompts[0]["metadata"]["analysis_role"] = "oracle_hidden"
            paths["prompts"].write_text(json.dumps(prompts), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PublicationError, "not an explicit"):
                MODULE.verify_semantic_contract(paths)

            prompts[0]["metadata"]["analysis_role"] = "explicit_contaminated_control"
            paths["prompts"].write_text(json.dumps(prompts), encoding="utf-8")
            prompts[1]["id"] = prompts[0]["id"]
            prompts[1]["metadata"]["stage"]["id"] = "S0"
            paths["prompts"].write_text(json.dumps(prompts), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PublicationError, "IDs/order"):
                MODULE.verify_semantic_contract(paths)

            prompts[1]["id"] = ids[1]
            prompts[1]["metadata"]["stage"]["id"] = stages[1]
            paths["prompts"].write_text(json.dumps(prompts), encoding="utf-8")
            analysis["source_prompt_bundle_sha256"] = "0" * 64
            paths["analysis"].write_text(json.dumps(analysis), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PublicationError, "analysis lineage"):
                MODULE.verify_semantic_contract(paths)


if __name__ == "__main__":
    unittest.main()
