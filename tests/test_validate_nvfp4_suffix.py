#!/usr/bin/env python3
"""Focused tests for the offline NVFP4 late-suffix validator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nvfp4_attention as ATTENTION  # noqa: E402
import nvfp4_gdn as GDN  # noqa: E402
import nvfp4_ste as STE  # noqa: E402
import validate_nvfp4_suffix as VALIDATOR  # noqa: E402


@dataclass(frozen=True)
class FakeRawWeight:
    value: torch.Tensor

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.value.to(dtype)


class FakeCheckpoint:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values
        self.requested: list[str] = []

    def load_nvfp4(self, module_name: str) -> FakeRawWeight:
        self.requested.append(module_name)
        return FakeRawWeight(self.values[module_name])


class SyntheticSuffix:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(1107)
        self.tokens = 4
        self.config = ATTENTION.QwenFullAttentionConfig(
            hidden_size=8,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=4,
            rotary_dim=4,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            mrope_section=(1, 1, 0),
        )
        self.layout = GDN.GdnLayout(
            key_heads=1,
            value_heads=2,
            key_dim=2,
            value_dim=3,
        )
        hidden = self.config.hidden_size
        intermediate = 5
        q_size = self.layout.key_heads * self.layout.key_dim
        value_size = self.layout.value_heads * self.layout.value_dim
        qkv_size = 2 * q_size + value_size

        def fp8_weight(out_features: int, in_features: int):
            postload = (
                torch.randn(in_features, out_features, generator=generator) * 0.15
            ).to(torch.float8_e4m3fn)
            return postload, postload.T.float()

        qkvz_runtime, qkvz = fp8_weight(qkv_size + value_size, hidden)
        gdn_out_runtime, gdn_out = fp8_weight(hidden, value_size)
        qkv_runtime, qkv = fp8_weight(self.config.qkv_projection_size, hidden)
        attention_out_runtime, attention_out = fp8_weight(
            hidden, self.config.query_size
        )
        ba = torch.randn(2 * self.layout.value_heads, hidden, generator=generator) * 0.1
        conv = torch.randn(qkv_size, 3, generator=generator) * 0.1
        a_log = torch.randn(self.layout.value_heads, generator=generator) * 0.1
        dt_bias = torch.randn(self.layout.value_heads, generator=generator) * 0.1
        gdn_norm = torch.randn(self.layout.value_dim, generator=generator) * 0.1
        l62_gate_up = torch.randn(2 * intermediate, hidden, generator=generator) * 0.1
        l62_down = torch.randn(hidden, intermediate, generator=generator) * 0.1
        l63_gate_up = torch.randn(2 * intermediate, hidden, generator=generator) * 0.1
        l63_down = torch.randn(hidden, intermediate, generator=generator) * 0.1
        norm_weights = {
            (layer, name): torch.randn(hidden, generator=generator) * 0.05
            for layer in (62, 63)
            for name in ("input_layernorm", "post_attention_layernorm")
        }
        q_norm = torch.randn(self.config.head_dim, generator=generator) * 0.05
        k_norm = torch.randn(self.config.head_dim, generator=generator) * 0.05

        h61 = torch.randn(self.tokens, hidden, generator=generator) * 0.2
        l62_attention_input = ATTENTION.qwen_rms_norm(
            h61, norm_weights[(62, "input_layernorm")], 1e-6
        )
        qkvz_output = l62_attention_input @ qkvz.T
        ba_output = l62_attention_input @ ba.T
        mixed, z_flat = qkvz_output.split((qkv_size, value_size), dim=-1)
        conv_output = STE.causal_depthwise_conv1d_silu(mixed, conv)
        q_raw, k_raw, value_raw = conv_output.split(
            (q_size, q_size, value_size), dim=-1
        )
        query = STE.l2_normalize(
            q_raw.reshape(self.tokens, self.layout.key_heads, self.layout.key_dim)
        )
        key = STE.l2_normalize(
            k_raw.reshape(self.tokens, self.layout.key_heads, self.layout.key_dim)
        )
        value = value_raw.reshape(
            self.tokens, self.layout.value_heads, self.layout.value_dim
        )
        b, a = ba_output.chunk(2, dim=-1)
        log_decay = STE.gdn_log_decay(a, a_log, dt_bias)
        beta = torch.sigmoid(b.float())
        initial_state = torch.zeros(
            self.layout.value_heads, self.layout.value_dim, self.layout.key_dim
        )
        recurrence = STE.gdn_reference_forward(
            query, key, value, log_decay, beta, initial_state
        )
        z = z_flat.reshape(
            self.tokens, self.layout.value_heads, self.layout.value_dim
        )
        gated_norm = STE.gated_rms_norm(
            recurrence.output, z, gdn_norm, self.layout.norm_eps
        )
        gdn_out_input = gated_norm.flatten(-2)
        gdn_branch = gdn_out_input @ gdn_out.T
        l62_after_attention = h61 + gdn_branch
        l62_mlp_input = ATTENTION.qwen_rms_norm(
            l62_after_attention,
            norm_weights[(62, "post_attention_layernorm")],
            1e-6,
        )
        l62_gate_up_output = l62_mlp_input @ l62_gate_up.T
        l62_swiglu = ATTENTION.qwen_swiglu(l62_gate_up_output)
        l62_mlp_output = l62_swiglu @ l62_down.T
        h62 = l62_after_attention + l62_mlp_output

        l63_linears = ATTENTION.QwenBlockLinears(
            qkv=lambda inputs: inputs @ qkv.T,
            attention_out=lambda inputs: inputs @ attention_out.T,
            gate_up=lambda inputs: inputs @ l63_gate_up.T,
            down=lambda inputs: inputs @ l63_down.T,
        )
        l63 = ATTENTION.replay_qwen_full_attention_suffix(
            h62,
            torch.arange(self.tokens),
            self.config,
            input_norm_weight=norm_weights[(63, "input_layernorm")],
            post_attention_norm_weight=norm_weights[
                (63, "post_attention_layernorm")
            ],
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            linears=l63_linears,
        )
        h63 = l63.output

        tensors: dict[str, torch.Tensor] = {
            "h61_post_block": h61,
            "h62_post_block": h62,
            "h63_post_block": h63,
            "linear.layers.62.linear_attn.in_proj_qkvz.input": l62_attention_input,
            "linear.layers.62.linear_attn.in_proj_qkvz.output": qkvz_output,
            "linear.layers.62.linear_attn.in_proj_ba.input": l62_attention_input,
            "linear.layers.62.linear_attn.in_proj_ba.output": ba_output,
            "linear.layers.62.linear_attn.out_proj.input": gdn_out_input,
            "linear.layers.62.linear_attn.out_proj.output": gdn_branch,
            "linear.layers.62.mlp.gate_up_proj.input": l62_mlp_input,
            "linear.layers.62.mlp.gate_up_proj.output": l62_gate_up_output,
            "linear.layers.62.mlp.down_proj.input": l62_swiglu,
            "linear.layers.62.mlp.down_proj.output": l62_mlp_output,
            "layers.62.mlp.swiglu_output": l62_swiglu,
            "gdn.layer62.conv_output_prefill": conv_output,
            "gdn.layer62.q": query.unsqueeze(0),
            "gdn.layer62.k": key.unsqueeze(0),
            "gdn.layer62.v": value.unsqueeze(0),
            "gdn.layer62.log_g": log_decay.unsqueeze(0),
            "gdn.layer62.beta": beta.unsqueeze(0),
            "gdn.layer62.initial_state": initial_state.unsqueeze(0),
            "gdn.layer62.chunk_output": recurrence.output.unsqueeze(0),
            "gdn.layer62.final_state": recurrence.final_state.unsqueeze(0),
            "gdn.layer62.core_output": recurrence.output,
            "gdn.layer62.norm_core_input": recurrence.output.flatten(0, 1),
            "gdn.layer62.norm_z_input": z.flatten(0, 1),
            "gdn.layer62.norm_output": gated_norm.flatten(0, 1),
            "linear.layers.63.self_attn.qkv_proj.input": l63.attention_input,
            "linear.layers.63.self_attn.qkv_proj.output": l63.attention.qkv,
            "linear.layers.63.self_attn.o_proj.input": l63.attention.gated_output,
            "linear.layers.63.self_attn.o_proj.output": l63.attention.output,
            "linear.layers.63.mlp.gate_up_proj.input": l63.mlp_input,
            "linear.layers.63.mlp.gate_up_proj.output": l63.gate_up,
            "linear.layers.63.mlp.down_proj.input": l63.activated,
            "linear.layers.63.mlp.down_proj.output": l63.hidden_states,
            "layers.63.mlp.swiglu_output": l63.activated,
            "attention.layer63.q_post_rope": l63.attention.query.flatten(1),
            "attention.layer63.k_post_rope": l63.attention.key.flatten(1),
            "attention.layer63.v": l63.attention.value.flatten(1),
            "attention.layer63.core_output": l63.attention.core_output.flatten(1),
            "replay.gdn.layers.62.in_proj_ba.weight": ba,
            "replay.gdn.layers.62.conv1d.weight": conv.unsqueeze(1),
            "replay.gdn.layers.62.A_log": a_log,
            "replay.gdn.layers.62.dt_bias": dt_bias,
            "replay.gdn.layers.62.norm.weight": gdn_norm,
            "replay.norm.layers.63.self_attn.q_norm.weight": q_norm,
            "replay.norm.layers.63.self_attn.k_norm.weight": k_norm,
        }
        for layer in (62, 63):
            for name in ("input_layernorm", "post_attention_layernorm"):
                tensors[f"replay.norm.layers.{layer}.{name}.weight"] = norm_weights[
                    (layer, name)
                ]
        for name, runtime in (
            ("layers.62.linear_attn.in_proj_qkvz", qkvz_runtime),
            ("layers.62.linear_attn.out_proj", gdn_out_runtime),
            ("layers.63.self_attn.qkv_proj", qkv_runtime),
            ("layers.63.self_attn.o_proj", attention_out_runtime),
        ):
            prefix = f"replay.fp8.{name}"
            tensors[f"{prefix}.weight"] = runtime
            tensors[f"{prefix}.weight_scale"] = torch.tensor(1.0)
            tensors[f"{prefix}.input_scale"] = torch.tensor(1.0)

        self.payload = {
            "schema_version": 1,
            "mode": "eager",
            "model_revision": VALIDATOR.MODEL_REVISION,
            "prompt": "synthetic",
            "prompt_token_ids": list(range(self.tokens)),
            "tensors": tensors,
        }
        prefix = VALIDATOR.CHECKPOINT_PREFIX
        self.checkpoint = FakeCheckpoint(
            {
                f"{prefix}.62.mlp.gate_up_proj": l62_gate_up,
                f"{prefix}.62.mlp.down_proj": l62_down,
                f"{prefix}.63.mlp.gate_up_proj": l63_gate_up,
                f"{prefix}.63.mlp.down_proj": l63_down,
            }
        )


class PayloadValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = SyntheticSuffix()

    def test_missing_runtime_boundary_is_rejected(self) -> None:
        payload = {**self.case.payload, "tensors": dict(self.case.payload["tensors"])}
        del payload["tensors"]["attention.layer63.q_post_rope"]
        with self.assertRaisesRegex(ValueError, "missing 1 required tensors"):
            VALIDATOR.validate_payload_schema(
                payload,
                attention_config=self.case.config,
                gdn_layout=self.case.layout,
            )

    def test_tensor_payload_round_trip_uses_weights_only_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.pt"
            torch.save(self.case.payload, path)
            loaded = VALIDATOR.load_capture_payload(path)
        self.assertEqual(loaded["prompt_token_ids"], list(range(self.case.tokens)))
        self.assertTrue(torch.equal(
            loaded["tensors"]["h63_post_block"],
            self.case.payload["tensors"]["h63_post_block"],
        ))


class VjpEstimatorTest(unittest.TestCase):
    def test_compiled_residual_accumulates_before_bfloat16_cast(self) -> None:
        hidden = torch.tensor([[100.0]], dtype=torch.bfloat16)
        attention = torch.tensor([[0.25]], dtype=torch.bfloat16)
        mlp = torch.tensor([[0.25]], dtype=torch.bfloat16)
        eager = (hidden + attention) + mlp
        compiled = VALIDATOR.compiled_residual_sum(hidden, attention, mlp)
        torch.testing.assert_close(eager, torch.tensor([[100.0]], dtype=torch.bfloat16))
        torch.testing.assert_close(
            compiled, torch.tensor([[100.5]], dtype=torch.bfloat16)
        )

    def test_batched_rows_match_sequential_rows(self) -> None:
        source = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        middle = torch.cumsum(source, dim=0).tanh()
        target = torch.cumsum(middle, dim=0).sin()
        valid = torch.arange(1, 4)
        batched = VALIDATOR.future_summed_vjp_rows(
            target, (source, middle), valid, 0, 3, retain_graph=True
        )
        sequential = VALIDATOR.sequential_future_summed_vjp_rows(
            target, (source, middle), valid, 0, 3
        )
        for actual, expected in zip(batched, sequential, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class ObserverProofValidationTest(unittest.TestCase):
    def _write_fixture(self, directory: Path) -> dict[str, object]:
        prompt_text = "proof fixture"
        token_ids = [7, 11, 13]
        shared = {
            name: torch.tensor([index], dtype=torch.bfloat16)
            for index, name in enumerate(
                sorted(VALIDATOR.EXPECTED_SHARED_PROOF_NAMES), start=1
            )
        }
        baseline_payload = {
            "schema_version": 1,
            "mode": "compiled",
            "model_revision": VALIDATOR.MODEL_REVISION,
            "prompt": prompt_text,
            "prompt_token_ids": token_ids,
            "tensors": shared,
        }
        observer_payload = {
            **baseline_payload,
            "mode": "compiled-observer",
            "tensors": {name: value.clone() for name, value in shared.items()},
        }
        baseline_tensor_path = directory / "baseline.pt"
        observer_tensor_path = directory / "observer.pt"
        torch.save(baseline_payload, baseline_tensor_path)
        torch.save(observer_payload, observer_tensor_path)

        generation = {
            "generated_token_id": 42,
            "prompt_token_ids": token_ids,
            "top_logprobs": [
                {"token_id": index, "logprob": -float(index)}
                for index in range(20)
            ],
        }
        common_report = {
            "schema_version": 1,
            "status": "passed",
            "model": {"revision": VALIDATOR.MODEL_REVISION},
            "prompt": {"text": prompt_text, "token_ids": token_ids},
        }
        baseline_report = {
            **common_report,
            "runtime": {"mode": "compiled"},
            "authoritative_compiled_generation": generation,
        }
        observer_report = {
            **common_report,
            "runtime": {"mode": "compiled-observer"},
            "instrumented_generation": generation,
        }
        baseline_report_path = directory / "baseline.json"
        observer_report_path = directory / "observer.json"
        baseline_report_path.write_text(
            json.dumps(baseline_report), encoding="utf-8"
        )
        observer_report_path.write_text(
            json.dumps(observer_report), encoding="utf-8"
        )

        exact_metric = {
            "comparable": True,
            "exact": True,
            "max_abs": 0.0,
            "rms": 0.0,
            "relative_rms": 0.0,
            "nonfinite_pair_count": 0,
        }
        proof = {
            "schema_version": 1,
            "status": "passed",
            "authority": {
                "accepted": True,
                "basis": "isolated synthetic processes",
                "ordinary_post_init_hooks_authoritative": False,
            },
            "scope": {
                "model_revision": VALIDATOR.MODEL_REVISION,
                "main_model_prefill": True,
                "mtp_enabled": False,
                "prompt": {
                    "text": prompt_text,
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                },
            },
            "artifacts": {
                "baseline_json": str(baseline_report_path),
                "baseline_tensors": str(baseline_tensor_path),
                "baseline_tensor_bytes": baseline_tensor_path.stat().st_size,
                "observer_json": str(observer_report_path),
                "observer_tensors": str(observer_tensor_path),
                "observer_tensor_bytes": observer_tensor_path.stat().st_size,
            },
            "endpoint_parity": {
                "generated_token_equal": True,
                "shared_top_logprob_tokens": 20,
                "max_abs_shared_logprob_delta": 0.0,
                "logprob_deltas": {str(index): 0.0 for index in range(20)},
            },
            "shared_compiled_boundary_parity": {
                "shared_tensor_count": len(shared),
                "all_exact": True,
                "all_within_bf16_boundary_tolerance": True,
                "eager_only": [],
                "metrics": {name: dict(exact_metric) for name in shared},
            },
        }
        proof_path = directory / "proof.json"
        proof_path.write_text(json.dumps(proof), encoding="utf-8")
        return {
            "proof_path": proof_path,
            "capture_path": observer_tensor_path,
            "capture_report_path": observer_report_path,
            "payload": observer_payload,
            "capture_report": observer_report,
        }

    def test_exact_isolated_proof_is_bound_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._write_fixture(Path(temporary))
            result = VALIDATOR.validate_observer_proof(**fixture)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["shared_tensor_count"], 30)
        self.assertEqual(len(result["proof_sha256"]), 64)
        self.assertTrue(result["shared_tensors_all_exact_on_independent_reload"])
        self.assertIn("surrogate only", result["derivative_claim"])

    def test_independent_tensor_reload_rejects_in_memory_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._write_fixture(Path(temporary))
            fixture["payload"]["tensors"]["gdn.layer62.q"] += 1
            with self.assertRaisesRegex(ValueError, "independent reload"):
                VALIDATOR.validate_observer_proof(**fixture)

    def test_nonzero_endpoint_delta_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._write_fixture(Path(temporary))
            proof = json.loads(fixture["proof_path"].read_text(encoding="utf-8"))
            proof["endpoint_parity"]["max_abs_shared_logprob_delta"] = 0.01
            fixture["proof_path"].write_text(json.dumps(proof), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "endpoint parity"):
                VALIDATOR.validate_observer_proof(**fixture)

    def test_memory_bounded_pair_proof_schema_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._write_fixture(Path(temporary))
            capture_path = fixture["capture_path"]
            capture_report_path = fixture["capture_report_path"]
            capture_report = fixture["capture_report"]
            capture_report["status"] = "captured"
            capture_report["runtime"].update(
                {
                    "target_profile": "all",
                    "target_layers": list(range(64)),
                    "mtp_enabled": False,
                }
            )
            resolved_model_path = (
                "/cache/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/"
                + VALIDATOR.MODEL_REVISION
            )
            capture_report["model"].update(
                {
                    "repo_id": VALIDATOR.MODEL_REPO,
                    "resolved_path": resolved_model_path,
                    "identity": {
                        "policy": VALIDATOR.MODEL_IDENTITY_POLICY,
                        "repo_id": VALIDATOR.MODEL_REPO,
                        "revision": VALIDATOR.MODEL_REVISION,
                        "resolved_path": resolved_model_path,
                        "metadata_sha256": VALIDATOR.PINNED_METADATA_SHA256,
                        "strict_pinned_validation": True,
                        "validator": "ModelOptCheckpoint(strict_pinned=True)",
                    },
                }
            )
            capture_report_path.write_text(
                json.dumps(capture_report), encoding="utf-8"
            )

            def record(path: Path) -> dict[str, object]:
                return {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            verifier_path = ROOT / "scripts" / "prove_nvfp4_capture_pair.py"
            proof = {
                "schema_version": 1,
                "status": "passed",
                "claim": {
                    "mtp": "off",
                    "observer_graph_modified": True,
                    "observer_modification_discharged": True,
                    "discharge_basis": ["isolated pair"],
                },
                "configuration": {
                    "model_revision": VALIDATOR.MODEL_REVISION,
                    "target_profile": "all",
                    "target_layers": list(range(64)),
                    "mtp_enabled": False,
                    "language_model_only": True,
                    "prompt": {
                        "token_ids": fixture["payload"]["prompt_token_ids"]
                    },
                },
                "generation_record_parity": {"exact": True},
                "shared_internal_tensor_parity": {
                    "shared_tensor_count": 688,
                    "observer_only_tensor_count": 432,
                    "all_shared_bit_exact": True,
                },
                "replay_parameter_parity": {
                    "parameter_count": 785,
                    "all_names_equal": True,
                    "all_shapes_equal": True,
                    "all_dtypes_equal": True,
                    "all_content_hashes_equal": True,
                    "json_provenance_equal": True,
                },
                "observer_capture_completeness": {
                    "required_missing": [],
                    "truncated": [],
                },
                "artifacts": {
                    "observer_json": record(capture_report_path),
                    "observer_tensors": record(capture_path),
                },
                "verifier": record(verifier_path),
            }
            fixture["proof_path"].write_text(
                json.dumps(proof), encoding="utf-8"
            )
            result = VALIDATOR.validate_observer_proof(**fixture)
        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["proof_schema"], "memory-bounded-manifest-pair-v1"
        )
        self.assertEqual(result["shared_tensor_count"], 688)


class EndToEndSuffixValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = SyntheticSuffix()

    def test_real_schema_mixed_weight_replay_and_vjps(self) -> None:
        observer_payload = {
            **self.case.payload,
            "tensors": {
                name: value
                for name, value in self.case.payload["tensors"].items()
                if name not in VALIDATOR._optional_boundary_tensor_names()
            },
        }
        result = VALIDATOR.validate_suffix_payload(
            observer_payload,
            self.case.checkpoint,
            device="cpu",
            attention_config=self.case.config,
            gdn_layout=self.case.layout,
            weight_dtype=torch.float32,
            vjp_rows=2,
            skip_first=0,
            forward_atol=1e-6,
            forward_rtol=1e-6,
            vjp_atol=1e-5,
            vjp_rtol=1e-5,
            checkpoint_interval=2,
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["forward"]["h62_reconstructed"]["exact"])
        self.assertTrue(result["forward"]["h63_reconstructed"]["exact"])
        self.assertTrue(result["vjp_validation"]["sources"]["J61"]["finite"])
        self.assertTrue(result["vjp_validation"]["sources"]["J62"]["finite"])
        self.assertEqual(
            set(result["unavailable_optional_boundaries"]),
            VALIDATOR._optional_boundary_tensor_names(),
        )
        self.assertEqual(
            result["contract"]["gdn_gated_norm_forward_value"],
            "recomputed surrogate from captured core/z",
        )
        self.assertEqual(
            set(self.case.checkpoint.requested),
            {
                f"{VALIDATOR.CHECKPOINT_PREFIX}.62.mlp.gate_up_proj",
                f"{VALIDATOR.CHECKPOINT_PREFIX}.62.mlp.down_proj",
                f"{VALIDATOR.CHECKPOINT_PREFIX}.63.mlp.gate_up_proj",
                f"{VALIDATOR.CHECKPOINT_PREFIX}.63.mlp.down_proj",
            },
        )


if __name__ == "__main__":
    unittest.main()
