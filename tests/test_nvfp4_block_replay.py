#!/usr/bin/env python3
"""Synthetic multi-block tests for generic Qwen3.6 captured replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_jlens_nvfp4_ste as FIT  # noqa: E402
import nvfp4_attention as ATTENTION  # noqa: E402
import nvfp4_block_replay as REPLAY  # noqa: E402
import nvfp4_gdn as GDN  # noqa: E402
import nvfp4_ste as STE  # noqa: E402


@dataclass(frozen=True)
class FakeRawWeight:
    value: torch.Tensor

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.value.to(dtype=dtype)


@dataclass(frozen=True)
class FakePackedRawWeight:
    packed_weight: torch.Tensor
    block_scales: torch.Tensor
    global_scale: torch.Tensor
    block_size: int

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return STE.dequantize_nvfp4_weight(
            self.packed_weight,
            self.block_scales,
            self.global_scale,
            block_size=self.block_size,
            dtype=dtype,
        )


class FakeCheckpoint:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.requested: list[str] = []

    def load_nvfp4(self, module_name: str) -> object:
        self.requested.append(module_name)
        value = self.values[module_name]
        return FakeRawWeight(value) if isinstance(value, torch.Tensor) else value


class SyntheticMultiBlock:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(36027)
        self.tokens = 4
        self.hidden = 8
        self.intermediate = 6
        self.attention = ATTENTION.QwenFullAttentionConfig(
            hidden_size=self.hidden,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=4,
            rotary_dim=4,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            mrope_section=(1, 1, 0),
        )
        self.gdn = GDN.GdnLayout(
            key_heads=1,
            value_heads=2,
            key_dim=2,
            value_dim=3,
        )
        self.spec = REPLAY.QwenReplaySpec(
            (
                REPLAY.LINEAR_ATTENTION,
                REPLAY.LINEAR_ATTENTION,
                REPLAY.FULL_ATTENTION,
                REPLAY.LINEAR_ATTENTION,
            ),
            self.attention,
            self.gdn,
        )
        self.tensors: dict[str, torch.Tensor] = {}
        self.checkpoint_values: dict[str, torch.Tensor] = {}

        current = torch.randn(
            self.tokens, self.hidden, generator=generator
        ) * 0.15
        self.tensors["h0_post_block"] = current
        for layer in range(1, 4):
            if self.spec.layer_types[layer] == REPLAY.LINEAR_ATTENTION:
                current = self._add_gdn_layer(layer, current, generator)
            else:
                current = self._add_attention_layer(layer, current, generator)
            self.tensors[f"h{layer}_post_block"] = current

        self.payload = {
            "schema_version": 1,
            "mode": "compiled-observer",
            "model_revision": "synthetic",
            "prompt": "synthetic multi-block prompt",
            "prompt_token_ids": [3, 5, 7, 11],
            "target_profile": "all",
            "target_layers": [1, 2, 3],
            "tensors": self.tensors,
        }
        self.checkpoint = FakeCheckpoint(self.checkpoint_values)

    def packed_checkpoint(self) -> FakeCheckpoint:
        generator = torch.Generator().manual_seed(481516)
        raw: dict[str, object] = {}
        scale_values = torch.tensor([0.5, 0.75, 1.0], dtype=torch.float32)
        for name, dense in self.checkpoint_values.items():
            rows, width = dense.shape
            if width % 2:
                raise AssertionError("synthetic W4 input widths must be even")
            packed = torch.randint(
                0,
                256,
                (rows, width // 2),
                generator=generator,
                dtype=torch.uint8,
            )
            indices = torch.randint(
                0,
                len(scale_values),
                (rows, width // 2),
                generator=generator,
            )
            raw[name] = FakePackedRawWeight(
                packed_weight=packed,
                block_scales=scale_values[indices].to(torch.float8_e4m3fn),
                global_scale=torch.tensor(0.125, dtype=torch.float32),
                block_size=2,
            )
        return FakeCheckpoint(raw)

    def _random_fp8(
        self,
        out_features: int,
        in_features: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        postload = (
            torch.randn(in_features, out_features, generator=generator) * 0.12
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.75, dtype=torch.float32)
        effective = postload.T.float() * scale
        return postload, scale, effective

    def _store_fp8(
        self,
        module: str,
        postload: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        prefix = f"replay.fp8.{module}"
        self.tensors[f"{prefix}.weight"] = postload
        self.tensors[f"{prefix}.weight_scale"] = scale
        self.tensors[f"{prefix}.input_scale"] = torch.tensor(1.0)

    def _common_parameters(
        self,
        layer: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_norm = torch.randn(self.hidden, generator=generator) * 0.03
        post_norm = torch.randn(self.hidden, generator=generator) * 0.03
        gate_up = (
            torch.randn(
                2 * self.intermediate, self.hidden, generator=generator
            )
            * 0.08
        )
        down = (
            torch.randn(
                self.hidden, self.intermediate, generator=generator
            )
            * 0.08
        )
        self.tensors[
            f"replay.norm.layers.{layer}.input_layernorm.weight"
        ] = input_norm
        self.tensors[
            f"replay.norm.layers.{layer}.post_attention_layernorm.weight"
        ] = post_norm
        prefix = REPLAY.CHECKPOINT_PREFIX
        self.checkpoint_values[
            f"{prefix}.{layer}.mlp.gate_up_proj"
        ] = gate_up
        self.checkpoint_values[f"{prefix}.{layer}.mlp.down_proj"] = down
        return input_norm, post_norm, gate_up, down

    def _add_gdn_layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        input_norm, post_norm, gate_up, down = self._common_parameters(
            layer, generator
        )
        q_size = self.gdn.key_heads * self.gdn.key_dim
        value_size = self.gdn.value_heads * self.gdn.value_dim
        qkv_size = 2 * q_size + value_size
        qkvz_module = f"layers.{layer}.linear_attn.in_proj_qkvz"
        out_module = f"layers.{layer}.linear_attn.out_proj"
        qkvz_runtime, qkvz_scale, qkvz_weight = self._random_fp8(
            qkv_size + value_size, self.hidden, generator
        )
        out_runtime, out_scale, out_weight = self._random_fp8(
            self.hidden, value_size, generator
        )
        self._store_fp8(qkvz_module, qkvz_runtime, qkvz_scale)
        self._store_fp8(out_module, out_runtime, out_scale)

        ba_weight = (
            torch.randn(
                2 * self.gdn.value_heads, self.hidden, generator=generator
            )
            * 0.08
        )
        conv_weight = (
            torch.randn(qkv_size, 3, generator=generator) * 0.08
        )
        a_log = torch.randn(self.gdn.value_heads, generator=generator) * 0.05
        dt_bias = torch.randn(self.gdn.value_heads, generator=generator) * 0.05
        norm_weight = torch.randn(self.gdn.value_dim, generator=generator) * 0.03
        gdn_prefix = f"replay.gdn.layers.{layer}"
        self.tensors[f"{gdn_prefix}.in_proj_ba.weight"] = ba_weight
        self.tensors[f"{gdn_prefix}.conv1d.weight"] = conv_weight.unsqueeze(1)
        self.tensors[f"{gdn_prefix}.A_log"] = a_log
        self.tensors[f"{gdn_prefix}.dt_bias"] = dt_bias
        self.tensors[f"{gdn_prefix}.norm.weight"] = norm_weight

        attention_input = ATTENTION.qwen_rms_norm(hidden, input_norm, 1e-6)
        qkvz = attention_input @ qkvz_weight.T
        ba = attention_input @ ba_weight.T
        mixed, z_flat = qkvz.split((qkv_size, value_size), dim=-1)
        conv = STE.causal_depthwise_conv1d_silu(mixed, conv_weight)
        query_raw, key_raw, value_raw = conv.split(
            (q_size, q_size, value_size), dim=-1
        )
        query = STE.l2_normalize(
            query_raw.reshape(self.tokens, self.gdn.key_heads, self.gdn.key_dim)
        )
        key = STE.l2_normalize(
            key_raw.reshape(self.tokens, self.gdn.key_heads, self.gdn.key_dim)
        )
        value = value_raw.reshape(
            self.tokens, self.gdn.value_heads, self.gdn.value_dim
        )
        b, a = ba.chunk(2, dim=-1)
        log_decay = STE.gdn_log_decay(a, a_log, dt_bias)
        beta = torch.sigmoid(b.float())
        initial_state = torch.zeros(
            self.gdn.value_heads, self.gdn.value_dim, self.gdn.key_dim
        )
        recurrence = STE.gdn_reference_forward(
            query, key, value, log_decay, beta, initial_state
        )
        z = z_flat.reshape(
            self.tokens, self.gdn.value_heads, self.gdn.value_dim
        )
        normalized = STE.gated_rms_norm(
            recurrence.output, z, norm_weight, self.gdn.norm_eps
        )
        attention_output = normalized.flatten(-2) @ out_weight.T
        after_attention = hidden + attention_output
        mlp_input = ATTENTION.qwen_rms_norm(after_attention, post_norm, 1e-6)
        gate_up_output = mlp_input @ gate_up.T
        activated = ATTENTION.qwen_swiglu(gate_up_output)
        mlp_output = activated @ down.T

        linear_prefix = f"linear.layers.{layer}"
        self.tensors[
            f"{linear_prefix}.linear_attn.in_proj_qkvz.output"
        ] = qkvz
        self.tensors[f"{linear_prefix}.linear_attn.in_proj_ba.output"] = ba
        self.tensors[
            f"{linear_prefix}.linear_attn.out_proj.output"
        ] = attention_output
        self.tensors[f"{linear_prefix}.mlp.gate_up_proj.output"] = gate_up_output
        self.tensors[f"{linear_prefix}.mlp.down_proj.output"] = mlp_output
        self.tensors[f"layers.{layer}.mlp.swiglu_output"] = activated
        capture_prefix = f"gdn.layer{layer}"
        self.tensors[f"{capture_prefix}.conv_output_prefill"] = conv
        self.tensors[f"{capture_prefix}.q"] = query.unsqueeze(0)
        self.tensors[f"{capture_prefix}.k"] = key.unsqueeze(0)
        self.tensors[f"{capture_prefix}.v"] = value.unsqueeze(0)
        self.tensors[f"{capture_prefix}.log_g"] = log_decay.unsqueeze(0)
        self.tensors[f"{capture_prefix}.beta"] = beta.unsqueeze(0)
        self.tensors[f"{capture_prefix}.initial_state"] = initial_state.unsqueeze(0)
        self.tensors[
            f"{capture_prefix}.chunk_output"
        ] = recurrence.output.unsqueeze(0)
        self.tensors[f"{capture_prefix}.core_output"] = recurrence.output
        self.tensors[
            f"{capture_prefix}.final_state"
        ] = recurrence.final_state.unsqueeze(0)
        if layer == 1:
            self.tensors[f"{capture_prefix}.norm_output"] = normalized.flatten(0, 1)
        return after_attention + mlp_output

    def _add_attention_layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        input_norm, post_norm, gate_up, down = self._common_parameters(
            layer, generator
        )
        qkv_module = f"layers.{layer}.self_attn.qkv_proj"
        out_module = f"layers.{layer}.self_attn.o_proj"
        qkv_runtime, qkv_scale, qkv_weight = self._random_fp8(
            self.attention.qkv_projection_size, self.hidden, generator
        )
        out_runtime, out_scale, out_weight = self._random_fp8(
            self.hidden, self.attention.query_size, generator
        )
        self._store_fp8(qkv_module, qkv_runtime, qkv_scale)
        self._store_fp8(out_module, out_runtime, out_scale)
        q_norm = torch.randn(self.attention.head_dim, generator=generator) * 0.03
        k_norm = torch.randn(self.attention.head_dim, generator=generator) * 0.03
        self.tensors[
            f"replay.norm.layers.{layer}.self_attn.q_norm.weight"
        ] = q_norm
        self.tensors[
            f"replay.norm.layers.{layer}.self_attn.k_norm.weight"
        ] = k_norm

        linears = ATTENTION.QwenBlockLinears(
            qkv=lambda inputs: inputs @ qkv_weight.T,
            attention_out=lambda inputs: inputs @ out_weight.T,
            gate_up=lambda inputs: inputs @ gate_up.T,
            down=lambda inputs: inputs @ down.T,
        )
        block = ATTENTION.replay_qwen_full_attention_suffix(
            hidden,
            torch.arange(self.tokens),
            self.attention,
            input_norm_weight=input_norm,
            post_attention_norm_weight=post_norm,
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            linears=linears,
        )
        linear_prefix = f"linear.layers.{layer}"
        self.tensors[f"{linear_prefix}.self_attn.qkv_proj.output"] = block.attention.qkv
        self.tensors[f"{linear_prefix}.self_attn.o_proj.output"] = block.attention.output
        self.tensors[f"{linear_prefix}.mlp.gate_up_proj.output"] = block.gate_up
        self.tensors[f"{linear_prefix}.mlp.down_proj.output"] = block.hidden_states
        self.tensors[f"layers.{layer}.mlp.swiglu_output"] = block.activated
        attention_prefix = f"attention.layer{layer}"
        self.tensors[f"{attention_prefix}.q_post_rope"] = block.attention.query.flatten(1)
        self.tensors[f"{attention_prefix}.k_post_rope"] = block.attention.key.flatten(1)
        self.tensors[f"{attention_prefix}.v"] = block.attention.value.flatten(1)
        self.tensors[
            f"{attention_prefix}.core_output"
        ] = block.attention.core_output.flatten(1)
        return block.output

    def factory(self) -> REPLAY.CapturedQwenBlockReplayFactory:
        backend = REPLAY.DenseDequantW4Backend(
            self.checkpoint,
            dtype=torch.float32,
        )
        return REPLAY.CapturedQwenBlockReplayFactory(
            self.payload,
            backend,
            spec=self.spec,
            fp8_weight_dtype=torch.float32,
            checkpoint_interval=2,
        )


class ReplaySpecTest(unittest.TestCase):
    def test_qwen36_schedule_and_config_parser(self) -> None:
        self.assertEqual(len(REPLAY.QWEN36_27B_LAYER_TYPES), 64)
        self.assertEqual(
            [
                index
                for index, kind in enumerate(REPLAY.QWEN36_27B_LAYER_TYPES)
                if kind == REPLAY.FULL_ATTENTION
            ],
            list(range(3, 64, 4)),
        )
        spec = REPLAY.QwenReplaySpec.from_model_config(
            {
                "text_config": {
                    "hidden_size": 8,
                    "layer_types": ["linear_attention", "full_attention"],
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 2,
                    "linear_key_head_dim": 2,
                    "linear_value_head_dim": 3,
                    "rms_norm_eps": 1e-6,
                    "attn_output_gate": True,
                    "rope_parameters": {
                        "partial_rotary_factor": 1.0,
                        "rope_theta": 10_000.0,
                        "mrope_section": [1, 1, 0],
                        "mrope_interleaved": True,
                    },
                }
            }
        )
        self.assertEqual(spec.attention.rotary_dim, 4)
        self.assertEqual(spec.attention.mrope_section, (1, 1, 0))
        self.assertEqual(spec.gdn.value_heads, 2)


class CapturedReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = SyntheticMultiBlock()

    def test_gdn_attention_schedule_replays_every_block_exactly(self) -> None:
        factory = self.case.factory()
        current = self.case.tensors["h0_post_block"]
        for layer in range(1, 4):
            logical_input = current.detach().requires_grad_(True)
            output = factory(layer, logical_input)
            self.assertTrue(
                torch.equal(output.detach(), self.case.tensors[f"h{layer}_post_block"])
            )
            (gradient,) = torch.autograd.grad(output.sum(), logical_input)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient != 0).any())
            current = output.detach()
        self.assertEqual(
            self.case.checkpoint.requested,
            [
                f"{REPLAY.CHECKPOINT_PREFIX}.{layer}.mlp.{module}"
                for layer in range(1, 4)
                for module in ("gate_up_proj", "down_proj")
            ],
        )

    def test_reverse_replay_rows_matches_one_retained_multiblock_graph(self) -> None:
        factory = self.case.factory()
        positions = torch.tensor([0, 1, 2])
        actual = REPLAY.reverse_replay_captured_rows(
            factory,
            positions,
            1,
            4,
            target_layer=3,
        )

        root = self.case.tensors["h0_post_block"].detach().requires_grad_(True)
        retained = {0: root}
        current = root
        for layer in range(1, 4):
            current = factory(layer, current)
            retained[layer] = current
        cotangents = FIT.target_cotangent_rows(
            tokens=self.case.tokens,
            hidden=self.case.hidden,
            valid_positions=positions,
            row_start=1,
            row_stop=4,
            dtype=current.dtype,
            device=current.device,
        )
        for source in range(3):
            (gradient,) = torch.autograd.grad(
                current,
                retained[source],
                grad_outputs=cotangents,
                is_grads_batched=True,
                retain_graph=source < 2,
            )
            expected = gradient[:, positions].float().mean(dim=1)
            torch.testing.assert_close(
                actual.rows[source], expected, rtol=1e-5, atol=1e-6
            )

    def test_packed_w4_live_fp8_matches_dense_multiblock_rows(self) -> None:
        dense_checkpoint = self.case.packed_checkpoint()
        packed_checkpoint = FakeCheckpoint(dense_checkpoint.values)
        dense = REPLAY.CapturedQwenBlockReplayFactory(
            self.case.payload,
            REPLAY.DenseDequantW4Backend(
                dense_checkpoint, dtype=torch.bfloat16
            ),
            fp8_backend=REPLAY.DenseDequantFp8Backend(dtype=torch.bfloat16),
            spec=self.case.spec,
            fp8_weight_dtype=torch.bfloat16,
            checkpoint_interval=2,
        )
        packed = REPLAY.CapturedQwenBlockReplayFactory(
            self.case.payload,
            REPLAY.PackedNvFp4W4Backend(packed_checkpoint),
            fp8_backend=REPLAY.LiveFp8Backend(),
            spec=self.case.spec,
            fp8_weight_dtype=torch.bfloat16,
            checkpoint_interval=2,
        )

        for factory in (dense, packed):
            current = self.case.tensors["h0_post_block"]
            for layer in range(1, 4):
                current = factory(layer, current)
                self.assertTrue(
                    torch.equal(
                        current.detach(),
                        self.case.tensors[f"h{layer}_post_block"],
                    )
                )

        positions = torch.tensor([0, 1, 2])
        dense_rows = dense.reverse_replay_rows(
            positions, 0, 3, target_layer=3
        ).rows
        packed_rows = packed.reverse_replay_rows(
            positions, 0, 3, target_layer=3
        ).rows
        self.assertEqual(set(dense_rows), set(packed_rows))
        for layer in dense_rows:
            difference = packed_rows[layer].float() - dense_rows[layer].float()
            relative_rms = float(
                difference.square().mean().sqrt()
                / dense_rows[layer].float().square().mean().sqrt().clamp_min(1e-12)
            )
            self.assertLess(relative_rms, 0.03, f"layer {layer}")

    def test_outer_capture_injects_compiled_residual_value(self) -> None:
        payload = {
            **self.case.payload,
            "tensors": dict(self.case.tensors),
            "target_layers": [1],
        }
        payload["tensors"]["h1_post_block"] = (
            payload["tensors"]["h1_post_block"] + 0.25
        )
        backend = REPLAY.DenseDequantW4Backend(
            FakeCheckpoint(self.case.checkpoint_values), dtype=torch.float32
        )
        factory = REPLAY.CapturedQwenBlockReplayFactory(
            payload,
            backend,
            spec=self.case.spec,
            fp8_weight_dtype=torch.float32,
        )
        logical_input = payload["tensors"]["h0_post_block"].detach().requires_grad_(True)
        output = factory(1, logical_input)
        self.assertTrue(torch.equal(output.detach(), payload["tensors"]["h1_post_block"]))
        (gradient,) = torch.autograd.grad(output.sum(), logical_input)
        self.assertTrue((gradient != 0).any())

    def test_missing_type_specific_capture_is_rejected(self) -> None:
        payload = {
            **self.case.payload,
            "tensors": dict(self.case.tensors),
        }
        del payload["tensors"]["attention.layer2.core_output"]
        backend = REPLAY.DenseDequantW4Backend(
            FakeCheckpoint(self.case.checkpoint_values), dtype=torch.float32
        )
        factory = REPLAY.CapturedQwenBlockReplayFactory(
            payload,
            backend,
            spec=self.case.spec,
            fp8_weight_dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "layer 2 capture is missing"):
            factory(2, payload["tensors"]["h1_post_block"])


if __name__ == "__main__":
    unittest.main()
