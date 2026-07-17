#!/usr/bin/env python3
"""Tests for full exact-forward GDN branch replay."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nvfp4_gdn as GDN  # noqa: E402
import nvfp4_ste as STE  # noqa: E402


class GdnBranchReplayTest(unittest.TestCase):
    def make_case(self):
        generator = torch.Generator().manual_seed(810)
        tokens, hidden = 4, 8
        layout = GDN.GdnLayout(
            key_heads=1,
            value_heads=2,
            key_dim=2,
            value_dim=3,
        )
        q_size = layout.key_heads * layout.key_dim
        v_size = layout.value_heads * layout.value_dim
        qkv_size = q_size * 2 + v_size
        qkvz_size = qkv_size + v_size
        weights = GDN.GdnWeights(
            qkvz_out_in=torch.randn(qkvz_size, hidden, generator=generator) * 0.2,
            qkvz_input_scale=torch.tensor(0.5),
            ba_out_in=torch.randn(
                layout.value_heads * 2, hidden, generator=generator
            )
            * 0.2,
            conv=torch.randn(qkv_size, 3, generator=generator) * 0.2,
            a_log=torch.randn(layout.value_heads, generator=generator) * 0.1,
            dt_bias=torch.randn(layout.value_heads, generator=generator) * 0.1,
            norm=torch.randn(layout.value_dim, generator=generator) * 0.1,
            out_out_in=torch.randn(hidden, v_size, generator=generator) * 0.2,
            out_input_scale=torch.tensor(0.5),
        )
        hidden_states = torch.randn(tokens, hidden, generator=generator)

        qkvz = hidden_states @ weights.qkvz_out_in.T
        ba = hidden_states @ weights.ba_out_in.T
        mixed_qkv, z_flat = qkvz.split((qkv_size, v_size), dim=-1)
        conv_qkv = STE.causal_depthwise_conv1d_silu(mixed_qkv, weights.conv)
        query_raw, key_raw, value_raw = conv_qkv.split(
            (q_size, q_size, v_size), dim=-1
        )
        query = STE.l2_normalize(
            query_raw.reshape(tokens, layout.key_heads, layout.key_dim)
        )
        key = STE.l2_normalize(
            key_raw.reshape(tokens, layout.key_heads, layout.key_dim)
        )
        value = value_raw.reshape(tokens, layout.value_heads, layout.value_dim)
        b, a = ba.chunk(2, dim=-1)
        log_decay = STE.gdn_log_decay(a, weights.a_log, weights.dt_bias)
        beta = torch.sigmoid(b.float())
        initial_state = torch.zeros(
            layout.value_heads, layout.value_dim, layout.key_dim
        )
        trace = STE.gdn_reference_forward(
            query, key, value, log_decay, beta, initial_state
        )
        z = z_flat.reshape(tokens, layout.value_heads, layout.value_dim)
        gated_norm = STE.gated_rms_norm(
            trace.output, z, weights.norm, layout.norm_eps
        )
        branch_output = gated_norm.flatten(-2) @ weights.out_out_in.T
        capture = GDN.GdnCapture(
            qkvz=qkvz,
            ba=ba,
            conv_qkv=conv_qkv,
            query=query,
            key=key,
            value=value,
            log_decay=log_decay,
            beta=beta,
            core_output=trace.output,
            final_state=trace.final_state,
            gated_norm=gated_norm,
            branch_output=branch_output,
        )
        return hidden_states, layout, weights, capture, initial_state

    def canonical(self, hidden_states, layout, weights, initial_state):
        q_size = layout.key_heads * layout.key_dim
        v_size = layout.value_heads * layout.value_dim
        qkv_size = q_size * 2 + v_size
        qkvz = hidden_states @ weights.qkvz_out_in.T
        ba = hidden_states @ weights.ba_out_in.T
        mixed_qkv, z_flat = qkvz.split((qkv_size, v_size), dim=-1)
        conv_qkv = STE.causal_depthwise_conv1d_silu(mixed_qkv, weights.conv)
        query_raw, key_raw, value_raw = conv_qkv.split(
            (q_size, q_size, v_size), dim=-1
        )
        query = STE.l2_normalize(
            query_raw.reshape(-1, layout.key_heads, layout.key_dim)
        )
        key = STE.l2_normalize(
            key_raw.reshape(-1, layout.key_heads, layout.key_dim)
        )
        value = value_raw.reshape(-1, layout.value_heads, layout.value_dim)
        b, a = ba.chunk(2, dim=-1)
        trace = STE.gdn_reference_forward(
            query,
            key,
            value,
            STE.gdn_log_decay(a, weights.a_log, weights.dt_bias),
            torch.sigmoid(b.float()),
            initial_state,
        )
        z = z_flat.reshape(-1, layout.value_heads, layout.value_dim)
        normalized = STE.gated_rms_norm(
            trace.output, z, weights.norm, layout.norm_eps
        )
        return normalized.flatten(-2) @ weights.out_out_in.T

    def test_forward_is_captured_and_backward_matches_canonical_branch(self):
        hidden, layout, weights, capture, initial_state = self.make_case()
        captured_override = GDN.GdnCapture(
            **{
                **capture.__dict__,
                "branch_output": capture.branch_output + 123.0,
            }
        )
        actual_input = hidden.detach().requires_grad_(True)
        actual = GDN.replay_gdn_branch(
            actual_input,
            layout,
            weights,
            captured_override,
            initial_state=initial_state,
        )
        torch.testing.assert_close(
            actual, captured_override.branch_output, rtol=0, atol=0
        )
        cotangent = torch.randn_like(actual)
        (actual_grad,) = torch.autograd.grad(actual, actual_input, cotangent)

        reference_input = hidden.detach().requires_grad_(True)
        reference = self.canonical(reference_input, layout, weights, initial_state)
        (reference_grad,) = torch.autograd.grad(
            reference, reference_input, cotangent
        )
        torch.testing.assert_close(
            actual_grad, reference_grad, rtol=4e-5, atol=4e-5
        )

    def test_capture_shape_mismatch_is_rejected(self):
        hidden, layout, weights, capture, initial_state = self.make_case()
        bad = GDN.GdnCapture(
            **{**capture.__dict__, "beta": capture.beta[:, :-1]}
        )
        with self.assertRaisesRegex(ValueError, "capture beta shape"):
            GDN.replay_gdn_branch(
                hidden,
                layout,
                weights,
                bad,
                initial_state=initial_state,
            )

    def test_complete_block_replay_preserves_residual_semantics(self):
        hidden, layout, weights, capture, initial_state = self.make_case()
        generator = torch.Generator().manual_seed(811)
        intermediate = 5
        gate_up_weight = torch.randn(
            intermediate * 2, hidden.shape[-1], generator=generator
        )
        down_weight = torch.randn(
            hidden.shape[-1], intermediate, generator=generator
        )
        input_norm = torch.randn(hidden.shape[-1], generator=generator) * 0.1
        post_norm = torch.randn(hidden.shape[-1], generator=generator) * 0.1
        replay = GDN.replay_qwen_gdn_block(
            hidden,
            layout,
            weights,
            capture,
            input_norm_weight=input_norm,
            post_attention_norm_weight=post_norm,
            gate_up_linear=lambda inputs: inputs @ gate_up_weight.T,
            down_linear=lambda inputs: inputs @ down_weight.T,
            initial_state=initial_state,
        )
        torch.testing.assert_close(
            replay.after_attention,
            hidden + capture.branch_output,
        )
        torch.testing.assert_close(
            replay.output,
            replay.after_attention + replay.mlp_output,
        )
        self.assertEqual(replay.gate_up.shape, (hidden.shape[0], 2 * intermediate))
        self.assertEqual(replay.activated.shape, (hidden.shape[0], intermediate))


if __name__ == "__main__":
    unittest.main()
