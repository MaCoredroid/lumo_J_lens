#!/usr/bin/env python3
"""Generic exact-forward Qwen3.6 block replay for NVFP4 J-lens fits.

The capture payload supplies deployed forward values and live post-load FP8
weights. Raw ModelOpt W4 tensors are dequantized one block at a time for a
frozen-weight validation VJP. Every returned block output is replaced by the
captured compiled residual, while gradients follow the declared FP8/W4/GDN
surrogate. This is not the literal derivative of quantized rounding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from fp8_live_vjp import exact_live_fp8_linear
from nvfp4_attention import (
    QWEN36_27B_LAYER63,
    QwenBlockLinears,
    QwenFullAttentionConfig,
    replay_qwen_full_attention_suffix,
)
from nvfp4_gdn import (
    GdnCapture,
    GdnLayout,
    GdnWeights,
    replay_qwen_gdn_block,
)
from nvfp4_packed_vjp import exact_packed_nvfp4_linear
from nvfp4_ste import (
    dequantize_runtime_fp8_weight,
    exact_fp8_linear_ste,
    exact_value,
    exact_w4a16_linear,
    gated_rms_norm,
)


LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"
SUPPORTED_LAYER_TYPES = frozenset((LINEAR_ATTENTION, FULL_ATTENTION))
CHECKPOINT_PREFIX = "model.language_model.layers"
QWEN36_27B_LAYER_TYPES = tuple(
    FULL_ATTENTION if (layer + 1) % 4 == 0 else LINEAR_ATTENTION
    for layer in range(64)
)
QWEN36_27B_GDN_LAYOUT = GdnLayout(
    key_heads=16,
    value_heads=48,
    key_dim=128,
    value_dim=128,
    norm_eps=1e-6,
)


class CheckpointWeight(Protocol):
    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor: ...


class CheckpointReader(Protocol):
    def load_nvfp4(self, module_name: str) -> CheckpointWeight: ...


class W4Backend(Protocol):
    def exact_linear(
        self,
        layer: int,
        module: str,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
    ) -> torch.Tensor: ...


class PackedCheckpointWeight(CheckpointWeight, Protocol):
    packed_weight: torch.Tensor
    block_scales: torch.Tensor
    global_scale: torch.Tensor
    block_size: int


def _w4_module_name(checkpoint_prefix: str, layer: int, module: str) -> str:
    if module not in {"gate_up_proj", "down_proj"}:
        raise ValueError(f"unsupported W4 MLP module: {module}")
    return f"{checkpoint_prefix}.{layer}.mlp.{module}"


def _require_rank_two_weight(weight: Any, *, name: str) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"dequantized W4 weight is not rank two: {name}")
    return weight


def _require_cpu_raw_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"raw packed W4 field is not a tensor: {name}")
    if value.device.type != "cpu":
        raise ValueError(f"raw packed W4 field must originate on CPU: {name}")
    return value


@dataclass(frozen=True)
class DenseDequantW4Backend:
    """On-demand dense W4 validation backend.

    The effective ``[out, in]`` matrix is retained only by the current
    projection's autograd node. Reverse block replay releases it with that
    block graph before materializing the preceding layer.
    """

    checkpoint: CheckpointReader
    dtype: torch.dtype = torch.bfloat16
    checkpoint_prefix: str = CHECKPOINT_PREFIX

    def exact_linear(
        self,
        layer: int,
        module: str,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
    ) -> torch.Tensor:
        name = _w4_module_name(self.checkpoint_prefix, layer, module)
        raw = self.checkpoint.load_nvfp4(name)
        weight = _require_rank_two_weight(
            raw.dequantize(dtype=self.dtype), name=name
        ).to(device=inputs.device)
        return exact_w4a16_linear(inputs, exact_output, weight)


@dataclass(frozen=True)
class PackedNvFp4W4Backend:
    """Exact-forward W4 backend with a packed, streaming input VJP.

    Raw ModelOpt tensors are loaded on CPU for only the invoked projection,
    moved to the input device, and saved by the custom autograd node. No
    dense ``[out, in]`` matrix is materialized.
    """

    checkpoint: CheckpointReader
    checkpoint_prefix: str = CHECKPOINT_PREFIX

    def exact_linear(
        self,
        layer: int,
        module: str,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
    ) -> torch.Tensor:
        name = _w4_module_name(self.checkpoint_prefix, layer, module)
        raw = self.checkpoint.load_nvfp4(name)
        packed = _require_cpu_raw_tensor(
            getattr(raw, "packed_weight", None), name=f"{name}.weight"
        ).to(device=inputs.device)
        scales = _require_cpu_raw_tensor(
            getattr(raw, "block_scales", None), name=f"{name}.weight_scale"
        ).to(device=inputs.device)
        global_scale = _require_cpu_raw_tensor(
            getattr(raw, "global_scale", None), name=f"{name}.weight_scale_2"
        ).to(device=inputs.device)
        block_size = getattr(raw, "block_size", None)
        if not isinstance(block_size, int):
            raise TypeError(f"raw packed W4 block size is invalid: {name}")
        return exact_packed_nvfp4_linear(
            inputs,
            exact_output,
            packed,
            scales,
            global_scale,
            block_size=block_size,
        )


class Fp8Backend(Protocol):
    def exact_linear(
        self,
        module: str,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
        postload_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        *,
        ste_policy: str,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class DenseDequantFp8Backend:
    """Validation FP8 backend using a dense effective ``[out, in]`` matrix."""

    dtype: torch.dtype = torch.bfloat16

    def exact_linear(
        self,
        module: str,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
        postload_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        *,
        ste_policy: str,
    ) -> torch.Tensor:
        del module
        weight = dequantize_runtime_fp8_weight(
            postload_weight,
            weight_scale,
            transposed=True,
            dtype=self.dtype,
        ).to(device=inputs.device)
        return exact_fp8_linear_ste(
            inputs,
            exact_output,
            weight,
            input_scale.to(device=inputs.device),
            ste_policy=ste_policy,
        )


@dataclass(frozen=True)
class LiveFp8Backend:
    """Exact-forward FP8 backend with a live E4M3 streaming input VJP."""

    def exact_linear(
        self,
        module: str,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
        postload_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        *,
        ste_policy: str,
    ) -> torch.Tensor:
        del module
        return exact_live_fp8_linear(
            inputs,
            exact_output,
            postload_weight.to(device=inputs.device),
            weight_scale.to(device=inputs.device),
            input_scale.to(device=inputs.device),
            ste_policy=ste_policy,
        )


@dataclass(frozen=True)
class QwenReplaySpec:
    """Architecture details needed to dispatch and replay captured blocks."""

    layer_types: tuple[str, ...]
    attention: QwenFullAttentionConfig
    gdn: GdnLayout

    def __post_init__(self) -> None:
        if len(self.layer_types) < 2:
            raise ValueError("replay spec must contain at least two decoder layers")
        unsupported = sorted(set(self.layer_types) - SUPPORTED_LAYER_TYPES)
        if unsupported:
            raise ValueError(f"unsupported decoder layer types: {unsupported}")
        if self.attention.hidden_size <= 0:
            raise ValueError("attention hidden size must be positive")

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> QwenReplaySpec:
        """Build a replay spec from a Qwen3.6 Hugging Face config mapping."""

        raw_text = model_config.get("text_config", model_config)
        if not isinstance(raw_text, Mapping):
            raise ValueError("model text_config must be a mapping")
        rope = raw_text.get("rope_parameters", {})
        if not isinstance(rope, Mapping):
            raise ValueError("rope_parameters must be a mapping")

        required = (
            "hidden_size",
            "layer_types",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "rms_norm_eps",
        )
        missing = [name for name in required if name not in raw_text]
        if missing:
            raise ValueError(f"model config is missing replay geometry: {missing}")

        head_dim = int(raw_text["head_dim"])
        partial = float(
            rope.get(
                "partial_rotary_factor",
                raw_text.get("partial_rotary_factor", 1.0),
            )
        )
        rotary_float = head_dim * partial
        rotary_dim = int(rotary_float)
        if rotary_float != rotary_dim:
            raise ValueError("partial rotary factor does not produce an integer width")
        raw_section = rope.get("mrope_section", raw_text.get("mrope_section"))
        section = (
            tuple(int(value) for value in raw_section)
            if raw_section is not None
            else None
        )
        if section is not None and len(section) != 3:
            raise ValueError("mrope_section must contain exactly three values")

        attention = QwenFullAttentionConfig(
            hidden_size=int(raw_text["hidden_size"]),
            num_query_heads=int(raw_text["num_attention_heads"]),
            num_kv_heads=int(raw_text["num_key_value_heads"]),
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rope_theta=float(
                rope.get("rope_theta", raw_text.get("rope_theta", 10_000.0))
            ),
            rms_norm_eps=float(raw_text["rms_norm_eps"]),
            mrope_section=section,
            mrope_interleaved=bool(
                rope.get(
                    "mrope_interleaved",
                    raw_text.get("mrope_interleaved", True),
                )
            ),
            attention_output_gate=bool(raw_text.get("attn_output_gate", False)),
        )
        gdn = GdnLayout(
            key_heads=int(raw_text["linear_num_key_heads"]),
            value_heads=int(raw_text["linear_num_value_heads"]),
            key_dim=int(raw_text["linear_key_head_dim"]),
            value_dim=int(raw_text["linear_value_head_dim"]),
            norm_eps=float(raw_text["rms_norm_eps"]),
        )
        raw_layer_types = raw_text["layer_types"]
        if not isinstance(raw_layer_types, Sequence) or isinstance(
            raw_layer_types, (str, bytes)
        ):
            raise ValueError("layer_types must be a sequence")
        return cls(tuple(str(value) for value in raw_layer_types), attention, gdn)


QWEN36_27B_REPLAY_SPEC = QwenReplaySpec(
    QWEN36_27B_LAYER_TYPES,
    QWEN36_27B_LAYER63,
    QWEN36_27B_GDN_LAYOUT,
)


def _squeeze_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim < 1 or value.shape[0] != 1:
        raise ValueError(f"{name} must have a singleton batch dimension")
    return value.squeeze(0)


class CapturedQwenBlockReplayFactory:
    """Create one exact-forward surrogate block from an all-layer capture."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        w4_backend: W4Backend,
        *,
        fp8_backend: Fp8Backend | None = None,
        spec: QwenReplaySpec = QWEN36_27B_REPLAY_SPEC,
        positions: torch.Tensor | None = None,
        fp8_weight_dtype: torch.dtype = torch.bfloat16,
        ste_policy: str = "identity",
        checkpoint_interval: int = 16,
        require_exact_input: bool = True,
    ) -> None:
        if payload.get("schema_version") != 1:
            raise ValueError("capture payload must use schema version 1")
        tensors = payload.get("tensors")
        if not isinstance(tensors, Mapping):
            raise ValueError("capture payload tensors must be a mapping")
        non_tensors = [
            name for name, value in tensors.items() if not isinstance(value, torch.Tensor)
        ]
        if non_tensors:
            raise TypeError(f"capture values are not tensors: {non_tensors[:4]}")
        if ste_policy not in {"identity", "clipped"}:
            raise ValueError("ste_policy must be 'identity' or 'clipped'")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

        token_ids = payload.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("capture prompt_token_ids must be a non-empty list")
        tokens = len(token_ids)
        if positions is None:
            positions = torch.arange(tokens, dtype=torch.long)
        if positions.dtype != torch.long or positions.ndim not in {1, 2}:
            raise ValueError("positions must be a rank-one or rank-two long tensor")
        if positions.shape[-1] != tokens:
            raise ValueError("positions token dimension does not match the capture")
        if positions.ndim == 2 and positions.shape[0] != 3:
            raise ValueError("rank-two MRoPE positions must have shape [3, tokens]")

        raw_targets = payload.get("target_layers")
        if raw_targets is None:
            targets = tuple(
                layer
                for layer in range(1, len(spec.layer_types))
                if f"h{layer}_post_block" in tensors
            )
        else:
            if not isinstance(raw_targets, list) or not all(
                isinstance(layer, int) for layer in raw_targets
            ):
                raise ValueError("capture target_layers must be an integer list")
            targets = tuple(raw_targets)
        if len(set(targets)) != len(targets):
            raise ValueError("capture target_layers contains duplicates")
        invalid_targets = [
            layer for layer in targets if not 0 <= layer < len(spec.layer_types)
        ]
        if invalid_targets:
            raise ValueError(f"capture target_layers are out of range: {invalid_targets}")
        if not any(layer >= 1 for layer in targets):
            raise ValueError("capture has no replayable layers in 1..N-1")

        self.payload = payload
        self.tensors: Mapping[str, torch.Tensor] = tensors
        self.w4_backend = w4_backend
        self.fp8_backend = (
            DenseDequantFp8Backend(dtype=fp8_weight_dtype)
            if fp8_backend is None
            else fp8_backend
        )
        self.spec = spec
        self.positions = positions.detach().cpu()
        self.fp8_weight_dtype = fp8_weight_dtype
        self.ste_policy = ste_policy
        self.checkpoint_interval = checkpoint_interval
        self.require_exact_input = require_exact_input
        self.target_layers = frozenset(targets)
        self.tokens = tokens

    def layer_type(self, layer: int) -> str:
        self._check_layer_number(layer)
        return self.spec.layer_types[layer]

    def _check_layer_number(self, layer: int) -> None:
        if not isinstance(layer, int) or not 1 <= layer < len(self.spec.layer_types):
            raise ValueError(
                f"replay layer must be in 1..{len(self.spec.layer_types) - 1}"
            )
        if layer not in self.target_layers:
            raise ValueError(f"layer {layer} is outside the capture target set")

    def _common_names(self, layer: int) -> set[str]:
        return {
            f"h{layer - 1}_post_block",
            f"h{layer}_post_block",
            f"linear.layers.{layer}.mlp.gate_up_proj.output",
            f"linear.layers.{layer}.mlp.down_proj.output",
            f"layers.{layer}.mlp.swiglu_output",
            f"replay.norm.layers.{layer}.input_layernorm.weight",
            f"replay.norm.layers.{layer}.post_attention_layernorm.weight",
        }

    def required_tensor_names(self, layer: int) -> set[str]:
        """Return every tensor required to replay one selected block."""

        self._check_layer_number(layer)
        names = self._common_names(layer)
        if self.layer_type(layer) == LINEAR_ATTENTION:
            names.update(
                {
                    f"linear.layers.{layer}.linear_attn.in_proj_qkvz.output",
                    f"linear.layers.{layer}.linear_attn.in_proj_ba.output",
                    f"linear.layers.{layer}.linear_attn.out_proj.output",
                }
            )
            names.update(
                f"gdn.layer{layer}.{suffix}"
                for suffix in (
                    "conv_output_prefill",
                    "q",
                    "k",
                    "v",
                    "log_g",
                    "beta",
                    "initial_state",
                    "chunk_output",
                    "final_state",
                    "core_output",
                )
            )
            names.update(
                f"replay.gdn.layers.{layer}.{suffix}"
                for suffix in (
                    "in_proj_ba.weight",
                    "conv1d.weight",
                    "A_log",
                    "dt_bias",
                    "norm.weight",
                )
            )
            fp8_modules = (
                f"layers.{layer}.linear_attn.in_proj_qkvz",
                f"layers.{layer}.linear_attn.out_proj",
            )
        else:
            names.update(
                {
                    f"linear.layers.{layer}.self_attn.qkv_proj.output",
                    f"linear.layers.{layer}.self_attn.o_proj.output",
                    f"replay.norm.layers.{layer}.self_attn.q_norm.weight",
                    f"replay.norm.layers.{layer}.self_attn.k_norm.weight",
                }
            )
            names.update(
                f"attention.layer{layer}.{suffix}"
                for suffix in ("q_post_rope", "k_post_rope", "v", "core_output")
            )
            fp8_modules = (
                f"layers.{layer}.self_attn.qkv_proj",
                f"layers.{layer}.self_attn.o_proj",
            )
        for module in fp8_modules:
            names.update(
                f"replay.fp8.{module}.{suffix}"
                for suffix in ("weight", "weight_scale", "input_scale")
            )
        return names

    def validate_layer(self, layer: int) -> None:
        missing = sorted(self.required_tensor_names(layer) - set(self.tensors))
        if missing:
            preview = ", ".join(missing[:6])
            suffix = " ..." if len(missing) > 6 else ""
            raise ValueError(
                f"layer {layer} capture is missing {len(missing)} tensors: "
                f"{preview}{suffix}"
            )

    def captured_hidden_states(
        self,
        *,
        first_block: int = 1,
        target_layer: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """Return the hidden-state mapping consumed by reverse replay."""

        target = max(self.target_layers) if target_layer is None else target_layer
        if not 1 <= first_block <= target:
            raise ValueError("invalid first_block/target_layer interval")
        for layer in range(first_block, target + 1):
            self.validate_layer(layer)
        names = {
            layer: f"h{layer}_post_block"
            for layer in range(first_block - 1, target + 1)
        }
        missing = [name for name in names.values() if name not in self.tensors]
        if missing:
            raise ValueError(f"capture is missing hidden states: {missing}")
        return {layer: self.tensors[name] for layer, name in names.items()}

    def _runtime_fp8_linear(
        self,
        module: str,
        inputs: torch.Tensor,
        exact_output: torch.Tensor,
    ) -> torch.Tensor:
        prefix = f"replay.fp8.{module}"
        return self.fp8_backend.exact_linear(
            module,
            inputs,
            exact_output,
            self.tensors[f"{prefix}.weight"],
            self.tensors[f"{prefix}.weight_scale"],
            self.tensors[f"{prefix}.input_scale"],
            ste_policy=self.ste_policy,
        )

    def __call__(self, layer: int, logical_input: torch.Tensor) -> torch.Tensor:
        """Replay one block and inject its exact compiled post-block residual."""

        self.validate_layer(layer)
        if logical_input.ndim != 2 or logical_input.shape != (
            self.tokens,
            self.spec.attention.hidden_size,
        ):
            raise ValueError("logical_input shape does not match the capture geometry")
        device = logical_input.device
        moved: dict[str, torch.Tensor] = {}

        def captured(name: str) -> torch.Tensor:
            if name not in moved:
                moved[name] = self.tensors[name].to(device=device)
            return moved[name]

        exact_input = captured(f"h{layer - 1}_post_block")
        if self.require_exact_input and not torch.equal(
            logical_input.detach(), exact_input
        ):
            difference = logical_input.detach().float() - exact_input.float()
            raise ValueError(
                f"layer {layer} logical input differs from its deployed capture: "
                f"max_abs={float(difference.abs().max())}"
            )

        if self.layer_type(layer) == LINEAR_ATTENTION:
            surrogate = self._replay_gdn(
                layer,
                logical_input,
                captured,
            )
        else:
            surrogate = self._replay_attention(
                layer,
                logical_input,
                captured,
            )
        return exact_value(surrogate, captured(f"h{layer}_post_block"))

    def _replay_gdn(
        self,
        layer: int,
        logical_input: torch.Tensor,
        captured: Any,
    ) -> torch.Tensor:
        layout = self.spec.gdn
        qkvz_module = f"layers.{layer}.linear_attn.in_proj_qkvz"
        out_module = f"layers.{layer}.linear_attn.out_proj"

        qkvz = captured(f"linear.{qkvz_module}.output")
        value_size = layout.value_heads * layout.value_dim
        z = qkvz[..., -value_size:].reshape(
            self.tokens, layout.value_heads, layout.value_dim
        )
        core = captured(f"gdn.layer{layer}.core_output")
        chunk = _squeeze_batch(
            captured(f"gdn.layer{layer}.chunk_output"),
            name=f"gdn.layer{layer}.chunk_output",
        )
        if not torch.equal(core, chunk):
            raise ValueError(f"layer {layer} GDN core/chunk captures differ")
        norm_name = f"gdn.layer{layer}.norm_output"
        if norm_name in self.tensors:
            gated_norm = captured(norm_name).reshape_as(core)
        else:
            gated_norm = gated_rms_norm(
                core,
                z,
                captured(f"replay.gdn.layers.{layer}.norm.weight"),
                layout.norm_eps,
            ).detach()

        gdn_capture = GdnCapture(
            qkvz=qkvz,
            ba=captured(
                f"linear.layers.{layer}.linear_attn.in_proj_ba.output"
            ),
            conv_qkv=captured(f"gdn.layer{layer}.conv_output_prefill"),
            query=_squeeze_batch(
                captured(f"gdn.layer{layer}.q"), name=f"gdn.layer{layer}.q"
            ),
            key=_squeeze_batch(
                captured(f"gdn.layer{layer}.k"), name=f"gdn.layer{layer}.k"
            ),
            value=_squeeze_batch(
                captured(f"gdn.layer{layer}.v"), name=f"gdn.layer{layer}.v"
            ),
            log_decay=_squeeze_batch(
                captured(f"gdn.layer{layer}.log_g"),
                name=f"gdn.layer{layer}.log_g",
            ),
            beta=_squeeze_batch(
                captured(f"gdn.layer{layer}.beta"),
                name=f"gdn.layer{layer}.beta",
            ),
            core_output=core,
            final_state=_squeeze_batch(
                captured(f"gdn.layer{layer}.final_state"),
                name=f"gdn.layer{layer}.final_state",
            ),
            gated_norm=gated_norm,
            branch_output=captured(f"linear.{out_module}.output"),
        )
        weights = GdnWeights(
            qkvz_out_in=None,
            qkvz_input_scale=None,
            ba_out_in=captured(
                f"replay.gdn.layers.{layer}.in_proj_ba.weight"
            ).to(self.fp8_weight_dtype),
            conv=captured(f"replay.gdn.layers.{layer}.conv1d.weight"),
            a_log=captured(f"replay.gdn.layers.{layer}.A_log"),
            dt_bias=captured(f"replay.gdn.layers.{layer}.dt_bias"),
            norm=captured(f"replay.gdn.layers.{layer}.norm.weight"),
            out_out_in=None,
            out_input_scale=None,
        )
        block = replay_qwen_gdn_block(
            logical_input,
            layout,
            weights,
            gdn_capture,
            input_norm_weight=captured(
                f"replay.norm.layers.{layer}.input_layernorm.weight"
            ),
            post_attention_norm_weight=captured(
                f"replay.norm.layers.{layer}.post_attention_layernorm.weight"
            ),
            qkvz_linear=lambda inputs: self._runtime_fp8_linear(
                qkvz_module,
                inputs,
                qkvz,
            ),
            out_linear=lambda inputs: self._runtime_fp8_linear(
                out_module,
                inputs,
                captured(f"linear.{out_module}.output"),
            ),
            gate_up_linear=lambda inputs: self.w4_backend.exact_linear(
                layer,
                "gate_up_proj",
                inputs,
                captured(f"linear.layers.{layer}.mlp.gate_up_proj.output"),
            ),
            down_linear=lambda inputs: self.w4_backend.exact_linear(
                layer,
                "down_proj",
                inputs,
                captured(f"linear.layers.{layer}.mlp.down_proj.output"),
            ),
            exact_swiglu_output=captured(f"layers.{layer}.mlp.swiglu_output"),
            initial_state=_squeeze_batch(
                captured(f"gdn.layer{layer}.initial_state"),
                name=f"gdn.layer{layer}.initial_state",
            ),
            ste_policy=self.ste_policy,
            checkpoint_interval=self.checkpoint_interval,
        )
        return block.output

    def _replay_attention(
        self,
        layer: int,
        logical_input: torch.Tensor,
        captured: Any,
    ) -> torch.Tensor:
        qkv_module = f"layers.{layer}.self_attn.qkv_proj"
        out_module = f"layers.{layer}.self_attn.o_proj"
        linears = QwenBlockLinears(
            qkv=lambda inputs: self._runtime_fp8_linear(
                qkv_module,
                inputs,
                captured(f"linear.{qkv_module}.output"),
            ),
            attention_out=lambda inputs: self._runtime_fp8_linear(
                out_module,
                inputs,
                captured(f"linear.{out_module}.output"),
            ),
            gate_up=lambda inputs: self.w4_backend.exact_linear(
                layer,
                "gate_up_proj",
                inputs,
                captured(f"linear.layers.{layer}.mlp.gate_up_proj.output"),
            ),
            down=lambda inputs: self.w4_backend.exact_linear(
                layer,
                "down_proj",
                inputs,
                captured(f"linear.layers.{layer}.mlp.down_proj.output"),
            ),
        )
        block = replay_qwen_full_attention_suffix(
            logical_input,
            self.positions.to(device=logical_input.device),
            self.spec.attention,
            input_norm_weight=captured(
                f"replay.norm.layers.{layer}.input_layernorm.weight"
            ),
            post_attention_norm_weight=captured(
                f"replay.norm.layers.{layer}.post_attention_layernorm.weight"
            ),
            q_norm_weight=captured(
                f"replay.norm.layers.{layer}.self_attn.q_norm.weight"
            ),
            k_norm_weight=captured(
                f"replay.norm.layers.{layer}.self_attn.k_norm.weight"
            ),
            linears=linears,
            exact_query=captured(f"attention.layer{layer}.q_post_rope"),
            exact_key=captured(f"attention.layer{layer}.k_post_rope"),
            exact_value=captured(f"attention.layer{layer}.v"),
            exact_attention_output=captured(
                f"attention.layer{layer}.core_output"
            ),
            exact_swiglu_output=captured(f"layers.{layer}.mlp.swiglu_output"),
        )
        return block.output

    def reverse_replay_rows(
        self,
        valid_positions: torch.Tensor,
        row_start: int,
        row_stop: int,
        *,
        first_block: int = 1,
        target_layer: int | None = None,
        device: torch.device | str | None = None,
    ) -> Any:
        """Run the fit estimator directly against this captured replay."""

        from fit_jlens_nvfp4_ste import reverse_replay_rows

        target = max(self.target_layers) if target_layer is None else target_layer
        hidden_states = self.captured_hidden_states(
            first_block=first_block,
            target_layer=target,
        )
        return reverse_replay_rows(
            hidden_states,
            self,
            valid_positions,
            row_start,
            row_stop,
            first_block=first_block,
            target_layer=target,
            device=device,
        )


def reverse_replay_captured_rows(
    factory: CapturedQwenBlockReplayFactory,
    valid_positions: torch.Tensor,
    row_start: int,
    row_stop: int,
    *,
    first_block: int = 1,
    target_layer: int | None = None,
    device: torch.device | str | None = None,
) -> Any:
    """Functional bridge to ``fit_jlens_nvfp4_ste.reverse_replay_rows``."""

    return factory.reverse_replay_rows(
        valid_positions,
        row_start,
        row_stop,
        first_block=first_block,
        target_layer=target_layer,
        device=device,
    )
