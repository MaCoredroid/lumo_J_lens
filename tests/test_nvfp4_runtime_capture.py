from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "check_nvfp4_runtime_capture.py"
SPEC = importlib.util.spec_from_file_location("check_nvfp4_runtime_capture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def qwen36_text_config() -> dict[str, object]:
    return {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "layer_types": [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(64)
        ],
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }


class TargetLayerTests(unittest.TestCase):
    def test_cli_defaults_to_late(self) -> None:
        with mock.patch("sys.argv", [str(MODULE_PATH)]):
            args = MODULE._parse_args()
        self.assertEqual(args.target_layers, "late")
        self.assertFalse(args.preflight_only)

    def test_late_is_default_proven_profile(self) -> None:
        self.assertEqual(MODULE._resolve_target_layers("late", 64), (61, 62, 63))

    def test_all_selects_every_main_model_layer(self) -> None:
        self.assertEqual(MODULE._resolve_target_layers("all", 64), tuple(range(64)))

    def test_partition_matches_qwen36_hybrid_schedule(self) -> None:
        layer_types = qwen36_text_config()["layer_types"]
        gdn, full = MODULE._partition_target_layers(tuple(range(64)), layer_types)
        self.assertEqual(len(gdn), 48)
        self.assertEqual(len(full), 16)
        self.assertEqual(full, tuple(range(3, 64, 4)))

    def test_prefix_selection_does_not_confuse_layer_6_and_61(self) -> None:
        self.assertTrue(MODULE._is_target_prefix("model.layers.61.mlp", (61,)))
        self.assertFalse(MODULE._is_target_prefix("model.layers.6.mlp", (61,)))

    def test_isolated_child_keeps_all_layer_profile(self) -> None:
        args = Namespace(
            prompt="p",
            prompt_manifest=None,
            prompt_index=0,
            gpu_memory_utilization=0.8,
            max_model_len=32768,
            max_num_batched_tokens=4096,
            max_num_seqs=2,
            capture_capacity=128,
            target_layers="all",
            model_path=None,
            no_weight_digests=False,
            total_gpu_memory_gib=None,
            allow_unsafe_capture_memory=False,
        )
        command = MODULE._child_command(
            args, "compiled-observer", Path("capture.json"), Path("capture.pt")
        )
        target_position = command.index("--target-layers")
        self.assertEqual(command[target_position + 1], "all")


class ModelIdentityTests(unittest.TestCase):
    def test_same_shaped_alternate_snapshot_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = (
                root
                / "cache"
                / "models--nvidia--Qwen3.6-27B-NVFP4"
                / "snapshots"
                / MODULE.MODEL_REVISION
            )
            alternate = (
                root
                / "alternate"
                / "models--nvidia--Qwen3.6-27B-NVFP4"
                / "snapshots"
                / MODULE.MODEL_REVISION
            )
            pinned.mkdir(parents=True)
            alternate.mkdir(parents=True)
            config = json.dumps({"text_config": qwen36_text_config()})
            (pinned / "config.json").write_text(config)
            (alternate / "config.json").write_text(config)

            with mock.patch.object(
                MODULE, "_download_pinned_model_snapshot", return_value=pinned
            ), mock.patch.object(MODULE, "_validate_pinned_checkpoint") as validate:
                with self.assertRaisesRegex(ValueError, "exact pinned NVIDIA snapshot"):
                    MODULE._resolve_model_path(alternate)
            validate.assert_not_called()

    def test_exact_snapshot_runs_strict_validation_and_returns_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pinned = Path(temporary) / MODULE.MODEL_REVISION
            pinned.mkdir()
            identity = {
                "policy": MODULE.MODEL_IDENTITY_POLICY,
                "repo_id": MODULE.MODEL_REPO,
                "revision": MODULE.MODEL_REVISION,
                "resolved_path": str(pinned.resolve()),
                "metadata_sha256": MODULE.PINNED_METADATA_SHA256,
                "strict_pinned_validation": True,
                "validator": "ModelOptCheckpoint(strict_pinned=True)",
            }
            with mock.patch.object(
                MODULE,
                "_download_pinned_model_snapshot",
                return_value=pinned.resolve(),
            ), mock.patch.object(
                MODULE, "_validate_pinned_checkpoint", return_value=identity
            ) as validate:
                path, provenance = MODULE._resolve_model_path(pinned)

            self.assertEqual(path, pinned.resolve())
            self.assertEqual(provenance, identity)
            validate.assert_called_once_with(pinned.resolve())

    def test_metadata_identity_rejects_modified_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "config.json").write_bytes(b"same-shaped but modified")
            expected = hashlib.sha256(b"pinned metadata").hexdigest()
            with mock.patch.object(
                MODULE, "PINNED_METADATA_SHA256", {"config.json": expected}
            ):
                with self.assertRaisesRegex(ValueError, "metadata SHA-256 mismatch"):
                    MODULE._metadata_identity(snapshot)

    def test_preflight_carries_validated_model_identity(self) -> None:
        identity = {
            "policy": MODULE.MODEL_IDENTITY_POLICY,
            "strict_pinned_validation": True,
        }
        args = Namespace(
            target_layers="all",
            capture_capacity=128,
            gpu_memory_utilization=0.8,
            total_gpu_memory_gib=33_635_434_496 / 1024**3,
        )
        _, estimate = MODULE._build_preflight(
            args,
            model_path=Path("/pinned") / MODULE.MODEL_REVISION,
            model_identity=identity,
            text_config=qwen36_text_config(),
        )
        self.assertEqual(estimate["model_identity"], identity)


class FrozenPromptTests(unittest.TestCase):
    def test_frozen_prompt_preserves_declared_truncated_ids(self) -> None:
        text = "a deliberately longer prompt"
        entry = {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids": [10, 20, 30],
            "token_count": 3,
            "row_index": 7,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            path.write_text(json.dumps({"schema_version": 1, "prompts": [entry]}))
            actual = MODULE._load_frozen_prompt(path, 0)
        self.assertEqual(actual["token_ids"], [10, 20, 30])
        self.assertEqual(actual["row_index"], 7)
        self.assertEqual(len(actual["manifest_sha256"]), 64)

    def test_frozen_prompt_rejects_text_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "prompts": [
                            {
                                "text": "x",
                                "text_sha256": "0" * 64,
                                "token_ids": [1],
                                "token_count": 1,
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "text SHA-256"):
                MODULE._load_frozen_prompt(path, 0)


class CaptureContractTests(unittest.TestCase):
    def test_geometry_matches_deployed_projection_widths(self) -> None:
        geometry = MODULE._capture_geometry(qwen36_text_config())
        self.assertEqual(geometry["full_q"], 6144)
        self.assertEqual(geometry["full_kv"], 1024)
        self.assertEqual(geometry["full_qkv"], 14336)
        self.assertEqual(geometry["gdn_mixed"], 10240)
        self.assertEqual(geometry["gdn_qkvz"], 16384)
        self.assertEqual(geometry["gdn_ba"], 96)

    def test_required_names_expand_across_all_layers(self) -> None:
        config = qwen36_text_config()
        late = MODULE._required_capture_names((61, 62, 63), config["layer_types"])
        all_layers = MODULE._required_capture_names(
            tuple(range(64)), config["layer_types"]
        )
        self.assertEqual(len(late), 30)
        self.assertEqual(len(all_layers), 672)
        self.assertIn("gdn.layer0.conv_output_prefill", all_layers)
        self.assertIn("attention.layer3.q_post_rope", all_layers)
        self.assertIn("h63_post_block", all_layers)
        self.assertIn("layers.0.mlp.swiglu_output", all_layers)

    def test_nested_text_config_is_loaded(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            expected = qwen36_text_config()
            (path / "config.json").write_text(json.dumps({"text_config": expected}))
            self.assertEqual(MODULE._load_text_config(path), expected)


class MemoryPreflightTests(unittest.TestCase):
    TOTAL_5090_BYTES = 33_635_434_496

    def estimate(self, *, profile: str, capacity: int) -> dict[str, object]:
        config = qwen36_text_config()
        return MODULE._estimate_capture_memory(
            text_config=config,
            target_layers=MODULE._resolve_target_layers(profile, 64),
            capture_capacity=capacity,
            gpu_memory_utilization=0.80,
            total_gpu_bytes=self.TOTAL_5090_BYTES,
        )

    def test_all_layer_capacity_128_fits_5090_with_margin(self) -> None:
        estimate = self.estimate(profile="all", capacity=128)
        self.assertTrue(estimate["safe"])
        self.assertEqual(estimate["gdn_layer_count"], 48)
        self.assertEqual(estimate["full_attention_layer_count"], 16)
        self.assertEqual(estimate["compile_visible_observer_buffer_count"], 480)
        self.assertEqual(estimate["postload_capture_buffer_count"], 1568)
        self.assertEqual(estimate["replay_parameter_tensor_count"], 785)
        self.assertEqual(estimate["compile_visible_observer_gpu_bytes"], 1_494_353_792)
        self.assertEqual(estimate["postload_capture_gpu_bytes"], 3_104_315_520)
        self.assertEqual(estimate["replay_parameter_host_bytes"], 7_266_685_440)
        self.assertEqual(estimate["empirical_runtime_overhead_bytes"], 1536 * 1024**2)
        self.assertGreater(estimate["estimated_gpu_headroom_bytes"], 1024**3)

    def test_large_capacity_is_rejected(self) -> None:
        estimate = self.estimate(profile="all", capacity=512)
        self.assertFalse(estimate["safe"])
        self.assertLess(estimate["estimated_gpu_headroom_bytes"], 0)

    def test_all_layer_buffers_exceed_late_profile(self) -> None:
        late = self.estimate(profile="late", capacity=128)
        all_layers = self.estimate(profile="all", capacity=128)
        self.assertEqual(late["replay_parameter_tensor_count"], 37)
        for key in (
            "compile_visible_observer_gpu_bytes",
            "postload_capture_gpu_bytes",
            "replay_parameter_host_bytes",
        ):
            self.assertGreater(all_layers[key], late[key])


if __name__ == "__main__":
    unittest.main()
