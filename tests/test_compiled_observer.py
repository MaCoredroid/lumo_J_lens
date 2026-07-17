from __future__ import annotations

import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "probe_compiled_observer.py"
SPEC = importlib.util.spec_from_file_location("probe_compiled_observer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompiledObserverTests(unittest.TestCase):
    def test_same_shaped_alternate_snapshot_cannot_claim_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pinned = (
                root
                / "pinned-cache"
                / "models--nvidia--Qwen3.6-27B-NVFP4"
                / "snapshots"
                / MODULE.MODEL_REVISION
            )
            alternate = (
                root
                / "alternate-cache"
                / "models--nvidia--Qwen3.6-27B-NVFP4"
                / "snapshots"
                / MODULE.MODEL_REVISION
            )
            pinned.mkdir(parents=True)
            alternate.mkdir(parents=True)
            (pinned / "config.json").write_text("{}")
            (alternate / "config.json").write_text("{}")

            with mock.patch.object(
                MODULE, "_download_pinned_model_snapshot", return_value=pinned
            ), mock.patch.object(MODULE, "_validate_pinned_checkpoint") as validate:
                with self.assertRaisesRegex(ValueError, "exact pinned NVIDIA snapshot"):
                    MODULE._resolve_model_path(alternate)
            validate.assert_not_called()

    def test_exact_snapshot_records_strict_modelopt_identity(self) -> None:
        calls: list[tuple[Path, bool]] = []

        class FakeCheckpoint:
            def __init__(self, snapshot: Path, *, strict_pinned: bool) -> None:
                calls.append((snapshot, strict_pinned))

        with tempfile.TemporaryDirectory() as temp:
            pinned = Path(temp).resolve() / MODULE.MODEL_REVISION
            pinned.mkdir()
            with mock.patch.object(
                MODULE, "_download_pinned_model_snapshot", return_value=pinned
            ), mock.patch.object(
                MODULE,
                "_metadata_identity",
                return_value=dict(MODULE.PINNED_METADATA_SHA256),
            ), mock.patch.object(
                MODULE,
                "_modelopt_checkpoint_api",
                return_value=(
                    FakeCheckpoint,
                    MODULE.MODEL_REVISION,
                    dict(MODULE.PINNED_METADATA_SHA256),
                ),
            ):
                path, identity = MODULE._resolve_model_path(pinned)

        self.assertEqual(path, pinned)
        self.assertEqual(calls, [(pinned, True)])
        self.assertEqual(identity["revision"], MODULE.MODEL_REVISION)
        self.assertEqual(identity["resolved_path"], str(pinned))
        self.assertTrue(identity["strict_pinned_validation"])
        self.assertEqual(
            identity["validator"], "ModelOptCheckpoint(strict_pinned=True)"
        )

    def test_modelopt_revision_drift_fails_before_revision_is_recorded(self) -> None:
        constructed = False

        class FakeCheckpoint:
            def __init__(self, snapshot: Path, *, strict_pinned: bool) -> None:
                nonlocal constructed
                constructed = True

        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            with mock.patch.object(
                MODULE,
                "_metadata_identity",
                return_value=dict(MODULE.PINNED_METADATA_SHA256),
            ), mock.patch.object(
                MODULE,
                "_modelopt_checkpoint_api",
                return_value=(
                    FakeCheckpoint,
                    "alternate-revision",
                    dict(MODULE.PINNED_METADATA_SHA256),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "revisions diverged"):
                    MODULE._validate_pinned_checkpoint(snapshot)
        self.assertFalse(constructed)

    def test_modified_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            (snapshot / "config.json").write_bytes(b"same shape, different identity")
            expected = hashlib.sha256(b"pinned config").hexdigest()
            with mock.patch.object(
                MODULE, "PINNED_METADATA_SHA256", {"config.json": expected}
            ):
                with self.assertRaisesRegex(ValueError, "metadata SHA-256 mismatch"):
                    MODULE._metadata_identity(snapshot)

    def test_only_selected_linear_prefixes_are_canonicalized(self) -> None:
        self.assertEqual(
            MODULE._canonical_linear_name(
                "model.language_model.model.layers.62.linear_attn.in_proj_qkvz"
            ),
            "layers.62.linear_attn.in_proj_qkvz",
        )
        self.assertEqual(
            MODULE._canonical_linear_name("model.layers.63.mlp.down_proj"),
            "layers.63.mlp.down_proj",
        )
        self.assertIsNone(
            MODULE._canonical_linear_name("model.layers.62.linear_attn.conv1d")
        )
        self.assertIsNone(
            MODULE._canonical_linear_name("model.layers.60.mlp.down_proj")
        )

    def test_observer_copy_is_capacity_bounded_and_nonintrusive(self) -> None:
        source = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
        original = source.clone()
        destination = torch.full((3, 4), float("nan"), dtype=torch.bfloat16)
        count = torch.zeros((), dtype=torch.int32)

        result = MODULE._observe_impl(source, destination, count)

        self.assertIsNone(result)
        torch.testing.assert_close(source, original, rtol=0, atol=0)
        torch.testing.assert_close(destination, source[:3], rtol=0, atol=0)
        self.assertEqual(count.item(), 5)

    def test_endpoint_parity_requires_exact_reported_logprobs(self) -> None:
        generation = {
            "prompt_token_ids": [1, 2],
            "generated_token_id": 3,
            "generated_text": "x",
            "finish_reason": "length",
            "top_logprobs": [
                {"token_id": 3, "logprob": -0.25, "rank": 1, "decoded_token": "x"}
            ],
        }
        exact = MODULE._endpoint_parity(
            {"generation": generation}, {"generation": dict(generation)}
        )
        self.assertTrue(exact["exact"])

        changed_generation = {
            **generation,
            "top_logprobs": [
                {"token_id": 3, "logprob": -0.2501, "rank": 1, "decoded_token": "x"}
            ],
        }
        changed = MODULE._endpoint_parity(
            {"generation": generation}, {"generation": changed_generation}
        )
        self.assertFalse(changed["exact"])
        self.assertFalse(changed["top_logprobs_exact"])

    def test_expected_linear_boundary_count(self) -> None:
        self.assertEqual(len(MODULE.EXPECTED_LINEAR_NAMES), 14)
        self.assertEqual(len(set(MODULE.EXPECTED_LINEAR_NAMES)), 14)

    def test_observer_scopes_select_expected_boundaries(self) -> None:
        linears, h_layers = MODULE._parse_observer_scope("h-only")
        self.assertEqual(linears, set())
        self.assertEqual(h_layers, {61, 62, 63})

        name = "layers.62.linear_attn.in_proj_qkvz"
        linears, h_layers = MODULE._parse_observer_scope(f"linear:{name}")
        self.assertEqual(linears, {name})
        self.assertEqual(h_layers, set())

        linears, h_layers = MODULE._parse_observer_scope("h:63")
        self.assertEqual(linears, set())
        self.assertEqual(h_layers, {63})

        with self.assertRaisesRegex(ValueError, "unknown observer linear"):
            MODULE._parse_observer_scope("linear:layers.62.linear_attn.conv1d")

    def test_debug_scan_counts_unique_pre_grad_observer_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graph = root / "model.BEFORE_PRE_GRAD.7.py"
            graph.write_text(
                "a = torch.ops.vllm.lumo_compiled_observe_v1.default(x, b, c)\n"
                "d = torch.ops.vllm.lumo_compiled_observe_v1.default(y, e, f)\n"
                "g = torch.ops._C.marlin_gemm(x)\n"
            )
            # A later compiler snapshot must not inflate the pre-grad node count.
            (root / "model.AFTER_POST_GRAD.7.py").write_text(graph.read_text())

            scan = MODULE._scan_debug_dump(root)

        self.assertEqual(scan["pre_grad_fx_file_count"], 1)
        self.assertEqual(scan["pre_grad_fx_node_occurrences"]["observer"], 2)
        self.assertEqual(scan["pre_grad_fx_node_occurrences"]["marlin_gemm"], 1)


if __name__ == "__main__":
    unittest.main()
