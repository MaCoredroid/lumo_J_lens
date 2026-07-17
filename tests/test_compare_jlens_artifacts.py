#!/usr/bin/env python3
"""Focused tests for streamed Jacobian Lens matrix comparison."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
if torch is not None:
    SPEC = importlib.util.spec_from_file_location(
        "compare_jlens_artifacts", SCRIPTS / "compare_jlens_artifacts.py"
    )
    assert SPEC and SPEC.loader
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)
else:
    MODULE = None


@unittest.skipIf(torch is None, "torch is installed in .venv-fit")
class LayerComparisonTest(unittest.TestCase):
    def test_identical_matrix_metrics(self) -> None:
        matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        record, totals = MODULE.compare_layer_matrices(
            matrix, matrix, layer=0, d_model=2, row_chunk=1
        )
        self.assertEqual(record["relative_frobenius_difference"], 0.0)
        self.assertAlmostEqual(record["frobenius_cosine"], 1.0)
        self.assertAlmostEqual(record["local"]["trace"], 5.0)
        self.assertAlmostEqual(record["local"]["frobenius_norm"], math.sqrt(15))
        self.assertAlmostEqual(
            record["local"]["best_scalar_identity"]["scalar"], 2.5
        )
        self.assertEqual(record["row_wise_cosine"]["quantiles"]["p050"], 1.0)
        self.assertEqual(totals.difference_norm_squared, 0.0)

    def test_known_difference_and_row_cosines(self) -> None:
        local = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
        public = torch.eye(2)
        record, _ = MODULE.compare_layer_matrices(
            local, public, layer=4, d_model=2, row_chunk=2
        )
        self.assertAlmostEqual(record["frobenius_cosine"], -0.5)
        self.assertAlmostEqual(record["relative_frobenius_difference"], math.sqrt(3))
        row = record["row_wise_cosine"]
        self.assertAlmostEqual(row["quantiles"]["p000"], -1.0)
        self.assertAlmostEqual(row["quantiles"]["p050"], -0.5)
        self.assertAlmostEqual(row["quantiles"]["p100"], 0.0)

    def test_nonfinite_and_zero_rows_fail_closed(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            MODULE.compare_layer_matrices(
                torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
                torch.eye(2),
                layer=0,
                d_model=2,
                row_chunk=1,
            )
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            MODULE.compare_layer_matrices(
                torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                torch.eye(2),
                layer=0,
                d_model=2,
                row_chunk=1,
            )


@unittest.skipIf(torch is None, "torch is installed in .venv-fit")
class ArtifactComparisonTest(unittest.TestCase):
    def write_checkpoint(
        self, path: Path, matrices: dict[int, torch.Tensor]
    ) -> None:
        torch.save(
            {
                "J": matrices,
                "d_model": 2,
                "n_prompts": 1,
                "source_layers": [0, 1],
            },
            path,
        )

    def test_tiny_memory_mapped_artifacts_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public.pt"
            local = root / "local.pt"
            self.write_checkpoint(public, {0: torch.eye(2), 1: torch.eye(2)})
            self.write_checkpoint(
                local, {0: torch.eye(2), 1: 2.0 * torch.eye(2)}
            )
            layers, aggregate = MODULE.compare_artifact_matrices(
                local,
                public,
                d_model=2,
                source_layers=(0, 1),
                row_chunk=1,
            )

        self.assertEqual([record["layer"] for record in layers], [0, 1])
        self.assertEqual(layers[0]["relative_frobenius_difference"], 0.0)
        self.assertEqual(layers[1]["relative_frobenius_difference"], 1.0)
        self.assertAlmostEqual(
            aggregate["global_relative_frobenius_difference"], math.sqrt(0.5)
        )
        self.assertAlmostEqual(aggregate["global_frobenius_cosine"], 3 / math.sqrt(10))
        self.assertEqual(aggregate["all_rows_cosine"]["count"], 4)

    def test_checkpoint_layer_contract_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.pt"
            second = root / "second.pt"
            torch.save(
                {"J": {0: torch.eye(2)}, "d_model": 2, "n_prompts": 1},
                first,
            )
            self.write_checkpoint(second, {0: torch.eye(2), 1: torch.eye(2)})
            with self.assertRaisesRegex(ValueError, "source layers mismatch"):
                MODULE.compare_artifact_matrices(
                    first,
                    second,
                    d_model=2,
                    source_layers=(0, 1),
                    row_chunk=1,
                )

    def test_artifact_verifiers_are_mandatory(self) -> None:
        calls = []

        def local_verifier(path, **kwargs):
            calls.append(("local", path, kwargs))
            return {"sha256": "a" * 64, "n_prompts": 10}

        def public_file_verifier(path):
            calls.append(("public_file", path, {}))
            return {"sha256": "b" * 64, "size_bytes": 123}

        def public_checkpoint_verifier(path, **kwargs):
            calls.append(("public_checkpoint", path, kwargs))
            return {"d_model": 5120, "source_layers": list(range(63))}

        records = MODULE.verify_artifacts(
            local_path=Path("local.pt"),
            local_sha256="a" * 64,
            local_provenance=Path("local.provenance.json"),
            public_path=Path("public.pt"),
            local_verifier=local_verifier,
            public_file_verifier=public_file_verifier,
            public_checkpoint_verifier=public_checkpoint_verifier,
        )
        self.assertEqual(records["local"]["sha256"], "a" * 64)
        self.assertEqual(records["public"]["sha256"], "b" * 64)
        self.assertEqual(calls[0][2]["expected_sha256"], "a" * 64)
        self.assertTrue(calls[0][2]["check_finite"])
        self.assertFalse(calls[2][2]["check_finite"])

    def test_native_artifact_verifier_is_selected_explicitly(self) -> None:
        calls = []

        def reject_nf4(*args, **kwargs):
            self.fail("NF4 verifier must not inspect a native NVFP4 STE artifact")

        def native_verifier(path, **kwargs):
            calls.append((path, kwargs))
            return {"kind": "native_nvfp4_ste_fit", "sha256": "c" * 64}

        records = MODULE.verify_artifacts(
            local_path=Path("native.pt"),
            local_sha256="c" * 64,
            local_provenance=Path("native.final.json"),
            local_state=Path("state.json"),
            local_state_sha256="e" * 64,
            public_path=Path("public.pt"),
            local_kind="nvfp4-ste",
            nf4_verifier=reject_nf4,
            native_verifier=native_verifier,
            public_file_verifier=lambda path: {
                "sha256": "d" * 64,
                "size_bytes": 123,
            },
            public_checkpoint_verifier=lambda path, **kwargs: {
                "d_model": 5120,
                "source_layers": list(range(63)),
            },
        )

        self.assertEqual(records["local"]["kind"], "native_nvfp4_ste_fit")
        self.assertEqual(calls[0][0], Path("native.pt"))
        self.assertEqual(calls[0][1]["expected_sha256"], "c" * 64)
        self.assertEqual(
            calls[0][1]["provenance_path"], Path("native.final.json")
        )
        self.assertEqual(calls[0][1]["state_path"], Path("state.json"))
        self.assertEqual(calls[0][1]["expected_state_sha256"], "e" * 64)
        self.assertTrue(calls[0][1]["check_finite"])

    def test_native_verifier_requires_exact_fit_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires local fit state"):
            MODULE.verify_artifacts(
                local_path=Path("native.pt"),
                local_sha256="c" * 64,
                local_provenance=Path("native.final.json"),
                public_path=Path("public.pt"),
                local_kind="nvfp4-ste",
            )

    def test_geometry_can_read_a_held_inode_after_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local.pt"
            replacement = root / "replacement.pt"
            public = root / "public.pt"
            matrices = {0: torch.eye(2), 1: torch.eye(2) * 2}
            self.write_checkpoint(local, matrices)
            self.write_checkpoint(public, matrices)
            self.write_checkpoint(
                replacement, {0: torch.eye(2) * 10, 1: torch.eye(2) * 20}
            )
            descriptor = os.open(local, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                replacement.replace(local)
                _layers, aggregate = MODULE.compare_artifact_matrices(
                    Path(f"/proc/self/fd/{descriptor}"),
                    public,
                    d_model=2,
                    source_layers=(0, 1),
                    row_chunk=1,
                )
            finally:
                os.close(descriptor)
            self.assertEqual(aggregate["global_relative_frobenius_difference"], 0.0)

    def test_public_verification_and_geometry_share_held_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local.pt"
            public = root / "public.pt"
            replacement = root / "replacement.pt"
            matrices = {0: torch.eye(2), 1: torch.eye(2) * 2}
            self.write_checkpoint(local, matrices)
            self.write_checkpoint(public, matrices)
            self.write_checkpoint(
                replacement, {0: torch.eye(2) * 10, 1: torch.eye(2) * 20}
            )
            verifier_paths = []

            def public_file_verifier(path):
                verifier_paths.append(path)
                replacement.replace(public)
                return {"sha256": "b" * 64, "size_bytes": path.stat().st_size}

            def public_checkpoint_verifier(path, **_kwargs):
                verifier_paths.append(path)
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=True, mmap=True
                )
                self.assertTrue(torch.equal(checkpoint["J"][0], matrices[0]))
                return {"d_model": 2, "source_layers": [0, 1]}

            with MODULE.open_held_regular_file(
                public,
                label="public lens checkpoint",
                expected_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),
            ) as held:
                public_fd_path = Path(held.fd_path)
                records = MODULE.verify_artifacts(
                    local_path=local,
                    local_sha256="a" * 64,
                    local_provenance=root / "local.provenance.json",
                    public_path=public_fd_path,
                    local_verifier=lambda *_args, **_kwargs: {"sha256": "a" * 64},
                    public_file_verifier=public_file_verifier,
                    public_checkpoint_verifier=public_checkpoint_verifier,
                )
                _layers, aggregate = MODULE.compare_artifact_matrices(
                    local,
                    public_fd_path,
                    d_model=2,
                    source_layers=(0, 1),
                    row_chunk=1,
                )
                with self.assertRaisesRegex(
                    ValueError, "changed while it was hashed|content changed"
                ):
                    held.require_unchanged()

            self.assertEqual(records["public"]["sha256"], "b" * 64)
            self.assertEqual(verifier_paths, [public_fd_path, public_fd_path])
            self.assertEqual(aggregate["global_relative_frobenius_difference"], 0.0)

    def test_public_symlink_is_rejected_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public.pt"
            link = root / "public-link.pt"
            self.write_checkpoint(public, {0: torch.eye(2), 1: torch.eye(2)})
            link.symlink_to(public)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                MODULE.open_held_regular_file(
                    link,
                    label="public lens checkpoint",
                    expected_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),
                )

    def test_public_in_place_mutation_is_rejected_with_restored_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "public.pt"
            self.write_checkpoint(path, {0: torch.eye(2), 1: torch.eye(2)})
            expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            before = path.stat()
            with MODULE.open_held_regular_file(
                path,
                label="public lens checkpoint",
                expected_sha256=expected_sha256,
            ) as held:
                with path.open("r+b") as handle:
                    original = handle.read(1)
                    self.assertEqual(len(original), 1)
                    handle.seek(0)
                    handle.write(bytes([original[0] ^ 1]))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.utime(
                    path,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                )
                with self.assertRaisesRegex(
                    ValueError, "changed while it was hashed|content changed"
                ):
                    held.require_unchanged()

    def test_atomic_write_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"
            output.parent.mkdir(parents=True)
            output.write_text("stale", encoding="utf-8")
            payload = {"schema_version": 1, "finite": 1.25}
            MODULE.atomic_write_json(output, payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*")), [])

    def test_cli_parser_accepts_native_exact_state_arguments(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "--local-path",
                "lens.pt",
                "--local-sha256",
                "a" * 64,
                "--local-provenance",
                "metadata.json",
                "--local-state",
                "state.json",
                "--local-state-sha256",
                "b" * 64,
                "--local-kind",
                "nvfp4-ste",
                "--output",
                "comparison.json",
            ]
        )
        self.assertEqual(args.local_kind, "nvfp4-ste")
        self.assertEqual(args.local_state, Path("state.json"))
        self.assertEqual(args.local_state_sha256, "b" * 64)


if __name__ == "__main__":
    unittest.main()
