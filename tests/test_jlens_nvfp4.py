#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_jlens_nvfp4", SCRIPTS / "run_jlens_nvfp4.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class JacobianLensHelpersTest(unittest.TestCase):
    def test_parse_all_layers(self):
        self.assertEqual(MODULE.parse_integer_list("all", allow_all=True), list(range(63)))

    def test_parse_integer_list_rejects_duplicates(self):
        with self.assertRaises(Exception):
            MODULE.parse_integer_list("1,2,1")

    def test_validate_layers_sorts_and_bounds_checks(self):
        self.assertEqual(MODULE.validate_layers([62, 0, 31]), [0, 31, 62])
        with self.assertRaises(ValueError):
            MODULE.validate_layers([63])

    def test_resolve_negative_positions(self):
        self.assertEqual(MODULE.resolve_positions([-1, -3, 0], 8), [7, 5, 0])

    def test_resolve_positions_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            MODULE.resolve_positions([-9], 8)

    def test_target_tokens_use_teacher_forcing_and_generation(self):
        self.assertEqual(
            MODULE.target_token_ids_for_positions([10, 20, 30], [0, 2], 99),
            (20, 99),
        )

    def test_target_tokens_require_final_position(self):
        with self.assertRaises(ValueError):
            MODULE.target_token_ids_for_positions([10, 20, 30], [0, 1], 99)

    def test_capture_positions_add_final_implicitly(self):
        self.assertEqual(MODULE.capture_positions_with_final([1, 3], 6), [1, 3, 5])

    def test_capture_positions_do_not_duplicate_final(self):
        self.assertEqual(MODULE.capture_positions_with_final([1, 5], 6), [1, 5])

    def test_post_block_reconstructs_branch_plus_residual(self):
        self.assertEqual(MODULE.reconstruct_post_block((7, 11)), 18)

    def test_transport_uses_transposed_jacobian(self):
        transpose_marker = object()

        class Jacobian:
            @property
            def T(self):
                return transpose_marker

        class Residual:
            def __matmul__(self, operand):
                self.operand = operand
                return "transported"

        residual = Residual()
        self.assertEqual(MODULE.transport_residual(residual, Jacobian()), "transported")
        self.assertIs(residual.operand, transpose_marker)

    def test_default_prompt_matches_reference(self):
        self.assertEqual(
            MODULE.DEFAULT_PROMPT,
            "Fact: The currency used in the country shaped like a boot is",
        )


if __name__ == "__main__":
    unittest.main()
