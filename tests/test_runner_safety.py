import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_swe_verified import _safe_task_dir, _validated_commit  # noqa: E402


class RunnerSafetyTests(unittest.TestCase):
    def test_official_instance_id_stays_below_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tasks"
            self.assertEqual(_safe_task_dir(root, "sympy__sympy-13480").parent, root.resolve())

    def test_path_traversal_instance_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _safe_task_dir(Path(tmp), "../../outside")

    def test_absolute_instance_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _safe_task_dir(Path(tmp), "/tmp/other")

    def test_sha_commit_is_accepted(self):
        commit = "f57fe3f4b3f2cab225749e1b3b38ae1bf80b62f0"
        self.assertEqual(_validated_commit(commit), commit)

    def test_shell_expression_commit_is_rejected(self):
        with self.assertRaises(ValueError):
            _validated_commit("HEAD; touch /tmp/bad")
