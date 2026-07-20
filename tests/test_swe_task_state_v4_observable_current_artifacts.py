from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "swe_task_state_v4_observable_current_artifacts.py"
)
spec = importlib.util.spec_from_file_location(
    "swe_task_state_v4_observable_current_artifacts", MODULE_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ObservableCurrentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = module.validate_config(
            module.DECODER.load_json(module.CONFIG_PATH)
        )

    def test_current_routes_only_to_v2f_and_nested_v2(self) -> None:
        current = self.config["current"]
        self.assertTrue(
            current["feature_bundle"]["manifest"]["path"].endswith("-v2f.json")
        )
        self.assertTrue(
            current["feature_bundle"]["data"]["path"].endswith("-v2f.npz")
        )
        self.assertTrue(current["decoder_report"]["path"].endswith("-v2f.json"))
        self.assertTrue(current["nested_inference"]["path"].endswith("-v2.json"))
        self.assertEqual(
            [item["artifact_id"] for item in self.config["historical_superseded"]],
            module.HISTORICAL_IDS,
        )
        self.assertTrue(
            all(
                item["status"] == "historical_superseded"
                for item in self.config["historical_superseded"]
            )
        )
        self.assertTrue(
            self.config["routing_policy"][
                "current_consumers_must_use_only_current_records"
            ]
        )
        self.assertFalse(
            self.config["routing_policy"]["historical_artifacts_deleted"]
        )

    def test_all_records_authenticate_and_current_chain_is_exact(self) -> None:
        records = module._flatten_records(self.config)
        bindings = module.authenticate_records(records)
        self.assertEqual(len(bindings), 17)
        module.validate_current_chain(self.config)

    def test_historical_entry_cannot_be_promoted_or_used_as_fallback(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["historical_superseded"][0]["status"] = "current"
        with self.assertRaisesRegex(module.ArtifactIndexError, "routing changed"):
            module.validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["routing_policy"]["historical_records_are_not_current_fallbacks"] = False
        with self.assertRaisesRegex(module.ArtifactIndexError, "routing policy"):
            module.validate_config(changed)

    def test_index_writer_is_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.json"
            module._write_no_clobber(output, {"first": True}, before_publish=lambda: None)
            with self.assertRaisesRegex(module.ArtifactIndexError, "output exists"):
                module._write_no_clobber(
                    output, {"second": True}, before_publish=lambda: None
                )
            self.assertEqual(module.DECODER.load_json(output), {"first": True})


if __name__ == "__main__":
    unittest.main()
