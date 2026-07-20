#!/usr/bin/env python3

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import swe_task_state_v4_extract as extract
from tests import test_analyze_swe_task_state_v3 as v3_fixtures


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/swe_task_state_interpreter_v3.json"
ACTION_PROTOCOL = ROOT / "configs/swe_task_state_v3_action_probes.json"


def replay_fixture(actions: list[str | None]) -> tuple[list[dict], dict]:
    declared = list(range(1, len(actions) + 1))
    prompts = [
        v3_fixtures.prompt(
            f"p{index}", index, action, declared=declared
        )
        for index, action in enumerate(actions, start=1)
    ]
    for index, row in enumerate(prompts, start=1):
        row["text"] = f"prompt {index}"
        row["token_ids"] = [10, 20 + index]
        row["score_token_ids"] = [1, 2, 3]
        row["metadata"]["cohort"] = {"id": "development_a"}
        row["metadata"]["selection"]["checkpoint_ordinal"] = index
    experiments = [
        {
            "id": row["id"],
            "prompt": row["text"],
            "prompt_token_ids": row["token_ids"],
            "metadata": row["metadata"],
            "capture_positions_resolved": [1],
            "scored_vocabulary": {"token_ids": row["score_token_ids"]},
        }
        for row in prompts
    ]
    return prompts, {"experiments": experiments}


def raw_sensor(row_id: str, method: str) -> np.ndarray:
    row_number = int(row_id.removeprefix("p"))
    method_offset = 100.0 if method == "public_jacobian" else -100.0
    return (
        np.arange(extract.RAW_SENSOR_WIDTH, dtype=np.float64) * row_number
        + method_offset
        + row_number**2
    )


@contextmanager
def stub_v3_helpers(*, unstable_ids: set[str] | None = None):
    unstable = set() if unstable_ids is None else set(unstable_ids)

    def numerical_stability(experiment, _protocol):
        is_unstable = str(experiment["id"]) in unstable
        return (not is_unstable, ["fixture_unstable"] if is_unstable else [])

    def layer_features(experiment, *, method, **_kwargs):
        return raw_sensor(str(experiment["id"]), method).tolist()

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                extract.V3.HISTORICAL_V1,
                "_validate_report_provenance",
                lambda *_args, **_kwargs: None,
            )
        )
        stack.enter_context(
            mock.patch.object(
                extract.V3.HISTORICAL_V1,
                "_numerically_stable",
                numerical_stability,
            )
        )
        stack.enter_context(
            mock.patch.object(
                extract.V3.HISTORICAL_V1,
                "_layer_class_features",
                layer_features,
            )
        )
        yield


class TestSweTaskStateV4Extract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = extract.V3.validate_protocol(
            json.loads(PROTOCOL.read_text(encoding="utf-8")),
            action_protocol_value=json.loads(
                ACTION_PROTOCOL.read_text(encoding="utf-8")
            ),
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def write_inputs(self, prompts: object, report: object) -> tuple[Path, Path]:
        prompts_path = self.root / "prompts.json"
        report_path = self.root / "report.json"
        prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return prompts_path, report_path

    def test_dependency_is_exact_hash_pinned_v3(self) -> None:
        self.assertEqual(
            extract.sha256_file(extract.V3_ANALYZER_PATH),
            extract.V3_ANALYZER_SHA256,
        )
        self.assertEqual(
            Path(extract.V3.__file__).resolve(),
            extract.V3_ANALYZER_PATH.resolve(),
        )
        self.assertIs(extract.V4_FEATURES._V3, extract.V3)
        with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
            extract._load_pinned_v3(expected_sha256="0" * 64)

    def test_in_memory_streaming_and_v3_current_feature_parity(self) -> None:
        prompts, report = replay_fixture(["inspect", "validate", None])
        prompts_path, report_path = self.write_inputs(prompts, report)

        with stub_v3_helpers():
            in_memory = extract.extract_stable_rows(
                prompts, report, protocol=self.protocol
            )
            streaming = extract.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=self.protocol
            )
            v3 = extract.V3.extract_stable_rows(
                prompts, report, protocol=self.protocol
            )

        self.assertEqual(streaming, in_memory)
        self.assertEqual(in_memory["eligibility"], v3["eligibility"])
        self.assertEqual(len(in_memory["rows"]), len(v3["rows"]))
        metadata_keys = (
            "row_id",
            "task_id",
            "repo",
            "cohort_id",
            "task_request_index",
            "checkpoint_ordinal",
            "source_action_label_status",
            "source_action_class_id",
            "label_status",
            "label",
            "metric_evaluable",
            "auxiliary_diagnostics",
        )
        for v4_row, v3_row in zip(
            in_memory["rows"], v3["rows"], strict=True
        ):
            self.assertEqual(
                {key: v4_row[key] for key in metadata_keys},
                {key: v3_row[key] for key in metadata_keys},
            )
            np.testing.assert_array_equal(
                v4_row["history"], v3_row["features"]["history_only"]
            )
            np.testing.assert_array_equal(
                v4_row["features"]["history_only"],
                v3_row["features"]["history_only"],
            )
            np.testing.assert_array_equal(
                np.asarray(v4_row["features"]["sequence_j"])[14:54],
                np.asarray(v3_row["features"]["history_j"])[14:54],
            )
            np.testing.assert_array_equal(
                np.asarray(v4_row["features"]["sequence_logit"])[14:54],
                np.asarray(v3_row["features"]["history_logit"])[14:54],
            )
            self.assertEqual(len(v4_row["public_jacobian"]), 96)
            self.assertEqual(len(v4_row["ordinary_logit"]), 96)

        self.assertEqual(in_memory["rows"][1]["label"], "check_or_finish")
        self.assertEqual(in_memory["rows"][1]["source_action_class_id"], "validate")
        self.assertIsNone(in_memory["rows"][2]["label"])
        self.assertFalse(in_memory["rows"][2]["metric_evaluable"])

    def test_unknown_stable_rows_update_state_but_unstable_rows_do_not(self) -> None:
        prompts, report = replay_fixture(["inspect", None, "edit"])

        with stub_v3_helpers():
            all_stable = extract.extract_stable_rows(
                prompts, report, protocol=self.protocol
            )
        self.assertEqual(len(all_stable["rows"]), 3)
        third_all = np.asarray(
            all_stable["rows"][2]["features"]["sequence_j"], dtype=np.float64
        )
        np.testing.assert_array_equal(
            third_all[54:94],
            extract.V3.compact_layer_shape(
                raw_sensor("p3", "public_jacobian")
                - raw_sensor("p2", "public_jacobian")
            ),
        )

        with stub_v3_helpers(unstable_ids={"p2"}):
            unstable_middle = extract.extract_stable_rows(
                prompts, report, protocol=self.protocol
            )
        self.assertEqual(
            [row["row_id"] for row in unstable_middle["rows"]], ["p1", "p3"]
        )
        third_after_exclusion = np.asarray(
            unstable_middle["rows"][1]["features"]["sequence_j"],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            third_after_exclusion[54:94],
            extract.V3.compact_layer_shape(
                raw_sensor("p3", "public_jacobian")
                - raw_sensor("p1", "public_jacobian")
            ),
        )
        eligibility = unstable_middle["eligibility"]
        self.assertEqual(eligibility["numerically_stable_prompt_count"], 2)
        self.assertEqual(eligibility["stable_feature_complete_prediction_count"], 2)
        self.assertEqual(
            eligibility["exclusion_counts"], {"numerically_unstable": 1}
        )
        self.assertEqual(
            eligibility["exclusions"][0]["details"], ["fixture_unstable"]
        )
        # Sensor state excludes p2, while exact V3 action history still records
        # its unknown preceding completion for the p3 boundary.
        self.assertEqual(unstable_middle["rows"][1]["history"][8], math.log1p(1))
        self.assertEqual(unstable_middle["rows"][1]["history"][9], 1.0)

    def test_capture_vocabulary_and_payload_binding_fail_closed(self) -> None:
        prompts, report = replay_fixture(["inspect", "edit"])
        cases: list[tuple[dict, str]] = []
        bad_capture = copy.deepcopy(report)
        bad_capture["experiments"][0]["capture_positions_resolved"] = [0]
        cases.append((bad_capture, "final prompt token"))
        bad_vocabulary = copy.deepcopy(report)
        bad_vocabulary["experiments"][0]["scored_vocabulary"]["token_ids"] = [9]
        cases.append((bad_vocabulary, "scored vocabulary differs"))
        bad_payload = copy.deepcopy(report)
        bad_payload["experiments"][0]["prompt"] = "different"
        cases.append((bad_payload, "not bound to supplied prompt"))

        with stub_v3_helpers():
            for value, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        extract.extract_stable_rows(
                            prompts, value, protocol=self.protocol
                        )

    def test_streaming_rejects_misaligned_trailing_and_extra_json(self) -> None:
        prompts, report = replay_fixture(["inspect", "edit"])
        prompts_path, report_path = self.write_inputs(prompts, report)

        misaligned = copy.deepcopy(report)
        misaligned["experiments"].reverse()
        report_path.write_text(json.dumps(misaligned), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "IDs or order differ"):
            extract.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=self.protocol
            )

        report_path.write_text(json.dumps({"experiments": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trailing prompt rows"):
            extract.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=self.protocol
            )

        prompts_path.write_text("[]\n", encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trailing experiment rows"):
            extract.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=self.protocol
            )

        prompts_path.write_text(
            json.dumps(prompts) + "\n{}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "could not stream prompt bundle|trailing"):
            extract.extract_stable_rows_streaming(
                prompts_path, report_path, protocol=self.protocol
            )


if __name__ == "__main__":
    unittest.main()
