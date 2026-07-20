from __future__ import annotations

import copy
import hashlib
import math
import unittest
from unittest import mock

import numpy as np

from scripts import analyze_swe_task_state_v3 as V3
from scripts import swe_task_state_v4_features as V4


def sensor(offset: float, *, scale: float = 1.0) -> list[float]:
    return (offset + scale * np.arange(V4.SENSOR_WIDTH, dtype=np.float64)).tolist()


def stable_row(
    task_id: str,
    request_index: int,
    *,
    jacobian: list[float] | None = None,
    logit: list[float] | None = None,
    history_offset: float = 0.0,
    action_status: str = "available",
    action_label: str | None = "inspect",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_request_index": request_index,
        "history": (
            history_offset + np.arange(V4.HISTORY_WIDTH, dtype=np.float64)
        ).tolist(),
        "public_jacobian": sensor(100.0) if jacobian is None else jacobian,
        "ordinary_logit": sensor(-100.0) if logit is None else logit,
        # Feature construction must treat these as arbitrary inert metadata.
        "label_status": action_status,
        "label": action_label,
    }


def compact(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(V3.compact_layer_shape(values), dtype=np.float64)


class TestV4FeatureContract(unittest.TestCase):
    def test_frozen_v3_dependency_is_byte_authenticated(self):
        self.assertFalse(V4.V3_ANALYZER_PATH.is_symlink())
        self.assertEqual(
            hashlib.sha256(V4.V3_ANALYZER_PATH.read_bytes()).hexdigest(),
            V4.V3_ANALYZER_SHA256,
        )
        V4._authenticate_v3_source()
        with mock.patch.object(type(V4.V3_ANALYZER_PATH), "read_bytes", return_value=b"tampered"):
            with self.assertRaisesRegex(RuntimeError, "byte identity changed"):
                V4._authenticate_v3_source()

    def test_exact_width_order_and_v3_current_compact_parity(self):
        row = stable_row(
            "task-a",
            1,
            jacobian=sensor(7.0, scale=0.25),
            logit=sensor(-11.0, scale=0.5),
            history_offset=3.0,
        )
        features = V4.build_sequence_features([row])[0]
        history = np.asarray(row["history"], dtype=np.float64)
        jacobian = np.asarray(row["public_jacobian"], dtype=np.float64)
        logit = np.asarray(row["ordinary_logit"], dtype=np.float64)
        zero = compact(np.zeros(V4.SENSOR_WIDTH, dtype=np.float64))

        self.assertEqual(tuple(features), V4.VARIANTS)
        self.assertEqual(
            {name: len(values) for name, values in features.items()},
            V4.VARIANT_WIDTHS,
        )
        np.testing.assert_array_equal(features["history_only"], history)

        sequence_j = np.asarray(features["sequence_j"])
        np.testing.assert_array_equal(sequence_j[:14], history)
        np.testing.assert_array_equal(sequence_j[14:54], compact(jacobian))
        np.testing.assert_array_equal(sequence_j[54:94], zero)
        np.testing.assert_array_equal(sequence_j[94:134], zero)
        np.testing.assert_array_equal(sequence_j[-2:], [0.0, 1.0])

        sequence_logit = np.asarray(features["sequence_logit"])
        np.testing.assert_array_equal(sequence_logit[:14], history)
        np.testing.assert_array_equal(sequence_logit[14:54], compact(logit))
        np.testing.assert_array_equal(sequence_logit[54:94], zero)
        np.testing.assert_array_equal(sequence_logit[94:134], zero)
        np.testing.assert_array_equal(sequence_logit[-2:], [0.0, 1.0])

        hybrid = np.asarray(features["sequence_logit_j"])
        np.testing.assert_array_equal(hybrid[:14], history)
        # Preserve V3's hybrid order: ordinary-logit triplet, then J triplet.
        np.testing.assert_array_equal(hybrid[14:54], compact(logit))
        np.testing.assert_array_equal(hybrid[54:94], zero)
        np.testing.assert_array_equal(hybrid[94:134], zero)
        np.testing.assert_array_equal(hybrid[134:174], compact(jacobian))
        np.testing.assert_array_equal(hybrid[174:214], zero)
        np.testing.assert_array_equal(hybrid[214:254], zero)
        np.testing.assert_array_equal(hybrid[-2:], [0.0, 1.0])

        for variant in V4.VARIANTS:
            self.assertEqual(len(V4.feature_names(variant)), V4.VARIANT_WIDTHS[variant])
        self.assertEqual(
            V4.feature_names("sequence_logit_j")[-2:],
            ["log1p_request_gap", "no_previous_stable_row"],
        )

    def test_unknown_rows_update_previous_and_prior_ema(self):
        j1 = np.asarray(sensor(1.0, scale=0.1))
        j2 = np.asarray(sensor(10.0, scale=0.2))
        j3 = np.asarray(sensor(-4.0, scale=0.3))
        l1 = np.asarray(sensor(-2.0, scale=0.4))
        l2 = np.asarray(sensor(8.0, scale=0.5))
        l3 = np.asarray(sensor(20.0, scale=-0.1))
        rows = [
            stable_row("task-a", 1, jacobian=j1.tolist(), logit=l1.tolist()),
            stable_row(
                "task-a",
                2,
                jacobian=j2.tolist(),
                logit=l2.tolist(),
                action_status="unknown",
                action_label=None,
            ),
            stable_row("task-a", 3, jacobian=j3.tolist(), logit=l3.tolist()),
        ]
        features = V4.build_sequence_features(rows)
        third_j = np.asarray(features[2]["sequence_j"])
        third_l = np.asarray(features[2]["sequence_logit"])

        np.testing.assert_allclose(third_j[54:94], compact(j3 - j2), rtol=0, atol=0)
        np.testing.assert_allclose(
            third_j[94:134], compact(j3 - 0.5 * (j1 + j2)), rtol=0, atol=0
        )
        np.testing.assert_allclose(third_l[54:94], compact(l3 - l2), rtol=0, atol=0)
        np.testing.assert_allclose(
            third_l[94:134], compact(l3 - 0.5 * (l1 + l2)), rtol=0, atol=0
        )

        without_unknown = V4.build_sequence_features([rows[0], rows[2]])[1]
        self.assertNotEqual(
            features[2]["sequence_j"][54:134],
            without_unknown["sequence_j"][54:134],
        )

    def test_future_sensor_or_label_mutation_cannot_change_earlier_features(self):
        original = [
            stable_row("task-a", 1, jacobian=sensor(1.0), logit=sensor(-1.0)),
            stable_row("task-a", 2, jacobian=sensor(2.0), logit=sensor(-2.0)),
            stable_row("task-a", 3, jacobian=sensor(3.0), logit=sensor(-3.0)),
        ]
        mutated = copy.deepcopy(original)
        mutated[2]["history"] = sensor(900.0)[: V4.HISTORY_WIDTH]
        mutated[2]["public_jacobian"] = sensor(1000.0, scale=-7.0)
        mutated[2]["ordinary_logit"] = sensor(-1000.0, scale=9.0)
        mutated[2]["label_status"] = "available"
        mutated[2]["label"] = "finalize"

        before = V4.build_sequence_features(original)
        after = V4.build_sequence_features(mutated)
        self.assertEqual(before[:2], after[:2])

        labels_only = copy.deepcopy(original)
        for position, row in enumerate(labels_only):
            row["label_status"] = "mutated"
            row["label"] = {"arbitrary": position}
        self.assertEqual(before, V4.build_sequence_features(labels_only))

    def test_task_state_is_isolated_when_tasks_are_interleaved(self):
        a1 = stable_row("task-a", 1, jacobian=sensor(1.0), logit=sensor(-1.0))
        a2 = stable_row("task-a", 4, jacobian=sensor(4.0), logit=sensor(-4.0))
        baseline = V4.build_sequence_features([a1, a2])

        b1 = stable_row(
            "task-b", 1, jacobian=sensor(10000.0), logit=sensor(-10000.0)
        )
        interleaved = V4.build_sequence_features([a1, b1, a2])
        self.assertEqual(baseline[0], interleaved[0])
        self.assertEqual(baseline[1], interleaved[2])
        np.testing.assert_array_equal(
            np.asarray(interleaved[1]["sequence_j"])[54:134],
            np.concatenate(
                [
                    compact(np.zeros(V4.SENSOR_WIDTH)),
                    compact(np.zeros(V4.SENSOR_WIDTH)),
                ]
            ),
        )
        self.assertEqual(interleaved[1]["sequence_j"][-2:], [0.0, 1.0])

    def test_unstable_rows_are_excluded_by_the_caller(self):
        rows = [
            {**stable_row("task-a", 1, jacobian=sensor(1.0)), "stable": True},
            {
                **stable_row("task-a", 2, jacobian=sensor(100000.0)),
                "stable": False,
            },
            {**stable_row("task-a", 3, jacobian=sensor(3.0)), "stable": True},
        ]
        supplied_stable_rows = [row for row in rows if row["stable"] is True]
        built = V4.build_feature_rows(supplied_stable_rows)
        self.assertEqual(len(built), 2)
        second = np.asarray(built[1]["features"]["sequence_j"])
        np.testing.assert_array_equal(
            second[54:94],
            compact(
                np.asarray(rows[2]["public_jacobian"])
                - np.asarray(rows[0]["public_jacobian"])
            ),
        )
        np.testing.assert_array_equal(second[-2:], [math.log1p(2), 0.0])

    def test_request_gaps_and_no_previous_are_per_task(self):
        rows = [
            stable_row("task-a", 2),
            stable_row("task-b", 10),
            stable_row("task-a", 5),
            stable_row("task-b", 11),
        ]
        features = V4.build_sequence_features(rows)
        self.assertEqual(features[0]["sequence_j"][-2:], [0.0, 1.0])
        self.assertEqual(features[1]["sequence_j"][-2:], [0.0, 1.0])
        np.testing.assert_allclose(
            features[2]["sequence_j"][-2:], [math.log1p(3), 0.0], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            features[3]["sequence_j"][-2:], [math.log1p(1), 0.0], rtol=0, atol=0
        )

    def test_rejects_bad_width_nonfinite_duplicate_and_out_of_order_rows(self):
        bad_width = stable_row("task-a", 1)
        bad_width["history"] = [0.0] * 13
        with self.assertRaisesRegex(ValueError, "history width must be 14"):
            V4.build_feature_rows([bad_width])

        nonfinite = stable_row("task-a", 1)
        nonfinite["ordinary_logit"][4] = math.inf  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "ordinary_logit.*finite"):
            V4.build_feature_rows([nonfinite])

        boolean_sensor = stable_row("task-a", 1)
        boolean_sensor["public_jacobian"] = [True] * V4.SENSOR_WIDTH
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            V4.build_feature_rows([boolean_sensor])

        string_history = stable_row("task-a", 1)
        string_history["history"] = ["1"] * V4.HISTORY_WIDTH
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            V4.build_feature_rows([string_history])

        with self.assertRaisesRegex(ValueError, "duplicate stable request index"):
            V4.build_feature_rows(
                [stable_row("task-a", 1), stable_row("task-a", 1)]
            )

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            V4.build_feature_rows(
                [stable_row("task-a", 2), stable_row("task-a", 1)]
            )

        with self.assertRaisesRegex(ValueError, "integer >= 1"):
            V4.build_feature_rows([stable_row("task-a", True)])

        # Request indices need only be unique within a task.
        self.assertEqual(
            len(
                V4.build_feature_rows(
                    [stable_row("task-a", 1), stable_row("task-b", 1)]
                )
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
