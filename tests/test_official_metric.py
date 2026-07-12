import unittest

import numpy as np
import pandas as pd

from official_metric import CAPACITY_KWH, TARGET_COLS, group_scores, metric


class OfficialMetricTests(unittest.TestCase):
    def test_thresholds_and_validity_mask(self):
        capacity = 100.0
        actual = np.array([50.0, 50.0, 50.0, 9.9])
        forecast = np.array([56.0, 58.0, 58.01, 0.0])

        nmae, ficr = group_scores(actual, forecast, capacity)

        self.assertAlmostEqual(nmae, (0.06 + 0.08 + 0.0801) / 3)
        self.assertAlmostEqual(ficr, 7 / 12)

    def test_group_helper_matches_full_metric(self):
        answers = {}
        predictions = {}
        group_nmae = []
        group_ficr = []
        for name in TARGET_COLS:
            capacity = CAPACITY_KWH[name]
            actual = np.array([0.5 * capacity, 0.7 * capacity, 0.05 * capacity])
            forecast = np.array([0.55 * capacity, 0.79 * capacity, 0.9 * capacity])
            answers[name] = actual
            predictions[name] = forecast
            nmae, ficr = group_scores(actual, forecast, capacity)
            group_nmae.append(nmae)
            group_ficr.append(ficr)

        score, one_minus_nmae, ficr = metric(
            pd.DataFrame(answers), pd.DataFrame(predictions)
        )

        expected_one_minus = 1 - np.mean(group_nmae)
        expected_ficr = np.mean(group_ficr)
        self.assertAlmostEqual(one_minus_nmae, expected_one_minus)
        self.assertAlmostEqual(ficr, expected_ficr)
        self.assertAlmostEqual(score, 0.5 * (expected_one_minus + expected_ficr))


if __name__ == "__main__":
    unittest.main()
