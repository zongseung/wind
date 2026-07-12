import tempfile
import unittest
from pathlib import Path

import pandas as pd

import wind_lib as W


class SpatialJoinTests(unittest.TestCase):
    def make_spatial(self, timestamps):
        values = {"kst_dtm": timestamps}
        values.update({column: range(len(timestamps)) for column in W.SPATIAL_COLS})
        return pd.DataFrame(values)

    def test_spatial_join_preserves_one_to_one_rows(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="h")
        base = pd.DataFrame({"kst_dtm": timestamps, "value": [1.0, 2.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.make_spatial(timestamps).to_parquet(
                path / "spatial_v2_train.parquet", index=False
            )

            joined = W.add_spatial(base, "train", path)

        self.assertEqual(len(joined), len(base))
        self.assertTrue(joined[W.SPATIAL_COLS].notna().all().all())

    def test_spatial_join_rejects_duplicate_keys(self):
        timestamp = pd.Timestamp("2024-01-01")
        base = pd.DataFrame({"kst_dtm": [timestamp], "value": [1.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.make_spatial([timestamp, timestamp]).to_parquet(
                path / "spatial_v2_train.parquet", index=False
            )

            with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
                W.add_spatial(base, "train", path)


if __name__ == "__main__":
    unittest.main()
