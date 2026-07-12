import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import wind_pipeline
from submission_validation import KEY_COLUMNS, validate_submission


ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_ROOT = ROOT / "submission"


class SubmissionLayoutTests(unittest.TestCase):
    def test_all_version_directories_exist(self):
        for version in range(1, 17):
            self.assertTrue((SUBMISSION_ROOT / f"ver_{version}").is_dir())

    def test_registry_is_complete_and_references_existing_artifacts(self):
        registry = json.loads((SUBMISSION_ROOT / "registry.json").read_text())
        versions = registry["versions"]
        self.assertEqual([entry["version"] for entry in versions], list(range(1, 17)))
        for entry in versions:
            submission = entry["submission"]
            if submission is not None:
                self.assertTrue((ROOT / submission).is_file(), submission)

    def test_all_stored_submission_csv_files_are_valid(self):
        paths = sorted(SUBMISSION_ROOT.glob("ver_*/*.csv"))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path):
                validate_submission(pd.read_csv(path))

    def test_v16_preserves_v12_groups_one_and_two(self):
        champion = pd.read_csv(SUBMISSION_ROOT / "ver_12" / "submission.csv")
        candidate = pd.read_csv(SUBMISSION_ROOT / "ver_16" / "submission.csv")
        self.assertTrue(champion[KEY_COLUMNS].equals(candidate[KEY_COLUMNS]))
        for group in (1, 2):
            column = f"kpx_group_{group}"
            self.assertTrue(
                np.array_equal(champion[column].to_numpy(), candidate[column].to_numpy())
            )

    def test_current_pipeline_import_does_not_load_data(self):
        wind_pipeline.clear_context()

        from submission.ver_16 import pipeline  # noqa: F401

        self.assertIsNone(wind_pipeline._CONTEXT)


if __name__ == "__main__":
    unittest.main()
