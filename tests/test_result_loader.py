import unittest
import tempfile
import shutil
import yaml
from pathlib import Path
from neuropilot.results import ResultLoader


class TestResultLoader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)

        # Create two structured result files
        (self.root / "run1-taskA-ds1").mkdir()
        (self.root / "run2-taskA-ds2").mkdir()
        (self.root / "run1-taskA-ds1" / "results.yaml").write_text(
            yaml.safe_dump({"results": {"accuracy": 0.9}, "status": "success"})
        )
        (self.root / "run2-taskA-ds2" / "results.yaml").write_text(
            yaml.safe_dump({"results": {"accuracy": 0.85}, "status": "success"})
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_collection_and_metadata(self):
        pattern = "{run_id}-taskA-{dataset_name}/results.yaml"  # Added '**/' for recursive matching
        collector = ResultLoader(
            root_dir=self.root,
            pattern=pattern,
            fmt="yaml",
            infer_metadata=True
        )
        results = collector.collect(strict=True)

        self.assertEqual(len(results), 2)
        self.assertTrue(all("dataset_name" in r for r in results))
        ds_names = sorted(r["dataset_name"] for r in results)
        self.assertEqual(ds_names, ["ds1", "ds2"])


    def test_strict_failure_on_bad_file(self):
        # Add an invalid file
        bad_file = self.root / "run1-taskA-ds1" / "corrupt.yaml"
        bad_file.write_text("not: : : yaml")

        collector = ResultLoader(
            root_dir=self.root,
            pattern="**/*.yaml",
            fmt=None,
            infer_metadata=True
        )

        with self.assertRaises(ValueError):
            collector.collect(strict=True)

        # Should skip in non-strict mode
        results = collector.collect(strict=False)
        self.assertGreaterEqual(len(results), 2)

    def test_non_matching_pattern(self):
        collector = ResultLoader(
            root_dir=self.root,
            pattern="missing-{job_id}/results.json",
            fmt="json",
            infer_metadata=True
        )
        results = collector.collect(strict=False)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
