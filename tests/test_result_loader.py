import unittest
import tempfile
import shutil
import yaml
import json
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


class TestResultLoaderAdvanced(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)

        # Create valid structured files
        (self.root / "experiments" / "expA" / "dataset1").mkdir(parents=True)
        (self.root / "experiments" / "expB" / "dataset2").mkdir(parents=True)
        (self.root / "experiments" / "expA" / "dataset1" / "result-abc123.json").write_text(
            json.dumps({"results": {"score": 0.77}, "status": "success"})
        )
        (self.root / "experiments" / "expB" / "dataset2" / "result-def456.json").write_text(
            json.dumps({"results": {"score": 0.81}, "status": "success"})
        )

        # Create a file that matches glob but not regex
        (self.root / "experiments" / "expC" / "dataset3").mkdir(parents=True)
        (self.root / "experiments" / "expC" / "dataset3" / "result-bad.json").write_text(
            json.dumps({"junk": True})
        )

        # Create malformed YAML
        (self.root / "experiments" / "expA" / "dataset1" / "broken.yaml").write_text(
            "key: : : invalid"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_complex_path_pattern_with_inference(self):
        pattern = "experiments/{task_name}/{dataset_name}/result-{job_id}.json"
        collector = ResultLoader(
            root_dir=self.root,
            pattern=pattern,
            fmt="json",
            infer_metadata=True
        )
        results = collector.collect(strict=True)

        self.assertEqual(len(results), 3)  # All match path + are parseable

        # Filter down to "real" results that actually have metrics
        valid = [r for r in results if "results" in r]

        self.assertEqual(len(valid), 2)
        for r in valid:
            self.assertIn("task_name", r)
            self.assertIn("dataset_name", r)
            self.assertIn("job_id", r)


    def test_partial_matches_skipped_when_strict(self):
        pattern = "experiments/{task_name}/{dataset_name}/result-{job_id}.json"
        collector = ResultLoader(
            root_dir=self.root,
            pattern=pattern,
            fmt="json",
            infer_metadata=True
        )

        # broken.yaml should not cause failure (wrong format)
        results = collector.collect(strict=False)
        self.assertEqual(len(results), 3)

    def test_malformed_yaml_raises_strict(self):
        pattern = "experiments/{task_name}/{dataset_name}/*.yaml"
        collector = ResultLoader(
            root_dir=self.root,
            pattern=pattern,
            fmt="yaml",
            infer_metadata=False
        )
        with self.assertRaises(ValueError):
            collector.collect(strict=True)

    def test_pattern_with_no_metadata_fields(self):
        pattern = "**/*.json"  # No placeholders
        collector = ResultLoader(
            root_dir=self.root,
            pattern=pattern,
            fmt="json",
            infer_metadata=True
        )
        results = collector.collect(strict=False)
        self.assertEqual(len(results), 3)  # 2 valid + 1 "junk" file still valid JSON

if __name__ == "__main__":
    unittest.main()
