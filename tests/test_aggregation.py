import unittest
import pandas as pd
from neuropilot.aggregation import aggregate_results

class TestAggregateResults(unittest.TestCase):

    def setUp(self):
        self.results = [
            {
                "job_id": "job1",
                "task_name": "finetune",
                "dataset_name": "ds1",
                "results": {"accuracy": 0.91, "f1": 0.87},
                "params": {"lr": 0.001, "dropout": 0.2}
            },
            {
                "job_id": "job2",
                "task_name": "finetune",
                "dataset_name": "ds2",
                "results": {"accuracy": 0.88, "f1": 0.83},
                "params": {"lr": 0.002, "dropout": 0.1}
            }
        ]

        self.results_dirty = [
            {
                "job_id": "job1",
                "task_name": "finetune",
                "dataset_name": "ds1",
                "results": {"accuracy": 0.91},
                "params": {"lr": 0.001, "dropout": 0.2}
            },
            {
                "job_id": "job2",
                "task_name": "finetune",
                "dataset_name": "ds2",
                "results": {"accuracy": 0.88, "f1": 0.83},
                "params": {"dropout": 0.1}
            }
        ]

    def test_basic_index_extraction(self):
        rs = aggregate_results(self.results, index_fields=["job_id", "task_name"])
        df = rs.to_dataframe()
        self.assertIn("job_id", df.columns)
        self.assertIn("task_name", df.columns)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["job_id"].iloc[0], "job1")

    def test_auto_detect_metrics(self):
        rs = aggregate_results(
            self.results,
            index_fields=["job_id"],
            auto_detect_metrics=True
        )
        df = rs.to_dataframe()
        self.assertIn("accuracy", df.columns)
        self.assertIn("f1", df.columns)
        self.assertEqual(df.loc[0, "accuracy"], 0.91)

    def test_auto_detect_params(self):
        rs = aggregate_results(
            self.results,
            index_fields=["job_id"],
            auto_detect_params=True
        )
        df = rs.to_dataframe()
        self.assertIn("lr", df.columns)
        self.assertIn("dropout", df.columns)
        self.assertAlmostEqual(df.loc[1, "lr"], 0.002)

    def test_auto_detect_both(self):
        rs = aggregate_results(
            self.results,
            index_fields=["dataset_name"],
            auto_detect_metrics=True,
            auto_detect_params=True
        )
        df = rs.to_dataframe()
        self.assertIn("f1", df.columns)
        self.assertIn("lr", df.columns)
        self.assertEqual(df.shape[0], 2)

    def test_strict_mode_missing_results_raises(self):
        bad_data = [{"job_id": "missing_results"}]
        with self.assertRaises(ValueError) as ctx:
            aggregate_results(
                bad_data,
                index_fields=["job_id"],
                metric_fields=["accuracy"],
                strict=True
            )
        self.assertIn("Missing 'accuracy'", str(ctx.exception))

    def test_non_strict_mode_missing_results_fills_nan(self):
        bad_data = [{"job_id": "missing_results"}]
        rs = aggregate_results(bad_data, index_fields=["job_id"], metric_fields=["accuracy"], strict=False)
        df = rs.to_dataframe()
        self.assertTrue(pd.isna(df.loc[0, "accuracy"]))

    def test_strict_mode_missing_metric_key_raises(self):
        partial = [{
            "job_id": "partial",
            "results": {"loss": 0.9}
        }]
        with self.assertRaises(ValueError) as ctx:
            aggregate_results(partial, metric_fields=["accuracy"], strict=True)
        self.assertIn("Missing 'accuracy'", str(ctx.exception))

    def test_strict_mode_missing_param_key_raises(self):
        partial = [{
            "job_id": "partial",
            "params": {"dropout": 0.1}
        }]
        with self.assertRaises(ValueError):
            aggregate_results(partial, param_fields=["lr"], strict=True)

    def test_non_strict_mode_fills_missing_fields(self):
        partial = [{
            "job_id": "partial",
            "params": {"dropout": 0.1}
        }]
        rs = aggregate_results(partial, param_fields=["lr", "dropout"], strict=False)
        df = rs.to_dataframe()
        self.assertTrue(pd.isna(df.loc[0, "lr"]))
        self.assertEqual(df.loc[0, "dropout"], 0.1)

    def test_empty_results_raises(self):
        with self.assertRaises(ValueError):
            aggregate_results([])

    def test_dirty_results_strict_raises_on_missing_fields(self):
        with self.assertRaises(ValueError) as ctx:
            aggregate_results(
                self.results_dirty,
                index_fields=["job_id"],
                metric_fields=["accuracy", "f1"],
                param_fields=["lr", "dropout"],
                strict=True
            )
        self.assertIn("Missing 'f1'", str(ctx.exception))

    def test_dirty_results_non_strict_fills_missing_fields(self):
        rs = aggregate_results(
            self.results_dirty,
            index_fields=["job_id"],
            metric_fields=["accuracy", "f1"],
            param_fields=["lr", "dropout"],
            strict=False
        )
        df = rs.to_dataframe()
        self.assertEqual(df.shape[0], 2)
        self.assertTrue(pd.isna(df.loc[0, "f1"]))  # job1 missing f1
        self.assertTrue(pd.isna(df.loc[1, "lr"]))  # job2 missing lr
        self.assertAlmostEqual(df.loc[0, "dropout"], 0.2)
        self.assertAlmostEqual(df.loc[1, "accuracy"], 0.88)


class TestAggregateResultsNested(unittest.TestCase):

    def setUp(self):
        self.nested_results = [
            {
                "job_id": "job1",
                "task_name": "finetune",
                "dataset_name": "ds1",
                "results": {
                    "var1": {"accuracy": 0.91, "f1": 0.85},
                    "var2": {"accuracy": 0.88, "f1": 0.83}
                },
                "params": {"lr": 0.001}
            },
            {
                "job_id": "job2",
                "task_name": "finetune",
                "dataset_name": "ds2",
                "results": {
                    "var1": {"accuracy": 0.87, "f1": 0.82}
                },
                "params": {"lr": 0.002}
            }
        ]

    def test_flatten_nested_results(self):
        rs = aggregate_results(
            self.nested_results,
            index_fields=["job_id"],
            flatten_nested=True,
            auto_detect_metrics=True,
            auto_detect_params=True,
            strict=True
        )
        df = rs.to_dataframe()
        expected_columns = {
            "job_id", "var1.accuracy", "var1.f1", "var2.accuracy", "var2.f1", "lr"
        }
        self.assertEqual(set(df.columns), expected_columns)
        self.assertEqual(len(df), 2)

    def test_long_format_nested_results(self):
        rs = aggregate_results(
            self.nested_results,
            index_fields=["job_id"],
            long_format=True,
            long_output_field="label",
            auto_detect_metrics=True,
            auto_detect_params=True,
            strict=True
        )
        df = rs.to_dataframe()
        self.assertEqual(len(df), 3)  # 2 from job1 (var1, var2), 1 from job2 (var1)
        self.assertIn("job_id", df.columns)
        self.assertIn("label", df.columns)
        self.assertIn("lr", df.columns)
        self.assertIn("accuracy", df.columns)
        self.assertIn("f1", df.columns)

        self.assertCountEqual(df["label"].unique(), ["var1", "var2"])
        self.assertCountEqual(df["job_id"].unique(), ["job1", "job2"])

    def test_flatten_with_non_nested_results(self):
        flat_results = [
            {"job_id": "job1", "results": {"acc": 0.9, "loss": 0.1}, "params": {}},
            {"job_id": "job2", "results": {"acc": 0.85, "loss": 0.15}, "params": {}}
        ]
        rs = aggregate_results(flat_results, index_fields=["job_id"], flatten_nested=True)
        df = rs.to_dataframe()
        self.assertIn("acc", df.columns)
        self.assertIn("loss", df.columns)
        self.assertEqual(len(df), 2)

    def test_long_format_with_non_nested_results(self):
        flat_results = [
            {"job_id": "job1", "results": {"acc": 0.9, "loss": 0.1}, "params": {}},
            {"job_id": "job2", "results": {"acc": 0.85, "loss": 0.15}, "params": {}}
        ]
        rs = aggregate_results(flat_results, index_fields=["job_id"], long_format=True)
        df = rs.to_dataframe()
        self.assertIn("job_id", df.columns)
        self.assertIn("acc", df.columns)
        self.assertIn("loss", df.columns)
        self.assertAlmostEqual(0.9, df['acc'].iloc[0])
        self.assertEqual(len(df), 2)


class TestResultSetFiltering(unittest.TestCase):

    def setUp(self):
        self.results = [
            {
                "job_id": "job1",
                "task_name": "finetune",
                "dataset_name": "mnist",
                "results": {"accuracy": 0.91, "f1": 0.87},
                "params": {"lr": 0.001, "dropout": 0.2}
            },
            {
                "job_id": "job2",
                "task_name": "pretrain",
                "dataset_name": "imagenet",
                "results": {"accuracy": 0.88},
                "params": {"lr": 0.002}
            },
            {
                "job_id": "job3",
                "task_name": "finetune",
                "dataset_name": "imagenet",
                "results": {"f1": 0.82},
                "params": {"dropout": 0.3}
            }
        ]

    def test_filter_by_task_name(self):
        rs = aggregate_results(
            self.results,
            index_fields=["job_id", "task_name", "dataset_name"],
            auto_detect_metrics=True,
            auto_detect_params=True,
            strict=False
        )
        filtered = rs.filter_by(task_name="finetune")
        df = filtered.to_dataframe()

        # Should include job1 and job3
        self.assertEqual(set(df["job_id"]), {"job1", "job3"})
        # Only metrics present across these jobs: accuracy (job1), f1 (job1, job3)
        self.assertIn("f1", df.columns)
        self.assertIn("accuracy", df.columns)  # job3 has no accuracy
        # Only params present: lr (job1), dropout (both)
        self.assertIn("dropout", df.columns)
        self.assertIn("lr", df.columns)  # job3 has no lr

    def test_filter_by_dataset_name(self):
        rs = aggregate_results(
            self.results,
            index_fields=["job_id", "dataset_name"],
            auto_detect_metrics=True,
            auto_detect_params=True,
            strict=False
        )
        filtered = rs.filter_by(dataset_name="imagenet")
        df = filtered.to_dataframe()

        # Should include job2 and job3
        self.assertEqual(set(df["job_id"]), {"job2", "job3"})
        self.assertIn("f1", df.columns)
        self.assertIn("accuracy", df.columns)  # job2 has accuracy
        self.assertIn("lr", df.columns)  # job3 lacks lr

    def test_filter_no_match_returns_empty(self):
        rs = aggregate_results(
            self.results,
            index_fields=["job_id", "task_name"],
            auto_detect_metrics=True,
            auto_detect_params=True,
            strict=False
        )
        filtered = rs.filter_by(task_name="nonexistent")
        df = filtered.to_dataframe()
        self.assertEqual(len(df), 0)
        self.assertEqual(df.columns.tolist(), rs.index_fields)  # no metrics/params retained

    def test_filter_preserves_output_field(self):
        nested_results = [
            {
                "job_id": "job1",
                "task_name": "finetune",
                "dataset_name": "ds1",
                "results": {
                    "var1": {"acc": 0.91, "f1": 0.85},
                    "var2": {"acc": 0.88, "f1": 0.83}
                },
                "params": {"lr": 0.001}
            },
            {
                "job_id": "job2",
                "task_name": "finetune",
                "dataset_name": "ds2",
                "results": {
                    "var1": {"acc": 0.87, "f1": 0.82}
                },
                "params": {"lr": 0.002}
            }
        ]
        rs = aggregate_results(
            nested_results,
            index_fields=["job_id", "dataset_name"],
            long_format=True,
            long_output_field="label",
            auto_detect_metrics=True,
            auto_detect_params=True,
            strict=True
        )
        filtered = rs.filter_by(dataset_name="ds2")
        df = filtered.to_dataframe()
        self.assertEqual(len(df), 1)
        self.assertIn("label", df.columns)
        self.assertEqual(df["label"].iloc[0], "var1")
        self.assertIn("acc", df.columns)
        self.assertIn("f1", df.columns)



if __name__ == "__main__":
    unittest.main()
