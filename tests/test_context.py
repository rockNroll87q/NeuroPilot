import unittest
import os
import tempfile
from pathlib import Path

from neuropilot.runtime import JobContext

class TestJobContext(unittest.TestCase):
    def setUp(self):
        self.job = {
            "job_id": "job123",
            "task_name": "finetune",
            "dataset_name": "my_dataset",
            "params": {"lr": 0.01, "epochs": 10},
            "data_root": "/data/my_dataset",
            "task_description": "finetune task",
            "dataset_description": "my dataset description",
        }

    def test_from_job_and_env(self):
        ctx = JobContext.from_job(self.job)
        self.assertEqual(ctx.job_id, "job123")
        self.assertEqual(ctx.task_name, "finetune")
        self.assertEqual(ctx.dataset_name, "my_dataset")
        self.assertIn("lr", ctx.params)
        self.assertEqual(ctx.data_root, "/data/my_dataset")

        os.environ["NEUROPILOT_JOB_ID"] = "job_env"
        os.environ["NEUROPILOT_DATASET"] = "env_ds"
        os.environ["NEUROPILOT_TASK"] = "env_task"
        ctx_env = JobContext.from_env()
        self.assertEqual(ctx_env.job_id, "job_env")

    def test_to_from_file_roundtrip(self):
        ctx = JobContext.from_job(self.job)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ctx.json"
            ctx.to_file(path)
            ctx2 = JobContext.from_file(path)
            self.assertEqual(ctx2.job_id, ctx.job_id)
            self.assertEqual(ctx2.params["lr"], ctx.params["lr"])

    def test_to_env(self):
        ctx = JobContext.from_job(self.job)
        env = ctx.to_env()
        self.assertIn("NEUROPILOT_JOB_ID", env)
        self.assertEqual(env["NEUROPILOT_TASK"], "finetune")