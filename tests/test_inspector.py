import io
import unittest
from contextlib import redirect_stdout
from neuropilot.utils.inspector import JobInspector
from neuropilot import ConfigLoader


class TestJobInspector(unittest.TestCase):
    def setUp(self):
        self.config = {
            "tasks": {
                "base": {"script": "train.py", "epochs": 10},
                "child": {"extends": "base", "learning_rate": 0.01},
                "unused": {"script": "unused.py"}
            },
            "datasets": {
                "dataset1": {
                    "root": "/data/dataset1",
                    "tasks": [
                        {"name": "child"}
                    ]
                }
            }
        }
        loader = ConfigLoader(self.config)
        validated = loader.load()
        self.inspector = JobInspector(validated)

    def test_summary_output_contains_expected_info(self):
        f = io.StringIO()
        with redirect_stdout(f):
            self.inspector.summarize()
        output = f.getvalue()
        self.assertIn("Total tasks defined: 3", output)
        self.assertIn("Total datasets: 1", output)
        self.assertIn("dataset1 / child", output)

    def test_warning_for_unused_task(self):
        f = io.StringIO()
        with redirect_stdout(f):
            self.inspector.check_warnings()
        output = f.getvalue()
        self.assertIn("unused", output)

    def test_estimate_param_keys(self):
        keys = self.inspector.estimate_param_columns()
        self.assertIn("learning_rate", keys)
        self.assertIn("epochs", keys)
