import unittest
import sys

from loguru import logger

from neuropilot import ConfigLoader, JobCreator

class TestParamSetDeletion(unittest.TestCase):
    def setUp(self):
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

    def test_paramset_deletion_using_none(self):
        config = {
            "tasks": {
                "base_task": {
                    "script": "train.py",
                    "param_set": {
                        "dropout": [0.1, 0.2],
                        "lr": [0.001]
                    }
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data/ds1",
                    "param_set": {
                        "dropout": None,  # This should remove 'dropout'
                        "batch_size": [32, 64]
                    },
                    "tasks": [
                        {"name": "base_task"}
                    ]
                }
            }
        }

        loader = ConfigLoader(config)
        validated = loader.load()
        creator = JobCreator(validated)
        jobs = creator.create()

        # Should produce 2 jobs (from batch_size sweep only)
        self.assertEqual(len(jobs), 2)

        for job in jobs:
            params = job["params"]
            self.assertIn("batch_size", params)
            self.assertIn("lr", params)
            self.assertNotIn("dropout", params)  

class TestParamSetMergingBehavior(unittest.TestCase):
    def setUp(self):
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

    def test_task_and_dataset_merge(self):
        config = {
            "tasks": {
                "base_task": {
                    "script": "train.py",
                    "param_set": {
                        "lr": [0.001],
                        "dropout": [0.1]
                    }
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data/ds1",
                    "param_set": {
                        "batch_size": [16, 32]
                    },
                    "tasks": [
                        {"name": "base_task"}
                    ]
                }
            }
        }

        loader = ConfigLoader(config)
        creator = JobCreator(loader.load())
        jobs = creator.create()

        # Expected params: dropout × lr × batch_size = 1×1×2 = 2 jobs
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertIn("dropout", job["params"])
            self.assertIn("lr", job["params"])
            self.assertIn("batch_size", job["params"])

    def test_entry_overrides_dataset_and_task(self):
        config = {
            "tasks": {
                "taskA": {
                    "script": "train.py",
                    "param_set": {
                        "lr": [0.001],
                        "dropout": [0.1]
                    }
                }
            },
            "datasets": {
                "dsX": {
                    "root": "/data/x",
                    "param_set": {
                        "lr": [0.01],  # dataset override
                        "batch_size": [64]
                    },
                    "tasks": [
                        {
                            "name": "taskA",
                            "param_set": {
                                "dropout": [0.2]  # entry override
                            }
                        }
                    ]
                }
            }
        }

        loader = ConfigLoader(config)
        creator = JobCreator(loader.load())
        jobs = creator.create()

        # Final param_set should be: dropout=0.2 (entry), lr=0.01 (dataset), batch_size=64
        self.assertEqual(len(jobs), 1)
        params = jobs[0]["params"]
        self.assertEqual(params["dropout"], 0.2)
        self.assertEqual(params["lr"], 0.01)
        self.assertEqual(params["batch_size"], 64)

    def test_paramset_deletion_at_entry_level(self):
        config = {
            "tasks": {
                "taskB": {
                    "script": "train.py",
                    "param_set": {
                        "dropout": [0.1, 0.2],
                        "lr": [0.001]
                    }
                }
            },
            "datasets": {
                "dsY": {
                    "root": "/data/y",
                    "tasks": [
                        {
                            "name": "taskB",
                            "param_set": {
                                "dropout": None,
                                "batch_size": [128]
                            }
                        }
                    ]
                }
            }
        }

        loader = ConfigLoader(config)
        creator = JobCreator(loader.load())
        jobs = creator.create()

        # Should sweep only over batch_size and lr
        self.assertEqual(len(jobs), 1)
        params = jobs[0]["params"]
        self.assertIn("lr", params)
        self.assertIn("batch_size", params)
        self.assertNotIn("dropout", params) 
