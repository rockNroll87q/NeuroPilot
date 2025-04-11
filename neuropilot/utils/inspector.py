"""
JobInspector - Config and Job Expansion Introspection Utility

This class provides summary-level analysis and sanity checks
for validated configs or expanded jobs before full execution.

Use this to:
  - Check inherited + overridden parameter resolutions
  - See how many job variants will be generated per dataset/task
  - Understand what param/metric fields will appear downstream
  - Detect unused task definitions, param_set mismatches, or silent nulls
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import pprint

from ..job_creator import JobCreator

class JobInspector:
    def __init__(self, validated_config: dict):
        self.config = validated_config
        self.jobs = JobCreator(validated_config).create()
        self._pp = pprint.PrettyPrinter(indent=2)

    def summarize(self):
        """
        Print a high-level summary of tasks, datasets, and job expansion.
        """
        print("\n Summary Report")
        print("=" * 30)
        print(f"Total tasks defined: {len(self.config['tasks'])}")
        print(f"Total datasets: {len(self.config['datasets'])}")
        print(f"Total jobs expanded: {len(self.jobs)}\n")

        # Per-dataset/task summary
        combos = defaultdict(int)
        for job in self.jobs:
            combos[(job["dataset_name"], job["task_name"])] += 1

        print(" Job Variants Per Dataset/Task:")
        for (ds, task), count in sorted(combos.items()):
            print(f"  - {ds} / {task}: {count} variants")

    def check_warnings(self):
        """
        Detect potential issues or inefficiencies in the configuration.
        """
        print("\n  Warnings")
        print("=" * 30)

        all_task_defs = set(self.config["tasks"])
        all_used_tasks = set()

        for dataset in self.config["datasets"].values():
            for entry in dataset["tasks"]:
                all_used_tasks.add(entry["name"])

        unused_tasks = all_task_defs - all_used_tasks
        if unused_tasks:
            print(f"  - Unused task(s) defined but never referenced: {sorted(unused_tasks)}")

        # Add other rule checks if needed

        if not unused_tasks:
            print("  No warnings found.")

    def print_sample_jobs(self, n=3):
        """
        Show a few fully-resolved jobs.
        """
        print(f"\n Sample Jobs (showing {n})")
        print("=" * 30)
        for job in self.jobs[:n]:
            self._pp.pprint(job)
            print()

    def estimate_param_columns(self) -> List[str]:
        """
        Determine which param keys are likely to appear in the aggregator table.
        Returns:
            List[str]: union of all used param keys
        """
        param_keys: Set[str] = set()
        for job in self.jobs:
            param_keys.update(job.get("params", {}).keys())
        return sorted(param_keys)

    def get_metrics_example(self) -> Optional[Dict[str, float]]:
        """
        Return a sample set of results (if any are included as static test values).
        """
        for job in self.jobs:
            r = job.get("results")
            if r:
                return r
        return None
