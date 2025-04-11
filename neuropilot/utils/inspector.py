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
        self.printer = print # Can be overridden

    def summarize(self):
        """
        Print a high-level summary of tasks, datasets, and job expansion.
        """
        self.printer("\n Summary Report")
        self.printer("=" * 30)
        self.printer(f"Total tasks defined: {len(self.config['tasks'])}")
        self.printer(f"Total datasets: {len(self.config['datasets'])}")
        self.printer(f"Total jobs expanded: {len(self.jobs)}\n")

        # Per-dataset/task summary
        combos = defaultdict(int)
        for job in self.jobs:
            combos[(job["dataset_name"], job["task_name"])] += 1

        self.printer(" Job Variants Per Dataset/Task:")
        for (ds, task), count in sorted(combos.items()):
            self.printer(f"  - {ds} / {task}: {count} variants")

    def check_warnings(self):
        """
        Detect potential issues or inefficiencies in the configuration.
        Suppresses unused warnings for tasks extended by used ones.
        """
        self.printer("\n  Warnings")
        self.printer("=" * 30)

        tasks = self.config["tasks"]
        all_task_defs = set(tasks)
        all_used_tasks = set()

        # 1. Get directly used task names from datasets
        for dataset in self.config["datasets"].values():
            for entry in dataset["tasks"]:
                all_used_tasks.add(entry["name"])

        # 2. Build reverse inheritance tree
        reverse_inherits = {k: set() for k in tasks}
        for child, details in tasks.items():
            parent = details.get("extends")
            if parent:
                reverse_inherits[parent].add(child)

        # 3. Recursively gather all indirectly used via inheritance
        def gather_descendants(task):
            children = reverse_inherits.get(task, set())
            return children | {c for child in children for c in gather_descendants(child)}

        indirectly_used = set()
        for used in all_used_tasks:
            indirectly_used |= gather_descendants(used)

        final_used = all_used_tasks | indirectly_used
        unused_tasks = all_task_defs - final_used

        if unused_tasks:
            self.printer(f"  - Unused task(s) defined but never referenced: {sorted(unused_tasks)}")
        else:
            self.printer("  No warnings found.")


    def print_sample_jobs(self, n: int = 5):
        if not self.jobs:
            self.printer("No jobs available.")
            return

        self.printer(f"\n Sample Jobs (showing {n})\n" + "="*30 + "\n")
        for job in self.jobs[:n]:
            formatted = pprint.pformat(job, indent=2)
            self.printer(formatted)

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
