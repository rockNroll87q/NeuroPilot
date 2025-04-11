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
        Suppresses warnings about tasks that are only used via inheritance.
        """
        self.printer("\n  Warnings")
        self.printer("=" * 30)

        inheritance_map = self.config.get("_task_inheritance", {})
        all_tasks = set(self.config["tasks"])
        used_tasks = {entry["name"] for ds in self.config["datasets"].values() for entry in ds["tasks"]}

        # Collect all transitive parents of used tasks
        def get_all_parents(task, inherit_map):
            seen = set()
            while task in inherit_map and inherit_map[task] and inherit_map[task] not in seen:
                parent = inherit_map[task]
                seen.add(parent)
                task = parent
            return seen

        transitive_parents = set()
        for task in used_tasks:
            transitive_parents.update(get_all_parents(task, inheritance_map))

        considered_used = used_tasks | transitive_parents
        unused_tasks = all_tasks - considered_used

        if unused_tasks:
            self.printer(f"  - Unused task(s) defined but never referenced or extended: {sorted(unused_tasks)}")
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
    
    def print_task_inheritance(self):
        """
        Display the task inheritance structure in a readable format.
        """
        self.printer("\n  Task Inheritance Structure")
        self.printer("=" * 30)

        inheritance = self.config.get("_task_inheritance", {})
        if not inheritance:
            self.printer("  (No inheritance relationships detected.)")
            return

        # Build inverted map: parent -> list of children
        from collections import defaultdict
        tree = defaultdict(list)
        for child, parent in inheritance.items():
            tree[parent].append(child)

        def print_branch(parent, level=0):
            children = sorted(tree.get(parent, []))
            for child in children:
                self.printer("  " + "  " * level + f"- {child}")
                print_branch(child, level + 1)

        roots = sorted(tree.get(None, []))
        if not roots:
            self.printer("  (No root tasks found — possible cycle?)")
        else:
            for root in roots:
                self.printer(f"- {root}")
                print_branch(root, 1)


    def get_metrics_example(self) -> Optional[Dict[str, float]]:
        """
        Return a sample set of results (if any are included as static test values).
        """
        for job in self.jobs:
            r = job.get("results")
            if r:
                return r
        return None
