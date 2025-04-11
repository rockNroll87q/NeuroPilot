from dataclasses import dataclass
import os
from typing import Optional, Dict


from dataclasses import dataclass
import os
from typing import Optional, Dict


@dataclass
class JobContext:
    """
    Structured runtime metadata for a job, accessible from within training scripts.

    This object allows jobs to access information like job ID, dataset name, task name,
    and other relevant identifiers in a standardized way. It can be created from environment
    variables (e.g., injected by JobRunner) or from the job dictionary directly.

    This context object is optional, but useful for:
    - Setting WandB run names
    - Logging or debugging
    - Creating reproducible output directory structures
    - Avoiding manual CLI or env var parsing in training code

    Fields:
        job_id (str): Unique job identifier (required)
        dataset_name (str): Name of dataset for this job
        task_name (str): Name of the task/experiment for this job
        group (str): Optional grouping identifier (e.g., experiment group or sweep ID)

    Usage:
        ctx = JobContext.from_env()
        if ctx.is_set():
            wandb.init(name=ctx.job_id)
        print(ctx.summary())
    """
    job_id: str
    dataset_name: Optional[str] = None
    task_name: Optional[str] = None
    group: Optional[str] = None

    @classmethod
    def from_env(cls) -> "JobContext":
        return cls(
            job_id=os.environ.get("NEUROPILOT_JOB_ID", "unknown"),
            dataset_name=os.environ.get("NEUROPILOT_DATASET"),
            task_name=os.environ.get("NEUROPILOT_TASK"),
            group=os.environ.get("NEUROPILOT_GROUP")
        )

    @classmethod
    def from_job(cls, job: Dict) -> "JobContext":
        return cls(
            job_id=job["job_id"],
            dataset_name=job.get("dataset_name"),
            task_name=job.get("task_name") or job.get("experiment_name"),
            group=job.get("group")
        )

    def to_env(self) -> Dict[str, str]:
        """
        Converts this context into an environment dictionary.
        Useful for subprocess execution.
        """
        env = {
            "NEUROPILOT_JOB_ID": self.job_id,
        }
        if self.dataset_name:
            env["NEUROPILOT_DATASET"] = self.dataset_name
        if self.task_name:
            env["NEUROPILOT_TASK"] = self.task_name
        if self.group:
            env["NEUROPILOT_GROUP"] = self.group
        return env

    def summary(self) -> str:
        return f"[{self.task_name or 'task'}] {self.dataset_name or 'dataset'} ({self.job_id})"

    def log_prefix(self) -> str:
        return f"{self.task_name or 'job'}:{self.dataset_name or 'data'}:{self.job_id}"

    def is_set(self) -> bool:
        """
        Returns True if the context appears to be meaningfully initialized.
        """
        if self.job_id == "unknown":
            return False
        return any([self.dataset_name, self.task_name, self.group])
    