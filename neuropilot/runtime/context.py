from dataclasses import dataclass, asdict
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
import yaml

@dataclass
class JobContext:
    """
    Structured runtime metadata for a job, accessible from within training scripts.

    This object allows jobs to access information like job ID, dataset name, task name,
    and other relevant identifiers in a standardized way. It can be created from environment
    variables (e.g., injected by JobRunner), from a .json file, or from the job dictionary directly.

    This context object is optional, but useful for:
    - Setting WandB run names
    - Logging or debugging
    - Creating reproducible output directory structures
    - Avoiding manual CLI or env var parsing in training code

    Fields (carried from job schema given in JobCreator):
        job_id (str): Unique job identifier (required)
        dataset_name (str): Name of dataset for this job
        task_name (str): Name of the task/experiment for this job
        group (str): Optional grouping identifier (e.g., experiment group or sweep ID)
        params (dict): Dictionary of param key/value pairs
        data_root (str): Path to dataset root directory
        task_description (str): String description of task
        dataset_description (str): String description of dataset

    Usage:
        ctx = JobContext.from_env()
        if ctx.is_set():
            wandb.init(name=ctx.job_id)
        print(ctx.summary())

        ctx.to_file("/path/to/job/file.json")
    """
    job_id: str
    task_name: Optional[str] = None
    dataset_name: Optional[str] = None
    group: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    data_root: Optional[str] = None
    task_description: Optional[str] = None
    dataset_description: Optional[str] = None

    @classmethod
    def from_env(cls) -> "JobContext":
        return cls(
            job_id=os.environ.get("NEUROPILOT_JOB_ID", "unknown"),
            dataset_name=os.environ.get("NEUROPILOT_DATASET"),
            task_name=os.environ.get("NEUROPILOT_TASK"),
            group=os.environ.get("NEUROPILOT_GROUP")
        )

    @classmethod
    def from_job(cls, job: Dict[str, Any]) -> "JobContext":
        return cls(
            job_id=job["job_id"],
            dataset_name=job.get("dataset_name"),
            task_name=job.get("task_name") or job.get("experiment_name"),
            group=job.get("group"),
            params=job.get("params"),
            data_root=job.get("data_root"),
            task_description=job.get("task_description"),
            dataset_description=job.get("dataset_description"),
        )
    
    def to_job(self) -> Dict[str, Any]:
        """
        Returns a dictionary job of key context values for use in logging or experiment tracking.

        This includes all core fields, including params.
        """
        job = {
            "job_id": self.job_id,
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
            "group": self.group,
            "params": self.params,
            "data_root": self.data_root,
            "task_description": self.task_description,
            "dataset_description": self.dataset_description
        }
        return job

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
    
    def to_file(self, path: Union[str, Path]):
        """
        Save the JobContext to a JSON or YAML file based on the file extension.

        Args:
            path (str or Path): Output file path. Must end in .json, .yaml, or .yml
        """
        path = Path(path)
        data = asdict(self)

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.suffix in {".yaml", ".yml"}:
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .json, .yaml, or .yml")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "JobContext":
        """
        Load a JobContext from a JSON or YAML file based on the file extension.

        Args:
            path (str or Path): Path to input file (.json, .yaml, or .yml)

        Returns:
            JobContext instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        elif path.suffix in {".yaml", ".yml"}:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .json, .yaml, or .yml")

        return cls(**data)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Returns a flat dictionary of key context values for use in logging or experiment tracking.

        This includes all core fields, including params, merged into one flat dictionary.
        """
        flat = {
            "job_id": self.job_id,
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
            "group": self.group,
            "data_root": self.data_root,
            "task_description": self.task_description,
            "dataset_description": self.dataset_description
        }
        if self.params:
            flat.update(self.params)
        return {k: v for k, v in flat.items() if v is not None}

    def is_set(self) -> bool:
        """
        Returns True if the context appears to be meaningfully initialized.
        """
        if self.job_id == "unknown":
            return False
        return any([self.dataset_name, self.task_name, \
                    self.group, self.params, self.data_root, \
                        self.dataset_description, self.task_description])
    