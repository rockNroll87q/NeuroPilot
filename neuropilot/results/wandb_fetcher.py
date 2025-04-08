try:
    import wandb
except ImportError:
    wandb = None

from typing import Optional, Dict
from .result_fetcher import RemoteResultFetcher

class WandbFetcher(RemoteResultFetcher):
    """
    A ResultFetcher implementation that retrieves job results using the WandB API.

    This requires the `wandb` library to be installed (`pip install wandb`).

    The fetcher assumes that job["job_id"] was used as the run name or is otherwise
    part of the run ID in WandB.

    Args:
        project (str): WandB project name.
        entity (str): WandB entity (user or team).
    """

    def __init__(self, project: str, entity: str):
        if wandb is None:
            raise ImportError("wandb must be installed to use WandbFetcher.")
        self.api = wandb.Api()
        self.project = project
        self.entity = entity

    def fetch_result(self, job: Dict) -> Optional[Dict]:
        job_id = job.get("job_id")
        if not job_id:
            return None

        try:
            run = self.api.run(f"{self.entity}/{self.project}/{job_id}")
            summary = dict(run.summary)

            return {
                "job_id": job_id,
                "results": summary,
                "status": "success",
                "params": job.get("params", {}),
                "dataset_name": job.get("dataset_name"),
                "task_name": job.get("task_name"),
            }
        except Exception as e:
            # Log or re-raise as needed
            return None
