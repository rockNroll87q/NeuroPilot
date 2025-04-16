from typing import Dict, Optional, List

class RemoteResultFetcher:
    """
    Abstract base class for fetching result data for jobs via external APIs or services.

    This interface enables integration with tools such as:
    - Weights & Biases (WandB)
    - MLflow
    - Neptune
    - Custom remote logs or databases

    Subclasses should override `fetch_result(job)` to return a result object
    conforming to the schema used by neuropilot's result and aggregation systems.

    -------------------------
    Expected return structure:
    -------------------------
    A valid result object is a dictionary containing at minimum:

        {
            "job_id": str,
            "results": dict,
            "status": str,                # e.g., "success", "failed"
            "params": dict,               # optional: training parameters
            "task_name": str,             # optional: experiment or task name
            "dataset_name": str           # optional: dataset key
        }

    Returns:
        - A dict matching the above schema if a result is found
        - `None` if the job should be skipped or not found

    -------------------------
    Usage:
    -------------------------
        fetcher = MyFetcher()
        results = fetcher.collect(jobs)

    Notes:
        - This interface is **read-only**: no emission or file writing is performed.
        - For disk-based parsing, see `ResultCollector`.
        - For structured emission, see `ResultManager`.
    """

    def fetch_result(self, job: Dict) -> Optional[Dict]:
        """
        Fetch the result for a single job.

        Args:
            job (dict): The job dictionary.

        Returns:
            dict or None: Parsed result object or None if skipped.
        """
        raise NotImplementedError("Subclasses must implement fetch_result(job)")

    def collect(self, jobs: List[Dict]) -> List[Dict]:
        """
        Fetch and compile results for a batch of jobs.

        Args:
            jobs (List[dict]): List of job specs.

        Returns:
            List[dict]: List of parsed result dictionaries.
        """
        results = []
        for job in jobs:
            try:
                result = self.fetch_result(job)
                if result is not None:
                    results.append(result)
            except Exception as e:
                self._handle_fetch_error(job, e)
        return results

    def _handle_fetch_error(self, job: Dict, error: Exception):
        """
        Hook for handling errors during result fetching.
        Override this to implement custom logging or error suppression.

        Default behavior: raise the error.
        """
        raise error
