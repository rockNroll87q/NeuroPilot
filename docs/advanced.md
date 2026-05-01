# Advanced Usage

This page covers patterns for more complex NeuroPilot setups, including sharing `JobContext` across scripts, multi-processing, and integrating with third-party platforms.

---

## Sharing JobContext Across Scripts

`JobContext` is the recommended way to pass job state between a job-generation script and a job-execution script (e.g., in a Slurm pipeline where the two run in separate processes).

**In your generation script:**

```python
from neuropilot import ConfigLoader, JobCreator
from neuropilot.runtime import JobContext

loader = ConfigLoader("config.yml")
creator = JobCreator(loader.load())
jobs = creator.create()

for job in jobs:
    JobContext.from_job(job).to_file(f"job_files/{job['job_id']}.json")
```

**In your job script:**

```python
from neuropilot.runtime import JobContext
from neuropilot import ResultEmitter

ctx = JobContext.from_file("job_files/<job_id>.json")

# Use context to drive training
# ...

manager = ResultEmitter(
    root_dir="results",
    output_pattern="{task_name}/{dataset_name}/{job_id}.json",
    fmt="json"
)
manager.create_and_emit_result(ctx, results={"accuracy": 0.91}, status="success")
```

---

## Multiprocessing with JobRunner

Set `max_workers > 1` to run jobs in parallel using Python's `multiprocessing` module:

```python
from neuropilot import JobRunner

class MyRunner(JobRunner):
    def run_one(self, job: dict):
        # Your execution logic here
        pass

runner = MyRunner(jobs, max_workers=4)
runner.run()
```

> **Note:** Make sure your job scripts and any resources they access are safe for concurrent use.

---

## Slurm Integration

NeuroPilot doesn't ship a Slurm backend, but it's straightforward to wire one up via `JobRunner`:

```python
import subprocess
from neuropilot import JobRunner
from neuropilot.runtime import JobContext

class SlurmRunner(JobRunner):
    def run_one(self, job: dict):
        ctx = JobContext.from_job(job)
        ctx.to_file(f"job_files/{job['job_id']}.json")

        subprocess.run([
            "sbatch",
            "--job-name", job["job_id"],
            "my_slurm_script.sh",
            f"job_files/{job['job_id']}.json"
        ], check=True)
```

Your `my_slurm_script.sh` can then load the context with `JobContext.from_file(...)`.

---

## WandB Integration

### Logging to WandB inside a job script

```python
import wandb
from neuropilot.runtime import JobContext

ctx = JobContext.from_file("job_files/<job_id>.json")

wandb.init(
    project="my_project",
    name=ctx.job_id,
    config=ctx.params
)

# ... run training, log metrics ...
wandb.log({"accuracy": 0.91})
```

### Collecting WandB results back into NeuroPilot

```python
from neuropilot.results import WandbFetcher
from neuropilot.aggregation import aggregate_results

fetcher = WandbFetcher(project="my_project", entity="my_team")
results = fetcher.collect(jobs)

df = aggregate_results(results, index_fields=["job_id", "task_name", "dataset_name"])
print(df)
```

> **Requires:** `wandb` package installed in your environment.

---

## Custom Result Collection via API

For full control, subclass `RemoteResultFetcher`:

```python
from neuropilot.results import RemoteResultFetcher

class MLflowFetcher(RemoteResultFetcher):
    def fetch_result(self, job):
        import mlflow
        run = mlflow.get_run(job["mlflow_run_id"])
        return {
            "job_id": job["job_id"],
            "results": run.data.metrics,
            "status": run.info.status,
            "params": run.data.params,
            "dataset_name": job.get("dataset_name"),
            "task_name": job.get("task_name"),
        }

fetcher = MLflowFetcher()
results = fetcher.collect(jobs)
```

Any results returned from a `RemoteResultFetcher` subclass are fully compatible with `aggregate_results()`.

---

## Tips

- **Everything is a suggestion.** The components (`ConfigLoader`, `JobCreator`, `JobRunner`, `ResultEmitter`) are intentionally decoupled. Use as many or as few as you need.
- **Override freely.** Most behavior (output paths, schema fields, metadata inference) can be overridden or subclassed.
- **Config inheritance is powerful.** See `examples/example5-inheritance.yml` for multi-level task inheritance with param merging and deletion.