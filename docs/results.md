# Result Management

NeuroPilot provides a modular result management system through `ResultEmitter`, `ResultLoader`, and `RemoteResultFetcher`. This guide covers emitting results from jobs, collecting them afterward, and aggregating into a comparison table.

---

## Emitting Results

Use `ResultEmitter` inside your job script to write structured result files to disk.

```python
from neuropilot import ResultEmitter
from neuropilot.runtime import JobContext

ctx = JobContext.from_file("<path to job file>")

manager = ResultEmitter(
    root_dir="results",
    output_pattern="{task_name}/{dataset_name}/{job_id}.json",
    fmt="json"
)

manager.create_and_emit_result(
    ctx,
    results={"accuracy": 0.91, "f1": 0.88},
    status="success",
    extra={"notes": "trial_1"}
)
```

Supported formats: `json`, `yaml`.

---

## Collecting Results

Once all jobs have finished, collect results into a list:

```python
manager = ResultEmitter(
    root_dir="results",
    output_pattern="{task_name}/{dataset_name}/{job_id}.json",
    fmt="json"
)

all_results = manager.collect()
```

By default, any metadata inferable from the file path (e.g., `task_name`, `dataset_name`) is automatically attached to each result. You can override this behavior:

```python
manager.collect(root_dir="custom_results", infer_metadata=False)
```

---

## Aggregating into a DataFrame

Once results are collected, aggregate them into a structured table for easy comparison:

```python
from neuropilot import aggregate_results

df = aggregate_results(
    all_results,
    index_fields=["job_id", "task_name", "dataset_name"],
    auto_detect_metrics=True,
    auto_detect_params=True,
    strict=False  # lenient mode fills missing fields with NaN
)

print(df)
```

**Sample output:**

| job_id | task_name | dataset_name | accuracy | f1   | lr    | dropout |
|--------|-----------|--------------|----------|------|-------|---------|
| job1   | finetune  | ds1          | 0.91     | NaN  | 0.001 | 0.2     |
| job2   | finetune  | ds2          | 0.88     | 0.83 | NaN   | 0.1     |

- Missing fields are filled with `NaN` when `strict=False`
- The returned object is a standard `pandas` DataFrame — sort, filter, group, or export as usual

---

## Collecting from 3rd-Party Frameworks

### From Disk (e.g., WandB output files)

Use `ResultLoader` with a custom glob pattern to pick up result files written by external tools:

```python
from neuropilot.results import ResultLoader

collector = ResultLoader(
    root_dir="wandb/",
    pattern="*/files/results-{job_id}.yaml",
    fmt="yaml",
    infer_metadata=True
)

results = collector.collect(strict=True)
```

### From a Remote API (e.g., WandB, MLflow)

Subclass `RemoteResultFetcher` to pull results from any API. The only requirement is that `fetch_result` returns a dict matching the expected result schema.

```python
from neuropilot.results import RemoteResultFetcher

class MyAPIClient(RemoteResultFetcher):
    def fetch_result(self, job):
        job_id = job["job_id"]
        result_data = my_api.get_result_by_id(job_id)

        return {
            "job_id": job_id,
            "results": result_data["metrics"],
            "status": result_data["status"],
            "params": job.get("params", {}),
            "dataset_name": job.get("dataset_name"),
            "task_name": job.get("task_name"),
        }

client = MyAPIClient()
results = client.collect(jobs)
```

### WandB Built-in Fetcher

A convenience fetcher is included if WandB is already installed in your environment:

```python
from neuropilot.results import WandbFetcher

fetcher = WandbFetcher(project="my_project", entity="my_team")
results = fetcher.collect(jobs)  # `jobs` is a list of job dicts
```

> **Note:** `WandbFetcher` is an optional utility and requires the `wandb` package.

---

Results from any collection method — disk or API — are compatible with `aggregate_results()`.