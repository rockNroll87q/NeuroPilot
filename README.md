# NeuroPilot

**NeuroPilot** is a lightweight, extensible framework for automating deep learning/machine learning/data science experiments. If you have multiple datasets and multiple experimental methods for each--this framework might be for you! Just create a YAML/JSON file defining the datasets and methods you're working with, write a few lines of Python code, and you're off!

## Why?

Basically, the existing toolchains for automating deep learning experiments didn't *quite* fit everything I needed to do for my research. I generally use Wandb, and that works for a lot of cases. However, I found myself looking for a way to unify all of my experimentation across different datasets. That involves a number of different datasets with (1) unique formats, (2) unique target variables, (3) unique model parameters, (4) comparing results across datasets, etc. I was doing all of that manually, so my PI said to me "there has to be a better way!" So, I decided to try and write a lightweight framework which could solve this problem! I present to you: [a new standard](https://xkcd.com/927/). All in all, I wrote this for my own research, but if **you** happen to find it helpful or would like to use it, that would be great! If I can save others time (or service subscription costs), then I'm all for that.

### What this framework is

NeurPilot is an automation tool. It is intended to be plugged into your existing DL/ML/data science/whatever code without too much fuss, and reduce your workload by automating all your experimental pipeline. It can either be very simple, or quite complex if you used the advanced features. If you're at the start of a new project, that's even better! Integration will be a bit easier.

The general idea behind the framework is: everything is a suggestion. I provide a lot of useful tools, but they're all fairly decoupled from each other so you're free to take (or leave) whichever parts you want.

### What this framework isn't

This framework isn't (currently) a visual one, with a dashboard or interface. It's just python code. It can absolutely be paired with dashboard tools, but you'll have to tie them together yourself. This framework is also not a one-size-fits-all solution. I've made it as flexible as I think is reasonable, but it's bound to miss some use-cases.

## 📦 Project Structure

```
NeuroPilot/
├── neuropilot/            # Core logic (config loader, job creator, etc.)
│   ├── config_loader.py
│   ├── ...
│
├── examples/               # Example YAML configs
│   ├── example1-basic.yml
│   ├── example2-sweep.yml
│   └── ...
│
├── tests/                   # Unit tests
│   ├── test_config_loader.py
│   ├── test_job_creator.py
│
├── __init__.py             # Exposes main classes to the package
├── spec.md                 # Markdown version of the YAML protocol spec
└── README.md               # This file
```


## 🚀 Quick Start

1. **Install requirements** (if any):

```bash
pip install -r requirements.txt  # currently none strictly required
```

2. **Create a config file (YAML example)**:

```yaml
tasks:
  base_finetune:
    script: train.py
    epochs: 40

  finetune:
    extends: base_finetune
    learning_rate: 1e-4
    param_set:
      dropout: [0.1, 0.2]

datasets:
  my_dataset:
    root: /data/my_dataset
    tasks:
      - name: finetune
        output_vars: [label1, label2]
        param_set:
          dropout: null  # Remove inherited sweep
```

If you want to get in the weeds, see our [`spec.md`](spec.md)

3. **Generate a job list**:

```python
from neuropilot import ConfigLoader, JobCreator
from neuropilot.runtime import JobContext

loader = ConfigLoader("examples/basic.yml") # alternatively, load from .json
validated_config = loader.load()

creator = JobCreator(validated_config)
jobs = creator.create()

for job in jobs:
    print(job)
    # *Optionally*, write job description to disk for your job script
    JobContext.from_job(job).to_file("<path to job file>")
```

4. **Use JobRunner to run your jobs**

You can extend the `JobRunner` interface as a clean way to run the defined jobs:

```python
from neuropilot import JobRunner

class MyRunner(JobRunner):
    def run_one(self, job:dict):
        # Run your job script here!
        return

# Set max_workers > 1 for multiprocessing
runner = MyRunner(jobs, max_workers=1) 
runner.run()
```

5. **Store your results**

Use `ResultEmitter` to produce result files.

```python
from neuropilot import ResultEmitter
from neuropilot.runtime import JobContext

ctx = JobContext.from_file("<path to job file>")

# Define output layout and format inside your job script
manager = ResultEmitter(
    root_dir="results",
    output_pattern="{task_name}/{dataset_name}/{job_id}.json",
    fmt="json"
)

# Emit result for a single job
manager.create_and_emit_result(
    ctx,
    results={"accuracy": 0.91, "f1": 0.88},
    status="success",
    extra={"notes": "trial_1"}
)
```

6. **Collect your results**

... And then collect them into a list automagically once all your jobs are finished.

```python
manager = ResultEmitter(
    root_dir="results",
    output_pattern="{task_name}/{dataset_name}/{job_id}.json",
    fmt="json"
)

all_results = manager.collect()
```

7. **Aggregate into a single dataframe**

Once you’ve emitted results for each job, you can aggregate them into a structured table for easy comparison:

```python
from neuropilot import aggregate_results

manager = # One of the job collectors
results = # manager collection function

# Aggregate into a comparison table
df = aggregate_results(
    results,
    index_fields=["job_id", "task_name", "dataset_name"],
    auto_detect_metrics=True,
    auto_detect_params=True,
    strict=False  # lenient mode fills missing fields with NaN
)

print(df)
```

Sample Output:

| job_id | task_name | dataset_name | accuracy | f1   | lr     | dropout |
|--------|-----------|--------------|----------|------|--------|---------|
| job1   | finetune  | ds1          | 0.91     | NaN  | 0.001  | 0.2     |
| job2   | finetune  | ds2          | 0.88     | 0.83 | NaN    | 0.1     |


- Missing fields are filled with `NaN` when `strict=False`
- You can sort, filter, group, or export the table using normal `pandas` operations


**For more advanced examples (3rd party tool integration, sharing contexts across scripts, etc) see below or check out our examples.**

## Features Overview

- Simple, declarative YAML/JSON task definitions  
- Dataset-specific overrides and parameter sweeps  
- Reusable task definitions with support for inheritance and merging  
- Automatic expansion into per-job configurations  
- Platform-agnostic — you plug in the execution backend (Slurm, subprocess, etc.)  
- Built-in validation with informative error messages
- Clean, composable class interfaces (ConfigLoader, JobCreator, JobRunner, ResultEmitter, etc.)
- As unopinionated as possible (most things can be overridden, though we encourage certain schema)

## Examples

Explore the `examples/` directory for working YAML configuration templates:

- `example1-basic.yml`: One task on one dataset
- `example2-sweep.yml`: Parameterized task with a `param_set`
- `example3-override.yml`: Per-dataset overrides of task parameters
- `example4-multi-dataset.yml`: Multiple datasets sharing and customizing tasks
- `example5-inheritance.yml`: Multi-level inheritance, overrides, and param_set merging and deletion

Run any of the above using `examples/proc_example.py <path>` to see the produced job objects.

## 📊 Result Creation & Collection

NeuroPilot includes a flexible and modular result management system through classes like `ResultEmitter`, `ResultLoader`, and `RemoteResultFetcher`.

✅ Automatically emit per-job result files in `.json` or `.yaml`  
✅ Structure your output using configurable file path patterns  
✅ Collect and aggregate results across jobs  
✅ Infer job metadata from file paths (e.g., task name, dataset name)  

### Collecting Results

By default, any metadata that can be inferred from the file path (e.g., `task_name`, `dataset_name`) will be added back to each result object when using `ResultEmitter`.

You can also opt-out of metadata inference, or override the root directory:

```python
manager.collect(root_dir="custom_results", infer_metadata=False)
```

### Collecting Results from 3rd Party Frameworks

**On Disk**

If using a 3rd-party framework (like Wandb), then you can instead use the `ResultLoader`
with custom collection patterns to grab result files from disk:

```python
collector = ResultLoader(
    root_dir="wandb/",
    pattern="*/files/results-{job_id}.yaml",
    fmt="yaml",
    infer_metadata=True
)

results = collector.collect(strict=True)

```

**From a Custom API**

For full flexibility, you can subclass `ResultFetcher` to pull results from APIs
like WandB, MLflow, or even your own service. The only requirement is that you return
a dictionary that matches the expected result schema.

✅ Using WandB (Requires wandb package to be already installed)

```python
from neuropilot.results import WandbFetcher

fetcher = WandbFetcher(project="my_project", entity="my_team")
results = fetcher.collect(jobs)  # `jobs` is a list of job dicts
```

    ℹ️ `WandbFetcher` is an optional utility and only works if wandb is installed.

✅ Using a Custom API Client

```python
from neuropilot.results import RemoteResultFetcher

class MyAPIClient(RemoteResultFetcher):
    def fetch_result(self, job):
        job_id = job["job_id"]
        # Simulate fetching result from your custom API
        result_data = my_api.get_result_by_id(job_id)

        return {
            "job_id": job_id,
            "results": result_data["metrics"],
            "status": result_data["status"],
            "params": job.get("params", {}),
            "dataset_name": job.get("dataset_name"),
            "task_name": job.get("task_name"),
        }

# Usage
client = MyAPIClient()
results = client.collect(jobs)
```

Results from either method (disk or API) are compatible with `aggregate_results()`
(see below) for building summary tables, comparisons, and filtering.

## Running Tests

```bash
python -m unittest discover tests
```

**Author**: Austin Dibble / Brain Imaging and Artificial Intelligence Research Lab

**License**: MIT
