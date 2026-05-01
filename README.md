# NeuroPilot

**NeuroPilot** is a lightweight, extensible framework for automating deep learning, machine learning, and data science experiments. Define your datasets and methods in a YAML/JSON config, write a few lines of Python, and let the framework handle the rest.

> 💡 **Used in published research!** NeuroPilot was a training dependency for:
> **[NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation](https://doi.org/10.64898/2026.03.27.26349489)** — *Dibble et al., 2026 (medRxiv)*
>
> See also: [neurofm-training](https://github.com/rockNroll87q/neurofm-training) — the training repository built on NeuroPilot.

---

## Why NeuroPilot?

Existing toolchains didn't quite cover the needs of cross-dataset experimentation — things like unique formats per dataset, unique target variables, unique model parameters, and comparing results across datasets. NeuroPilot was built to unify that pipeline without heavy vendor lock-in. I present to you: [a new standard](https://xkcd.com/927/).

**What it is:** An automation tool that plugs into your existing DL/ML code with minimal friction. Most components are decoupled, so you can take what you need and leave the rest.

**What it isn't:** A visual tool with a dashboard (though it pairs well with them), or a one-size-fits-all solution. It's Python code that gets out of your way.

---

## 📦 Project Structure

```
NeuroPilot/
├── neuropilot/            # Core logic (config loader, job creator, etc.)
│   ├── config_loader.py
│   └── ...
├── examples/              # Example YAML configs
│   ├── example1-basic.yml
│   ├── example2-sweep.yml
│   └── ...
├── tests/                 # Unit tests
│   ├── test_config_loader.py
│   └── test_job_creator.py
├── docs/                  # Extended documentation
│   ├── results.md         # Result emission, collection, and aggregation
│   ├── spec.md            # Full YAML protocol specification
│   └── advanced.md        # 3rd-party integrations and advanced patterns
├── __init__.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Install
 

**Directly from GitHub:**
 
```bash
pip install git+https://github.com/your-org/NeuroPilot.git
```
 
With optional integrations:
 
```bash
pip install "neuropilot[wandb] @ git+https://github.com/your-org/NeuroPilot.git"
pip install "neuropilot[mlflow] @ git+https://github.com/your-org/NeuroPilot.git"
```
 
**From a local clone:**
 
```bash
git clone https://github.com/your-org/NeuroPilot.git
pip install ./NeuroPilot
```


### 2. Create a config file

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

See [`spec.md`](spec.md) for the full configuration reference.

### 3. Generate a job list

```python
from neuropilot import ConfigLoader, JobCreator
from neuropilot.runtime import JobContext

loader = ConfigLoader("examples/basic.yml")  # also accepts .json
config = loader.load()

creator = JobCreator(config)
jobs = creator.create()

for job in jobs:
    print(job)
    # Optionally persist the job description for use in your job script
    JobContext.from_job(job).to_file("<path to job file>")
```

### 4. Run your jobs

Extend `JobRunner` to plug in any execution backend (subprocess, Slurm, etc.):

```python
from neuropilot import JobRunner

class MyRunner(JobRunner):
    def run_one(self, job: dict):
        # Your execution logic here
        pass

runner = MyRunner(jobs, max_workers=1)  # Set max_workers > 1 for multiprocessing
runner.run()
```

### 5. Emit, collect, and aggregate results

See **[docs/results.md](docs/results.md)** for the full guide, including:
- Emitting per-job result files (`.json` / `.yaml`)
- Collecting results with path-pattern inference
- Aggregating into a `pandas` DataFrame for comparison

---

## ✨ Features

- Declarative YAML/JSON task definitions
- Dataset-specific parameter overrides and sweeps
- Task inheritance and merging
- Automatic expansion into per-job configurations
- Platform-agnostic execution (Slurm, subprocess, or custom)
- Built-in config validation with informative error messages
- Composable interfaces: `ConfigLoader`, `JobCreator`, `JobRunner`, `ResultEmitter`, `ResultLoader`, `RemoteResultFetcher`, and more
- Unopinionated — most defaults can be overridden

---

## 📖 Examples

See the `examples/` directory for working YAML configs:

| File | Description |
|---|---|
| `example1-basic.yml` | One task on one dataset |
| `example2-sweep.yml` | Parameterized task with `param_set` |
| `example3-override.yml` | Per-dataset overrides |
| `example4-multi-dataset.yml` | Multiple datasets sharing tasks |
| `example5-inheritance.yml` | Multi-level inheritance, merges, and deletions |

Run any example with:

```bash
python examples/proc_example.py <path>
```

For 3rd-party integrations (WandB, MLflow, custom APIs) and advanced patterns, see **[docs/advanced.md](docs/advanced.md)**.

---

## 🧪 Running Tests

```bash
python -m unittest discover tests
```

---

**Author:** Austin Dibble / Brain Imaging and Artificial Intelligence Research Lab  
**License:** MIT