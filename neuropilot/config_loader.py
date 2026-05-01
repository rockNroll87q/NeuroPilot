"""
Created on Friday, 28 March 2025.

@authors:
* Austin Dibble, University of Glasgow

-------------------------------------
YAML Task Protocol Specification in Brief
-------------------------------------

Purpose:
    Define a structured YAML format to run deep learning tasks across multiple datasets,
    allowing parameterized variants, per-dataset overrides, and reusable task definitions
    via inheritance.

Structure:

1. tasks:
    A dictionary of globally defined task templates.
    Each task includes:
        - static parameters (e.g., epochs, lr)
        - param_set (optional): defines keys with lists of values to sweep
        - extends (optional): inherit fields from another task
        - description (optional): string to describe this task

    Inheritance:
        - A task may extend another task using the 'extends' key
        - All static fields are inherited and overridden by the child
        - param_sets are **merged** across inheritance and dataset levels:
            * Keys from lower levels (e.g., dataset, task entry) override higher levels (e.g., base task)
            * Keys not overridden are preserved
            * The resulting param_set is the merged union of all levels
        - If a param_set key is set to null (YAML `~` or Python `None`), it will explicitly remove that key from the inherited sweep
        - Inheritance is resolved before validation
        - Cycles or undefined parents raise validation errors


    Example:
    tasks:
      base_finetune:
        script: train.py
        epochs: 50
        learning_rate: 1e-5

      finetune_all:
        extends: base_finetune
        finetune_all_layers: true
        param_set:
          dropout: [0.1, 0.2]

2. datasets:
    Dataset-specific job instructions. Each dataset includes:
        - root (str): base path to the dataset
        - description (optional): string to describe this dataset
        - override (optional): static parameter overrides applied to all tasks
        - param_set (optional): param sweep definitions applied to all tasks
        - additional static parameters may also be defined
        - tasks: list of task runs, each including:
            - name: reference to a task defined above
            - override (optional): static parameter overrides
            - param_set (optional): param sweep for this specific run
            - description (optional): optional description for this run

    Example:
    datasets:
      dataset_alpha:
        root: /mnt/data/alpha
        learning_rate: 0.001
        param_set:
          dropout: [0.1, 0.2]
        tasks:
          - name: finetune_all
            output_vars: [label1]
          - name: finetune_all
            output_vars: [label2]
            override:
              learning_rate: 2e-5
            param_set:
              finetune_depth: [1, 3]
              dropout: null  # Explicitly remove dropout sweep inherited from above

Precedence:
    - override > entry-level param_set > dataset-level param_set > global (task-level) param_set > static params
    - param_sets are merged, not replaced — later definitions override earlier ones only at the key level
    - param_set expands to the cartesian product of the final merged sweep
    - param_set keys must not conflict with static fields at the same scope
    - Inherited param_set keys can be removed by assigning `null` (YAML) or `None` (Python)


Validation Rules:
    - All task references in datasets must match defined tasks
    - param_set keys must not conflict with static fields at the same level
    - task inheritance must be acyclic and refer to existing tasks only
"""

import yaml
import json
from typing import List, Dict, Union, Any, Set, Optional, Tuple
from pathlib import Path

class InheritanceError(Exception):
    """Raised when inheritance resolution fails (e.g. due to cycles or missing base)."""
    pass

def _resolve_task_inheritance(tasks: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
    resolved = {}
    inherited_keys = {}
    inheritance_map = {}

    def resolve(key: str, trail: Set[str]) -> Dict[str, Any]:
        if key in resolved:
            return resolved[key]

        if key not in tasks:
            raise InheritanceError(f"Experiment '{key}' is not defined.")

        if key in trail:
            raise InheritanceError(f"Inheritance cycle detected: {' -> '.join(list(trail) + [key])}")

        exp = tasks[key]
        base_key = exp.get("extends")

        inheritance_map[key] = base_key

        if not base_key:
            resolved[key] = dict(exp)
            inherited_keys[key] = set()
            return resolved[key]

        trail.add(key)
        base = resolve(base_key, trail)
        trail.remove(key)

        # Merge base and child
        merged = dict(base)
        inherited = set(merged.keys())

        # Handle 'param_set' merge manually
        child_param_set = exp.get("param_set", {})
        base_param_set = base.get("param_set", {})

        # Merge param_set with child override
        if isinstance(base_param_set, dict) and isinstance(child_param_set, dict):
            merged_param_set = dict(base_param_set)
            merged_param_set.update(child_param_set)
            merged["param_set"] = merged_param_set
        elif "param_set" in exp:
            merged["param_set"] = child_param_set
        elif "param_set" in base:
            merged["param_set"] = base_param_set

        # Update the rest (excluding extends and param_set)
        merged.update({k: v for k, v in exp.items() if k not in {"extends", "param_set"}})


        resolved[key] = merged
        inherited_keys[key] = inherited
        return merged

    for key in tasks:
        resolve(key, set())

    return resolved, inherited_keys, inheritance_map


class ConfigValidationError(Exception):
    """Raised when the configuration is invalid."""
    pass

class ConfigLoader:
    # In our overlap and param checks, these are the built-in keywords which cannot be used as user-defined variables.
    _g_inbuilt_keywords = {"param_set", "description", "extends", \
                            "name", "override", "tasks", "root"}


    def __init__(self, config_source: Union[str, Path, Dict]):
        """
        Initializes the configuration loader.

        Args:
            config_source (str or dict): Path to a YAML config file or a pre-parsed dictionary.
        """
        self._source = config_source
        self._raw_config = None
        self._validated_config = None

    def load(self) -> dict:
        """
        Load and validate the configuration.

        Returns:
            dict: A validated and normalized configuration dictionary.

        Raises:
            ConfigValidationError: If validation fails.
        """
        self._raw_config = self._load_source()
        self._validated_config = self._validate(self._raw_config.copy())
        return self._validated_config
    
    def save_raw(self, path: Union[str, Path]):
        """
        Save the raw (unvalidated) config to a file (.json or .yaml).

        Args:
            path (str or Path): Path to write to. Extension determines format.
        """
        if self._raw_config is None:
            raise ValueError("Raw config is not loaded yet. Run .load() before calling this function.")

        path = Path(path)
        with open(path, 'w') as f:
            if path.suffix == ".json":
                json.dump(self._raw_config, f, indent=2)
            elif path.suffix in {".yml", ".yaml"}:
                yaml.safe_dump(self._raw_config, f)
            else:
                raise ValueError("Unsupported file extension for saving. Use .json or .yaml")

    def save_validated(self, path: Union[str, Path]):
        """
        Save the validated (fully resolved) config to a file (.json or .yaml).

        Args:
            path (str or Path): Path to write to. Extension determines format.
        """
        if self._validated_config is None:
            raise ValueError("Validated config is not available. Call `load()` first.")

        path = Path(path)
        with open(path, 'w') as f:
            if path.suffix == ".json":
                json.dump(self._validated_config, f, indent=2)
            elif path.suffix in {".yml", ".yaml"}:
                yaml.safe_dump(self._validated_config, f)
            else:
                raise ValueError("Unsupported file extension for saving. Use .json or .yaml")


    def _load_source(self) -> dict:
        """
        Load the configuration from file or return directly if already a dict.

        Returns:
            dict: Raw configuration dictionary.

        Raises:
            TypeError: If input is not a str or dict.
        """
        if isinstance(self._source, (str, Path)):
            path = Path(self._source)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, 'r') as f:
                if path.suffix == ".json":
                    return json.load(f)
                return yaml.safe_load(f)
        elif isinstance(self._source, dict):
            return self._source
        else:
            raise TypeError("Config source must be a file path or a dictionary.")

    def _validate(self, config: dict) -> dict:
        """
        Validate and normalize the configuration.

        Args:
            config (dict): Raw configuration dictionary.

        Returns:
            dict: Normalized configuration dictionary.

        Raises:
            ConfigValidationError: If validation fails.
        """
        if not isinstance(config, dict):
            raise ConfigValidationError("Config must be a dictionary.")

        tasks = config.get("tasks")
        datasets = config.get("datasets")

        if not isinstance(tasks, dict):
            raise ConfigValidationError("Missing or invalid 'tasks' section. Must be a dictionary.")

        if not isinstance(datasets, dict):
            raise ConfigValidationError("Missing or invalid 'datasets' section. Must be a dictionary.")

        tasks, inherited_keys, inheritance_map = _resolve_task_inheritance(tasks)
        config["tasks"] = tasks
        self._inherited_keys = inherited_keys
        config["_task_inheritance"] = inheritance_map

        self._validate_tasks(tasks)

        for dataset_name, dataset_def in datasets.items():
            if not isinstance(dataset_def, dict):
                raise ConfigValidationError(f"Dataset '{dataset_name}' must be a dictionary.")
            if "root" not in dataset_def:
                raise ConfigValidationError(f"Dataset '{dataset_name}' is missing required field: 'root'.")
            if "tasks" not in dataset_def or not isinstance(dataset_def["tasks"], list):
                raise ConfigValidationError(f"Dataset '{dataset_name}' must have a list under 'tasks'.")

            # Check for param_set conflicts at the dataset level
            if "param_set" in dataset_def:
                self._check_param_conflicts(dataset_name, dataset_def)
        
            # Make sure the overrides, if present, are a dictionary object
            dataset_override = dataset_def.get('override', {})
            if not isinstance(dataset_override, dict):
                raise ConfigValidationError(f"In '{dataset_name}', 'override' must be a dictionary.")

            for i, exp_entry in enumerate(dataset_def["tasks"]):
                if not isinstance(exp_entry, dict):
                    raise ConfigValidationError(f"Experiment entry #{i} in dataset '{dataset_name}' must be a dictionary.")
                name = exp_entry.get("name")
                if name not in tasks:
                    raise ConfigValidationError(f"Dataset '{dataset_name}' references undefined task '{name}'.")
                global_exp = tasks[name]

                if "param_set" in exp_entry:
                    self._check_param_conflicts(name, exp_entry)

                exp_override = exp_entry.get('override', {})
                if not isinstance(exp_override, dict):
                    raise ConfigValidationError(f"In '{dataset_name}:{name}', 'override' must be a dictionary.")
                
                # Check user variables of this task listing under the dataset against the global task def
                conflicts = self._check_user_var_conflicts(global_exp, exp_entry)
                if conflicts:
                    raise ConfigValidationError(f"In {dataset_name}:{name}, user-provided variables [{', '.join(conflicts)}] \
conflict with those given in the task definition. Use override if required.")
                
                # Check the user variables of the broader dataset def against the global task def
                conflicts = self._check_user_var_conflicts(global_exp, dataset_def)
                if conflicts:
                    raise ConfigValidationError(f"In {dataset_name}, user-provided variables [{', '.join(conflicts)}] \
conflict with those given in the task definition '{name}'. Use override if required.")
                
                # Check the user variables of the dataset-task def against the dataset def
                conflicts = self._check_user_var_conflicts(dataset_def, exp_entry)
                if conflicts:
                    raise ConfigValidationError(f"In {dataset_name}:{name}, user-provided variables [{', '.join(conflicts)}] \
conflict with those given in the dataset definition '{dataset_name}'. Use override if required.")

        return config
    
    def _validate_tasks(self, tasks:dict):
        for task_name, task_def in tasks.items():
            inherited = self._inherited_keys.get(task_name, set())
            self._check_param_conflicts(task_name, task_def, inherited_keys=inherited)

    def _check_param_conflicts(self, scope_name: str, exp_def: dict, inherited_keys: Optional[Set[str]] = None):
        """
        Ensure no overlap between static params and param_set keys, unless they're inherited (and intended to be overridden).

        Args:
            scope_name (str): Name of the task or dataset+task scope
            exp_def (dict): The definition containing static keys and param_set
            inherited_keys (set[str] or None): If provided, keys from a parent that are considered defaultable

        Raises:
            ConfigValidationError: If conflicts are found that are not inherited
        """
        static_keys = ConfigLoader._get_object_user_vars(exp_def)
        param_set = exp_def.get("param_set", {})

        if not isinstance(param_set, dict):
            raise ConfigValidationError(f"In '{scope_name}', 'param_set' must be a dictionary.")

        param_keys = set(param_set.keys())

        # Exclude inherited keys from the conflict set
        check_keys = static_keys if inherited_keys is None else (static_keys - inherited_keys)
        conflicts = check_keys & param_keys

        if conflicts:
            raise ConfigValidationError(
                f"In '{scope_name}', the following keys appear in both static fields and param_set: {sorted(conflicts)}"
            )
        
        # Also, verify that if a param set item isn't a list, we yield an error
        for p_key in param_keys:
            if not isinstance(param_set[p_key], (list, tuple, set)):
                raise ConfigValidationError(
                    f"In '{scope_name}, 'param_set' entry {p_key} must be a list, tuple or set! It is currently a {type(param_set[p_key])}."
                )

    
    def _check_user_var_conflicts(self, global_def:dict, local_def:dict):
        """
        Get the overlapping variables between some user definition (like in a dataset definition, or dataset-task)
        and the more global-level variables from the global task definition.

        Args:
            global_def (dict): Object from the task's definition
            local_def (dict): dataset or dataset-task object.

        Returns:
            a set of overlapping keys, if any
        """
        global_var_keys = ConfigLoader._get_object_user_vars(global_def)
        exp_keys = ConfigLoader._get_object_user_vars(local_def)
        return global_var_keys & exp_keys

    @staticmethod
    def _get_object_user_vars(obj: dict):
        return set(obj.keys()) - ConfigLoader._g_inbuilt_keywords

    @property
    def raw(self) -> dict:
        """Access the raw configuration before validation."""
        return self._raw_config

    @property
    def validated(self) -> dict:
        """Access the validated and normalized configuration."""
        return self._validated_config
