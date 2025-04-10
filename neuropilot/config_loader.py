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
        - static parameters (epochs, lr, etc.)
        - param_set (optional): keys with lists of values to sweep
        - extends (optional): inherit fields from another task
        - description (optional): string to describe this task

    Inheritance:
        - An task may extend another by key name
        - Inherited fields are merged, with child fields overriding parent fields. 
            This also applies to param_sets overriding static fields. param_sets will be merged.
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

2. datasets:
    Dataset-specific job instructions. Each dataset includes:
        - root (str): base path to dataset
        - description (str) (optional): string to describe this dataset
        - override (optional): dataset-wide param override for all tasks
        - param_set (optional): dataset-wide param sweep for all tasks
        - other static params may be defined here
        - tasks: list of tasks to run
            - name: task key from global section
            - override (optional): static parameter overrides
            - param_set (optional): replaces higher-level param_set
            - description (optional): string to describe this specific run

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

Precedence:
    - override > local param_set > dataset-level param_set > global param_set > static params
    - param_set expands to cartesian product of values
    - Conflicts between static and param_set keys in the same scope raise validation errors

Validation Rules:
    - All task names in datasets must match defined tasks
    - param_set keys must not also appear as static params in the same block
    - task inheritance trees must be acyclic and refer only to defined tasks


"""

import yaml
from typing import List, Dict, Union, Any, Set, Optional, Tuple
from pathlib import Path

class InheritanceError(Exception):
    """Raised when inheritance resolution fails (e.g. due to cycles or missing base)."""
    pass

def _resolve_task_inheritance(tasks: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
    resolved = {}
    inherited_keys = {}

    def resolve(key: str, trail: Set[str]) -> Dict[str, Any]:
        if key in resolved:
            return resolved[key]

        if key not in tasks:
            raise InheritanceError(f"Experiment '{key}' is not defined.")

        if key in trail:
            raise InheritanceError(f"Inheritance cycle detected: {' -> '.join(list(trail) + [key])}")

        exp = tasks[key]
        base_key = exp.get("extends")

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

    return resolved, inherited_keys


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
        self._validated_config = self._validate(self._raw_config)
        return self._validated_config

    def _load_source(self) -> dict:
        """
        Load the configuration from file or return directly if already a dict.

        Returns:
            dict: Raw configuration dictionary.

        Raises:
            TypeError: If input is not a str or dict.
        """
        if isinstance(self._source, str) or isinstance(self._source, Path):
            with open(self._source, 'r') as f:
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

        tasks, inherited_keys = _resolve_task_inheritance(tasks)
        config["tasks"] = tasks
        self._inherited_keys = inherited_keys

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
