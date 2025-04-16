"""
Created on Wednesday, 2 April 2025.

@authors:
* Austin Dibble, University of Glasgow

---------------------------------
Utility for aggregating job results
---------------------------------

example:
    df = aggregate_results(
        results,
        index_fields=['job_id', 'task_name'],
        auto_detect_metrics=True,
        param_fields=['lr', 'dropout']
    )

See `aggregate_results` docstring for details.

"""
import pandas as pd
from typing import List, Dict, Optional


def aggregate_results(
    results: List[Dict],
    index_fields: Optional[List[str]] = None,
    metric_fields: Optional[List[str]] = None,
    param_fields: Optional[List[str]] = None,
    auto_detect_metrics: bool = False,
    auto_detect_params: bool = False,
    strict: bool = True,
    flatten_nested: bool = False,
    long_format: bool = False,
    long_output_field: str = "output_var"
) -> pd.DataFrame:
    """
    Aggregates a list of result dictionaries into a structured pandas DataFrame.

    Each result dictionary is expected to contain:
        - Top-level fields (e.g., job_id, task_name)
        - A 'results' dictionary (possibly nested)
        - A 'params' dictionary with parameter values

    This function supports multiple output modes depending on your result structure and analysis needs.

    Args:
        results (List[Dict]):
            A list of job result dictionaries, typically produced by a ResultEmitter or ResultLoader.

        index_fields (List[str], optional):
            Top-level keys from each result to include as identifying columns (e.g., ['job_id', 'task_name']).

        metric_fields (List[str], optional):
            Specific metric keys to extract from the 'results' dictionary. 
            If not provided and `auto_detect_metrics=True`, all keys across results will be inferred.

        param_fields (List[str], optional):
            Parameter keys to extract from the 'params' dictionary. 
            If not provided and `auto_detect_params=True`, they will be inferred.

        auto_detect_metrics (bool):
            If True, automatically detect all metric keys from the results.

        auto_detect_params (bool):
            If True, automatically detect all param keys from the results.

        strict (bool):
            If True, raise an error if any expected field is missing.
            If False, missing values are filled with `None` (NaN in the DataFrame).

        flatten_nested (bool):
            If True and the 'results' field contains nested dictionaries (e.g., per output variable),
            the metrics will be flattened into columns like `var1.accuracy`.

        long_format (bool):
            If True and the 'results' field contains nested dictionaries,
            a long-format table is returned with one row per output variable.
            In this mode:
                - `metric_fields` become separate columns
                - `long_output_field` is added as a column with the output variable key (e.g., 'label')

        long_output_field (str):
            Column name to use for the output variable identifier in long-format mode.

    Returns:
        pd.DataFrame:
            A DataFrame where each row corresponds to a job (or output variable if long_format=True).
            Columns include selected index fields, metrics, and parameters.

    Raises:
        ValueError:
            If expected keys are missing and `strict=True`, or if the results list is malformed.

    Example:
        >>> df = aggregate_results(
        ...     results,
        ...     index_fields=["job_id", "dataset_name"],
        ...     auto_detect_metrics=True,
        ...     auto_detect_params=True
        ... )

    Example (long format with nested metrics):
        >>> df = aggregate_results(
        ...     results,
        ...     long_format=True,
        ...     long_output_field="label",
        ...     auto_detect_metrics=True
        ... )
    """

    if not results:
        raise ValueError("No results provided for aggregation.")

    index_fields = index_fields or []

    metric_fields = metric_fields or (
        _auto_detect_fields(results, key="results", nested=long_format or flatten_nested)
        if auto_detect_metrics else None
    )
    param_fields = param_fields or (
        _auto_detect_fields(results, key="params") if auto_detect_params else None
    )

    rows = []
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"Result #{i} is not a dictionary (got type {type(result)}).")

        base_row = _extract_fields(result, index_fields, section="index", index=i, strict=strict)
        param_row = _extract_nested_fields(result.get("params", {}), param_fields, section="params", index=i, strict=strict)
        base_row.update(param_row)

        metrics = result.get("results", {})
        if not isinstance(metrics, dict):
            if strict:
                raise ValueError(f"Missing or invalid 'results' block in result #{i}.")
            metrics = {}

        if long_format and _is_nested_dict(metrics):
            rows.extend(_format_long_nested(metrics, base_row, metric_fields, long_output_field, i, strict))
        elif flatten_nested and _is_nested_dict(metrics):
            rows.append(_format_flat(metrics, base_row))
        else:
            metric_row = _extract_nested_fields(metrics, metric_fields, section="results", index=i, strict=strict)
            base_row.update(metric_row)
            rows.append(base_row)

    return pd.DataFrame(rows)


def _auto_detect_fields(results: List[Dict], key: str, nested: bool = False) -> List[str]:
    fields = set()
    for r in results:
        section = r.get(key)
        if isinstance(section, dict):
            if nested and _is_nested_dict(section):
                for sub in section.values():
                    if isinstance(sub, dict):
                        fields.update(sub.keys())
            else:
                fields.update(section.keys())
    return sorted(fields)


def _extract_fields(
    source: Dict,
    keys: List[str],
    section: str,
    index: int,
    strict: bool
) -> Dict[str, Optional[any]]:
    output = {}
    for key in keys:
        if key in source:
            output[key] = source[key]
        elif strict:
            raise ValueError(f"Missing '{key}' in result['{section}'] for result #{index}")
        else:
            output[key] = None
    return output


def _extract_nested_fields(
    nested: Dict,
    keys: Optional[List[str]],
    section: str,
    index: int,
    strict: bool
) -> Dict[str, Optional[any]]:
    if not isinstance(nested, dict):
        if strict:
            raise ValueError(f"Missing or invalid '{section}' block in result #{index}.")
        return {}

    output = {}
    for key in keys or nested.keys():
        if key in nested:
            output[key] = nested[key]
        elif strict:
            raise ValueError(f"Missing '{key}' in result['{section}'] for result #{index}")
        else:
            output[key] = None
    return output


def _format_flat(metrics: Dict, base_row: Dict) -> Dict:
    flat_metrics = {}
    for outer_key, subdict in metrics.items():
        if isinstance(subdict, dict):
            for inner_key, val in subdict.items():
                flat_metrics[f"{outer_key}.{inner_key}"] = val
    row = base_row.copy()
    row.update(flat_metrics)
    return row


def _format_long_nested(
    metrics: Dict,
    base_row: Dict,
    metric_fields: Optional[List[str]],
    output_field: str,
    index: int,
    strict: bool
) -> List[Dict]:
    """
    Format nested results into long format with one row per output_var.
    Produces a row with each metric as a column.
    """
    long_rows = []
    for outer_key, subdict in metrics.items():
        if not isinstance(subdict, dict):
            continue

        row = base_row.copy()
        row[output_field] = outer_key

        for metric_key in (metric_fields or subdict.keys()):
            if metric_key in subdict:
                row[metric_key] = subdict[metric_key]
            elif strict:
                raise ValueError(
                    f"Missing '{metric_key}' in result['results'][{outer_key}] for result #{index}"
                )
            else:
                row[metric_key] = None

        long_rows.append(row)

    return long_rows



def _is_nested_dict(d: Dict) -> bool:
    return all(isinstance(v, dict) for v in d.values())
