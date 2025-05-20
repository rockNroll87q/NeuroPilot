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
from typing import List, Dict, Optional, Any

class ResultSet:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        index_fields: List[str],
        metric_fields: List[str],
        param_fields: List[str],
        results_raw: List[Dict],
        other_fields: Optional[List[str]] = None,
        output_field: Optional[str] = None,
        flattened: Optional[bool] = False,
    ):
        self.df = dataframe
        self.index_fields = index_fields
        self.metric_fields = metric_fields
        self.param_fields = param_fields
        self.output_field = output_field
        self.results_raw = results_raw
        self.flattened = flattened
        self.other_fields = other_fields or []

    def filter_by(self, **kwargs) -> "ResultSet":
        mask = pd.Series(True, index=self.df.index)
        for key, value in kwargs.items():
            mask &= self.df[key] == value
        filtered_df = self.df[mask].copy()

        # Get the subset of original results matching the filter
        filtered_jobs = [
            r for r in self.results_raw
            if all(r.get(k) == v for k, v in kwargs.items())
        ]

        metric_fields = _auto_detect_fields(filtered_jobs, key="results", 
                                            nested=self.output_field is not None or self.flattened, 
                                            flatten=self.flattened)
        param_fields = _auto_detect_fields(filtered_jobs, key="params")

        relevant_cols = (
            self.index_fields +
            metric_fields +
            param_fields +
            self.other_fields + 
            ([self.output_field] if self.output_field else [])
        )
        relevant_cols = [c for c in relevant_cols if c in filtered_df.columns]

        return ResultSet(
            dataframe=filtered_df[relevant_cols],
            index_fields=self.index_fields,
            metric_fields=metric_fields,
            param_fields=param_fields,
            results_raw=filtered_jobs,
            other_fields=self.other_fields,
            output_field=self.output_field,
            flattened=self.flattened
        )
    
    def get_values_for(self, var_name: str) -> set:
        """
        Return a set of unique non-null values for a given variable/column.

        Args:
            var_name (str): Column name (e.g., 'dataset_name', 'task_name', etc.)

        Returns:
            Set[Any]: Set of unique values for that column in the current DataFrame.
        """
        if var_name not in self.df.columns:
            raise ValueError(f"Column '{var_name}' not found in the dataset.")
        return set(self.df[var_name].dropna().unique())

    def summary(self) -> pd.DataFrame:
        return self.df.describe(include="all")

    def to_dataframe(self) -> pd.DataFrame:
        return self.df.copy()


# ---- Modified aggregate_results ----
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
    long_output_field: str = "output_var",
    other_fields: Optional[List[str]] = None,
) -> ResultSet:

    if not results:
        raise ValueError("No results provided for aggregation.")

    index_fields = index_fields or []
    other_fields = other_fields or []

    metric_fields = metric_fields or (
        _auto_detect_fields(results, key="results", nested=long_format or flatten_nested, flatten=flatten_nested)
        if auto_detect_metrics else []
    )
    param_fields = param_fields or (
        _auto_detect_fields(results, key="params")
        if auto_detect_params else []
    )

    rows = []
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"Result #{i} is not a dictionary (got type {type(result)}).")

        base_row = _extract_fields(result, index_fields, section="index", index=i, strict=strict)
        param_row = _extract_nested_fields(result.get("params", {}), param_fields, section="params", index=i, strict=strict)
        base_row.update(param_row)

        base_row.update(_extract_fields(result, other_fields, section="other", index=i, strict=strict))

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

    df = pd.DataFrame(rows)

    return ResultSet(
        dataframe=df,
        index_fields=index_fields,
        metric_fields=metric_fields,
        param_fields=param_fields,
        results_raw=results,
        other_fields=other_fields,
        output_field=long_output_field if long_format else None,
        flattened=flatten_nested
    )


# ---- Helpers (unchanged except _auto_detect_fields) ----

def _auto_detect_fields(results: List[Dict], key: str, nested: bool = False, flatten: bool = False) -> List[str]:
    fields = set()
    for r in results:
        section = r.get(key)
        if isinstance(section, dict):
            if nested and _is_nested_dict(section):
                for outer_key, sub in section.items():
                    if isinstance(sub, dict):
                        if flatten:
                            for inner_key in sub:
                                fields.add(f"{outer_key}.{inner_key}")
                        else:
                            fields.update(sub.keys())
            else:
                fields.update(section.keys())
    return sorted(fields)


def _extract_fields(source: Dict, keys: List[str], section: str, index: int, strict: bool) -> Dict[str, Optional[Any]]:
    output = {}
    for key in keys:
        if key in source:
            output[key] = source[key]
        elif strict:
            raise ValueError(f"Missing '{key}' in result['{section}'] for result #{index}")
        else:
            output[key] = None
    return output

def _extract_nested_fields(nested: Dict, keys: Optional[List[str]], section: str, index: int, strict: bool) -> Dict[str, Optional[Any]]:
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
