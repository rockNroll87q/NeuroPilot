
import json
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
import re
from string import Formatter

from loguru import logger

class ResultLoader:
    """
    Collects and parses result files from disk using flexible path-matching and metadata inference.

    Supports:
        - Glob-style wildcards (e.g., **/*.json) for recursive pattern matching
        - Format-style placeholders (e.g., {job_id}, {dataset_name}) for extracting metadata
        - Automatic parsing of YAML or JSON result files
        - Optional metadata inference from file paths based on provided format pattern

    Path Matching Spec:
        The `pattern` argument is used both to locate files (via globbing) and to optionally
        extract metadata. It can include static components, wildcards (*), and format-style
        placeholders (e.g. {task_name}) to be converted into named fields.

        Example pattern: "{task_name}/{dataset_name}/results-{job_id}.yaml"
        Matches:         "finetune/ds1/results-abc123.yaml"
        Extracts:        task_name=finetune, dataset_name=ds1, job_id=abc123

        Wildcard-only pattern: "**/*.json"
        - Matches any JSON file recursively under root_dir
        - No metadata inference unless placeholder pattern is also used

    Arguments:
        root_dir (str | Path): Base directory to scan under.
        pattern (str): Path pattern to match files, relative to root_dir.
                    May include format-style placeholders and glob wildcards.
        fmt (str | None): File format to load ('json' or 'yaml'), or None to infer by extension.
        infer_metadata (bool): Whether to extract metadata from path using placeholders.

    Examples:
        ResultLoader("results", "**/*.json", fmt="json")

        ResultLoader(
            root_dir="results",
            pattern="{task}/{dataset}/result-{job_id}.yaml",
            fmt="yaml",
            infer_metadata=True
        )
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        pattern: str = "**/*.json",
        fmt: Optional[str] = None,
        infer_metadata: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.pattern = pattern
        self.fmt = fmt
        self.infer_metadata = infer_metadata

        if not self.root_dir.exists():
            raise ValueError(f"Provided root directory does not exist: {self.root_dir}")

        self._metadata_fields = self._extract_fields_from_pattern(pattern)
        self._compiled_regex = self._compile_pattern_to_regex(pattern)

    def collect(self, strict: bool = False) -> List[Dict]:
        """
        Collect and parse all result files matching the pattern.

        Args:
            strict (bool): If True, warn about skipped files or metadata mismatches.

        Returns:
            List[dict]: Parsed and annotated result objects.
        """
        results = []

        # Convert format-style pattern to glob-compatible pattern
        glob_pattern = self._pattern_to_glob(self.pattern)
        matched_files = list(self._glob_files(glob_pattern))

        for file in matched_files:
            if not file.is_file():
                continue
            if file.suffix not in {".json", ".yaml", ".yml"}:
                if strict:
                    logger.warning(f"Skipped unsupported file type: {file}")
                continue

            try:
                with open(file, "r") as f:
                    if self.fmt == "json" or file.suffix == ".json":
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
            except Exception as e:
                if strict:
                    raise ValueError(f"Failed to parse result file: {file}\n{e}")
                else:
                    logger.warning(f"Failed to parse file (skipped): {file}")
                    continue

            if self.infer_metadata and self._metadata_fields:
                try:
                    metadata = self._infer_metadata_from_path(file)
                    data.update(metadata)
                except Exception as e:
                    if strict:
                        logger.warning(f"Metadata inference failed for {file}: {e}")
                    continue

            results.append(data)

        return results

    def _glob_files(self, pattern: str):
        if pattern.startswith("**/") or pattern == "**":
            # Valid recursive pattern — use rglob directly
            return self.root_dir.rglob(pattern[3:] if pattern.startswith("**/") else "*")
        else:
            # Valid literal or non-recursive glob
            return self.root_dir.glob(pattern)


    def _extract_fields_from_pattern(self, pattern: str) -> List[str]:
        return [field for _, field, _, _ in Formatter().parse(pattern) if field]

    def _compile_pattern_to_regex(self, pattern: str) -> re.Pattern:
        """
        Convert a format-style path pattern to a regex that can extract metadata.
        Example: "{task}/{dataset}/results.yaml" => "(?P<task>[^/]+)/(?P<dataset>[^/]+)/results\.yaml"
        """
        regex_str = ""
        for literal, field, _, _ in Formatter().parse(pattern):
            if literal:
                regex_str += re.escape(literal)
            if field:
                regex_str += rf"(?P<{field}>[^/]+)"
        return re.compile(regex_str)

    def _infer_metadata_from_path(self, full_path: Path) -> Dict[str, str]:
        """
        Extract metadata by matching full path against compiled regex.

        Returns:
            dict of field values (e.g., {"job_id": "abc123"})

        Raises:
            ValueError if path does not match pattern
        """
        try:
            rel_path = full_path.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(f"File path {full_path} is not under root directory {self.root_dir}")

        rel_str = rel_path.as_posix() # We want to use / as part of our pattern system.
        match = self._compiled_regex.fullmatch(rel_str)
        if not match:
            raise ValueError(f"Path '{rel_str}' does not match the collection pattern structure.")

        return match.groupdict()
    
    def _pattern_to_glob(self, pattern: str) -> str:
        """
        Converts a format-style pattern (with placeholders) to a glob pattern
        by replacing each field with '*' and keeping all literal path structure.
        """
        glob_parts = []
        for literal, field, *_ in Formatter().parse(pattern):
            if literal:
                glob_parts.append(literal)
            if field:
                glob_parts.append("*")
        return "".join(glob_parts)

