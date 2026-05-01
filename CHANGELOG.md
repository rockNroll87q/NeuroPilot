# Changelog

All notable changes to this project will be documented here.

## [0.1.1] — 2026-05-01

Update release.

- Used as a training dependency in [Dibble et al., 2026 (medRxiv)](https://doi.org/10.64898/2026.03.27.26349489)
- Updated documentation, added improved pip support, github test workflows

## [0.1.0] — 2025-06-01

Initial release.

- Declarative YAML/JSON experiment configuration via `ConfigLoader`
- Task inheritance, parameter merging, and sweep expansion via `JobCreator`
- Platform-agnostic job execution with multiprocessing support via `JobRunner`
- Result emission and collection via `ResultEmitter` and `ResultLoader`
- Remote result fetching with built-in WandB support and extensible `RemoteResultFetcher`
- Result aggregation into pandas DataFrames via `aggregate_results`
- Config validation with informative error messages
