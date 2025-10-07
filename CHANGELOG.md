# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://github.com/vweevers/common-changelog),
and this project roughly adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **CI**: Fast tests run on Windows, Mac, and Linux in Github Actions.

### Added

- Add `sha256sum` parameter for `Runner().from_parquet(...)` to allow loading datasets that don't exist, if any previous runner has already materialized them to the store.
- Add implementation for `AssignTrainTestSplitOp` to assign train/test splits using a random state.

## 0.1.4, 0.1.5, 0.1.6 - 2025-10-01
_No changes. These releases are intended to test the security of our github actions publishing pipeline._

## 0.1.3 - 2025-10-01

### Changed

- **CI**: `make test-fast` now runs efficient (non-ML-model) tests.
- **CI**: Fast tests are run on every github workflow push.

## 0.1.2 - 2025-10-01

_Initial release._

### Changed

- **CLI (Breaking)**: [CLI] Change `--var-name` to have `-s` alias instead of `-n`.

### Added

- **CLI**: Add a `-n` parameter to limit the number of operations listed/selected/ran by the `krnel-graph` CLI.
