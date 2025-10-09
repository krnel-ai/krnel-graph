# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://github.com/vweevers/common-changelog),
and this project roughly adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **CI**: Fast tests run on Windows, Mac, and Linux in Github Actions.
- `krnel-graph[cli]` extras are now part of `krnel-graph`. This just adds `rich`, `cyclopts`, and `humanize`. The `ml` extras group is still separate for now.
- `VectorColumnType.train_classifier()`'s `negatives` argument now defaults to `~positives`. This makes the common case (where all examples are either positive or negative) easier.
- `ScoreColumnType.evaluate()`'s 'gt_negatives` argument now defaults to `~gt_positives`, as above.

### Added

- Ops now have `.to_json()`, `.to_numpy()`, `.to_arrow()` convenience functions.
  - *Before:* `runner.to_json(op)`
  - *After:* `op.to_json()`
- **New op**: Add a `LLMLogitScores` operation for getting a distribution of output logits from a forward pass. This is useful for guardrail models, e.g. to compare the score of the `safe` token with the `unsafe` token.
- **New op**: `VectorToScalarOp` which selects one column of a `VectorColumnType`. This is like running `runner.to_numpy(vector_column)[:, 3]`. Accessible by the fluent API as `vector_column_type.col(3)`.
- **New op**: `PairwiseArithmeticOp` which performs pairwise arithmetic on two `ScoreColumnType`s. Now, `score_a + score_b`, `score_c - score_d`, etc are supported. `+`, `-`, `*`, `/`.
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
