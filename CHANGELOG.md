# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://github.com/vweevers/common-changelog),
and this project roughly adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

## 0.1.4, 0.1.5, 0.1.6 - 2025-10-01
No changes. These releases are intended to test the security of our github actions publishing pipeline.

## 0.1.3 - 2025-10-01

### Changed
- `make test-fast` now runs efficient (non-ML-model) tests.
- Fast tests are run on every github workflow push.

## 0.1.2 - 2025-10-01

### Changed
- Initial release.
- [CLI] Changed `--var-name` to have `-s` alias, freeing up `-n`.
### Added
- [CLI] Added a `-n` parameter to limit the number of operations listed/selected/ran by the `krnel-graph` CLI.
### Fixed
