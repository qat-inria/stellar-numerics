# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [#6] added approximate Gaussian conversion bounds for mixed state with standard protocol (stellar rank <= 1>).

### Fixed

- [#4](https://github.com/qat-inria/stellar-numerics/pull/4) fixed typing issues by making `StellarProfile` covariant with the type of their state.

### Changed

- [#6] modified the logic of `max_trace_distance` in `conversion.py` to use pattern matching (`match`/`case` syntax) instead of `if`/`else`.

## [0.1] - 2026-02-11

### Added

First commit.

### Fixed

### Changed
