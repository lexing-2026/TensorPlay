# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Implemented `nn.Bilinear` module and functional interface.
- Improved `nn.Linear`, `nn.Conv*`, `nn.BatchNorm*` and other modules (added `__constants__`, type hints, `factory_kwargs`).
- Added factory kwargs support (`device`, `dtype`) to module initializers.
- Documentation building infrastructure using Sphinx.

### Changed
- Improved `nn.Linear` initialization.

## [1.0.0rc0] - 2025-12-19
### Added
- Initial release of TensorPlay.
- Basic Tensor operations (P10).
- Autograd engine (TPX).
- Static graph optimization (Stax).
- Neural network modules (`nn.Linear`, `nn.Conv2d`, etc.).
