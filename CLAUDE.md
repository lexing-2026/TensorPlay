# CLAUDE.md

Guidance for AI coding assistants (Claude Code and similar) and human
contributors working in this repository. All commands assume the repository
root as the working directory. See also [CONTRIBUTING.md](CONTRIBUTING.md)
for the contribution process and [RELEASE.md](RELEASE.md) for the release
handbook.

## Project Overview

TensorPlay is a learner-first, DIY-ready framework for tensors, kernels, and
custom hardware. The Python frontend (`tensorplay`) is backed by a C++ core:
the **p10** kernel library, the **TPX** autograd engine (records every
operation into an explicit DAG and replays the chain rule node by node), and
the **Stax** compiler stack. Op declarations live in `config/` and drive code
generation; the blog series under `docs/blogs/` is the guided tour, one
pillar per post.

## Repository Layout

| Path | Contents |
| --- | --- |
| `tensorplay/` | Python package (frontend API); the compiled `_C` extension lands here |
| `p10/` | Core C++ library: CPU/CUDA kernels, `include/` + `src/` |
| `tpx/` | Autograd engine (C++) |
| `stax/` | Compiler stack (graph capture, codegen) |
| `src/` | Python bindings, distributed support, version glue |
| `config/` | Op contract: `native_functions.yaml`, `derivatives.yaml`, `tags.yaml` |
| `test/` | pytest suite (`test/test_*.py`) |
| `benchmark/` | Benchmark harness |
| `docs/` | Sphinx documentation, blog series, release notes, whitepaper |
| `tools/` | Maintenance scripts (commit schema, release notes, versioning) |
| `.github/` | Workflows, labels, wheel platform matrix |

## Build

Prerequisites: Python >= 3.10, < 3.14; CMake >= 3.18 (< 4.0); a C++20
compiler (MSVC 2022 on Windows, GCC/Clang on Linux); optional CUDA Toolkit
(set `CMAKE_CUDA_ARCHITECTURES` to target specific GPUs) or ROCm 7.2.x
(HIP backend is built from source).

```bash
# Install the toolchain once for fast iterative builds
pip install -r requirements-build.txt

# Editable install for development — recompiles only what changed
pip install -e . --no-build-isolation

# Or produce a wheel without installing
python -m build --wheel
```

- All builds go through the PEP 517 interface declared in `pyproject.toml`
  (scikit-build-core + CMake). Do not invoke `cmake` by hand for packaging.
- Build options are driven by environment variables, not `-D` flags
  (e.g. `USE_CUDA=OFF pip install .` for a CPU-only build).
- On success, `import tensorplay` picks up the compiled `_C` extension from
  the installed package. If `import` fails with a stale-looking error, suspect
  an out-of-date build rather than a code bug, and rebuild.

## Generated Code

The op contract in `config/` (`native_functions.yaml`, `derivatives.yaml`,
`tags.yaml`) is the single source of truth for operator declarations.
Bindings and related artifacts are generated from it during the build —
never hand-edit generated files (e.g. `*Generated*` headers); change the
contract and rebuild instead.

## Testing

```bash
pytest test/                       # full suite
pytest test/test_nn_utils_norm.py  # single file
pytest test/test_nn_utils_norm.py -k relu   # single test by keyword
```

Match the existing test style in `test/`: plain pytest files named
`test_<topic>.py`. Numerical checks compare against explicit expected values.

## Documentation

```bash
cd docs
pip install -r requirements.txt
make html        # output: docs/build/html/index.html
```

## Lint

Ruff is configured in `pyproject.toml` and enforced by CI on every PR.

```bash
ruff check .      # lint
ruff check --fix . # apply automatic fixes
```

`pre-commit` hooks are available:

```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Commits and Pull Requests

Commits follow [Conventional Commits](https://www.conventionalcommits.org/),
enforced by `tools/commit_schema.py` on both the commit-msg hook and the PR
title (a PR title becomes the squash-commit subject on merge):

```
type(scope): subject          # subject <= 100 chars, no trailing period
feat(compiler)!: subject      # breaking change: add '!' and a footer
```

- Types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`,
  `revert`, `style`, `test`. `feat` and `fix` **require** a scope.
- Scopes map 1:1 to the `release notes: *` labels that compile the release
  notes: `frontend`, `autograd`, `compiler`, `kernels`, `cuda`, `build`,
  `docs`.
- Breaking changes: append `!` before `:`, add a `BREAKING CHANGE:` footer
  describing the migration, and set the `breaking change` label on the PR.
- Versions follow the semantic rules in `version.txt`; release-note drafts
  come from `cz changelog --dry-run` plus `tools/collect_release_notes.py`.
  Never bump versions by hand outside `version.txt`.

## CI Overview

| Trigger | Workflow | What runs |
| --- | --- | --- |
| Pull request | `pull` | Lint + one CPU smoke build (linux, Python 3.12) — the required check; plus docs build, labeler, dependency scanning |
| Push to `main` / `release/*` | `trunk` | Full platform/variant wheel matrix; skipped entirely for docs- and CI-meta-only pushes |
| Tag `v*.*.0` / `v*.*.0-rc*` | `publish` | cu130 wheel matrix, validation, GitHub Release |
| Manual dispatch | `nightly` | Rolling nightly wheels for the `nightly` channel |

Note that docs-, `.github/`-, and CI-meta-only pushes intentionally do not
start the build matrix; those changes ride along in the next code-bearing
wheel.

## Releases and Channels

| Channel | Version | Where |
| --- | --- | --- |
| stable | `X.Y.Z` | PyPI + stable wheel indexes |
| release candidate | `X.Y.0rcN` (tag `vX.Y.0-rcN`) | GitHub prerelease only |
| nightly | `X.Y.0.dev<UTC date>[+cuXXX\|+cpu]` | rolling `nightly` GitHub Release + `whl/nightly/<variant>/` indexes |

```bash
# CUDA 13.0 stable from the TensorPlay index (PyPI stays the extra index)
pip install tensorplay --index-url https://download.tensorplay.cn/whl/cu130/ \
  --extra-index-url https://pypi.org/simple

# Nightly preview
pip install --pre tensorplay \
  --index-url https://download.tensorplay.cn/whl/nightly/cu130/ \
  --extra-index-url https://pypi.org/simple
```

Wheel tags must match your Python version (e.g. `cp310` for 3.10). Only the
latest nightly build per variant is kept. The full process lives in
[RELEASE.md](RELEASE.md).
