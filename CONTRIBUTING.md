# Contributing to TensorPlay

We want to make contributing to TensorPlay as easy and transparent as possible.

## Development Process

1.  **Fork the repository**: Click the 'Fork' button on the repository page.
2.  **Clone your fork**:
    ```bash
    git clone https://github.com/your-username/tensorplay.git
    cd tensorplay
    ```
3.  **Create a branch**:
    ```bash
    git checkout -b my-new-feature
    ```
4.  **Make your changes**: Implement your feature or fix.
5.  **Run tests**: Ensure all tests pass.
    ```bash
    pytest test/
    ```
6.  **Commit your changes** using the Conventional Commits convention (see
    [Commit conventions](#commit-conventions)):
    ```bash
    git commit -am 'feat(kernels): add pooling backward kernel'
    ```
7.  **Push to the branch**:
    ```bash
    git push origin my-new-feature
    ```
8.  **Submit a Pull Request**: Go to the original repository and create a Pull
    Request. The PR title must follow the same convention (it becomes the
    squash-commit subject) and is checked by the `pr-title` workflow.

## Triage and Labels

New issues and PRs are labeled `needs triage` and routed onto the project
board automatically. During triage a maintainer will:

1. Confirm the report (reproduce bugs, check the measured numbers of
   performance reports).
2. Apply a `release notes: *` label to PRs (the labeler does this from paths;
   adjust when needed) — these labels bucket merged PRs for the release notes.
3. Assign a milestone (`v1.0.0`, `v1.1.0`, ...) that buckets the work per
   release; milestones carry no due dates.

Useful labels: `kind/flaky` (flaky test/build), `breaking change` (must be
documented in the release notes), `regression`, `performance`, and
`platform: linux|macos|windows`.

## Commit conventions

TensorPlay uses [Conventional Commits](https://www.conventionalcommits.org/),
enforced by `tools/commit_schema.py` (commit-msg hook and PR title check):

```
type(scope): subject          # subject <= 100 chars, no trailing period
feat(compiler)!: subject      # breaking change: add '!' and a footer
```

- Types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`,
  `revert`, `style`, `test`.
- `feat` and `fix` **require** a scope. Scopes map 1:1 to the
  `release notes: *` labels used to compile the release notes:

  | Scope | Label | Covers |
  | --- | --- | --- |
  | frontend | release notes: frontend | Python frontend, Tensor API, bindings |
  | autograd | release notes: autograd | TPX autograd engine, forward-mode AD |
  | compiler | release notes: compiler | tensorplay.compile, Stax, Triton codegen |
  | kernels | release notes: kernels | p10 core library, CPU/CUDA kernels |
  | cuda | release notes: cuda | CUDA runtime, streams, graphs, allocator |
  | build | release notes: build | Build system, packaging, CI, release tooling |
  | docs | release notes: docs | Documentation and examples |

- Breaking changes: append `!` before `:` and describe the migration in a
  `BREAKING CHANGE:` footer; also set the `breaking change` label on the PR.
- Local setup: `pip install pre-commit && pre-commit install --hook-type pre-commit --hook-type commit-msg`.
- Versions follow pytorch's rules (`version.txt`); `cz bump` is never used.
  Release notes are drafted with `cz changelog --dry-run` plus the
  `release notes: *` PR buckets (see RELEASE.md).

## Coding Style

- We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
- We use [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) for C++ code.
- Python code should be typed using type hints where possible.
- Use `black` for Python formatting and `clang-format` for C++.

## Building from Source

### Prerequisites
- Python 3.9+
- CMake 3.18+
- C++20 compatible compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- CUDA Toolkit (optional, for GPU support)

### Installation

TensorPlay is built with scikit-build-core through the standard PEP 517
interface declared in `pyproject.toml`:

```bash
# Install (add -v for verbose output)
pip install .

# CPU-only build (USE_*/BUILD_*/CMAKE_* env vars are forwarded to CMake,
# mirroring pytorch's EnvVarForwarding)
USE_CUDA=OFF pip install .

# Editable install for development
pip install -e .

# Build a wheel
python -m build --wheel
```

Build requirements (scikit-build-core, cmake, ninja, ...) are fetched
automatically in isolated PEP 517 builds. For `--no-build-isolation`
installs, install them first:

```bash
pip install -r requirements-build.txt
```

`MAX_JOBS=N` caps build parallelism; the package
version comes from `version.txt`.

## Running Tests

We use `pytest` for testing.

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_tensor_basic.py
```

Flaky test? Rerun automatically to confirm:

```bash
pytest --reruns 2 test/test_something.py
```

If a test only fails intermittently, open an issue with the `kind/flaky`
label rather than silently rerunning in CI.

## Documentation

Documentation is built using Sphinx.

```bash
cd docs
pip install -r requirements.txt
make html
```

The generated HTML files will be in `docs/_build/html`.

## License

By contributing, you agree that your contributions will be licensed under its Apache 2.0 License.
