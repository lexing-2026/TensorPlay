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
6.  **Commit your changes**:
    ```bash
    git commit -am 'Add some feature'
    ```
7.  **Push to the branch**:
    ```bash
    git push origin my-new-feature
    ```
8.  **Submit a Pull Request**: Go to the original repository and create a Pull Request.

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
