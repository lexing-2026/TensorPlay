Rolling preview builds of TensorPlay, published automatically from `main`.
Expect the newest capabilities — and the occasional rough edge.

## Install

Wheels cover Python 3.10–3.13. `--pre` is required to select dev builds.

### CPU (Linux x86_64 / aarch64, macOS arm64, Windows x86_64)

```bash
pip install --pre tensorplay \
  --index-url https://download.tensorplay.cn/whl/nightly/cpu/ \
  --extra-index-url https://pypi.org/simple
```

### CUDA 12.4 (Linux x86_64, Windows x86_64)

```bash
pip install --pre tensorplay \
  --index-url https://download.tensorplay.cn/whl/nightly/cu124/ \
  --extra-index-url https://pypi.org/simple
```

### CUDA 12.6 (Linux x86_64, Windows x86_64)

```bash
pip install --pre tensorplay \
  --index-url https://download.tensorplay.cn/whl/nightly/cu126/ \
  --extra-index-url https://pypi.org/simple
```

### CUDA 13.0 (Linux x86_64, Windows x86_64)

```bash
pip install --pre tensorplay \
  --index-url https://download.tensorplay.cn/whl/nightly/cu130/ \
  --extra-index-url https://pypi.org/simple
```

### Pin a specific build

```bash
pip install --pre tensorplay==1.0.0.dev20260907+cu130 \
  --index-url https://download.tensorplay.cn/whl/nightly/cu130/ \
  --extra-index-url https://pypi.org/simple
```

## About this channel

- Built and smoke-tested on every code-bearing `main` push; a scheduled run at 02:30 UTC fills any gaps.
- Only the newest build of each variant is kept here — older dev wheels are removed as new ones land.
- The same wheels are served from the plain indexes at `https://download.tensorplay.cn/whl/nightly/<variant>/`.
- For production work prefer the [stable releases](https://github.com/lexing-2026/TensorPlay/releases) or [PyPI](https://pypi.org/project/tensorplay/).
- Hit a problem? [Open a bug report](https://github.com/lexing-2026/TensorPlay/issues/new?template=bug-report.yml) and include the version printed by `python -c "import tensorplay; print(tensorplay.__version__)"`.
