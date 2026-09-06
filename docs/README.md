# TensorPlay docs

This directory hosts the Sphinx documentation system.

```
docs/
├── Makefile            # sphinx-build driver (SPHINXPROJ=TensorPlay)
├── requirements.txt    # doc toolchain
└── source/
    ├── conf.py         # Sphinx configuration
    ├── index.md        # master toctree
    └── *.md            # one reference page per module
```

## Build

```bash
cd docs
pip install -r requirements.txt
make html
# output: docs/build/html/index.html
```
