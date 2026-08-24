# TensorPlay docs

This directory hosts the Sphinx documentation system, laid out to mirror
`third_party/pytorch/docs`:

```
docs/
├── Makefile            # sphinx-build driver (SPHINXPROJ=TensorPlay)
├── requirements.txt    # doc toolchain
└── source/
    ├── conf.py         # Sphinx configuration (adapted from upstream)
    ├── index.md        # master toctree
    └── *.md            # one reference page per module, mirroring upstream
```

## Build

```bash
cd docs
pip install -r requirements.txt
make html
# output: docs/build/html/index.html
```

## Page alignment policy

Every `source/*.md` page that has an upstream counterpart
(`third_party/pytorch/docs/source/*.md`) mirrors it section-for-section:

1. Same section headings and order as the upstream page.
2. Each `autosummary`/`autofunction`/`autoclass` entry from upstream is kept if
   and only if the corresponding symbol exists in the current `tensorplay`
   package (resolved by name at generation time). Missing symbols are dropped.
3. Pages may end with an explicit **"TensorPlay-specific additions"** section
   listing public tensorplay-only symbols not covered by the upstream lists.

The pages are regenerated against the live package with:

```bash
python3 docs/source/scripts/gen_reference_pages.py
```

## Deviations from upstream

- **Theme**: upstream uses the private `pytorch_sphinx_theme2`; we use
  `sphinx_book_theme`.
- **Extensions**: `myst_nb`, `sphinxcontrib.katex`, `sphinx_design`,
  `jupyter_sphere`, `sphinx-tippy`, `sphinx-jinja` and friends are not used;
  notebooks/kernels are executed upstream only. We use plain `myst_parser`.
- **Makefile targets**: `figures`, `opset`, `exportdb`, `docset`,
  `html-stable` depend on upstream scripts/assets this repo does not carry.
- **Prose**: long upstream tutorials/prose inside reference pages are
  intentionally not copied; pages keep upstream structure + API listings plus a
  short intro. Narrative guides belong in the website, not here.
- **Static pages**: `compiler.md`, `stax.md` (partly), `vision.md`,
  `audio.md` have no upstream counterpart. `vision.md`, `audio.md` and parts of
  `compiler.md` avoid autodoc because their extension modules require optional
  dependencies or are still moving.
- **Engineering notes**: pre-existing `docs/*.md` handoff notes at this root
  are unrelated to the Sphinx tree and left untouched.
