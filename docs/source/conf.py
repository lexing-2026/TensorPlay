# TensorPlay documentation build configuration file, adapted from
# repo actually uses; see docs/README.md for the deviation list).

import os
import sys

# -- Path setup ------------------------------------------------------------

# Add the repository root so autodoc can import tensorplay.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tensorplay  # noqa: E402

# -- Project information -----------------------------------------------------

project = "TensorPlay"
copyright = "2025, zlx"
author = "TensorPlay contributors"
release = getattr(tensorplay, "__version__", "main")

version = release

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
# autosummary writes .rst stubs into source/generated, so both suffixes must
# be registered even though all handwritten pages are MyST markdown.
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = []
master_doc = "index"

# MyST options used by this documentation build.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
# dollarmath's default delimiter matching treats any pair of `$` as an equation,
# which swallows literal shell variables in prose. Requiring the delimiters to
# sit flush against the content keeps those as text while real formulas render.
myst_dmath_allow_space = False
myst_heading_anchors = 4

# The following options show the correct "next/previous" banners when supported
# by the selected theme.

# autodoc / autosummary options
autosummary_generate = True
numpydoc_show_class_members = False

# autosectionlabel throws warnings if section names are duplicated.
# Do not throw a warning for duplicated section names in different documents.
autosectionlabel_prefix_document = True

# Disable docstring inheritance.
autodoc_inherit_docstrings = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]

html_title = f"{project} documentation"

# -- Intersphinx -----------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Nitpick and warnings ----------------------------------------------------------

nitpicky = False


import re

_SIGNATURE_ONLY = re.compile(r"^\s*[\w.]+\([^()]*\)\s*->", re.S)


def _strip_reference_assets(app, what, name, obj, options, lines):
    # Generated assets under docs/source/scripts/*_images are not shipped
    # here; the directives would otherwise emit one unreadable-image warning
    # per scheduler/activation entry.
    lines[:] = [
        line for line in lines
        if not re.match(r"^\s*\.\.\s+image::\s+\S*(?:scripts/|_images/)", line)
        and "Henry2019" not in line
    ]
    # Native ops whose __doc__ is just the generated C++ signature
    # ("op(Tensor a, *, int dim=0) -> Tensor"): rendered as rst it trips
    # inline-markup warnings and carries no information beyond the stub
    # signature autodoc already prints.
    text = "\n".join(lines).strip()
    if text and "\n" not in text and _SIGNATURE_ONLY.match(text):
        lines[:] = []


def setup(app):
    app.connect("autodoc-process-docstring", _strip_reference_assets)
