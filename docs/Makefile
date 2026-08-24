# Minimal makefile for Sphinx documentation (mirrors upstream
# third_party/pytorch/docs/Makefile; torch-specific targets that depend on
# scripts this repo does not carry -- figures/opset/exportdb/docset -- are not
# included, see docs/README.md).

# You can set these variables from the command line.
SPHINXOPTS    ?= -j $(shell nproc) -WT --keep-going
SPHINXBUILD   ?= sphinx-build
SPHINXPROJ    ?= TensorPlay
SOURCEDIR     ?= source
BUILDDIR      ?= build
PYCMD         ?= python3

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Removing everything under 'build' and 'source/generated'.."
	@rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees $(SOURCEDIR)/generated
