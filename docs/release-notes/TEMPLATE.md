# TensorPlay X.Y.0 Release Notes

<!--
Compile these notes with the following workflow:

1. Bucket: every merged PR between the previous release tag and this one
   carries a "release notes: *" label (applied automatically from paths via
   .github/labeler.yml, adjust manually when needed). List them with:

     gh pr list --repo lexing-2026/TensorPlay --state merged --limit 200 \
       --search "label:\"release notes: compiler\" merged:>=<prev-release-date>" \
       --json number,title,labels

2. Curate: fold the PR list into the sections below. Every entry links its
   PR/issue. Backwards-incompatible entries show before/after code.

3. Publish: save this file as docs/release-notes/vX.Y.0.md and commit it
   before pushing the release tag. publish.yml picks it up automatically via
   `gh release create --notes-file`; without the file it falls back to
   GitHub's auto-generated notes.
-->

- [Highlights](#highlights)
- [Backwards Incompatible Changes](#backwards-incompatible-changes)
- [Deprecations](#deprecations)
- [New Features](#new-features)
- [Improvements](#improvements)
- [Bug fixes](#bug-fixes)
- [Performance](#performance)
- [Tracked Regressions](#tracked-regressions)
- [Documentation](#documentation)
- [Developers](#developers)

# Highlights

<!-- 3-10 headline items of the release, one line each. -->

# Backwards Incompatible Changes

<!-- Group by subsystem. For each change: what changed, why, and a minimal
before/after code block. -->

# Deprecations

<!-- APIs kept in this release but scheduled for removal; name the target
version. Write "None." when empty. -->

# New Features

<!-- Group by subsystem: Autograd / Custom operators / Compiler / CUDA /
Kernels / Frontend ... -->

# Improvements

# Bug fixes

# Performance

<!-- Measured numbers with hardware, shape and methodology
(e.g. "min-of-200, RTX 4090 D"), like the CHANGELOG entries. -->

# Tracked Regressions

<!-- Known regressions shipped with this release that are already tracked for
a future fix: link the issue/PR, state the affected workload and the plan
(fix version or workaround). Write "None." when empty. -->

# Documentation

# Developers

<!-- Build system, packaging, CI and release-tooling changes. -->

# Installation

```bash
# CPU wheels from PyPI
pip install tensorplay==X.Y.0

# CUDA wheels from the TensorPlay index (choose cu124, cu126, or cu130)
pip install tensorplay==X.Y.0 \
  --index-url https://download.tensorplay.cn/whl/cu124/ \
  --extra-index-url https://pypi.org/simple
```
