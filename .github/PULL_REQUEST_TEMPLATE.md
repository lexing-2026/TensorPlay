<!--
Title convention: this repository uses Conventional Commits for PR titles,
e.g. `feat(compiler): add triton fallback for stax autotune` or
`fix(cuda): guard stream capture reentry`. The scope must be one of:
frontend, autograd, compiler, kernels, cuda, build, docs.
The PR title is checked automatically (pr-title workflow) and becomes the
squash-commit subject on merge.
-->

## Summary

What does this PR do, and why? Link the issue it closes (`Fixes #123`) if any.

## Checklist

- [ ] PR title follows the Conventional Commits convention (type + scope).
- [ ] `pytest test/` passes locally (flaky test? rerun with `pytest --reruns 2` and label `kind/flaky`).
- [ ] Docs updated if this changes user-facing behavior (`docs/`, README).
- [ ] The `release notes: *` label applied by the labeler is correct (adjust if needed).
- [ ] Breaking change: the `breaking change` label is set and the footer
      `BREAKING CHANGE:` describes the migration.

## Performance

If this PR is performance-related, include measured numbers with hardware,
shapes and methodology (e.g. "min-of-200, RTX 4090 D").
