# TensorPlay Release Handbook

How versions, channels and releases work in this repository. The design
uses one version source: `version.txt` is the single version source, release notes
are compiled from the `release notes: *` PR label family, and wheels ship
through three channels.

## Channels and versioning

| Channel | Version rule | Where wheels land |
| --- | --- | --- |
| stable | `X.Y.0` from `version.txt` | PyPI (CPU) + `whl/cuXXX/` indexes on `pypi-pages` |
| release candidate | `X.Y.0rcN` (tag `vX.Y.0-rcN`) | GitHub Release prerelease only — never PyPI, never the stable indexes |
| nightly | `X.Y.0.dev<UTC date>[+cuXXX\|+cpu]` | rolling `nightly` GitHub Release + `whl/nightly/<variant>/` indexes |

- Version source of truth: `version.txt` (main currently carries an alpha
  suffix, e.g. `1.0.0a0`, like the main release). Dev builds append
  `+git<sha>` via `tools/generate_tensorplay_version.py`.
- Variant local labels follow the `binary_populate_env.sh` rule: CUDA
  wheels get `+cuXXX`, Linux/Windows CPU wheels get `+cpu`, macOS wheels get
  no suffix.
- `cz bump` is never used; commitizen only drafts changelogs.

## Nightly (manual preview channel)

Dispatch `.github/workflows/nightly.yml` (no cron):

- `build-from-source`: builds the wheel matrix from the current ref with the
  computed `X.Y.0.dev<date>` version (override with `version`, republish the
  same day with `build_number > 1`).
- `upload-wheels`: publishes locally built wheels uploaded beforehand as an
  Actions artifact named `wheels` (all wheels must share one base version).

The nightly index overwrites `whl/nightly/<variant>/` on `pypi-pages` and
prunes older assets from the rolling release (prune aborts if no wheel of the
new version made it). Only the latest published day is kept.

```bash
pip install --pre tensorplay --index-url https://download.tensorplay.cn/whl/nightly/cu124/ --extra-index-url https://pypi.org/simple
```

## Release candidate process

1. Make sure `docs/release-notes/vX.Y.0.md` is drafted (see below) — RCs can
   ship without it (GitHub auto-notes are used), but the stable must not.
2. Tag and push: `git tag vX.Y.0-rc1 && git push origin vX.Y.0-rc1`.
3. `publish.yml` builds the full wheel matrix, creates a **prerelease**
   GitHub Release and uploads every wheel. PyPI and the stable indexes are
   untouched.
4. Iterate (`-rc2`, ...) until the release is clean.

## Stable release process

1. **Aggregate release notes** (draft, then curate by hand):
   ```bash
   python tools/collect_release_notes.py --from <prev-tag> --to HEAD --output draft.md
   uvx --from commitizen cz changelog --dry-run   # commit-type grouped draft
   ```
   Fold both drafts into `docs/release-notes/TEMPLATE.md` and commit the
   result as `docs/release-notes/vX.Y.0.md`.
2. **Set the version**: `version.txt` -> `X.Y.0` (no suffix).
3. **Tag and push**: `git tag vX.Y.0 && git push origin vX.Y.0`.
4. `publish.yml` builds the matrix, creates the release with
   `--notes-file docs/release-notes/vX.Y.0.md` (falls back to auto-notes),
   publishes CPU wheels to PyPI (trusted publishing, `pypi` environment) and
   republishes the CUDA indexes on `pypi-pages`.
5. **Verify**: `pip install tensorplay==X.Y.0` from PyPI and each
   `whl/cuXXX/` index; check https://tensorplay.cn.
6. **Open the next cycle**: bump `version.txt` to the next `X.(Y+1).0a0`,
   close the `vX.Y.0` milestone, ensure the next milestone exists.

## Release notes mechanics

- The labeler workflow applies `release notes: *` labels from paths
  (`.github/labeler.yml`); adjust labels manually when a PR spans subsystems.
- Commit scopes map 1:1 to the label family (see CONTRIBUTING.md table), so
  direct pushes stay bucketable too.
- `BREAKING CHANGE:` footers are collected into the Backwards Incompatible
  Changes section by `tools/collect_release_notes.py`.
- Known regressions shipping with a release go into the Tracked Regressions
  section of the notes.

## Milestones

Milestones bucket work per release (`v1.0.0`, `v1.1.0`, ...) and carry **no
due dates**. `milestone-guard.yml` auto-assigns the milestone derived from
`version.txt` to PRs that lack one.

## Project board setup (manual, UI)

The automation token cannot create ProjectsV2 boards (GraphQL write returns
403), so create the board once in the GitHub UI:

1. New project (board) named **TensorPlay Roadmap**, owned by `lexing-2026`.
2. Fields:
   - `Status` (built-in): Todo / In Progress / Done.
   - `Priority`: single select — High / Medium / Low.
   - `Milestone`: single select — v1.0.0 / v1.1.0 (extend per release).
3. Views:
   - **Status board**: board layout grouped by `Status`.
   - **By milestone**: table layout grouped by `Milestone`.
   - **Priority**: table layout grouped by `Priority`.
4. Update the project URL in `.github/workflows/add-to-project.yml`
   (replace `<NUMBER>`).
5. Create a fine-grained PAT with read/write access to the project and store
   it as the `PROJECT_TOKEN` repository secret; `add-to-project.yml` no-ops
   until the secret exists.

## Branch protection setup (manual, UI)

The automation token cannot read or write branch protection (REST returns
403), so configure it once under Settings → Branches → Add classic branch
protection rule for `main`:- [ ] Require a pull request before merging
  - [ ] Require approvals: 1
  - [ ] Dismiss stale pull request approvals when new commits are pushed
  - [ ] Require review from Code Owners
- [ ] Require status checks to pass before merging
  - [ ] Require branches to be up to date before merging
  - [ ] Required checks: `lint`, `pr-title` (add wheel-matrix job names from
        `trunk` as they stabilize)
- [ ] Require conversation resolution before merging
- [ ] Do not allow force pushes
- [ ] Do not allow deletions
- [ ] Include administrators

## Spam defense

- `spam-guard.yml` closes unsolicited bot PRs (allowlisted: dependabot,
  github-actions) and spam-signature PRs from first-time authors; other
  first-time PRs get `needs triage`.
- The Vercel Speed Insights drive-by (PR #1) was closed manually; the guard
  covers repeats.

## One-time repository settings (manual, UI)

The automation token lacks these permissions (403), so enable them once in
the UI:

- **Private vulnerability reporting**: Security tab → Settings → enable
  "Private vulnerability reporting" (SECURITY.md points reporters there).
- **Branch protection**: see checklist above.
- **Projects board**: see setup above.
