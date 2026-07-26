# Contributing to rowvoi

Thanks for your interest. This is a small, focused package; the guidelines
below are correspondingly short.

## Setup

rowvoi follows the [py-canon](https://github.com/gojiplus/py-canon) fleet
standard, applied and checked with [preen](https://github.com/gojiplus/preen).
Development dependencies live in PEP 735 dependency groups, not extras:

```bash
uv sync --all-groups --all-extras
```

`--all-extras` matters: `optimization` (pulp) and `claude` (anthropic) gate
real tests. Without them those tests skip rather than fail, and you can ship a
break without seeing it.

## Before opening a PR

```bash
make ci      # ruff check, ruff format --check, pyright, pydoclint, pytest
make docs    # sphinx-build -W: warnings are errors
make check   # preen fleet-conformance check
```

All of these must pass. `make ci` mirrors what the canon reusable workflow
runs, so a green local run should mean a green CI run.

## Conventions

**Docstrings are Google style.** `arg-type-hints-in-docstring` is off, so
types belong in the signature and never in the docstring prose. `pydoclint`
enforces that arguments, returns, and raises actually match the code.

**Examples are executed, not asserted by eye.** Every example in `docs/` is a
`testcode`/`testoutput` block run by the docs build, and module docstring
examples run via `tests/test_docstrings.py`. If you write an example, write
its real output — don't guess it. The docs in this repo previously rotted into
documenting an API that no longer existed, precisely because nothing ran them.

**Version numbers are not edited by hand.** The version is derived from the
git tag via `uv-dynamic-versioning`. There is no `version =` in
`pyproject.toml` and no `__version__ = "..."` literal to bump.

## Releasing

Releases are tag-driven:

```bash
make release VERSION=x.y.z    # wraps `preen release`
```

`preen release` runs the conformance checks, refuses to proceed on any
critical issue, requires a `CHANGELOG.md` entry for the version, then tags and
pushes. The canon `release.yml` workflow does the build, attestations, and
PyPI trusted publishing.

Add your change to the `## [Unreleased]` section of `CHANGELOG.md` as part of
your PR; the release step renames that heading.

## Gotchas

**`preen adopt` and `preen update` overwrite `.github/workflows/ci.yml`.** Our
shim pins `python-versions` to `["3.11", "3.12", "3.13"]`; the canon default
is `["3.11", "3.14"]`. If you re-run either command, check that pin survived.

**pydoclint does not run in CI.** The canon workflow guards it behind
`if [ -d src ]`, and rowvoi uses a flat layout. `make ci` runs it locally, so
please don't skip that step.
