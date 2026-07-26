# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`rowvoi.rag`** — the two existing engines applied to retrieval, as an
  adapter layer rather than a second implementation:
  - `select_context` / `plan_context_path` cover the *claims* an answer needs
    with the fewest or cheapest *chunks*, with cost measured in tokens. Unlike
    top-k, which ranks each chunk independently, this optimizes the selection
    jointly, and a claim that no chunk supports is reported rather than
    silently dropped.
  - `next_question` / `question_values` / `observe_answer` pick the clarifying
    question that best splits an ambiguous candidate set per unit of user
    patience. The mutual information is `CandidateMIPolicy`, reused unchanged.
  - `RetrievalSession` sequences retrieval probes by value of information and
    stops on residual uncertainty rather than at a fixed *k*.
  - `rowvoi.rag.protocols` defines the LLM boundary; `rowvoi.rag.claude`
    implements it against the Anthropic SDK behind the optional `claude` extra
    and is never imported by `rowvoi.rag`.
- **`rowvoi.setcover`** — `SetCoverProblem`, the weighted set cover engine
  extracted from `KeyProblem`, now usable over any universe.
- `SolverUnavailableError`, raised when the `ilp` strategy has no usable LP
  solver.
- `CandidateState.reweight` for soft evidence: multiplies the posterior by a
  likelihood vector and retains every candidate, so a merely improbable
  observation cannot eliminate the right one.
- `CandidateMIPolicy.compute_mi` is public, so callers can get the whole
  ranking rather than only the argmax.

### Fixed

- **The `ilp` strategy no longer silently returns a greedy result.** It caught
  `ImportError` and fell back to greedy when pulp was missing. Greedy is an
  ln(m) approximation and ILP is exact, so this substituted one algorithm for
  the other under the name the caller asked for. Two concrete consequences:
  `test_ilp_algorithm` had a `pytest.skip` that could never fire, so it passed
  while asserting greedy behaviour and the ILP code was never executed; and
  `compute_gold_key` could return a greedy key as the "optimal" baseline,
  making `KeyEvalResult.optimality_gap` wrong — possibly negative, showing a
  method beating the optimum. `find_key(strategy="ilp")` now raises
  `SolverUnavailableError`. **This is a behaviour change** for anyone relying
  on the silent fallback.
- `compute_gold_key` no longer falls back to greedy either: it tries ILP then
  exhaustive search and raises if neither exact strategy succeeds. Pass
  `allow_approximate=True` for the old cascade. `evaluate_keys` needed no
  change — it already treated a failed gold solve as "no baseline" and
  reported no gap; that branch was simply never reached.
- pulp installed but with no runnable CBC previously raised `PulpSolverError`
  from deep inside `solve()`. The solver is now checked with `available()`
  first, so both no-solver cases give the same actionable error.
- `compute_gold_next_column_probabilistic` passed `candidate_features=` to a
  parameter named `candidate_cols` and raised `TypeError` on every call.
- `MIPolicy` never forwarded its objective or costs to the model. It took the
  raw-MI argmax and rescaled that column's score afterwards, so cost could not
  affect which column was chosen.
- The `exact` set-cover strategy minimized cardinality rather than cost,
  returning the first cover of the smallest size. With non-uniform costs it
  could pick one expensive set over two cheap ones.
- The ILP strategy's `epsilon` branch let an element covered by no set count
  toward the coverage target, overstating achieved coverage.
- `GreedyCoveragePolicy` and `CandidateMIPolicy` divided by cost unguarded, so
  a zero-cost column produced a `nan` (plus a numpy warning) that silently lost
  every comparison.
- `RandomPolicy` used `np.random.choice`, which coerces column labels into an
  array and mangles non-string ones.
- Several unguarded `Optional` accesses on `suggest_next_feature`, a possibly
  unbound `best_score`, and an in-place mutation of a read-only `Mapping`.
- Seven invisible non-breaking hyphens in `ml.py` docstrings.

### Changed

- **Versioning is derived from the git tag** (hatchling + uv-dynamic-versioning)
  rather than hand-maintained. `pyproject.toml` no longer carries a `version`,
  and there is no `__version__` literal. This removes a three-way drift in
  which `pyproject.toml` said 0.2.0 while `CITATION.cff` and `docs/conf.py`
  both said 0.1.0.
- Adopted the [py-canon](https://github.com/gojiplus/py-canon) fleet standard
  via [preen](https://github.com/gojiplus/preen): canon CI/docs/release
  workflows, `pyright` and `pydoclint` in the gate, PEP 735 dependency groups
  in place of the `dev` and `docs` extras, and PEP 639 license metadata.
- Docstrings converted from numpy to Google style. This is what made
  `pydoclint` able to parse them at all; in google mode it could not read
  numpy sections and so reported zero problems on a codebase with 22 real
  documentation gaps, all now fixed.
- `docs/index.rst` and `docs/examples.rst` rewritten. They documented an API
  deleted in an earlier refactor — `minimal_key_greedy`, `is_key` as a free
  function, `candidate_mi`, dict-valued posteriors, `expected_ig`. Every
  example is now executed by the docs build.
- `FeatureSuggestion.col` is typed `ColName | None`, which several policies
  already returned.
- Silently swallowed exceptions in `eval.py` are now logged at debug level.
- The ILP strategy creates variables through `LpProblem.add_variable` where
  available, and probes `COIN_CMD` before the deprecated `PULP_CBC_CMD`, so it
  works unchanged on PuLP 4 and stops emitting deprecation warnings wherever
  CBC is reachable.

### Removed

- `.github/workflows/python-publish.yml`, superseded by canon's tag-triggered
  `release.yml`. Keeping both would have double-published on a tag.
- The `dev` and `docs` extras, replaced by dependency groups. The
  `optimization` and `claude` extras remain — those are real optional features.

## [0.2.0] - 2025-11-25

### Added

- Probabilistic key finding (`find_key_probabilistic`,
  `plan_key_path_probabilistic`) and coverage estimation.
- Additional set-cover strategies: ILP, simulated annealing, genetic
  algorithm, LP relaxation, and a hybrid.
- Evaluation and benchmarking tools.

### Changed

- Reworked the public API and reorganized the examples directory.

## [0.1.0]

- Initial release: deterministic minimal keys via set cover, mutual-information
  policies, and interactive disambiguation sessions.

[Unreleased]: https://github.com/gojiplus/rowvoi/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/gojiplus/rowvoi/releases/tag/v0.2.0
