"""Top‑level package for rowvoi.

The :mod:`rowvoi` package provides a set of tools for interactive
disambiguation of rows in a dataset.  Given a small set of candidate
rows, it helps answer questions such as:

* Which columns (features) must be observed to uniquely distinguish
  these rows?  (See :func:`rowvoi.logical.minimal_key_exact` and
  :func:`rowvoi.logical.minimal_key_greedy`.)
* How much information does a given feature provide about which row
  is correct, given what we have seen so far?  (See
  :func:`rowvoi.mi.candidate_mi` and
  :func:`rowvoi.mi.best_feature_by_candidate_mi`.)
* Under a simple noise model and global frequency priors, which
  feature should we acquire next to maximize expected reduction in
  uncertainty?  (See :class:`rowvoi.ml.RowVoiModel`.)
* How does a greedy feature acquisition policy compare to the
  optimal minimal key in practice?  (See
  :func:`rowvoi.simulate.benchmark_policy`.)

Example usage::

    >>> import pandas as pd
    >>> from rowvoi import minimal_key_greedy, RowVoiModel, CandidateState
    >>> df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 4, 3], "C": [5, 6, 7]})
    >>> minimal_key_greedy(df, [0, 1])
    ['B']
    >>> # Fit a model to estimate expected information gain
    >>> model = RowVoiModel().fit(df)
    >>> state = CandidateState(candidate_rows=[0, 2], posterior={0: 0.5, 2: 0.5},
    ...                       observed_cols=[], observed_values={})
    >>> suggestion = model.suggest_next_feature(df, state)
    >>> suggestion.col
    'A'

The package is organized into submodules:

* :mod:`rowvoi.types` – basic dataclasses and type aliases.
* :mod:`rowvoi.logical` – deterministic functional dependency algorithms.
* :mod:`rowvoi.mi` – candidate‑set mutual information routines.
* :mod:`rowvoi.ml` – model‑based value‑of‑information policies.
* :mod:`rowvoi.simulate` – utilities for sampling and benchmarking.
"""

from .logical import (
    is_key,
    minimal_key_exact,
    minimal_key_greedy,
)
from .mi import (
    best_feature_by_candidate_mi,
    candidate_mi,
)
from .ml import (
    FeatureSuggestion,
    RowVoiModel,
)
from .simulate import (
    AcquisitionResult,
    benchmark_policy,
    sample_candidate_sets,
)
from .types import CandidateState, ColName, RowIndex  # re‑export types

__all__ = [
    # types
    "CandidateState",
    "ColName",
    "RowIndex",
    # logical
    "is_key",
    "minimal_key_exact",
    "minimal_key_greedy",
    # mutual information (candidate set)
    "candidate_mi",
    "best_feature_by_candidate_mi",
    # model
    "RowVoiModel",
    "FeatureSuggestion",
    # simulation
    "sample_candidate_sets",
    "benchmark_policy",
    "AcquisitionResult",
]
