"""Core deterministic algorithms for rowvoi.

This module provides the same deterministic routines as the original
`rowdisambiguator` package: a simple mutual information calculation
for distinguishing a subset of rows and a greedy set cover
implementation for selecting a minimal separating set of columns.  See
``rowvoi.ml`` for probabilistic extensions that incorporate dataset
frequency information.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import pandas as pd


def mutual_information(
    values: Sequence,
    row_indices: Sequence[int],
    *,
    normalize: bool = False,
) -> float:
    r"""Compute the mutual information between a single column and row identity.

    Given a sequence of values (one per row in the underlying dataset) and a
    subset of row indices ``row_indices``, this function returns the
    expected reduction in entropy of the row identity when the value of
    the column is observed.  Formally,

    .. math::

        I(R; X_j) = H(R) - E_{X_j}[H(R \mid X_j)]

    where ``R`` is the random variable representing which of the
    ``row_indices`` corresponds to the true row, and ``X_j`` is the
    column in question.

    Parameters
    ----------
    values : Sequence
        A sequence containing the values of the column for all rows in the
        DataFrame.  Only the entries at positions given by
        ``row_indices`` are used.
    row_indices : Sequence[int]
        The indices of the rows to disambiguate.  Each element of
        ``row_indices`` should be an integer between 0 and ``len(values) - 1``.
    normalize : bool, optional
        If ``True``, the mutual information will be divided by the
        maximum possible entropy (``log2(k)`` where ``k`` is the number
        of rows) so that the result lies in the range [0, 1].  Default
        is ``False``.

    Returns
    -------
    float
        The mutual information in bits (or as a fraction of the maximum
        entropy if ``normalize`` is ``True``).
    """
    subset = [values[i] for i in row_indices]
    k = len(subset)
    if k <= 1:
        return 0.0
    counts = {}
    for val in subset:
        counts[val] = counts.get(val, 0) + 1
    h_prior = math.log2(k)
    h_conditional = 0.0
    for count in counts.values():
        if count > 0:
            h_conditional += (count / k) * math.log2(count)
    mi = h_prior - h_conditional
    if normalize:
        return mi / h_prior if h_prior > 0 else 0.0
    return mi


def _column_covers_pairs(values: Sequence, rows: Sequence[int]) -> set[tuple[int, int]]:
    covered: set[tuple[int, int]] = set()
    bucket: dict[object, list[int]] = {}
    for idx in rows:
        val = values[idx]
        bucket.setdefault(val, []).append(idx)
    keys = list(bucket.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rows_i = bucket[keys[i]]
            rows_j = bucket[keys[j]]
            for r in rows_i:
                for s in rows_j:
                    covered.add((r, s) if r < s else (s, r))
    return covered


def greedy_minimal_set(
    dataframe: pd.DataFrame, row_indices: Sequence[int]
) -> list[str]:
    rows = list(row_indices)
    if len(rows) <= 1:
        return []
    universe: set[tuple[int, int]] = set()
    for a, b in itertools.combinations(sorted(rows), 2):
        universe.add((a, b) if a < b else (b, a))
    coverage: dict[str, set[tuple[int, int]]] = {}
    for col in dataframe.columns:
        values = dataframe[col].tolist()
        coverage[col] = _column_covers_pairs(values, rows)
    selected: list[str] = []
    uncovered = universe.copy()
    while uncovered:
        best_col = None
        best_count = 0
        for col, pairs in coverage.items():
            if col in selected:
                continue
            count = len(uncovered & pairs)
            if count > best_count:
                best_count = count
                best_col = col
        if best_col is None:
            break
        selected.append(best_col)
        uncovered -= coverage[best_col]
        if not uncovered:
            break
    return selected


def find_minimal_columns(dataframe: pd.DataFrame, rows: Sequence[int]) -> list[str]:
    positions: list[int] = []
    for r in rows:
        if isinstance(r, int):
            positions.append(r)
        else:
            positions.append(dataframe.index.get_loc(r))
    return greedy_minimal_set(dataframe, positions)
