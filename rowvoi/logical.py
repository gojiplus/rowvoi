"""Deterministic algorithms for rowvoi.

This module contains functions that operate in the purely logical
functional dependency (FD) world.  Given a pandas DataFrame and a
subset of row indices, these functions answer questions such as:

* Is a given set of columns sufficient to uniquely identify those
  rows (i.e., does it form a key for those rows)?
* What is the smallest set of columns that distinguishes the rows,
  either exactly (via brute force) or approximately (via a greedy
  hitting set approximation)?

No probabilities or models are used here; everything is treated
deterministically based on equality/inequality of values.  For the
probabilistic/ML version of the problem, see :mod:`rowvoi.ml`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import pandas as pd

from .types import ColName, RowIndex


def is_key(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    cols: Sequence[ColName],
) -> bool:
    """Check whether a set of columns uniquely identifies a set of rows.

    This function returns ``True`` if and only if for every pair of
    distinct rows in ``rows`` there exists at least one column in
    ``cols`` on which the two rows differ.  Equivalently, the
    projection of ``df`` onto ``cols`` has no duplicate rows when
    restricted to ``rows``.  If ``rows`` contains 0 or 1 element or
    ``cols`` is empty, the function returns ``False`` unless
    ``rows`` is empty or a singleton.

    Parameters
    ----------
    df : pandas.DataFrame
        The underlying data table.
    rows : Sequence[int]
        The indices of the rows to check.  These should be integer
        positions in ``df`` (not necessarily the DataFrame's
        index labels).  If your DataFrame uses a custom index,
        convert it to positions using ``df.index.get_loc``.
    cols : Sequence[ColName]
        The columns to test.  If empty, the function returns
        ``False`` for nontrivial ``rows``.

    Returns
    -------
    bool
        ``True`` if ``cols`` is a key for ``rows``; ``False``
        otherwise.
    """
    if len(rows) <= 1:
        # A single row is trivially unique.
        return True
    if not cols:
        return False
    # Build a set of tuples of column values for each candidate row.
    seen: set[tuple] = set()
    for r in rows:
        tup = tuple(df.iloc[r, df.columns.get_indexer_for(cols)])
        if tup in seen:
            return False
        seen.add(tup)
    return True


def minimal_key_greedy(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    columns: Sequence[ColName] | None = None,
) -> list[ColName]:
    """Approximate a minimal distinguishing column set via a greedy hit set.

    This function attempts to find a small set of columns that
    distinguishes the rows in ``rows`` by greedily selecting the
    column that separates the most currently indistinguishable pairs
    of rows.  It solves an instance of the minimum set cover
    (hitting set) problem with the usual logarithmic approximation
    guarantee.  If the candidate ``columns`` list is provided, only
    those columns are considered.  The function does **not** test
    whether the returned set is minimal in the strict sense; it
    provides a good approximation in practice.

    Parameters
    ----------
    df : pandas.DataFrame
        The data table containing all rows and columns.
    rows : Sequence[int]
        The row indices to distinguish.
    columns : Sequence[ColName], optional
        A subset of columns to consider.  If ``None``, all columns
        from ``df`` are considered.

    Returns
    -------
    List[ColName]
        A list of column names that, together, distinguish all pairs
        of rows.  If no columns separate all pairs, the returned list
        may not eliminate all ambiguity.  The caller can check
        ``is_key`` on the result to verify.
    """
    rows = list(rows)
    if len(rows) <= 1:
        return []
    # Universe of unordered pairs of rows that need to be separated.
    universe: set[tuple[int, int]] = set()
    for a, b in itertools.combinations(sorted(rows), 2):
        universe.add((a, b) if a < b else (b, a))
    # Determine which columns to consider.
    if columns is None:
        columns_to_consider = list(df.columns)
    else:
        columns_to_consider = list(columns)
    # Precompute, for each column, the set of row pairs it separates.
    coverage: dict[ColName, set[tuple[int, int]]] = {}
    for col in columns_to_consider:
        bucket: dict[object, list[int]] = {}
        # Group candidate rows by their value in this column.
        for idx in rows:
            val = df.iloc[idx][col]
            bucket.setdefault(val, []).append(idx)
        pairs: set[tuple[int, int]] = set()
        labels = list(bucket.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                rows_i = bucket[labels[i]]
                rows_j = bucket[labels[j]]
                for r in rows_i:
                    for s in rows_j:
                        pair = (r, s) if r < s else (s, r)
                        pairs.add(pair)
        coverage[col] = pairs
    # Greedily pick columns covering the most currently uncovered pairs.
    selected: list[ColName] = []
    uncovered = universe.copy()
    while uncovered:
        best_col = None
        best_count = -1
        for col, pairs in coverage.items():
            if col in selected:
                continue
            count = len(uncovered & pairs)
            if count > best_count:
                best_count = count
                best_col = col
        if best_col is None or best_count <= 0:
            # No column covers any new pairs; break to avoid infinite loop.
            break
        selected.append(best_col)
        uncovered -= coverage[best_col]
    return selected


def minimal_key_exact(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    columns: Sequence[ColName] | None = None,
    max_search_cols: int = 20,
) -> list[ColName]:
    """Find the smallest column set that uniquely identifies the rows.

    This brute force implementation searches over all subsets of the
    candidate columns (up to a maximum search size) to find a minimal
    set of columns that together form a key for the given rows.  The
    complexity is exponential in the number of columns, so the search
    is only feasible for modest numbers of columns (dozens at most).

    If ``max_search_cols`` is smaller than the number of columns to
    consider, the function raises ``ValueError`` to avoid explosive
    runtimes.  In practice, users should restrict the candidate
    columns to a smaller subset (for example, known key fields).

    Parameters
    ----------
    df : pandas.DataFrame
        The data table.
    rows : Sequence[int]
        Row indices to disambiguate.
    columns : Sequence[ColName], optional
        Specific columns to consider.  If ``None``, all columns are
        considered.
    max_search_cols : int, optional
        Maximum number of columns over which to perform brute force
        search.  Defaults to 20.  Adjusting this downward may be
        necessary for very wide tables.

    Returns
    -------
    List[ColName]
        The smallest set of columns (in ascending order of size) that
        forms a key for ``rows``.  If no such set exists within the
        search space, the function returns an empty list.

    Raises
    ------
    ValueError
        If the number of columns to search exceeds ``max_search_cols``.
    """
    rows = list(rows)
    # Determine columns to consider.
    if columns is None:
        candidate_cols = list(df.columns)
    else:
        candidate_cols = list(columns)
    if len(candidate_cols) > max_search_cols:
        raise ValueError(
            f"Too many columns to search ({len(candidate_cols)} > {max_search_cols}). "
            "Specify a smaller candidate set or increase max_search_cols."
        )
    # Try subsets of increasing size.
    for r in range(1, len(candidate_cols) + 1):
        for combo in itertools.combinations(candidate_cols, r):
            if is_key(df, rows, combo):
                return list(combo)
    # If no combination works, return empty list.
    return []
