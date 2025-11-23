"""Candidate‑set mutual information routines for rowvoi.

This module provides functions to compute conditional mutual
information on a *local* candidate set of rows.  Unlike the
deterministic routines in :mod:`rowvoi.logical`, these functions
explicitly operate on a probabilistic state over the candidate rows.
Given a :class:`~rowvoi.types.CandidateState` and a column, the
functions estimate how much observing that column’s value would
reduce uncertainty about which row is correct.  No global dataset
frequencies or noise models are used here – everything is derived
solely from the candidate set and its posterior.

These measures can be used to greedily choose the next feature in
interactive disambiguation when you do not wish to appeal to a
learned model of the data distribution.  For more sophisticated
policy decisions that incorporate noise and global priors, see
:mod:`rowvoi.ml`.
"""

import math
from collections.abc import Sequence

import pandas as pd

from .types import CandidateState, ColName


def candidate_mi(
    df: pd.DataFrame,
    state: CandidateState,
    col: ColName,
) -> float:
    r"""Compute the conditional mutual information on a candidate set.

    For a given :class:`~rowvoi.types.CandidateState` and a single
    column ``col``, this function calculates the expected reduction
    in entropy of the row identity when the value of ``col`` is
    observed.  Formally, letting ``R`` be the random variable for
    the true row (over the support ``state.candidate_rows``) and
    ``X_col`` be the value of column ``col``, the returned value is

    .. math::

        I(R; X_{col} \mid E) = H(R \mid E)
        \;\;\;\;-\;\mathbb{E}_{X_{col}}[H(R \mid E, X_{col})],

    where ``E`` represents the evidence already incorporated into
    ``state`` (i.e., the observed columns and their values).  The
    distribution of ``X_col`` given the current state is induced
    deterministically by the candidate rows: for each candidate row
    ``r``, ``X_col`` takes the value ``df.iloc[r][col]`` with
    probability equal to the posterior probability of ``r``.

    If all remaining candidate rows have the same value in ``col``,
    the mutual information is zero because no information is gained.

    Parameters
    ----------
    df : pandas.DataFrame
        The full data table.
    state : CandidateState
        The current candidate state, containing row indices and
        posterior probabilities.
    col : ColName
        The column name for which to compute mutual information.

    Returns
    -------
    float
        The mutual information in bits.  Values are nonnegative and
        bounded above by the prior entropy ``H(R | E)``.
    """
    rows = state.candidate_rows
    if len(rows) <= 1:
        return 0.0
    # Compute prior entropy H(R | E)
    h_prior = 0.0
    for r in rows:
        p_r = state.posterior.get(r, 0.0)
        if p_r > 0:
            h_prior -= p_r * math.log2(p_r)
    # Group candidate rows by their value in the selected column.
    # Build mapping from column value to list of rows with that value.
    value_groups: dict[object, list[int]] = {}
    for r in rows:
        val = df.iloc[r][col]
        value_groups.setdefault(val, []).append(r)
    if len(value_groups) <= 1:
        # All rows share the same value in this column; no information.
        return 0.0
    # Compute expected conditional entropy H(R | E, X_col)
    h_cond = 0.0
    # For each possible value of X_col among the candidates
    for _v, group in value_groups.items():
        # Probability that X_col takes this value under current posterior
        p_x = sum(state.posterior.get(r, 0.0) for r in group)
        if p_x <= 0.0:
            continue
        # Posterior over R given X_col = v: renormalize probabilities
        h_r_given_v = 0.0
        for r in group:
            p_r = state.posterior.get(r, 0.0)
            # conditional probability of r given value v
            p_r_given = p_r / p_x
            if p_r_given > 0.0:
                h_r_given_v -= p_r_given * math.log2(p_r_given)
        h_cond += p_x * h_r_given_v
    return h_prior - h_cond


def best_feature_by_candidate_mi(
    df: pd.DataFrame,
    state: CandidateState,
    candidate_cols: Sequence[ColName] | None = None,
) -> ColName | None:
    """Select the column with maximum conditional mutual information.

    This function iterates over the provided list of candidate columns
    (or all columns not yet observed if ``candidate_cols`` is
    ``None``) and computes :func:`candidate_mi` for each.  It returns
    the column with the highest mutual information value.  Ties are
    broken deterministically by the column name.  If no column
    provides positive information (e.g. all candidates have been
    exhausted or there is only one candidate row), the function
    returns ``None``.

    Parameters
    ----------
    df : pandas.DataFrame
        The data table.
    state : CandidateState
        The current candidate state.
    candidate_cols : Sequence[ColName], optional
        A list of columns to consider.  If ``None``, all columns in
        ``df`` that are not already in ``state.observed_cols`` are
        considered.

    Returns
    -------
    ColName or None
        The name of the column with highest mutual information, or
        ``None`` if none are suitable.
    """
    if candidate_cols is None:
        # Consider all columns except those already observed.
        cols_to_consider = [c for c in df.columns if c not in state.observed_cols]
    else:
        cols_to_consider = list(candidate_cols)
    best_col = None
    best_mi = -1.0
    for col in cols_to_consider:
        mi = candidate_mi(df, state, col)
        if mi > best_mi + 1e-12:  # allow tiny epsilon for floating comparison
            best_mi = mi
            best_col = col
        elif (
            abs(mi - best_mi) < 1e-12
            and best_col is not None
            and str(col) < str(best_col)
        ):
            # tie-break on lexical order for determinism
            best_col = col
    return best_col
