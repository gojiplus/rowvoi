"""Probabilistic and ε-relaxed set cover algorithms for row disambiguation.

This module implements probabilistic set cover algorithms that handle uncertainty
about unseen column values. Instead of requiring deterministic disambiguation,
these algorithms find minimal column sets that achieve disambiguation with
high probability (1-ε) based on learned coverage probabilities.

The key innovation is treating column coverage as a stochastic process where
each column has a probability p_{u,c} of distinguishing each pair of rows u,
estimated from historical data.

Key Functions
-------------
estimate_pair_separation_probs : Estimate P(column c distinguishes pair u)
greedy_epsilon_cover : Greedy algorithm for probabilistic set cover
probabilistic_minimal_key : Main API for ε-relaxed minimal key finding
suggest_next_feature_epsilon : Adaptive policy for sequential selection

Examples
--------
>>> import pandas as pd
>>> from rowvoi.probcover import probabilistic_minimal_key
>>>
>>> # Find columns that distinguish rows with 95% probability
>>> df = pd.DataFrame({'A': [1, 1, 2], 'B': [1, 2, 2], 'C': [1, 2, 3]})
>>> cols = probabilistic_minimal_key(df, rows=[0, 1, 2], eps=0.05)
>>> print(f"Selected columns for 95% coverage: {cols}")

Theory
------
This implements the Probabilistic Set Covering Problem where:
- Universe U = all unordered pairs of candidate rows
- For each column c and pair u, Y_{u,c} ~ Bernoulli(p_{u,c})
- Goal: minimize |C| s.t. E[covered pairs] ≥ (1-ε)|U|

The greedy algorithm achieves O(log(1/ε)) approximation due to
submodularity of the expected coverage function.

References
----------
- Luedtke & Ahmed (2008): Sample approximation for chance-constrained programming
- Golovin & Krause (2011): Adaptive submodularity
- Wolsey (1982): Analysis of the greedy algorithm for submodular set cover
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from rowvoi.types import CandidateState, ColName

# Type aliases for clarity
RowPair = tuple[int, int]
ProbDict = dict[ColName, dict[RowPair, float]]


@dataclass(frozen=True, slots=True)
class ProbabilisticCoverageResult:
    """Result from probabilistic set cover algorithms.

    Attributes
    ----------
    columns : list[ColName]
        Selected columns in order of selection
    expected_coverage : float
        Expected fraction of pairs covered (0 to 1)
    actual_eps : float
        Actual expected uncovered fraction achieved
    method : str
        Method used to estimate probabilities
    p_estimates : ProbDict
        The probability estimates used
    costs : dict[ColName, float] | None
        Column costs if provided
    """

    columns: list[ColName]
    expected_coverage: float
    actual_eps: float
    method: str
    p_estimates: ProbDict
    costs: dict[ColName, float] | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveFeatureSuggestion:
    """Suggestion from adaptive epsilon policy.

    Attributes
    ----------
    col : ColName
        Suggested column to query next
    expected_coverage_gain : float
        Expected number of newly covered pairs
    current_coverage : float
        Current fraction of pairs already distinguished
    target_coverage : float
        Target coverage (1 - eps)
    """

    col: ColName
    expected_coverage_gain: float
    current_coverage: float
    target_coverage: float


def estimate_pair_separation_probs(
    df: pd.DataFrame,
    rows: Iterable[int],
    cols: Iterable[ColName] | None = None,
    method: str = "empirical",
    observed_cols: list[ColName] | None = None,
    observed_values: dict[ColName, Any] | None = None,
) -> ProbDict:
    """Estimate p_{u,c} = P(column c will distinguish pair u).

    For each pair u = (i,j) of rows and each column c, estimate the
    probability that observing column c will reveal different values
    for rows i and j.

    Parameters
    ----------
    df : pd.DataFrame
        Training data for estimating probabilities
    rows : Iterable[int]
        Row indices to consider pairs from
    cols : Iterable[ColName] | None
        Columns to estimate for (default: all columns)
    method : str
        Estimation method:
        - "empirical": Use global column distributions
        - "conditional": Condition on observed columns (future)
        - "clustered": Use cluster-specific distributions (future)
    observed_cols : list[ColName] | None
        Already observed columns (for conditional methods)
    observed_values : dict[ColName, Any] | None
        Values of observed columns (for conditional methods)

    Returns
    -------
    ProbDict
        Nested dict: p[col][(i,j)] = probability of separation

    Notes
    -----
    The "empirical" method assumes:
    - Rows are iid samples from the column distribution
    - P(X_i ≠ X_j) = 1 - Σ_x P(X=x)²

    This is exact for independent sampling and provides a reasonable
    baseline for more sophisticated methods.
    """
    if cols is None:
        cols = df.columns

    rows_list = list(rows)
    p: ProbDict = {}

    if method == "empirical":
        # Simple global frequency approach
        for col in cols:
            if col not in df.columns:
                continue

            # Estimate column distribution from all data
            value_counts = df[col].value_counts(normalize=True, dropna=False)

            # P(two iid draws differ) = 1 - Σ p(x)²
            # value_counts.values is a numpy array
            prob_same = sum(p_val**2 for p_val in value_counts.to_numpy())
            prob_diff = 1.0 - prob_same

            # Apply same probability to all pairs
            col_probs = {}
            for i_idx, i in enumerate(rows_list):
                for j in rows_list[i_idx + 1 :]:
                    if i < j:
                        pair = (i, j)
                    else:
                        pair = (j, i)
                    col_probs[pair] = prob_diff

            p[col] = col_probs

    elif method == "conditional":
        # Future enhancement: condition on observed columns
        raise NotImplementedError(
            "Conditional probability estimation not yet implemented. "
            "Use method='empirical' for now."
        )

    elif method == "clustered":
        # Future enhancement: cluster-specific distributions
        raise NotImplementedError(
            "Clustered probability estimation not yet implemented. "
            "Use method='empirical' for now."
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return p


def greedy_epsilon_cover(
    rows: list[int],
    cols: list[ColName],
    p: ProbDict,
    eps: float = 0.0,
    costs: dict[ColName, float] | None = None,
) -> tuple[list[ColName], float]:
    """Greedy probabilistic set cover algorithm.

    Selects columns to maximize expected covered pairs until the
    expected uncovered fraction drops below eps.

    Parameters
    ----------
    rows : list[int]
        Candidate row indices
    cols : list[ColName]
        Available columns to choose from
    p : ProbDict
        p[c][(i,j)] = probability column c distinguishes pair (i,j)
    eps : float
        Maximum allowed expected uncovered fraction (0 to 1)
    costs : dict[ColName, float] | None
        Cost per column; if None, all costs = 1

    Returns
    -------
    tuple[list[ColName], float]
        (selected columns, achieved expected coverage fraction)

    Algorithm
    ---------
    This implements the greedy algorithm for submodular set cover:
    1. Initialize all pairs as uncovered
    2. Repeatedly select column with best coverage gain per cost
    3. Stop when expected coverage ≥ (1-ε)|U|

    The expected coverage function f(C) = Σ_u [1 - Π_{c∈C} (1-p_{u,c})]
    is monotone submodular, giving O(log(1/ε)) approximation.
    """
    if costs is None:
        costs = dict.fromkeys(cols, 1.0)

    # Universe: all unordered pairs
    U = [(i, j) for idx, i in enumerate(rows) for j in rows[idx + 1 :]]

    if not U:  # Single row or empty
        return [], 1.0

    # Track probability each pair remains uncovered
    remain_prob = dict.fromkeys(U, 1.0)
    chosen: list[ColName] = []

    def expected_covered() -> float:
        """Return current expected number of covered pairs."""
        return sum(1.0 - remain_prob[u] for u in U)

    def expected_coverage_fraction() -> float:
        """Return current expected fraction of covered pairs."""
        return expected_covered() / len(U) if U else 1.0

    target_coverage = (1.0 - eps) * len(U)

    # Greedy selection
    while expected_covered() < target_coverage and len(chosen) < len(cols):
        best_col = None
        best_score = 0.0

        for c in cols:
            if c in chosen:
                continue

            # Expected newly covered pairs if we add c
            gain = 0.0
            for u in U:
                puc = p.get(c, {}).get(u, 0.0)
                if puc > 0:
                    # Current uncovered prob: remain_prob[u]
                    # New uncovered prob: remain_prob[u] * (1 - puc)
                    gain += remain_prob[u] * puc

            score = gain / costs.get(c, 1.0)
            if score > best_score:
                best_score = score
                best_col = c

        if best_col is None or best_score <= 0:
            break  # No column improves coverage

        # Commit best column and update remain probabilities
        chosen.append(best_col)
        for u in U:
            puc = p.get(best_col, {}).get(u, 0.0)
            if puc > 0:
                remain_prob[u] *= 1.0 - puc

    return chosen, expected_coverage_fraction()


def probabilistic_minimal_key(
    df: pd.DataFrame,
    rows: list[int],
    eps: float = 0.0,
    method: str = "empirical",
    costs: dict[ColName, float] | None = None,
    candidate_cols: list[ColName] | None = None,
) -> ProbabilisticCoverageResult:
    """Find minimal column set for probabilistic row disambiguation.

    This is the main API for ε-relaxed set cover. It finds a small
    set of columns that disambiguates the given rows with expected
    coverage ≥ (1-ε), based on probabilistic coverage estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Data for probability estimation and candidate columns
    rows : list[int]
        Row indices to disambiguate
    eps : float
        Maximum expected uncovered fraction (0 for deterministic)
    method : str
        Probability estimation method (see estimate_pair_separation_probs)
    costs : dict[ColName, float] | None
        Cost per column for weighted optimization
    candidate_cols : list[ColName] | None
        Columns to consider (default: all non-constant columns)

    Returns
    -------
    ProbabilisticCoverageResult
        Selected columns and coverage statistics

    Examples
    --------
    >>> # 95% probabilistic coverage (allow 5% expected uncovered)
    >>> result = probabilistic_minimal_key(df, [0, 1, 2], eps=0.05)
    >>> print(f"Columns: {result.columns}")
    >>> print(f"Expected coverage: {result.expected_coverage:.1%}")

    >>> # Weighted by column costs
    >>> costs = {'expensive_col': 10.0, 'cheap_col': 1.0}
    >>> result = probabilistic_minimal_key(df, rows, eps=0.1, costs=costs)

    Notes
    -----
    When eps=0, this approximates deterministic set cover but uses
    probabilistic estimates. For true deterministic cover, use
    minimal_key_exact or minimal_key_greedy instead.
    """
    if candidate_cols is None:
        # Default: all columns that vary in the candidate rows
        candidate_cols = []
        for col in df.columns:
            if col in df.columns and len(df.loc[rows, col].unique()) > 1:
                candidate_cols.append(col)

    # Estimate separation probabilities
    p = estimate_pair_separation_probs(df, rows, candidate_cols, method=method)

    # Run greedy epsilon cover
    selected_cols, coverage_frac = greedy_epsilon_cover(
        rows, candidate_cols, p, eps=eps, costs=costs
    )

    return ProbabilisticCoverageResult(
        columns=selected_cols,
        expected_coverage=coverage_frac,
        actual_eps=1.0 - coverage_frac,
        method=method,
        p_estimates=p,
        costs=costs,
    )


def suggest_next_feature_epsilon(
    df: pd.DataFrame,
    state: CandidateState,
    p: ProbDict | None = None,
    eps: float = 0.1,
    costs: dict[ColName, float] | None = None,
    method: str = "empirical",
) -> AdaptiveFeatureSuggestion | None:
    """Adaptive epsilon-cover policy for sequential feature selection.

    Given current observations, suggests the next column that maximizes
    expected coverage gain per cost for still-ambiguous row pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing candidate rows and columns
    state : CandidateState
        Current state with observed columns and values
    p : ProbDict | None
        Pre-computed separation probabilities (computed if None)
    eps : float
        Target maximum uncovered fraction
    costs : dict[ColName, float] | None
        Cost per column
    method : str
        Probability estimation method if p is None

    Returns
    -------
    AdaptiveFeatureSuggestion | None
        Next column to query, or None if target coverage achieved

    Notes
    -----
    This implements adaptive stochastic minimum-cost coverage:
    - Track which pairs are already distinguished by observations
    - For remaining pairs, use expected marginal coverage gain
    - Stop when realized coverage ≥ (1-ε)

    Under standard assumptions, this greedy policy achieves
    logarithmic approximation to the optimal adaptive policy.
    """
    rows = state.candidate_rows
    observed = set(state.observed_cols)

    # Find pairs not yet distinguished by observed columns
    U = [(i, j) for idx, i in enumerate(rows) for j in rows[idx + 1 :]]
    unresolved = []

    for i, j in U:
        # Check if this pair is already distinguished
        distinguished = False
        for col in observed:
            if col in df.columns:
                if df.loc[i, col] != df.loc[j, col]:
                    distinguished = True
                    break
        if not distinguished:
            unresolved.append((i, j))

    # Check if we've achieved target coverage
    current_coverage = 1.0 - len(unresolved) / len(U) if U else 1.0
    target_coverage = 1.0 - eps

    if current_coverage >= target_coverage or not unresolved:
        return None  # Target achieved

    # Compute probabilities if not provided
    if p is None:
        candidate_cols = [c for c in df.columns if c not in observed]
        p = estimate_pair_separation_probs(
            df,
            rows,
            candidate_cols,
            method=method,
            observed_cols=list(observed),
            observed_values=state.observed_values,
        )

    if costs is None:
        costs = dict.fromkeys(df.columns, 1.0)

    # Find column with best expected coverage gain per cost
    best_col = None
    best_score = 0.0
    best_gain = 0.0

    for col in df.columns:
        if col in observed:
            continue

        # Expected coverage gain for unresolved pairs
        gain = 0.0
        for u in unresolved:
            puc = p.get(col, {}).get(u, 0.0)
            gain += puc  # Probability this pair gets resolved

        score = gain / costs.get(col, 1.0)
        if score > best_score:
            best_score = score
            best_col = col
            best_gain = gain

    if best_col is None:
        return None

    return AdaptiveFeatureSuggestion(
        col=best_col,
        expected_coverage_gain=best_gain,
        current_coverage=current_coverage,
        target_coverage=target_coverage,
    )


def evaluate_coverage(
    df: pd.DataFrame,
    rows: list[int],
    cols: list[ColName],
) -> tuple[float, list[RowPair]]:
    """Evaluate actual coverage achieved by a column set.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the rows and columns
    rows : list[int]
        Row indices
    cols : list[ColName]
        Columns to evaluate

    Returns
    -------
    tuple[float, list[RowPair]]
        (coverage fraction, list of uncovered pairs)
    """
    U = [(i, j) for idx, i in enumerate(rows) for j in rows[idx + 1 :]]
    if not U:
        return 1.0, []

    uncovered = []
    for i, j in U:
        distinguished = False
        for col in cols:
            if col in df.columns:
                if df.loc[i, col] != df.loc[j, col]:
                    distinguished = True
                    break
        if not distinguished:
            uncovered.append((i, j))

    coverage = 1.0 - len(uncovered) / len(U)
    return coverage, uncovered
