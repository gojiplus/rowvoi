"""Deterministic key and path algorithms for row disambiguation.

This module handles the deterministic case where all column values are known.
It solves the minimal set cover problem: find the smallest set of columns
that distinguishes all pairs of rows in a candidate set.

The universe of row pairs is built here; the solvers themselves live in
:mod:`rowvoi.setcover`, which knows nothing about rows.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .core import ColName, RowIndex
from .setcover import CoverPath, SetCoverProblem

Pair = tuple[int, int]


@dataclass
class KeyPathStep:
    """A single step in a key path showing incremental progress.

    Attributes
    ----------
    col : ColName
        The column added in this step
    newly_covered_pairs : int
        Number of pairs newly covered by this column
    cumulative_covered_pairs : int
        Total pairs covered up to and including this step
    total_pairs : int
        Total number of pairs that need to be covered
    marginal_cost : float
        Cost of adding this specific column
    cumulative_cost : float
        Total cost up to and including this step
    newly_covered_weight : float, optional
        Weighted coverage gain (for weighted objectives)
    cumulative_covered_weight : float, optional
        Total weighted coverage so far
    """

    col: ColName
    newly_covered_pairs: int
    cumulative_covered_pairs: int
    total_pairs: int
    marginal_cost: float
    cumulative_cost: float
    newly_covered_weight: float | None = None
    cumulative_covered_weight: float | None = None

    @property
    def coverage(self) -> float:
        """Fraction of pairs covered so far."""
        if self.total_pairs == 0:
            return 1.0
        return self.cumulative_covered_pairs / self.total_pairs


@dataclass
class KeyPath:
    """Ordered sequence of columns and their contribution to coverage/cost.

    Attributes
    ----------
    steps : list[KeyPathStep]
        Ordered list of steps showing incremental progress
    """

    steps: list[KeyPathStep]

    def columns(self) -> list[ColName]:
        """Return the ordered list of columns in the path."""
        return [step.col for step in self.steps]

    def prefix_for_budget(self, budget: float) -> list[ColName]:
        """Return the longest prefix of columns whose cumulative_cost <= budget.

        Parameters
        ----------
        budget : float
            Maximum allowed cumulative cost

        Returns
        -------
        list[ColName]
            Columns that fit within the budget
        """
        result = []
        for step in self.steps:
            if step.cumulative_cost <= budget:
                result.append(step.col)
            else:
                break
        return result

    def prefix_for_epsilon_pairs(self, epsilon: float) -> list[ColName]:
        """Return the shortest prefix that leaves <= epsilon fraction unresolved.

        Parameters
        ----------
        epsilon : float
            Maximum allowed fraction of unresolved pairs

        Returns
        -------
        list[ColName]
            Minimum columns needed to achieve (1-epsilon) coverage
        """
        target_coverage = 1.0 - epsilon
        for i, step in enumerate(self.steps):
            if step.coverage >= target_coverage:
                return self.columns()[: i + 1]
        return self.columns()

    def coverage_curve(self) -> list[tuple[float, float]]:
        """Return the coverage curve as (cumulative_cost, coverage_fraction) points."""
        return [(step.cumulative_cost, step.coverage) for step in self.steps]


def _to_key_path(path: CoverPath) -> KeyPath:
    """Reframe a generic CoverPath in the vocabulary of row pairs."""
    return KeyPath(
        steps=[
            KeyPathStep(
                col=step.name,
                newly_covered_pairs=step.newly_covered,
                cumulative_covered_pairs=step.cumulative_covered,
                total_pairs=step.total_elements,
                marginal_cost=step.marginal_cost,
                cumulative_cost=step.cumulative_cost,
                newly_covered_weight=step.newly_covered_weight,
                cumulative_covered_weight=step.cumulative_covered_weight,
            )
            for step in path.steps
        ]
    )


def pairwise_coverage(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    cols: Sequence[ColName],
) -> float:
    """Fraction of unordered row pairs in `rows` that are distinguished by `cols`.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame
    rows : Sequence[RowIndex]
        Row indices to consider
    cols : Sequence[ColName]
        Columns to use for distinguishing

    Returns
    -------
    float
        Fraction of pairs that differ on at least one column in cols
    """
    rows = list(rows)
    n = len(rows)
    if n <= 1:
        return 1.0

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 1.0

    covered_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            row_i = rows[i]
            row_j = rows[j]
            # Check if any column distinguishes this pair
            for col in cols:
                if df.iloc[row_i][col] != df.iloc[row_j][col]:
                    covered_pairs += 1
                    break

    return covered_pairs / total_pairs


class KeyProblem:
    """Deterministic key-finding problem for a fixed subset of rows.

    Under the hood: universe = row pairs; columns cover pairs they separate.
    Solving is delegated to :class:`~rowvoi.setcover.SetCoverProblem`.

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    rows : Sequence[RowIndex]
        Row indices to distinguish
    columns : Sequence[ColName], optional
        Columns to consider. If None, use all columns
    costs : Mapping[ColName, float], optional
        Cost of each column. If None, unit cost
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        *,
        columns: Sequence[ColName] | None = None,
        costs: Mapping[ColName, float] | None = None,
    ) -> None:
        self.df = df
        self.rows = list(rows)
        self.columns = list(columns) if columns is not None else list(df.columns)
        self.costs = costs or {}

        universe, separates = self._build_coverage()
        self._problem = SetCoverProblem(separates, universe=universe, costs=self.costs)

    def _build_coverage(self) -> tuple[set[Pair], dict[ColName, set[Pair]]]:
        """Build the universe of row pairs and the pairs each column separates."""
        n = len(self.rows)
        if n <= 1:
            # No pairs to distinguish; keep the columns so the solver still
            # reports them as available (it will select none).
            return set(), {col: set() for col in self.columns}

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        universe = set(pairs)

        separates: dict[ColName, set[Pair]] = {}
        for col in self.columns:
            separates[col] = {
                (i, j)
                for i, j in pairs
                if self.df.iloc[self.rows[i]][col] != self.df.iloc[self.rows[j]][col]
            }

        return universe, separates

    def is_key(
        self,
        cols: Sequence[ColName],
        *,
        epsilon_pairs: float = 0.0,
    ) -> bool:
        """Check if cols distinguish all but at most epsilon_pairs fraction.

        Parameters
        ----------
        cols : Sequence[ColName]
            Columns to check
        epsilon_pairs : float, default 0.0
            Maximum allowed fraction of unresolved pairs

        Returns
        -------
        bool
            True if cols form an epsilon-key
        """
        return self._problem.is_cover(cols, epsilon=epsilon_pairs)

    def pairwise_coverage(self, cols: Sequence[ColName]) -> float:
        """Compute pairwise coverage for this problem."""
        return self._problem.coverage(cols)

    def minimal_key(
        self,
        strategy: Literal[
            "greedy", "exact", "ilp", "sa", "ga", "lp", "hybrid"
        ] = "greedy",
        *,
        epsilon_pairs: float = 0.0,
        time_limit: float | None = None,
    ) -> list[ColName]:
        """Solve deterministic min-key / set-cover for this row set.

        Parameters
        ----------
        strategy : str, default "greedy"
            Algorithm to use:
            - "greedy": greedy set cover on row pairs
            - "exact": brute force enumeration (only for small problems)
            - "ilp": Integer Linear Programming (requires pulp)
            - "sa": Simulated Annealing metaheuristic
            - "ga": Genetic Algorithm metaheuristic
            - "lp": Linear Programming relaxation with rounding
            - "hybrid": Combined SA+GA approach
        epsilon_pairs : float, default 0.0
            Allow some unresolved pairs to remain
        time_limit : float, optional
            Maximum time in seconds

        Returns
        -------
        list[ColName]
            Minimal (or near-minimal) set of columns
        """
        return self._problem.solve(
            strategy, epsilon=epsilon_pairs, time_limit=time_limit
        )

    def plan_path(
        self,
        *,
        objective: Literal["pair_coverage", "entropy"] = "pair_coverage",
        weighting: Literal["uniform", "pair_idf"] = "uniform",
    ) -> KeyPath:
        """Build a greedy ordering of columns for this row set.

        Parameters
        ----------
        objective : str, default "pair_coverage"
            - "pair_coverage": gain = newly covered pairs
            - "entropy": gain = reduction in log cluster size
        weighting : str, default "uniform"
            - "uniform": all pairs weighted equally
            - "pair_idf": weight hard-to-separate pairs more

        Returns
        -------
        KeyPath
            Ordered sequence with coverage information
        """
        path = self._problem.plan_path(
            objective="coverage" if objective == "pair_coverage" else "entropy",
            weighting="idf" if weighting == "pair_idf" else "uniform",
        )
        return _to_key_path(path)


def find_key(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    *,
    columns: Sequence[ColName] | None = None,
    costs: Mapping[ColName, float] | None = None,
    strategy: Literal["greedy", "exact", "ilp", "sa", "ga", "lp", "hybrid"] = "greedy",
    epsilon_pairs: float = 0.0,
    time_limit: float | None = None,
) -> list[ColName]:
    """Find a minimal key for distinguishing a set of rows.

    Convenience wrapper around KeyProblem.minimal_key().

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    rows : Sequence[RowIndex]
        Row indices to distinguish
    columns : Sequence[ColName], optional
        Columns to consider
    costs : Mapping[ColName, float], optional
        Cost of each column
    strategy : str, default "greedy"
        Algorithm to use
    epsilon_pairs : float, default 0.0
        Allow some unresolved pairs
    time_limit : float, optional
        Maximum time in seconds

    Returns
    -------
    list[ColName]
        Minimal set of columns
    """
    problem = KeyProblem(df, rows, columns=columns, costs=costs)
    return problem.minimal_key(
        strategy=strategy, epsilon_pairs=epsilon_pairs, time_limit=time_limit
    )


def plan_key_path(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    *,
    columns: Sequence[ColName] | None = None,
    costs: Mapping[ColName, float] | None = None,
    objective: Literal["pair_coverage", "entropy"] = "pair_coverage",
    weighting: Literal["uniform", "pair_idf"] = "uniform",
) -> KeyPath:
    """Plan an ordered path of columns for disambiguation.

    Convenience wrapper around KeyProblem.plan_path().

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    rows : Sequence[RowIndex]
        Row indices to distinguish
    columns : Sequence[ColName], optional
        Columns to consider
    costs : Mapping[ColName, float], optional
        Cost of each column
    objective : str, default "pair_coverage"
        Objective function for ordering
    weighting : str, default "uniform"
        Weighting scheme for pairs

    Returns
    -------
    KeyPath
        Ordered sequence with coverage information
    """
    problem = KeyProblem(df, rows, columns=columns, costs=costs)
    return problem.plan_path(objective=objective, weighting=weighting)
