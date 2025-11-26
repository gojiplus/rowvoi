"""Deterministic key and path algorithms for row disambiguation.

This module handles the deterministic case where all column values are known.
It solves the minimal set cover problem: find the smallest set of columns
that distinguishes all pairs of rows in a candidate set.
"""

import itertools
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .core import ColName, RowIndex


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
        for step in self.steps:
            if step.coverage >= target_coverage:
                return self.columns()[: self.steps.index(step) + 1]
        return self.columns()

    def coverage_curve(self) -> list[tuple[float, float]]:
        """Return the coverage curve as (cumulative_cost, coverage_fraction) points."""
        return [(step.cumulative_cost, step.coverage) for step in self.steps]


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

        # Precompute the universe and coverage mapping
        self._universe, self._coverage = self._build_coverage()

    def _build_coverage(
        self,
    ) -> tuple[set[tuple[int, int]], dict[ColName, set[tuple[int, int]]]]:
        """Build the universe of pairs and column coverage mapping."""
        n = len(self.rows)
        if n <= 1:
            return set(), {}

        # Universe: all pairs of rows that need distinguishing
        universe = set()
        for i in range(n):
            for j in range(i + 1, n):
                universe.add((i, j))

        # Coverage: which pairs each column distinguishes
        coverage = {}
        for col in self.columns:
            covered = set()
            for i in range(n):
                for j in range(i + 1, n):
                    row_i = self.rows[i]
                    row_j = self.rows[j]
                    if self.df.iloc[row_i][col] != self.df.iloc[row_j][col]:
                        covered.add((i, j))
            coverage[col] = covered

        return universe, coverage

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
        coverage = self.pairwise_coverage(cols)
        return coverage >= 1.0 - epsilon_pairs

    def pairwise_coverage(self, cols: Sequence[ColName]) -> float:
        """Compute pairwise coverage for this problem."""
        if not self._universe:
            return 1.0

        covered = set()
        for col in cols:
            if col in self._coverage:
                covered |= self._coverage[col]

        return len(covered) / len(self._universe)

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
        if strategy == "greedy":
            return self._greedy_set_cover(epsilon_pairs)
        elif strategy == "exact":
            return self._exact_set_cover(epsilon_pairs, time_limit)
        elif strategy == "ilp":
            return self._ilp_set_cover(epsilon_pairs, time_limit)
        elif strategy == "sa":
            return self._simulated_annealing(epsilon_pairs, time_limit)
        elif strategy == "ga":
            return self._genetic_algorithm(epsilon_pairs, time_limit)
        elif strategy == "lp":
            return self._lp_relaxation(epsilon_pairs, time_limit)
        elif strategy == "hybrid":
            return self._hybrid_sa_ga(epsilon_pairs, time_limit)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _greedy_set_cover(self, epsilon_pairs: float) -> list[ColName]:
        """Greedy set cover algorithm."""
        if not self._universe:
            return []

        selected = []
        uncovered = self._universe.copy()
        target_covered = len(self._universe) * (1 - epsilon_pairs)

        while len(self._universe) - len(uncovered) < target_covered:
            best_col = None
            best_cost_ratio = float("inf")

            for col in self.columns:
                if col in selected:
                    continue
                gain = len(uncovered & self._coverage[col])
                if gain > 0:
                    cost = self.costs.get(col, 1.0)
                    cost_ratio = cost / gain
                    if cost_ratio < best_cost_ratio:
                        best_col = col
                        best_cost_ratio = cost_ratio

            if best_col is None:
                break

            selected.append(best_col)
            uncovered -= self._coverage[best_col]

        return selected

    def _exact_set_cover(
        self, epsilon_pairs: float, time_limit: float | None
    ) -> list[ColName]:
        """Exact solution via brute force enumeration."""
        if not self._universe:
            return []

        target_covered = len(self._universe) * (1 - epsilon_pairs)
        best = None
        best_cost = float("inf")
        start_time = time.time()

        # Try all subsets in order of increasing size
        for size in range(1, len(self.columns) + 1):
            if time_limit and time.time() - start_time > time_limit:
                break

            for subset in itertools.combinations(self.columns, size):
                covered = set()
                cost = 0.0
                for col in subset:
                    covered |= self._coverage[col]
                    cost += self.costs.get(col, 1.0)

                if len(covered) >= target_covered and cost < best_cost:
                    best = list(subset)
                    best_cost = cost
                    # If we found a solution of this size, no need to check larger
                    if epsilon_pairs == 0:
                        return best

            if best is not None and epsilon_pairs == 0:
                return best

        return best or self._greedy_set_cover(epsilon_pairs)

    def _ilp_set_cover(
        self, epsilon_pairs: float, time_limit: float | None
    ) -> list[ColName]:
        """Integer Linear Programming solution (requires pulp)."""
        try:
            import pulp
        except ImportError:
            # Fall back to greedy if pulp not available
            return self._greedy_set_cover(epsilon_pairs)

        if not self._universe:
            return []

        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)

        # Decision variables: x[c] = 1 if column c is selected
        x = {col: pulp.LpVariable(f"x_{col}", cat="Binary") for col in self.columns}

        # Objective: minimize total cost
        prob += pulp.lpSum(x[col] * self.costs.get(col, 1.0) for col in self.columns)

        # Constraints: each pair must be covered (with epsilon relaxation)
        target_pairs = len(self._universe) * (1 - epsilon_pairs)
        if epsilon_pairs == 0:
            # Exact coverage constraints
            for pair in self._universe:
                covering_cols = [
                    col for col in self.columns if pair in self._coverage[col]
                ]
                if covering_cols:
                    prob += pulp.lpSum(x[col] for col in covering_cols) >= 1
        else:
            # Relaxed: count covered pairs
            y = {
                pair: pulp.LpVariable(f"y_{pair}", cat="Binary")
                for pair in self._universe
            }
            for pair in self._universe:
                covering_cols = [
                    col for col in self.columns if pair in self._coverage[col]
                ]
                if covering_cols:
                    prob += y[pair] <= pulp.lpSum(x[col] for col in covering_cols)
            prob += pulp.lpSum(y[pair] for pair in self._universe) >= target_pairs

        # Solve
        if time_limit:
            prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
        else:
            prob.solve()

        if prob.status == pulp.LpStatusOptimal:
            return [col for col in self.columns if x[col].value() > 0.5]
        else:
            return self._greedy_set_cover(epsilon_pairs)

    def _simulated_annealing(
        self, epsilon_pairs: float, time_limit: float | None
    ) -> list[ColName]:
        """Use Simulated Annealing metaheuristic."""
        if not self._universe:
            return []

        # Start with greedy solution
        current = set(self._greedy_set_cover(epsilon_pairs))
        current_cost = sum(self.costs.get(col, 1.0) for col in current)
        best = current.copy()
        best_cost = current_cost

        target_covered = len(self._universe) * (1 - epsilon_pairs)
        temperature = 10.0
        cooling_rate = 0.95
        start_time = time.time()

        while temperature > 0.01:
            if time_limit and time.time() - start_time > time_limit:
                break

            # Generate neighbor
            neighbor = current.copy()
            if random.random() < 0.5 and len(neighbor) > 1:
                # Remove a random column
                col_to_remove = random.choice(list(neighbor))
                neighbor.remove(col_to_remove)
            else:
                # Add a random column not in the set
                candidates = [c for c in self.columns if c not in neighbor]
                if candidates:
                    neighbor.add(random.choice(candidates))

            # Check if neighbor is valid
            covered = set()
            for col in neighbor:
                covered |= self._coverage[col]

            if len(covered) >= target_covered:
                neighbor_cost = sum(self.costs.get(col, 1.0) for col in neighbor)
                delta = neighbor_cost - current_cost

                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current = neighbor
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best = current.copy()
                        best_cost = current_cost

            temperature *= cooling_rate

        return list(best)

    def _genetic_algorithm(
        self, epsilon_pairs: float, time_limit: float | None
    ) -> list[ColName]:
        """Genetic Algorithm metaheuristic."""
        if not self._universe:
            return []

        target_covered = len(self._universe) * (1 - epsilon_pairs)
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        start_time = time.time()

        # Initialize population
        population = []
        # Add greedy solution
        greedy = set(self._greedy_set_cover(epsilon_pairs))
        population.append(greedy)

        # Add random valid solutions
        while len(population) < population_size:
            if time_limit and time.time() - start_time > time_limit:
                break

            individual = set()
            uncovered = self._universe.copy()

            while len(self._universe) - len(uncovered) < target_covered:
                candidates = [c for c in self.columns if c not in individual]
                if not candidates:
                    break
                col = random.choice(candidates)
                individual.add(col)
                uncovered -= self._coverage[col]

            if len(self._universe) - len(uncovered) >= target_covered:
                population.append(individual)

        best = min(population, key=lambda x: sum(self.costs.get(col, 1.0) for col in x))

        for _ in range(generations):
            if time_limit and time.time() - start_time > time_limit:
                break

            # Selection and crossover
            new_population = [best]  # Elitism

            while len(new_population) < population_size:
                # Tournament selection
                parent1 = min(
                    random.sample(population, 3),
                    key=lambda x: sum(self.costs.get(col, 1.0) for col in x),
                )
                parent2 = min(
                    random.sample(population, 3),
                    key=lambda x: sum(self.costs.get(col, 1.0) for col in x),
                )

                # Crossover
                child = set()
                for col in parent1 | parent2:
                    if random.random() < 0.5:
                        child.add(col)

                # Mutation
                if random.random() < mutation_rate:
                    if random.random() < 0.5 and len(child) > 1:
                        child.remove(random.choice(list(child)))
                    else:
                        candidates = [c for c in self.columns if c not in child]
                        if candidates:
                            child.add(random.choice(candidates))

                # Check validity
                covered = set()
                for col in child:
                    covered |= self._coverage[col]

                if len(covered) >= target_covered:
                    new_population.append(child)

            population = new_population
            current_best = min(
                population, key=lambda x: sum(self.costs.get(col, 1.0) for col in x)
            )
            current_cost = sum(self.costs.get(col, 1.0) for col in current_best)
            best_cost = sum(self.costs.get(col, 1.0) for col in best)
            if current_cost < best_cost:
                best = current_best

        return list(best)

    def _lp_relaxation(
        self, epsilon_pairs: float, time_limit: float | None
    ) -> list[ColName]:
        """Linear Programming relaxation with rounding."""
        # For simplicity, fall back to greedy
        # A full implementation would use scipy.optimize.linprog or similar
        return self._greedy_set_cover(epsilon_pairs)

    def _hybrid_sa_ga(
        self, epsilon_pairs: float, time_limit: float | None
    ) -> list[ColName]:
        """Hybrid SA+GA approach."""
        if time_limit:
            half_time = time_limit / 2
            sa_result = set(self._simulated_annealing(epsilon_pairs, half_time))
            ga_result = set(self._genetic_algorithm(epsilon_pairs, half_time))

            sa_cost = sum(self.costs.get(col, 1.0) for col in sa_result)
            ga_cost = sum(self.costs.get(col, 1.0) for col in ga_result)

            return list(sa_result) if sa_cost <= ga_cost else list(ga_result)
        else:
            # Run both and pick the better one
            sa_result = set(self._simulated_annealing(epsilon_pairs, None))
            ga_result = set(self._genetic_algorithm(epsilon_pairs, None))

            sa_cost = sum(self.costs.get(col, 1.0) for col in sa_result)
            ga_cost = sum(self.costs.get(col, 1.0) for col in ga_result)

            return list(sa_result) if sa_cost <= ga_cost else list(ga_result)

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
        if not self._universe:
            return KeyPath(steps=[])

        # Compute pair weights if needed
        pair_weights = {}
        if weighting == "pair_idf":
            # IDF-like weighting: pairs covered by fewer columns get higher weight
            for pair in self._universe:
                covering_cols = [c for c in self.columns if pair in self._coverage[c]]
                if covering_cols:
                    pair_weights[pair] = 1.0 / len(covering_cols)
                else:
                    pair_weights[pair] = 1.0
        else:
            pair_weights = dict.fromkeys(self._universe, 1.0)

        selected = []
        uncovered = self._universe.copy()
        steps = []
        cumulative_cost = 0.0
        cumulative_covered = 0

        while uncovered and len(selected) < len(self.columns):
            best_col = None
            best_score = -float("inf")

            for col in self.columns:
                if col in selected:
                    continue

                newly_covered = uncovered & self._coverage[col]
                if not newly_covered:
                    continue

                cost = self.costs.get(col, 1.0)

                if objective == "pair_coverage":
                    # Score = weighted newly covered pairs / cost
                    weighted_gain = sum(pair_weights[p] for p in newly_covered)
                    score = weighted_gain / cost
                else:  # entropy
                    # Score = entropy reduction / cost
                    # Entropy before adding this column
                    n_clusters_before = len(uncovered) + len(selected) + 1
                    h_before = (
                        math.log2(n_clusters_before) if n_clusters_before > 1 else 0
                    )
                    # Entropy after adding this column
                    n_clusters_after = (
                        len(uncovered - newly_covered) + len(selected) + 2
                    )
                    h_after = math.log2(n_clusters_after) if n_clusters_after > 1 else 0
                    score = (h_before - h_after) / cost

                if score > best_score:
                    best_col = col
                    best_score = score

            if best_col is None:
                break

            newly_covered = uncovered & self._coverage[best_col]
            selected.append(best_col)
            uncovered -= newly_covered
            cost = self.costs.get(best_col, 1.0)
            cumulative_cost += cost
            cumulative_covered += len(newly_covered)

            step = KeyPathStep(
                col=best_col,
                newly_covered_pairs=len(newly_covered),
                cumulative_covered_pairs=cumulative_covered,
                total_pairs=len(self._universe),
                marginal_cost=cost,
                cumulative_cost=cumulative_cost,
                newly_covered_weight=(
                    sum(pair_weights[p] for p in newly_covered)
                    if weighting == "pair_idf"
                    else None
                ),
                cumulative_covered_weight=(
                    sum(pair_weights[p] for p in (self._universe - uncovered))
                    if weighting == "pair_idf"
                    else None
                ),
            )
            steps.append(step)

        return KeyPath(steps=steps)


def find_key(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    *,
    columns: Sequence[ColName] | None = None,
    costs: Mapping[ColName, float] | None = None,
    strategy: str = "greedy",
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
    objective: str = "pair_coverage",
    weighting: str = "uniform",
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
