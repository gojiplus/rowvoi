"""Advanced algorithms for the minimal set cover problem.

This module provides multiple algorithms for solving the NP-hard minimal set cover
problem, which arises when finding the smallest set of columns that can uniquely
identify all rows in a candidate set. Different algorithms offer various tradeoffs
between solution quality and computation time.

The set cover problem is formally defined as: given a universe U of elements and
a collection S of subsets of U, find the minimum number of subsets whose union
equals U. In the row disambiguation context, elements are pairs of rows that need
to be distinguished, and subsets represent which pairs each column can distinguish.

Algorithms available:
- Greedy: Fast O(log n) approximation (implemented in logical.py)
- Exact: Optimal solution via brute force (implemented in logical.py)
- ILP: Integer Linear Programming with branch & bound
- SA: Simulated Annealing metaheuristic
- GA: Genetic Algorithm metaheuristic
- Hybrid: Combined SA+GA approach
- LP: Linear Programming relaxation with rounding
"""

import abc
import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from .types import ColName, RowIndex


class SetCoverAlgorithm(Enum):
    """Available algorithms for minimal set cover problem."""

    GREEDY = "greedy"
    EXACT = "exact"
    ILP = "ilp"
    SIMULATED_ANNEALING = "sa"
    GENETIC_ALGORITHM = "ga"
    HYBRID_SA_GA = "hybrid"
    LP_RELAXATION = "lp"


@dataclass(frozen=True, slots=True)
class SetCoverResult:
    """Result of a set cover algorithm execution.

    Attributes
    ----------
    columns : list[ColName]
        The selected columns that form a set cover.
    algorithm : str
        Name of the algorithm used.
    is_optimal : bool
        Whether the solution is proven optimal.
    objective_value : float
        Size of the solution (number of columns).
    computation_time : float
        Time taken in seconds.
    iterations : int
        Number of iterations/generations for iterative algorithms.
    metadata : dict[str, Any]
        Algorithm-specific additional information.
    """

    columns: list[ColName]
    algorithm: str
    is_optimal: bool
    objective_value: float
    computation_time: float
    iterations: int
    metadata: dict[str, Any]


class BaseSetCoverSolver(abc.ABC):
    """Abstract base class for set cover algorithms."""

    def __init__(self, random_seed: int | None = None) -> None:
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

    @abc.abstractmethod
    def solve(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        candidate_cols: Sequence[ColName] | None = None,
        **kwargs: Any,
    ) -> SetCoverResult:
        """Solve the minimal set cover problem.

        Parameters
        ----------
        df : pd.DataFrame
            The data table.
        rows : Sequence[RowIndex]
            Row indices to distinguish.
        candidate_cols : Sequence[ColName], optional
            Columns to consider. If None, use all columns.
        **kwargs : Any
            Algorithm-specific parameters.

        Returns
        -------
        SetCoverResult
            Solution with metadata.
        """
        pass

    def _build_universe_and_coverage(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        candidate_cols: Sequence[ColName] | None = None,
    ) -> tuple[set[tuple[int, int]], dict[ColName, set[tuple[int, int]]]]:
        """Build the universe of pairs and column coverage mapping.

        Returns
        -------
        tuple
            (universe, coverage) where universe is set of row pairs
            and coverage maps columns to pairs they distinguish.
        """
        rows = list(rows)
        if len(rows) <= 1:
            return set(), {}

        # Universe: all pairs of rows that need distinguishing
        universe: set[tuple[int, int]] = set()
        for a, b in itertools.combinations(sorted(rows), 2):
            universe.add((a, b) if a < b else (b, a))

        # Determine columns to consider
        if candidate_cols is None:
            cols_to_consider = list(df.columns)
        else:
            cols_to_consider = list(candidate_cols)

        # Coverage: which pairs each column distinguishes
        coverage: dict[ColName, set[tuple[int, int]]] = {}
        for col in cols_to_consider:
            # Group rows by value in this column
            value_groups: dict[object, list[int]] = {}
            for r in rows:
                val = df.iloc[r][col]
                value_groups.setdefault(val, []).append(r)

            # Pairs distinguished by this column
            pairs: set[tuple[int, int]] = set()
            values = list(value_groups.keys())
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    rows_i = value_groups[values[i]]
                    rows_j = value_groups[values[j]]
                    for r in rows_i:
                        for s in rows_j:
                            pair = (r, s) if r < s else (s, r)
                            pairs.add(pair)

            coverage[col] = pairs

        return universe, coverage


def solve_set_cover(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    algorithm: SetCoverAlgorithm = SetCoverAlgorithm.GREEDY,
    candidate_cols: Sequence[ColName] | None = None,
    random_seed: int | None = None,
    **kwargs: Any,
) -> SetCoverResult:
    """Solve minimal set cover problem using specified algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        The data table.
    rows : Sequence[RowIndex]
        Row indices to distinguish.
    algorithm : SetCoverAlgorithm
        Algorithm to use.
    candidate_cols : Sequence[ColName], optional
        Columns to consider. If None, use all columns.
    random_seed : int, optional
        Random seed for reproducible results.
    **kwargs : Any
        Algorithm-specific parameters.

    Returns
    -------
    SetCoverResult
        Solution with metadata.
    """
    # Import here to avoid circular imports
    from .logical import minimal_key_exact, minimal_key_greedy

    if algorithm == SetCoverAlgorithm.GREEDY:
        import time

        start_time = time.time()
        columns = minimal_key_greedy(df, rows, candidate_cols)
        computation_time = time.time() - start_time

        return SetCoverResult(
            columns=columns,
            algorithm="greedy",
            is_optimal=False,
            objective_value=len(columns),
            computation_time=computation_time,
            iterations=1,
            metadata={"approximation_ratio": "O(log n)"},
        )

    elif algorithm == SetCoverAlgorithm.EXACT:
        import time

        start_time = time.time()
        columns = minimal_key_exact(
            df, rows, candidate_cols, kwargs.get("max_search_cols", 20)
        )
        computation_time = time.time() - start_time

        return SetCoverResult(
            columns=columns,
            algorithm="exact",
            is_optimal=True,
            objective_value=len(columns),
            computation_time=computation_time,
            iterations=1,
            metadata={"brute_force": True},
        )

    elif algorithm == SetCoverAlgorithm.ILP:
        solver = ILPSetCoverSolver(random_seed)
        return solver.solve(df, rows, candidate_cols, **kwargs)

    elif algorithm == SetCoverAlgorithm.SIMULATED_ANNEALING:
        solver = SimulatedAnnealingSetCoverSolver(random_seed)
        return solver.solve(df, rows, candidate_cols, **kwargs)

    elif algorithm == SetCoverAlgorithm.GENETIC_ALGORITHM:
        solver = GeneticAlgorithmSetCoverSolver(random_seed)
        return solver.solve(df, rows, candidate_cols, **kwargs)

    elif algorithm == SetCoverAlgorithm.HYBRID_SA_GA:
        solver = HybridSetCoverSolver(random_seed)
        return solver.solve(df, rows, candidate_cols, **kwargs)

    elif algorithm == SetCoverAlgorithm.LP_RELAXATION:
        solver = LPRelaxationSetCoverSolver(random_seed)
        return solver.solve(df, rows, candidate_cols, **kwargs)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


class ILPSetCoverSolver(BaseSetCoverSolver):
    """Integer Linear Programming solver using branch & bound."""

    def solve(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        candidate_cols: Sequence[ColName] | None = None,
        time_limit: float = 60.0,
        gap_tolerance: float = 0.0,
        **kwargs: Any,
    ) -> SetCoverResult:
        """Solve using ILP with branch & bound."""
        try:
            import pulp
        except ImportError:
            # Fallback to greedy if pulp not available
            import time

            from .logical import minimal_key_greedy

            start_time = time.time()
            columns = minimal_key_greedy(df, rows, candidate_cols)
            computation_time = time.time() - start_time

            return SetCoverResult(
                columns=columns,
                algorithm="ilp_fallback_greedy",
                is_optimal=False,
                objective_value=len(columns),
                computation_time=computation_time,
                iterations=1,
                metadata={"error": "pulp not available, used greedy fallback"},
            )

        import time

        start_time = time.time()

        universe, coverage = self._build_universe_and_coverage(df, rows, candidate_cols)

        if not universe:
            return SetCoverResult(
                columns=[],
                algorithm="ilp",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"trivial": True},
            )

        # Create ILP model
        prob = pulp.LpProblem("MinimalSetCover", pulp.LpMinimize)

        # Decision variables: x[col] = 1 if column is selected
        columns = list(coverage.keys())
        x = pulp.LpVariable.dicts("x", columns, cat="Binary")

        # Objective: minimize number of selected columns
        prob += pulp.lpSum([x[col] for col in columns])

        # Constraints: each pair must be covered by at least one column
        for pair in universe:
            prob += (
                pulp.lpSum([x[col] for col in columns if pair in coverage[col]]) >= 1
            )

        # Solve
        solver = pulp.PULP_CBC_CMD(
            timeLimit=int(time_limit), gapRel=gap_tolerance, msg=0
        )
        prob.solve(solver)

        computation_time = time.time() - start_time

        # Extract solution
        selected_columns = [col for col in columns if x[col].varValue == 1]

        is_optimal = prob.status == pulp.LpStatusOptimal

        return SetCoverResult(
            columns=selected_columns,
            algorithm="ilp",
            is_optimal=is_optimal,
            objective_value=len(selected_columns),
            computation_time=computation_time,
            iterations=1,
            metadata={
                "status": pulp.LpStatus[prob.status],
                "gap_tolerance": gap_tolerance,
                "time_limit": time_limit,
            },
        )


class SimulatedAnnealingSetCoverSolver(BaseSetCoverSolver):
    """Simulated Annealing metaheuristic for set cover."""

    def solve(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        candidate_cols: Sequence[ColName] | None = None,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 1e-6,
        max_iterations: int = 10000,
        **kwargs: Any,
    ) -> SetCoverResult:
        """Solve using Simulated Annealing."""
        import time

        start_time = time.time()

        universe, coverage = self._build_universe_and_coverage(df, rows, candidate_cols)

        if not universe:
            return SetCoverResult(
                columns=[],
                algorithm="simulated_annealing",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"trivial": True},
            )

        columns = list(coverage.keys())
        if not columns:
            return SetCoverResult(
                columns=[],
                algorithm="simulated_annealing",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"no_columns": True},
            )

        # Initialize with greedy solution
        current_solution = self._greedy_solution(universe, coverage)
        current_cost = len(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost

        temperature = initial_temperature
        iteration = 0

        while temperature > min_temperature and iteration < max_iterations:
            # Generate neighbor solution
            neighbor = self._get_neighbor(current_solution, columns, universe, coverage)
            neighbor_cost = len(neighbor)

            # Accept or reject
            if neighbor_cost < current_cost:
                current_solution = neighbor
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            else:
                # Accept with probability based on temperature
                delta = neighbor_cost - current_cost
                probability = math.exp(-delta / temperature)
                if random.random() < probability:
                    current_solution = neighbor
                    current_cost = neighbor_cost

            temperature *= cooling_rate
            iteration += 1

        computation_time = time.time() - start_time

        return SetCoverResult(
            columns=best_solution,
            algorithm="simulated_annealing",
            is_optimal=False,
            objective_value=best_cost,
            computation_time=computation_time,
            iterations=iteration,
            metadata={
                "final_temperature": temperature,
                "initial_temperature": initial_temperature,
                "cooling_rate": cooling_rate,
            },
        )

    def _greedy_solution(
        self,
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate initial solution using greedy algorithm."""
        selected: list[ColName] = []
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

            if best_col is None or best_count == 0:
                break

            selected.append(best_col)
            uncovered -= coverage[best_col]

        return selected

    def _get_neighbor(
        self,
        solution: list[ColName],
        all_columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate a neighbor solution."""
        if not solution:
            # Add a random column
            return [random.choice(all_columns)]

        operation = random.choice(["add", "remove", "swap"])
        neighbor = solution.copy()

        if operation == "add":
            # Add a column not in solution
            candidates = [col for col in all_columns if col not in neighbor]
            if candidates:
                neighbor.append(random.choice(candidates))

        elif operation == "remove" and len(neighbor) > 1:
            # Remove a column if solution remains valid
            col_to_remove = random.choice(neighbor)
            test_neighbor = [col for col in neighbor if col != col_to_remove]
            if self._is_valid_solution(test_neighbor, universe, coverage):
                neighbor = test_neighbor

        elif operation == "swap":
            # Replace one column with another
            if neighbor:
                col_to_remove = random.choice(neighbor)
                candidates = [col for col in all_columns if col not in neighbor]
                if candidates:
                    col_to_add = random.choice(candidates)
                    neighbor = [
                        col if col != col_to_remove else col_to_add for col in neighbor
                    ]

        return neighbor

    def _is_valid_solution(
        self,
        solution: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> bool:
        """Check if solution covers all pairs."""
        covered = set()
        for col in solution:
            covered |= coverage[col]
        return covered >= universe


class GeneticAlgorithmSetCoverSolver(BaseSetCoverSolver):
    """Genetic Algorithm for set cover."""

    def solve(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        candidate_cols: Sequence[ColName] | None = None,
        population_size: int = 50,
        max_generations: int = 1000,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        elitism_ratio: float = 0.1,
        **kwargs: Any,
    ) -> SetCoverResult:
        """Solve using Genetic Algorithm."""
        import time

        start_time = time.time()

        universe, coverage = self._build_universe_and_coverage(df, rows, candidate_cols)

        if not universe:
            return SetCoverResult(
                columns=[],
                algorithm="genetic_algorithm",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"trivial": True},
            )

        columns = list(coverage.keys())
        if not columns:
            return SetCoverResult(
                columns=[],
                algorithm="genetic_algorithm",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"no_columns": True},
            )

        # Initialize population
        population = self._initialize_population(
            population_size, columns, universe, coverage
        )
        best_individual = min(population, key=len)
        best_fitness = len(best_individual)

        generation = 0
        stagnation_count = 0
        max_stagnation = 50

        while generation < max_generations and stagnation_count < max_stagnation:
            # Evaluate fitness
            fitness_scores = [
                self._fitness(individual, universe, coverage)
                for individual in population
            ]

            # Track best solution
            current_best_idx = min(
                range(len(population)), key=lambda i: fitness_scores[i]
            )
            current_best = population[current_best_idx]
            current_fitness = fitness_scores[current_best_idx]

            if current_fitness < best_fitness:
                best_individual = current_best.copy()
                best_fitness = current_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Create next generation
            new_population = []

            # Elitism: keep best individuals
            elite_count = int(population_size * elitism_ratio)
            elite_indices = sorted(
                range(len(population)), key=lambda i: fitness_scores[i]
            )[:elite_count]
            for i in elite_indices:
                new_population.append(population[i].copy())

            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(
                    population, fitness_scores, tournament_size
                )
                parent2 = self._tournament_selection(
                    population, fitness_scores, tournament_size
                )

                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1, columns, universe, coverage)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2, columns, universe, coverage)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:population_size]
            generation += 1

        computation_time = time.time() - start_time

        return SetCoverResult(
            columns=best_individual,
            algorithm="genetic_algorithm",
            is_optimal=False,
            objective_value=best_fitness,
            computation_time=computation_time,
            iterations=generation,
            metadata={
                "population_size": population_size,
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate,
                "stagnation_generations": stagnation_count,
            },
        )

    def _initialize_population(
        self,
        size: int,
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[list[ColName]]:
        """Initialize population with valid solutions."""
        population = []

        # Add greedy solution
        greedy = self._greedy_solution(universe, coverage)
        population.append(greedy)

        # Add random valid solutions
        for _ in range(size - 1):
            individual = self._random_valid_solution(columns, universe, coverage)
            population.append(individual)

        return population

    def _random_valid_solution(
        self,
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate a random valid solution."""
        solution = []
        uncovered = universe.copy()
        available_columns = columns.copy()
        random.shuffle(available_columns)

        for col in available_columns:
            if uncovered & coverage[col]:  # If this column covers new pairs
                solution.append(col)
                uncovered -= coverage[col]
                if not uncovered:
                    break

        # Add some random columns with probability
        for col in columns:
            if col not in solution and random.random() < 0.1:
                solution.append(col)

        return solution

    def _fitness(
        self,
        individual: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> float:
        """Compute fitness (lower is better)."""
        if not individual:
            return float("inf")

        # Check coverage
        covered = set()
        for col in individual:
            covered |= coverage[col]

        uncovered_count = len(universe - covered)
        if uncovered_count > 0:
            # Heavy penalty for not covering all pairs
            return len(individual) + 1000 * uncovered_count

        # Objective is to minimize number of columns
        return len(individual)

    def _tournament_selection(
        self,
        population: list[list[ColName]],
        fitness_scores: list[float],
        tournament_size: int,
    ) -> list[ColName]:
        """Tournament selection."""
        tournament_indices = random.sample(
            range(len(population)), min(tournament_size, len(population))
        )
        winner_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_idx].copy()

    def _crossover(
        self,
        parent1: list[ColName],
        parent2: list[ColName],
    ) -> tuple[list[ColName], list[ColName]]:
        """Uniform crossover."""
        child1, child2 = [], []

        # Get all unique columns from both parents
        all_cols = list(set(parent1 + parent2))

        for col in all_cols:
            if random.random() < 0.5:
                if col in parent1:
                    child1.append(col)
                if col in parent2:
                    child2.append(col)
            else:
                if col in parent2:
                    child1.append(col)
                if col in parent1:
                    child2.append(col)

        return child1, child2

    def _mutate(
        self,
        individual: list[ColName],
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Mutate individual."""
        if not individual:
            return [random.choice(columns)]

        operation = random.choice(["add", "remove", "replace"])
        mutated = individual.copy()

        if operation == "add":
            # Add a column not in individual
            candidates = [col for col in columns if col not in mutated]
            if candidates:
                mutated.append(random.choice(candidates))

        elif operation == "remove" and len(mutated) > 1:
            # Remove a random column
            col_to_remove = random.choice(mutated)
            test_mutated = [col for col in mutated if col != col_to_remove]
            # Only remove if still valid
            if self._is_valid_solution(test_mutated, universe, coverage):
                mutated = test_mutated

        elif operation == "replace":
            # Replace one column with another
            if mutated:
                col_to_replace = random.choice(mutated)
                candidates = [col for col in columns if col not in mutated]
                if candidates:
                    col_to_add = random.choice(candidates)
                    mutated = [
                        col if col != col_to_replace else col_to_add for col in mutated
                    ]

        return mutated

    def _greedy_solution(
        self,
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate initial solution using greedy algorithm."""
        selected: list[ColName] = []
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

            if best_col is None or best_count == 0:
                break

            selected.append(best_col)
            uncovered -= coverage[best_col]

        return selected

    def _is_valid_solution(
        self,
        solution: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> bool:
        """Check if solution covers all pairs."""
        covered = set()
        for col in solution:
            covered |= coverage[col]
        return covered >= universe


class HybridSetCoverSolver(BaseSetCoverSolver):
    """Hybrid SA+GA solver combining the best of both approaches."""

    def solve(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        candidate_cols: Sequence[ColName] | None = None,
        population_size: int = 30,
        max_generations: int = 500,
        sa_iterations_per_generation: int = 50,
        initial_temperature: float = 50.0,
        cooling_rate: float = 0.98,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        sa_probability: float = 0.3,
        **kwargs: Any,
    ) -> SetCoverResult:
        """Solve using hybrid SA+GA approach.

        This hybrid algorithm combines genetic algorithm population-based search
        with simulated annealing local search. Some individuals undergo GA
        operations while others are improved via SA local search.
        """
        import time

        start_time = time.time()

        universe, coverage = self._build_universe_and_coverage(df, rows, candidate_cols)

        if not universe:
            return SetCoverResult(
                columns=[],
                algorithm="hybrid_sa_ga",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"trivial": True},
            )

        columns = list(coverage.keys())
        if not columns:
            return SetCoverResult(
                columns=[],
                algorithm="hybrid_sa_ga",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"no_columns": True},
            )

        # Initialize population
        population = self._initialize_population(
            population_size, columns, universe, coverage
        )
        best_individual = min(population, key=len)
        best_fitness = len(best_individual)

        temperature = initial_temperature
        generation = 0
        total_sa_iterations = 0

        while generation < max_generations:
            # Evaluate fitness
            fitness_scores = [
                self._fitness(individual, universe, coverage)
                for individual in population
            ]

            # Track best solution
            current_best_idx = min(
                range(len(population)), key=lambda i: fitness_scores[i]
            )
            current_best = population[current_best_idx]
            current_fitness = fitness_scores[current_best_idx]

            if current_fitness < best_fitness:
                best_individual = current_best.copy()
                best_fitness = current_fitness

            # Create next generation using hybrid approach
            new_population = []

            # Keep best individual (elitism)
            new_population.append(current_best.copy())

            for _i in range(1, population_size):
                if random.random() < sa_probability:
                    # Apply Simulated Annealing to random individual
                    individual_idx = random.randint(0, len(population) - 1)
                    improved_individual = self._simulated_annealing_local_search(
                        population[individual_idx],
                        columns,
                        universe,
                        coverage,
                        temperature,
                        sa_iterations_per_generation,
                    )
                    total_sa_iterations += sa_iterations_per_generation
                    new_population.append(improved_individual)
                else:
                    # Apply Genetic Algorithm operations
                    parent1_idx = self._tournament_selection_idx(fitness_scores, 3)
                    parent2_idx = self._tournament_selection_idx(fitness_scores, 3)

                    if random.random() < crossover_rate:
                        child1, child2 = self._crossover(
                            population[parent1_idx], population[parent2_idx]
                        )
                        child = child1 if len(child1) <= len(child2) else child2
                    else:
                        child = population[parent1_idx].copy()

                    if random.random() < mutation_rate:
                        child = self._mutate(child, columns, universe, coverage)

                    new_population.append(child)

            population = new_population
            temperature *= cooling_rate
            generation += 1

        computation_time = time.time() - start_time

        return SetCoverResult(
            columns=best_individual,
            algorithm="hybrid_sa_ga",
            is_optimal=False,
            objective_value=best_fitness,
            computation_time=computation_time,
            iterations=generation,
            metadata={
                "population_size": population_size,
                "sa_iterations_total": total_sa_iterations,
                "final_temperature": temperature,
                "sa_probability": sa_probability,
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate,
            },
        )

    def _initialize_population(
        self,
        size: int,
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[list[ColName]]:
        """Initialize population with diverse valid solutions."""
        population = []

        # Add greedy solution
        greedy = self._greedy_solution(universe, coverage)
        population.append(greedy)

        # Add reverse greedy (start from worst column)
        if size > 1:
            reverse_greedy = self._reverse_greedy_solution(universe, coverage)
            population.append(reverse_greedy)

        # Add random valid solutions
        for _ in range(size - len(population)):
            individual = self._random_valid_solution(columns, universe, coverage)
            population.append(individual)

        return population

    def _reverse_greedy_solution(
        self,
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate solution starting with least effective columns."""
        selected: list[ColName] = []
        uncovered = universe.copy()

        # Sort columns by their total coverage (ascending)
        columns_by_coverage = sorted(coverage.items(), key=lambda x: len(x[1]))

        for col, pairs in columns_by_coverage:
            if uncovered & pairs:  # If this column covers new pairs
                selected.append(col)
                uncovered -= pairs
                if not uncovered:
                    break

        return selected

    def _simulated_annealing_local_search(
        self,
        individual: list[ColName],
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
        temperature: float,
        iterations: int,
    ) -> list[ColName]:
        """Improve individual using simulated annealing."""
        current = individual.copy()
        current_cost = self._fitness(current, universe, coverage)

        for _ in range(iterations):
            neighbor = self._get_neighbor_sa(current, columns, universe, coverage)
            neighbor_cost = self._fitness(neighbor, universe, coverage)

            if neighbor_cost < current_cost:
                current = neighbor
                current_cost = neighbor_cost
            elif temperature > 0:
                delta = neighbor_cost - current_cost
                probability = math.exp(-delta / temperature)
                if random.random() < probability:
                    current = neighbor
                    current_cost = neighbor_cost

        return current

    def _get_neighbor_sa(
        self,
        solution: list[ColName],
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate neighbor for simulated annealing."""
        if not solution:
            return [random.choice(columns)]

        operation = random.choice(["add", "remove", "swap"])
        neighbor = solution.copy()

        if operation == "add":
            candidates = [col for col in columns if col not in neighbor]
            if candidates:
                neighbor.append(random.choice(candidates))

        elif operation == "remove" and len(neighbor) > 1:
            col_to_remove = random.choice(neighbor)
            test_neighbor = [col for col in neighbor if col != col_to_remove]
            if self._is_valid_solution(test_neighbor, universe, coverage):
                neighbor = test_neighbor

        elif operation == "swap":
            if neighbor:
                col_to_remove = random.choice(neighbor)
                candidates = [col for col in columns if col not in neighbor]
                if candidates:
                    col_to_add = random.choice(candidates)
                    neighbor = [
                        col if col != col_to_remove else col_to_add for col in neighbor
                    ]

        return neighbor

    def _tournament_selection_idx(
        self, fitness_scores: list[float], tournament_size: int
    ) -> int:
        """Tournament selection returning index."""
        tournament_indices = random.sample(
            range(len(fitness_scores)), min(tournament_size, len(fitness_scores))
        )
        return min(tournament_indices, key=lambda i: fitness_scores[i])

    def _crossover(
        self, parent1: list[ColName], parent2: list[ColName]
    ) -> tuple[list[ColName], list[ColName]]:
        """Uniform crossover."""
        child1, child2 = [], []
        all_cols = list(set(parent1 + parent2))

        for col in all_cols:
            if random.random() < 0.5:
                if col in parent1:
                    child1.append(col)
                if col in parent2:
                    child2.append(col)
            else:
                if col in parent2:
                    child1.append(col)
                if col in parent1:
                    child2.append(col)

        return child1, child2

    def _mutate(
        self,
        individual: list[ColName],
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Mutate individual."""
        if not individual:
            return [random.choice(columns)]

        operation = random.choice(["add", "remove", "replace"])
        mutated = individual.copy()

        if operation == "add":
            candidates = [col for col in columns if col not in mutated]
            if candidates:
                mutated.append(random.choice(candidates))

        elif operation == "remove" and len(mutated) > 1:
            col_to_remove = random.choice(mutated)
            test_mutated = [col for col in mutated if col != col_to_remove]
            if self._is_valid_solution(test_mutated, universe, coverage):
                mutated = test_mutated

        elif operation == "replace":
            if mutated:
                col_to_replace = random.choice(mutated)
                candidates = [col for col in columns if col not in mutated]
                if candidates:
                    col_to_add = random.choice(candidates)
                    mutated = [
                        col if col != col_to_replace else col_to_add for col in mutated
                    ]

        return mutated

    def _fitness(
        self,
        individual: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> float:
        """Compute fitness (lower is better)."""
        if not individual:
            return float("inf")

        covered = set()
        for col in individual:
            covered |= coverage[col]

        uncovered_count = len(universe - covered)
        if uncovered_count > 0:
            return len(individual) + 1000 * uncovered_count

        return len(individual)

    def _greedy_solution(
        self,
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate greedy solution."""
        selected: list[ColName] = []
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

            if best_col is None or best_count == 0:
                break

            selected.append(best_col)
            uncovered -= coverage[best_col]

        return selected

    def _random_valid_solution(
        self,
        columns: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Generate random valid solution."""
        solution = []
        uncovered = universe.copy()
        available_columns = columns.copy()
        random.shuffle(available_columns)

        for col in available_columns:
            if uncovered & coverage[col]:
                solution.append(col)
                uncovered -= coverage[col]
                if not uncovered:
                    break

        # Add some random columns with low probability
        for col in columns:
            if col not in solution and random.random() < 0.05:
                solution.append(col)

        return solution

    def _is_valid_solution(
        self,
        solution: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> bool:
        """Check if solution covers all pairs."""
        covered = set()
        for col in solution:
            covered |= coverage[col]
        return covered >= universe


class LPRelaxationSetCoverSolver(BaseSetCoverSolver):
    """LP Relaxation with randomized rounding."""

    def solve(
        self,
        df: pd.DataFrame,
        rows: Sequence[RowIndex],
        candidate_cols: Sequence[ColName] | None = None,
        rounding_iterations: int = 100,
        alpha_multiplier: float = 1.0,
        time_limit: float = 30.0,
        **kwargs: Any,
    ) -> SetCoverResult:
        """Solve using LP relaxation with randomized rounding.

        This algorithm relaxes the integer constraints to get fractional solution,
        then uses randomized rounding to convert to integer solution.
        """
        try:
            import pulp
        except ImportError:
            # Fallback to greedy if pulp not available
            import time

            from .logical import minimal_key_greedy

            start_time = time.time()
            columns = minimal_key_greedy(df, rows, candidate_cols)
            computation_time = time.time() - start_time

            return SetCoverResult(
                columns=columns,
                algorithm="lp_relaxation_fallback_greedy",
                is_optimal=False,
                objective_value=len(columns),
                computation_time=computation_time,
                iterations=1,
                metadata={"error": "pulp not available, used greedy fallback"},
            )

        import time

        start_time = time.time()

        universe, coverage = self._build_universe_and_coverage(df, rows, candidate_cols)

        if not universe:
            return SetCoverResult(
                columns=[],
                algorithm="lp_relaxation",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"trivial": True},
            )

        columns = list(coverage.keys())
        if not columns:
            return SetCoverResult(
                columns=[],
                algorithm="lp_relaxation",
                is_optimal=True,
                objective_value=0.0,
                computation_time=time.time() - start_time,
                iterations=0,
                metadata={"no_columns": True},
            )

        # Solve LP relaxation
        prob = pulp.LpProblem("SetCoverLP", pulp.LpMinimize)

        # Variables: x[col] ∈ [0,1] for fractional solution
        x = pulp.LpVariable.dicts("x", columns, lowBound=0, upBound=1, cat="Continuous")

        # Objective: minimize sum of variables
        prob += pulp.lpSum([x[col] for col in columns])

        # Constraints: each pair must be covered
        for pair in universe:
            prob += (
                pulp.lpSum([x[col] for col in columns if pair in coverage[col]]) >= 1
            )

        # Solve LP relaxation
        solver = pulp.PULP_CBC_CMD(timeLimit=int(time_limit), msg=0)
        prob.solve(solver)

        if prob.status != pulp.LpStatusOptimal:
            # Fallback to greedy if LP fails
            from .logical import minimal_key_greedy

            columns = minimal_key_greedy(df, rows, candidate_cols)
            computation_time = time.time() - start_time

            return SetCoverResult(
                columns=columns,
                algorithm="lp_relaxation_fallback_greedy",
                is_optimal=False,
                objective_value=len(columns),
                computation_time=computation_time,
                iterations=1,
                metadata={"lp_status": pulp.LpStatus[prob.status]},
            )

        # Extract fractional solution
        fractional_solution = {col: x[col].varValue for col in columns}
        lp_objective = sum(fractional_solution.values())

        # Randomized rounding
        best_solution = None
        best_cost = float("inf")

        for _iteration in range(rounding_iterations):
            # Compute rounding probability threshold
            # Use alpha * max(x_j) as threshold, where alpha is tunable
            max_frac = max(fractional_solution.values()) if fractional_solution else 0
            alpha = alpha_multiplier / max(1, max_frac) if max_frac > 0 else 1.0

            # Round using threshold
            candidate_solution = []
            for col, frac_val in fractional_solution.items():
                # Include column with probability proportional to its fractional value
                if frac_val >= alpha or random.random() < frac_val / alpha:
                    candidate_solution.append(col)

            # Check if solution is feasible (covers all pairs)
            if self._is_valid_solution(candidate_solution, universe, coverage):
                cost = len(candidate_solution)
                if cost < best_cost:
                    best_solution = candidate_solution
                    best_cost = cost

        # If no valid solution found, apply greedy completion to best fractional
        if best_solution is None:
            best_solution = self._greedy_completion(
                fractional_solution, universe, coverage
            )
            best_cost = len(best_solution)

        computation_time = time.time() - start_time

        return SetCoverResult(
            columns=best_solution,
            algorithm="lp_relaxation",
            is_optimal=False,
            objective_value=best_cost,
            computation_time=computation_time,
            iterations=rounding_iterations,
            metadata={
                "lp_objective": lp_objective,
                "integrality_gap": best_cost / max(lp_objective, 1e-6),
                "rounding_iterations": rounding_iterations,
                "alpha_multiplier": alpha_multiplier,
            },
        )

    def _greedy_completion(
        self,
        fractional_solution: dict[ColName, float],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> list[ColName]:
        """Complete solution greedily based on fractional values."""
        # Start with columns with highest fractional values
        sorted_cols = sorted(
            fractional_solution.items(), key=lambda x: x[1], reverse=True
        )

        selected = []
        uncovered = universe.copy()

        for col, _frac_val in sorted_cols:
            if uncovered & coverage[col]:  # If this column covers new pairs
                selected.append(col)
                uncovered -= coverage[col]
                if not uncovered:
                    break

        return selected

    def _is_valid_solution(
        self,
        solution: list[ColName],
        universe: set[tuple[int, int]],
        coverage: dict[ColName, set[tuple[int, int]]],
    ) -> bool:
        """Check if solution covers all pairs."""
        covered = set()
        for col in solution:
            covered |= coverage[col]
        return covered >= universe
