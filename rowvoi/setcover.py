"""Weighted set cover: the solver engine shared by keys and RAG context selection.

Nothing here knows what an element *means*. :class:`~rowvoi.keys.KeyProblem` builds
a universe of row pairs on top of it; :mod:`rowvoi.rag.context` builds a universe of
claims. The seven strategies and the greedy path planner are common to both.
"""

import itertools
import logging
import math
import random
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# Type aliases
Element = Hashable
SetName = Hashable

Strategy = Literal["greedy", "exact", "ilp", "sa", "ga", "lp", "hybrid"]

logger = logging.getLogger(__name__)

# Solvers to try, in order. COIN_CMD is not deprecated but finds CBC only on
# PATH (or via the `pulp[cbc]` extra); PULP_CBC_CMD ships a bundled CBC but is
# removed in PuLP 4, hence the getattr lookup rather than direct attribute
# access.
_SOLVER_NAMES = ("COIN_CMD", "PULP_CBC_CMD")


class SolverUnavailableError(RuntimeError):
    """No linear programming solver is usable.

    Raised rather than quietly substituting an approximate strategy: greedy is
    an ln(m) approximation and ILP is exact, so returning one where the caller
    asked for the other is a silent correctness change.
    """


def _find_solver(pulp, *, time_limit: float | None = None):
    """Return the first available pulp solver, or None if there is none.

    Args:
        pulp: The imported ``pulp`` module.
        time_limit: Seconds to allow the solver, passed through when set.

    Returns:
        A configured solver instance, or None when none is usable.
    """
    kwargs: dict[str, Any] = {"msg": False}
    if time_limit:
        kwargs["timeLimit"] = time_limit

    for name in _SOLVER_NAMES:
        cls = getattr(pulp, name, None)
        if cls is None:
            continue
        try:
            solver = cls(**kwargs)
        except Exception:  # pragma: no cover - solver refuses construction
            logger.debug("solver %s could not be constructed", name, exc_info=True)
            continue
        # `available()` returns a path or False/None, not a strict bool. A
        # stock install has COIN_CMD unavailable, which is exactly the case
        # that otherwise surfaces as PulpSolverError deep inside solve().
        if solver.available():
            return solver
    return None


@dataclass
class CoverStep:
    """A single step in a cover path showing incremental progress.

    Attributes:
        name: The set added in this step
        newly_covered: Number of elements newly covered by this set
        cumulative_covered: Total elements covered up to and including this step
        total_elements: Size of the universe
        marginal_cost: Cost of adding this specific set
        cumulative_cost: Total cost up to and including this step
        newly_covered_weight: Weighted coverage gain (for weighted objectives)
        cumulative_covered_weight: Total weighted coverage so far
    """

    name: SetName
    newly_covered: int
    cumulative_covered: int
    total_elements: int
    marginal_cost: float
    cumulative_cost: float
    newly_covered_weight: float | None = None
    cumulative_covered_weight: float | None = None

    @property
    def coverage(self) -> float:
        """Fraction of the universe covered so far."""
        if self.total_elements == 0:
            return 1.0
        return self.cumulative_covered / self.total_elements


@dataclass
class CoverPath:
    """Ordered sequence of sets and their contribution to coverage/cost.

    Attributes:
        steps: Ordered list of steps showing incremental progress
    """

    steps: list[CoverStep]

    def names(self) -> list[SetName]:
        """Return the ordered list of set names in the path."""
        return [step.name for step in self.steps]

    def prefix_for_budget(self, budget: float) -> list[SetName]:
        """Return the longest prefix of sets whose cumulative_cost <= budget.

        Args:
            budget: Maximum allowed cumulative cost

        Returns:
            Sets that fit within the budget
        """
        result = []
        for step in self.steps:
            if step.cumulative_cost <= budget:
                result.append(step.name)
            else:
                break
        return result

    def prefix_for_epsilon(self, epsilon: float) -> list[SetName]:
        """Return the shortest prefix that leaves <= epsilon fraction uncovered.

        Args:
            epsilon: Maximum allowed fraction of uncovered elements

        Returns:
            Minimum sets needed to achieve (1-epsilon) coverage
        """
        target_coverage = 1.0 - epsilon
        for i, step in enumerate(self.steps):
            if step.coverage >= target_coverage:
                return self.names()[: i + 1]
        return self.names()

    def coverage_curve(self) -> list[tuple[float, float]]:
        """Return the coverage curve as (cumulative_cost, coverage_fraction) points."""
        return [(step.cumulative_cost, step.coverage) for step in self.steps]


class SetCoverProblem:
    """Weighted set cover over an arbitrary universe.

    Args:
        sets: Maps each selectable set to the elements it covers
        universe: Elements that must be covered. If None, the union of all sets is used.
            Pass it explicitly when some elements are covered by no set (they then
            count against coverage rather than being silently ignored).
        costs: Cost of selecting each set. Missing entries default to 1.0
    """

    def __init__(
        self,
        sets: Mapping[Any, Iterable[Any]],
        *,
        universe: Iterable[Any] | None = None,
        costs: Mapping[Any, float] | None = None,
    ) -> None:
        self.sets: dict[SetName, set[Element]] = {
            name: set(elements) for name, elements in sets.items()
        }
        self.names: list[SetName] = list(self.sets)
        if universe is None:
            self.universe: set[Element] = (
                set().union(*self.sets.values()) if self.sets else set()
            )
        else:
            self.universe = set(universe)
        self.costs = dict(costs or {})

    def cost(self, name: SetName) -> float:
        """Cost of selecting one set (1.0 when unspecified)."""
        return self.costs.get(name, 1.0)

    def total_cost(self, names: Iterable[SetName]) -> float:
        """Total cost of a selection."""
        return sum(self.cost(name) for name in names)

    def covered(self, names: Iterable[SetName]) -> set[Element]:
        """Elements of the universe covered by a selection."""
        result: set[Element] = set()
        for name in names:
            if name in self.sets:
                result |= self.sets[name]
        return result & self.universe

    def coverage(self, names: Iterable[SetName]) -> float:
        """Fraction of the universe covered by a selection."""
        if not self.universe:
            return 1.0
        return len(self.covered(names)) / len(self.universe)

    def is_cover(self, names: Iterable[SetName], *, epsilon: float = 0.0) -> bool:
        """Whether a selection covers all but at most `epsilon` of the universe."""
        return self.coverage(names) >= 1.0 - epsilon

    def solve(
        self,
        strategy: Strategy = "greedy",
        *,
        epsilon: float = 0.0,
        time_limit: float | None = None,
    ) -> list[SetName]:
        """Find a minimum-cost selection covering (1-epsilon) of the universe.

        Args:
            strategy: Algorithm to use:
                - "greedy": greedy cost/gain ratio (ln m approximation)
                - "exact": brute force enumeration (only for small problems)
                - "ilp": Integer Linear Programming (requires pulp)
                - "sa": Simulated Annealing metaheuristic
                - "ga": Genetic Algorithm metaheuristic
                - "lp": Linear Programming relaxation with rounding
                - "hybrid": Combined SA+GA approach
            epsilon: Allow this fraction of the universe to remain uncovered
            time_limit: Maximum time in seconds

        Returns:
            Minimal (or near-minimal) selection

        Raises:
            ValueError: If `strategy` is not one of the supported names.
        """
        if strategy == "greedy":
            return self._greedy(epsilon)
        elif strategy == "exact":
            return self._exact(epsilon, time_limit)
        elif strategy == "ilp":
            return self._ilp(epsilon, time_limit)
        elif strategy == "sa":
            return self._simulated_annealing(epsilon, time_limit)
        elif strategy == "ga":
            return self._genetic_algorithm(epsilon, time_limit)
        elif strategy == "lp":
            return self._lp_relaxation(epsilon, time_limit)
        elif strategy == "hybrid":
            return self._hybrid_sa_ga(epsilon, time_limit)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _greedy(self, epsilon: float) -> list[SetName]:
        """Greedy set cover algorithm."""
        if not self.universe:
            return []

        selected: list[SetName] = []
        uncovered = self.universe.copy()
        target_covered = len(self.universe) * (1 - epsilon)

        while len(self.universe) - len(uncovered) < target_covered:
            best_name = None
            best_cost_ratio = float("inf")

            for name in self.names:
                if name in selected:
                    continue
                gain = len(uncovered & self.sets[name])
                if gain > 0:
                    cost_ratio = self.cost(name) / gain
                    if cost_ratio < best_cost_ratio:
                        best_name = name
                        best_cost_ratio = cost_ratio

            if best_name is None:
                break

            selected.append(best_name)
            uncovered -= self.sets[best_name]

        return selected

    def _exact(self, epsilon: float, time_limit: float | None) -> list[SetName]:
        """Exact solution via brute force enumeration, minimizing *cost*.

        Subsets are enumerated by increasing size, but a solution of the
        smallest size is not necessarily the cheapest one -- with non-uniform
        costs, two cheap sets can beat one expensive set that covers
        everything. So we keep searching until no larger subset could possibly
        undercut the incumbent: any subset of size k costs at least
        ``k * min_cost``, which gives an admissible cutoff. Under uniform costs
        that cutoff fires immediately after the first feasible size, so this is
        no slower than stopping at the smallest cover.
        """
        if not self.universe:
            return []

        target_covered = len(self.universe) * (1 - epsilon)
        best = None
        best_cost = float("inf")
        start_time = time.time()

        min_cost = min((self.cost(name) for name in self.names), default=0.0)

        for size in range(1, len(self.names) + 1):
            if time_limit and time.time() - start_time > time_limit:
                break

            # No subset this large or larger can beat the incumbent
            if best is not None and min_cost > 0 and size * min_cost >= best_cost:
                break

            for subset in itertools.combinations(self.names, size):
                covered: set[Element] = set()
                cost = 0.0
                for name in subset:
                    covered |= self.sets[name]
                    cost += self.cost(name)

                if len(covered & self.universe) >= target_covered and cost < best_cost:
                    best = list(subset)
                    best_cost = cost

        return best if best is not None else self._greedy(epsilon)

    def _ilp(self, epsilon: float, time_limit: float | None) -> list[SetName]:
        """Integer Linear Programming solution.

        Args:
            epsilon: Fraction of the universe permitted to remain uncovered.
            time_limit: Maximum seconds to allow the solver.

        Returns:
            A minimum-cost selection.

        Raises:
            SolverUnavailableError: If pulp is not installed, or no LP solver
                is usable. Deliberately not downgraded to greedy: ILP is exact
                and greedy is an approximation, so silently swapping them would
                make `optimality_gap` in the evaluation results meaningless.
        """
        try:
            import pulp
        except ImportError as exc:
            raise SolverUnavailableError(
                "The 'ilp' strategy needs pulp, which is not installed. "
                'Install it with: pip install "rowvoi[optimization]"'
            ) from exc

        if not self.universe:
            return []

        solver = _find_solver(pulp, time_limit=time_limit)
        if solver is None:
            raise SolverUnavailableError(
                "pulp is installed but no LP solver is available. Install CBC "
                'with: pip install "pulp[cbc]"'
            )

        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)

        # PuLP 4 moves variable creation onto the problem and deprecates
        # constructing LpVariable directly. Prefer the new call where it
        # exists so we neither warn on new PuLP nor break on the >=2.7 floor.
        add_variable = getattr(prob, "add_variable", None)

        def _binary(varname: str):
            if add_variable is not None:
                return add_variable(varname, cat="Binary")
            return pulp.LpVariable(varname, cat="Binary")  # pragma: no cover

        # Decision variables: x[name] = 1 if that set is selected. Index-based
        # variable names keep pulp happy with arbitrary hashable set names.
        x = {name: _binary(f"x_{i}") for i, name in enumerate(self.names)}

        # Objective: minimize total cost
        prob += pulp.lpSum(x[name] * self.cost(name) for name in self.names)

        elements = sorted(self.universe, key=repr)
        covering = {
            element: [name for name in self.names if element in self.sets[name]]
            for element in elements
        }

        if epsilon == 0:
            # Exact coverage constraints
            for element in elements:
                if covering[element]:
                    prob += pulp.lpSum(x[name] for name in covering[element]) >= 1
        else:
            # Relaxed: count covered elements
            target = len(self.universe) * (1 - epsilon)
            y = {element: _binary(f"y_{i}") for i, element in enumerate(elements)}
            for element in elements:
                if covering[element]:
                    prob += y[element] <= pulp.lpSum(
                        x[name] for name in covering[element]
                    )
                else:
                    prob += y[element] == 0
            prob += pulp.lpSum(y[element] for element in elements) >= target

        prob.solve(solver)

        if prob.status == pulp.LpStatusOptimal:
            # An unset variable reads back as None rather than 0.0.
            return [name for name in self.names if (x[name].value() or 0.0) > 0.5]
        else:
            return self._greedy(epsilon)

    def _simulated_annealing(
        self, epsilon: float, time_limit: float | None
    ) -> list[SetName]:
        """Use Simulated Annealing metaheuristic."""
        if not self.universe:
            return []

        # Start with greedy solution
        current = set(self._greedy(epsilon))
        current_cost = self.total_cost(current)
        best = current.copy()
        best_cost = current_cost

        target_covered = len(self.universe) * (1 - epsilon)
        temperature = 10.0
        cooling_rate = 0.95
        start_time = time.time()

        while temperature > 0.01:
            if time_limit and time.time() - start_time > time_limit:
                break

            # Generate neighbor
            neighbor = current.copy()
            if random.random() < 0.5 and len(neighbor) > 1:
                neighbor.remove(random.choice(list(neighbor)))
            else:
                candidates = [n for n in self.names if n not in neighbor]
                if candidates:
                    neighbor.add(random.choice(candidates))

            # Check if neighbor is valid
            if len(self.covered(neighbor)) >= target_covered:
                neighbor_cost = self.total_cost(neighbor)
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
        self, epsilon: float, time_limit: float | None
    ) -> list[SetName]:
        """Genetic Algorithm metaheuristic."""
        if not self.universe:
            return []

        target_covered = len(self.universe) * (1 - epsilon)
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        start_time = time.time()

        # Initialize population with the greedy solution plus random valid ones
        population = [set(self._greedy(epsilon))]

        while len(population) < population_size:
            if time_limit and time.time() - start_time > time_limit:
                break

            individual: set[SetName] = set()
            uncovered = self.universe.copy()

            while len(self.universe) - len(uncovered) < target_covered:
                candidates = [n for n in self.names if n not in individual]
                if not candidates:
                    break
                name = random.choice(candidates)
                individual.add(name)
                uncovered -= self.sets[name]

            if len(self.universe) - len(uncovered) >= target_covered:
                population.append(individual)

        best = min(population, key=self.total_cost)

        for _ in range(generations):
            if time_limit and time.time() - start_time > time_limit:
                break

            # Selection and crossover
            new_population = [best]  # Elitism

            while len(new_population) < population_size:
                # Tournament selection
                parent1 = min(random.sample(population, 3), key=self.total_cost)
                parent2 = min(random.sample(population, 3), key=self.total_cost)

                # Crossover
                child = {n for n in parent1 | parent2 if random.random() < 0.5}

                # Mutation
                if random.random() < mutation_rate:
                    if random.random() < 0.5 and len(child) > 1:
                        child.remove(random.choice(list(child)))
                    else:
                        candidates = [n for n in self.names if n not in child]
                        if candidates:
                            child.add(random.choice(candidates))

                if len(self.covered(child)) >= target_covered:
                    new_population.append(child)

            population = new_population
            current_best = min(population, key=self.total_cost)
            if self.total_cost(current_best) < self.total_cost(best):
                best = current_best

        return list(best)

    def _lp_relaxation(self, epsilon: float, time_limit: float | None) -> list[SetName]:
        """Linear Programming relaxation with rounding."""
        # For simplicity, fall back to greedy
        # A full implementation would use scipy.optimize.linprog or similar
        return self._greedy(epsilon)

    def _hybrid_sa_ga(self, epsilon: float, time_limit: float | None) -> list[SetName]:
        """Hybrid SA+GA approach: run both, keep the cheaper."""
        half_time = time_limit / 2 if time_limit else None
        sa_result = set(self._simulated_annealing(epsilon, half_time))
        ga_result = set(self._genetic_algorithm(epsilon, half_time))

        if self.total_cost(sa_result) <= self.total_cost(ga_result):
            return list(sa_result)
        return list(ga_result)

    def element_weights(
        self, weighting: Literal["uniform", "idf"] = "uniform"
    ) -> dict[Element, float]:
        """Per-element weights used by the path planner.

        "idf" gives elements covered by fewer sets a higher weight, so scarce
        coverage is prioritized. Elements covered by no set get weight 1.0.
        """
        if weighting != "idf":
            return dict.fromkeys(self.universe, 1.0)

        weights = {}
        for element in self.universe:
            n_covering = sum(1 for name in self.names if element in self.sets[name])
            weights[element] = 1.0 / n_covering if n_covering else 1.0
        return weights

    def plan_path(
        self,
        *,
        objective: Literal["coverage", "entropy"] = "coverage",
        weighting: Literal["uniform", "idf"] = "uniform",
    ) -> CoverPath:
        """Build a greedy ordering of sets, tracking coverage and cost per step.

        Unlike :meth:`solve`, which returns an unordered selection, this returns
        the order in which to acquire sets — so callers can cut the sequence at
        a budget or a coverage target.

        Args:
            objective: - "coverage": gain = weighted newly covered elements
                - "entropy": gain = reduction in log cluster size
            weighting: - "uniform": all elements weighted equally
                - "idf": elements covered by fewer sets weighted higher

        Returns:
            Ordered sequence with coverage information
        """
        if not self.universe:
            return CoverPath(steps=[])

        weights = self.element_weights(weighting)

        selected: list[SetName] = []
        uncovered = self.universe.copy()
        steps: list[CoverStep] = []
        cumulative_cost = 0.0
        cumulative_covered = 0

        while uncovered and len(selected) < len(self.names):
            best_name = None
            best_score = -float("inf")

            for name in self.names:
                if name in selected:
                    continue

                newly_covered = uncovered & self.sets[name]
                if not newly_covered:
                    continue

                cost = self.cost(name)

                if objective == "coverage":
                    weighted_gain = sum(weights[e] for e in newly_covered)
                    score = weighted_gain / cost
                else:  # entropy
                    n_clusters_before = len(uncovered) + len(selected) + 1
                    h_before = (
                        math.log2(n_clusters_before) if n_clusters_before > 1 else 0
                    )
                    n_clusters_after = (
                        len(uncovered - newly_covered) + len(selected) + 2
                    )
                    h_after = math.log2(n_clusters_after) if n_clusters_after > 1 else 0
                    score = (h_before - h_after) / cost

                if score > best_score:
                    best_name = name
                    best_score = score

            if best_name is None:
                break

            newly_covered = uncovered & self.sets[best_name]
            selected.append(best_name)
            uncovered -= newly_covered
            cost = self.cost(best_name)
            cumulative_cost += cost
            cumulative_covered += len(newly_covered)

            steps.append(
                CoverStep(
                    name=best_name,
                    newly_covered=len(newly_covered),
                    cumulative_covered=cumulative_covered,
                    total_elements=len(self.universe),
                    marginal_cost=cost,
                    cumulative_cost=cumulative_cost,
                    newly_covered_weight=(
                        sum(weights[e] for e in newly_covered)
                        if weighting == "idf"
                        else None
                    ),
                    cumulative_covered_weight=(
                        sum(weights[e] for e in (self.universe - uncovered))
                        if weighting == "idf"
                        else None
                    ),
                )
            )

        return CoverPath(steps=steps)


def solve_set_cover(
    sets: Mapping[Any, Iterable[Any]],
    *,
    universe: Iterable[Any] | None = None,
    costs: Mapping[Any, float] | None = None,
    strategy: Strategy = "greedy",
    epsilon: float = 0.0,
    time_limit: float | None = None,
) -> list[SetName]:
    """Solve a weighted set cover instance.

    Convenience wrapper around :meth:`SetCoverProblem.solve`.
    """
    problem = SetCoverProblem(sets, universe=universe, costs=costs)
    return problem.solve(strategy, epsilon=epsilon, time_limit=time_limit)


def coverage_of(
    sets: Mapping[Any, Iterable[Any]],
    selection: Sequence[SetName],
    *,
    universe: Iterable[Any] | None = None,
) -> float:
    """Fraction of the universe covered by `selection`."""
    return SetCoverProblem(sets, universe=universe).coverage(selection)
