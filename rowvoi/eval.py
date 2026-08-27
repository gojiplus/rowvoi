"""Simulation and evaluation tools for rowvoi.

This module provides comprehensive evaluation tools for comparing different
key-finding algorithms and policies, including gold standard computation
and systematic benchmarking.
"""

import logging
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .core import CandidateState, ColName, RowIndex
from .keys import KeyProblem, find_key
from .policies import Policy
from .session import DisambiguationSession, StopRules
from .setcover import SolverUnavailableError, Strategy

if TYPE_CHECKING:
    from .ml import RowVoiModel

logger = logging.getLogger(__name__)


def sample_candidate_sets(
    df: pd.DataFrame,
    *,
    subset_size: int,
    n_samples: int,
    random_state: int | None = None,
) -> list[list[RowIndex]]:
    """Randomly sample subsets of rows from a DataFrame.

    Args:
        df: The data frame to sample from
        subset_size: Number of rows in each subset
        n_samples: Number of subsets to generate
        random_state: Random seed for reproducibility

    Returns:
        List of row index lists

    Raises:
        ValueError: If subset_size exceeds the number of rows in the frame.
    """
    if random_state is not None:
        random.seed(random_state)

    n_rows = len(df)
    if subset_size > n_rows:
        raise ValueError(
            f"subset_size ({subset_size}) cannot exceed number of rows ({n_rows})"
        )

    candidate_sets = []
    for _ in range(n_samples):
        subset = random.sample(range(n_rows), subset_size)
        candidate_sets.append(sorted(subset))

    return candidate_sets


def compute_gold_key(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    *,
    columns: Sequence[ColName] | None = None,
    costs: Mapping[ColName, float] | None = None,
    epsilon_pairs: float = 0.0,
    time_limit: float = 10.0,
    allow_approximate: bool = False,
) -> list[ColName]:
    """Compute the provably optimal deterministic key.

    Tries ILP, then exhaustive search. Both are exact, so whichever answers
    first is the optimum.

    This deliberately does *not* fall back to greedy. Callers use the result as
    the baseline for `optimality_gap`, and a greedy baseline makes that number
    wrong -- it can even go negative, showing a method beating the "optimum".
    Failing is the honest outcome; `evaluate_keys` already treats a failed gold
    solve as "no baseline" and reports no gap.

    Args:
        df: The data table
        rows: Row indices to distinguish
        columns: Columns to consider
        costs: Cost of each column
        epsilon_pairs: Tolerance for unresolved pairs
        time_limit: Maximum time for each exact strategy
        allow_approximate: Permit a greedy result when no exact one is
            obtainable. The return value is then not necessarily optimal, so
            do not use it as an optimality baseline.

    Returns:
        Optimal key columns, or an approximate key when `allow_approximate`
        is set and no exact strategy succeeded.

    Raises:
        SolverUnavailableError: If no exact strategy could produce a key and
            `allow_approximate` is False.
    """
    # Both of these are exact; greedy is not, and is only reachable on request.
    strategies: list[Strategy] = ["ilp", "exact"]

    for strategy in strategies:
        try:
            return find_key(
                df,
                rows,
                columns=columns,
                costs=costs,
                strategy=strategy,
                epsilon_pairs=epsilon_pairs,
                time_limit=time_limit,
            )
        except Exception:
            # ILP needs a solver, exhaustive search can time out. Trying the
            # next exact strategy is the point, but stay diagnosable.
            logger.debug("gold key strategy %r failed, trying next", strategy)
            continue

    if not allow_approximate:
        raise SolverUnavailableError(
            "No exact strategy could compute a gold-standard key: ILP needs a "
            'solver (pip install "rowvoi[optimization]") and exhaustive '
            f"search did not finish within time_limit={time_limit}s. Pass "
            "allow_approximate=True to accept a greedy key, but note it is not "
            "a valid optimality baseline."
        )

    logger.debug("falling back to an approximate gold key for %d rows", len(rows))
    # No time_limit: SetCoverProblem.solve() routes "greedy" to _greedy(epsilon),
    # which takes no budget. The limit belongs to the exact and ILP strategies
    # that were already tried and exhausted above.
    return find_key(  # preen: allow-dropped-arg
        df,
        rows,
        columns=columns,
        costs=costs,
        strategy="greedy",
        epsilon_pairs=epsilon_pairs,
    )


def compute_gold_next_column_probabilistic(
    df: pd.DataFrame,
    state: CandidateState,
    model: "RowVoiModel",
    *,
    candidate_cols: Sequence[ColName] | None = None,
    # Part of the documented API; the current implementation delegates to the
    # model's suggestion, which computes MI, so only the default is honoured.
    objective: Literal["mi", "expected_entropy_reduction"] = "mi",  # noqa: ARG001
) -> ColName:
    """Compute the 'gold standard' next column under a model.

    Args:
        df: The data table
        state: Current state
        model: Trained model
        candidate_cols: Columns to consider
        objective: Objective function

    Returns:
        Optimal next column
    """
    # For now, use the model's suggestion as gold standard
    # A more sophisticated implementation could do exact computation
    suggestion = model.suggest_next_feature(df, state, candidate_cols=candidate_cols)
    return suggestion.col if suggestion is not None else None


@dataclass
class KeyEvalResult:
    """Result of evaluating a key-finding method.

    Attributes:
        method: Name of the method
        rows: The candidate row set
        key: Columns selected by the method
        key_cost: Total cost of the key
        pair_coverage: Fraction of pairs distinguished
        runtime_sec: Time taken to compute the key
        gold_key: Optimal key if computed
        gold_cost: Cost of optimal key
        optimality_gap: key_cost - gold_cost
    """

    method: str
    rows: tuple[RowIndex, ...]
    key: list[ColName]
    key_cost: float
    pair_coverage: float
    runtime_sec: float
    gold_key: list[ColName] | None = None
    gold_cost: float | None = None
    optimality_gap: float | None = None


def evaluate_keys(
    df: pd.DataFrame,
    candidate_sets: Sequence[Sequence[RowIndex]],
    methods: Mapping[
        str, Callable[[pd.DataFrame, Sequence[RowIndex]], Sequence[ColName]]
    ],
    *,
    costs: Mapping[ColName, float] | None = None,
    # Part of the documented API. Results report the exact pair coverage each
    # method achieved; the tolerance is for the methods under evaluation (and
    # any gold solver) to apply, not for the scoring here to absorb.
    epsilon_pairs: float = 0.0,  # noqa: ARG001
    gold_solver: Callable[..., Sequence[ColName]] | None = None,
) -> list[KeyEvalResult]:
    """Evaluate multiple key-finding methods on candidate sets.

    Args:
        df: The data table
        candidate_sets: Test cases (row subsets)
        methods: Methods to evaluate (name -> function)
        costs: Column costs
        epsilon_pairs: Coverage tolerance
        gold_solver: Function to compute optimal solution

    Returns:
        Evaluation results for each method and candidate set
    """
    results = []
    costs = costs or {}

    for candidate_rows in candidate_sets:
        rows_tuple = tuple(candidate_rows)

        # Compute gold standard if solver provided
        gold_key = None
        gold_cost = None
        if gold_solver:
            try:
                gold_key = list(gold_solver(df, candidate_rows))
                gold_cost = sum(costs.get(c, 1.0) for c in gold_key)
            except Exception:
                # A failed gold solve leaves gold_key None and the row is
                # scored without a baseline, rather than aborting the sweep.
                logger.debug("gold solver failed for %r", rows_tuple, exc_info=True)

        # Evaluate each method
        for method_name, method_func in methods.items():
            start_time = time.time()
            try:
                key = list(method_func(df, candidate_rows))
                runtime = time.time() - start_time

                # Compute metrics
                key_cost = sum(costs.get(c, 1.0) for c in key)
                problem = KeyProblem(df, candidate_rows, costs=costs)
                coverage = problem.pairwise_coverage(key)

                # Compute optimality gap if gold available
                gap = None
                if gold_cost is not None:
                    gap = key_cost - gold_cost

                result = KeyEvalResult(
                    method=method_name,
                    rows=rows_tuple,
                    key=key,
                    key_cost=key_cost,
                    pair_coverage=coverage,
                    runtime_sec=runtime,
                    gold_key=gold_key,
                    gold_cost=gold_cost,
                    optimality_gap=gap,
                )
                results.append(result)

            except Exception:
                # Record failure
                result = KeyEvalResult(
                    method=method_name,
                    rows=rows_tuple,
                    key=[],
                    key_cost=float("inf"),
                    pair_coverage=0.0,
                    runtime_sec=time.time() - start_time,
                    gold_key=gold_key,
                    gold_cost=gold_cost,
                    optimality_gap=None,
                )
                results.append(result)

    return results


@dataclass
class PolicyEvalStats:
    """Statistics for a policy's performance.

    Attributes:
        name: Policy name
        mean_steps: Average number of steps to termination
        mean_cost: Average total cost
        mean_final_entropy: Average final entropy
        mean_final_pair_coverage: Average final pairwise coverage
        std_steps: Standard deviation of steps
        std_cost: Standard deviation of cost
        success_rate: Fraction of runs that achieved uniqueness
    """

    name: str
    mean_steps: float
    mean_cost: float
    mean_final_entropy: float
    mean_final_pair_coverage: float
    std_steps: float = 0.0
    std_cost: float = 0.0
    success_rate: float = 0.0


def evaluate_policies(
    df: pd.DataFrame,
    candidate_sets: Sequence[Sequence[RowIndex]],
    policies: Mapping[str, Policy],
    *,
    feature_costs: Mapping[ColName, float] | None = None,
    stop: StopRules | None = None,
    n_repeats: int = 1,
) -> list[PolicyEvalStats]:
    """Evaluate disambiguation policies on candidate sets.

    Args:
        df: The data table
        candidate_sets: Test cases (row subsets)
        policies: Policies to evaluate (name -> policy)
        feature_costs: Cost of each feature
        stop: Stopping criteria (default: target_unique=True)
        n_repeats: Number of times to run each policy per candidate set

    Returns:
        Performance statistics for each policy
    """
    if stop is None:
        stop = StopRules(target_unique=True)

    results_by_policy = {name: [] for name in policies}

    for candidate_rows in candidate_sets:
        if len(candidate_rows) <= 1:
            continue

        for _ in range(n_repeats):
            # Pick a random true row for simulation
            true_row = random.choice(candidate_rows)

            for policy_name, policy in policies.items():
                # Create session
                session = DisambiguationSession(
                    df=df,
                    candidate_rows=candidate_rows,
                    policy=policy,
                    feature_costs=feature_costs,
                )

                # Run session
                steps = session.run(stop, true_row=true_row)

                # Collect metrics
                final_state = session.state
                n_steps = len(steps)
                total_cost = session.cumulative_cost
                final_entropy = final_state.entropy
                is_unique = final_state.is_unique

                # Compute final pair coverage
                from .keys import pairwise_coverage

                if len(final_state.candidate_rows) > 1:
                    final_coverage = pairwise_coverage(
                        df, final_state.candidate_rows, list(final_state.observed_cols)
                    )
                else:
                    final_coverage = 1.0

                results_by_policy[policy_name].append(
                    {
                        "steps": n_steps,
                        "cost": total_cost,
                        "entropy": final_entropy,
                        "coverage": final_coverage,
                        "success": is_unique,
                    }
                )

    # Aggregate statistics
    stats = []
    for policy_name, results in results_by_policy.items():
        if not results:
            continue

        steps_vals = [r["steps"] for r in results]
        cost_vals = [r["cost"] for r in results]
        entropy_vals = [r["entropy"] for r in results]
        coverage_vals = [r["coverage"] for r in results]
        success_vals = [r["success"] for r in results]

        stat = PolicyEvalStats(
            name=policy_name,
            mean_steps=float(np.mean(steps_vals)),
            mean_cost=float(np.mean(cost_vals)),
            mean_final_entropy=float(np.mean(entropy_vals)),
            mean_final_pair_coverage=float(np.mean(coverage_vals)),
            std_steps=float(np.std(steps_vals)),
            std_cost=float(np.std(cost_vals)),
            success_rate=float(np.mean(success_vals)),
        )
        stats.append(stat)

    return stats


@dataclass
class AcquisitionResult:
    """Result of a single feature acquisition simulation.

    Attributes:
        subset_size: Size of the candidate set
        steps_used: Number of queries made
        unique_identified: Whether unique row was found
        optimal_steps: Size of minimal key if computed
        cols_used: Sequence of columns queried
    """

    subset_size: int
    steps_used: int
    unique_identified: bool
    optimal_steps: int | None = None
    cols_used: list[ColName] | None = None


def benchmark_policy(
    df: pd.DataFrame,
    policy: Policy,
    subset_sizes: Sequence[int],
    n_samples: int,
    *,
    compute_optimal: bool = True,
    max_cols_for_exact: int = 10,
    feature_costs: Mapping[ColName, float] | None = None,
    random_state: int | None = None,
) -> dict[int, list[AcquisitionResult]]:
    """Benchmark a policy across different subset sizes.

    Args:
        df: The data table
        policy: Policy to benchmark
        subset_sizes: Different candidate set sizes to test
        n_samples: Number of samples per size
        compute_optimal: Whether to compute optimal key size
        max_cols_for_exact: Max columns for exact solution
        feature_costs: Column costs
        random_state: Random seed

    Returns:
        Results keyed by subset size
    """
    if random_state is not None:
        random.seed(random_state)

    results = {}

    for size in subset_sizes:
        size_results = []

        # Sample candidate sets
        # No random_state: both functions seed the global `random` module, and
        # this one already seeded it once above. Passing it here would re-seed
        # on every iteration, so each subset size would draw the same samples.
        candidate_sets = sample_candidate_sets(  # preen: allow-dropped-arg
            df, subset_size=size, n_samples=n_samples
        )

        for candidate_rows in candidate_sets:
            # Create session
            session = DisambiguationSession(
                df=df,
                candidate_rows=candidate_rows,
                policy=policy,
                feature_costs=feature_costs,
            )

            # Run until unique
            stop = StopRules(target_unique=True)
            true_row = random.choice(candidate_rows)
            steps = session.run(stop, true_row=true_row)

            # Compute optimal if requested and feasible
            optimal_steps = None
            if compute_optimal and len(df.columns) <= max_cols_for_exact:
                try:
                    optimal_key = compute_gold_key(
                        df, candidate_rows, costs=feature_costs, time_limit=1.0
                    )
                    optimal_steps = len(optimal_key)
                except Exception:
                    # Optimal is a nice-to-have comparison; leave it None.
                    logger.debug(
                        "optimal key computation failed for subset size %d",
                        size,
                        exc_info=True,
                    )

            result = AcquisitionResult(
                subset_size=size,
                steps_used=len(steps),
                unique_identified=session.state.is_unique,
                optimal_steps=optimal_steps,
                cols_used=[step.col for step in steps],
            )
            size_results.append(result)

        results[size] = size_results

    return results
