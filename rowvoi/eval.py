"""Simulation and evaluation tools for rowvoi.

This module provides comprehensive evaluation tools for comparing different
key-finding algorithms and policies, including gold standard computation
and systematic benchmarking.
"""

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

if TYPE_CHECKING:
    from .ml import RowVoiModel


def sample_candidate_sets(
    df: pd.DataFrame,
    *,
    subset_size: int,
    n_samples: int,
    random_state: int | None = None,
) -> list[list[RowIndex]]:
    """Randomly sample subsets of rows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to sample from
    subset_size : int
        Number of rows in each subset
    n_samples : int
        Number of subsets to generate
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    list[list[RowIndex]]
        List of row index lists
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
) -> list[ColName]:
    """Compute the optimal deterministic key using exact or ILP solver.

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
    epsilon_pairs : float, default 0.0
        Tolerance for unresolved pairs
    time_limit : float, default 10.0
        Maximum time for exact solution

    Returns
    -------
    list[ColName]
        Optimal key columns
    """
    # Try ILP first (if available), then exact, then fallback to greedy
    strategies = ["ilp", "exact", "greedy"]

    for strategy in strategies:
        try:
            return find_key(
                df,
                rows,
                columns=columns,
                costs=costs,
                strategy=strategy,
                epsilon_pairs=epsilon_pairs,
                time_limit=time_limit if strategy != "greedy" else None,
            )
        except Exception:
            continue

    # Final fallback
    return find_key(
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
    objective: Literal["mi", "expected_entropy_reduction"] = "mi",
) -> ColName:
    """Compute the 'gold standard' next column under a model.

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    state : CandidateState
        Current state
    model : RowVoiModel
        Trained model
    candidate_cols : Sequence[ColName], optional
        Columns to consider
    objective : str, default "mi"
        Objective function

    Returns
    -------
    ColName
        Optimal next column
    """
    # For now, use the model's suggestion as gold standard
    # A more sophisticated implementation could do exact computation
    suggestion = model.suggest_next_feature(
        df, state, candidate_features=candidate_cols
    )
    return suggestion.col


@dataclass
class KeyEvalResult:
    """Result of evaluating a key-finding method.

    Attributes
    ----------
    method : str
        Name of the method
    rows : tuple[RowIndex, ...]
        The candidate row set
    key : list[ColName]
        Columns selected by the method
    key_cost : float
        Total cost of the key
    pair_coverage : float
        Fraction of pairs distinguished
    runtime_sec : float
        Time taken to compute the key
    gold_key : list[ColName], optional
        Optimal key if computed
    gold_cost : float, optional
        Cost of optimal key
    optimality_gap : float, optional
        key_cost - gold_cost
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
    epsilon_pairs: float = 0.0,
    gold_solver: Callable[..., Sequence[ColName]] | None = None,
) -> list[KeyEvalResult]:
    """Evaluate multiple key-finding methods on candidate sets.

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    candidate_sets : Sequence[Sequence[RowIndex]]
        Test cases (row subsets)
    methods : Mapping[str, Callable]
        Methods to evaluate (name -> function)
    costs : Mapping[ColName, float], optional
        Column costs
    epsilon_pairs : float, default 0.0
        Coverage tolerance
    gold_solver : Callable, optional
        Function to compute optimal solution

    Returns
    -------
    list[KeyEvalResult]
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
                pass

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

    Attributes
    ----------
    name : str
        Policy name
    mean_steps : float
        Average number of steps to termination
    mean_cost : float
        Average total cost
    mean_final_entropy : float
        Average final entropy
    mean_final_pair_coverage : float
        Average final pairwise coverage
    std_steps : float
        Standard deviation of steps
    std_cost : float
        Standard deviation of cost
    success_rate : float
        Fraction of runs that achieved uniqueness
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

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    candidate_sets : Sequence[Sequence[RowIndex]]
        Test cases (row subsets)
    policies : Mapping[str, Policy]
        Policies to evaluate (name -> policy)
    feature_costs : Mapping[ColName, float], optional
        Cost of each feature
    stop : StopRules, optional
        Stopping criteria (default: target_unique=True)
    n_repeats : int, default 1
        Number of times to run each policy per candidate set

    Returns
    -------
    list[PolicyEvalStats]
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
            mean_steps=np.mean(steps_vals),
            mean_cost=np.mean(cost_vals),
            mean_final_entropy=np.mean(entropy_vals),
            mean_final_pair_coverage=np.mean(coverage_vals),
            std_steps=np.std(steps_vals),
            std_cost=np.std(cost_vals),
            success_rate=np.mean(success_vals),
        )
        stats.append(stat)

    return stats


@dataclass
class AcquisitionResult:
    """Result of a single feature acquisition simulation.

    Attributes
    ----------
    subset_size : int
        Size of the candidate set
    steps_used : int
        Number of queries made
    unique_identified : bool
        Whether unique row was found
    optimal_steps : int, optional
        Size of minimal key if computed
    cols_used : list[ColName]
        Sequence of columns queried
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

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    policy : Policy
        Policy to benchmark
    subset_sizes : Sequence[int]
        Different candidate set sizes to test
    n_samples : int
        Number of samples per size
    compute_optimal : bool, default True
        Whether to compute optimal key size
    max_cols_for_exact : int, default 10
        Max columns for exact solution
    feature_costs : Mapping[ColName, float], optional
        Column costs
    random_state : int, optional
        Random seed

    Returns
    -------
    dict[int, list[AcquisitionResult]]
        Results keyed by subset size
    """
    if random_state is not None:
        random.seed(random_state)

    results = {}

    for size in subset_sizes:
        size_results = []

        # Sample candidate sets
        candidate_sets = sample_candidate_sets(
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
                    pass

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
