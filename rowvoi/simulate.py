"""Simulation and benchmarking utilities for rowvoi.

This module contains helper functions to generate random candidate sets
and benchmark different feature acquisition policies.  These
utilities are intended to aid in testing and evaluating the
performance of :mod:`rowvoi` algorithms on real or synthetic
datasets.

The primary functions are:

* :func:`sample_candidate_sets` – sample random subsets of row
  indices for evaluation.
* :func:`benchmark_policy` – run repeated simulations of a
  feature acquisition policy and compare against the optimal
  minimal key size when feasible.

``benchmark_policy`` returns a dictionary of results keyed by
subset size.  Each entry is a list of :class:`AcquisitionResult`
dataclasses containing detailed information about each run.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .logical import minimal_key_exact
from .ml import RowVoiModel
from .types import CandidateState, ColName, RowIndex


def sample_candidate_sets(
    n_rows: int,
    subset_size: int,
    n_samples: int,
    rng: random.Random | None = None,
) -> list[list[RowIndex]]:
    """Sample random subsets of row indices without replacement.

    This helper returns ``n_samples`` distinct candidate sets, each
    consisting of ``subset_size`` unique row indices drawn from
    ``0, 1, ..., n_rows - 1``.  Sampling is performed without
    replacement within each subset (but subsets may overlap across
    samples).  For reproducibility a ``random.Random`` instance can
    be provided; otherwise Python's default randomness is used.

    Parameters
    ----------
    n_rows : int
        Total number of rows in the dataset.
    subset_size : int
        Number of rows in each candidate set.
    n_samples : int
        Number of candidate sets to generate.
    rng : random.Random, optional
        Source of randomness.  If ``None``, the default RNG is used.

    Returns
    -------
    List[List[int]]
        A list of lists of row indices.
    """
    if rng is None:
        rng = random.Random()
    sets: list[list[RowIndex]] = []
    for _ in range(n_samples):
        if subset_size > n_rows:
            raise ValueError(
                f"subset_size ({subset_size}) cannot exceed number of rows ({n_rows})"
            )
        subset = rng.sample(range(n_rows), subset_size)
        sets.append(sorted(subset))
    return sets


@dataclass
class AcquisitionResult:
    """Result of a single feature acquisition simulation.

    Attributes
    ----------
    subset_size : int
        Size of the candidate set at the start of the simulation.
    steps_used : int
        Number of queries made by the policy.
    unique_identified : bool
        Whether the simulation terminated with a single candidate
        remaining (i.e., the true row was uniquely identified).
    optimal_steps : Optional[int]
        The size of a minimal key for the candidate set if computed.
        ``None`` if the optimal key could not be computed (e.g., too
        many columns to perform exact search).
    cols_used : List[ColName]
        The sequence of columns queried by the policy.
    """

    subset_size: int
    steps_used: int
    unique_identified: bool
    optimal_steps: int | None
    cols_used: list[ColName]


def benchmark_policy(
    df: pd.DataFrame,
    model: RowVoiModel,
    subset_sizes: Sequence[int],
    *,
    n_samples: int = 50,
    max_exact_cols: int = 12,
    objective: str = "mi",
    feature_costs: dict[ColName, float] | None = None,
    rng: random.Random | None = None,
) -> dict[int, list[AcquisitionResult]]:
    """Evaluate a feature acquisition policy across multiple candidate sizes.

    For each subset size ``k`` in ``subset_sizes``, this function
    performs ``n_samples`` simulations.  For each simulation, it
    randomly selects ``k`` candidate rows, draws a true row uniformly
    at random from that set, computes the greedy sequence of feature
    queries using the model's :meth:`~RowVoiModel.run_acquisition`,
    and records the number of queries required and whether the
    disambiguation succeeded.  When feasible, it also computes the
    exact minimal key size using :func:`rowvoi.logical.minimal_key_exact`
    and records it.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset used both for computing keys and simulating
        feature values.
    model : RowVoiModel
        A fitted model used to suggest features.  See
        :meth:`~RowVoiModel.fit`.
    subset_sizes : Sequence[int]
        A sequence of candidate set sizes to test.
    n_samples : int, optional
        Number of simulations per subset size.  Default is 50.
    max_exact_cols : int, optional
        Maximum number of columns allowed in the brute force search
        for computing the minimal key.  If the number of columns in
        the DataFrame exceeds this threshold, exact minimal key
        computation is skipped and ``optimal_steps`` is set to
        ``None``.  Default is 12.
    objective : str, optional
        Objective passed through to
        :meth:`RowVoiModel.run_acquisition`.  Default is ``'mi'``.
    feature_costs : Dict[ColName, float], optional
        Feature costs used when ``objective='mi_over_cost'``.
    rng : random.Random, optional
        Source of randomness.  If ``None``, Python's default RNG is
        used.

    Returns
    -------
    Dict[int, List[AcquisitionResult]]
        A dictionary keyed by subset size, where each value is a list
        of :class:`AcquisitionResult` objects describing each run.
    """
    if rng is None:
        rng = random.Random()
    results: dict[int, list[AcquisitionResult]] = {}
    n_rows = len(df)
    for k in subset_sizes:
        runs: list[AcquisitionResult] = []
        # Sample candidate sets
        candidate_sets = sample_candidate_sets(n_rows, k, n_samples, rng=rng)
        for rows in candidate_sets:
            # Compute optimal minimal key size if possible
            optimal_steps: int | None = None
            if len(df.columns) <= max_exact_cols:
                try:
                    optimal_cols = minimal_key_exact(df, rows)
                    optimal_steps = len(optimal_cols)
                except Exception:
                    optimal_steps = None
            # Prepare initial state: uniform prior over candidates, no evidence yet
            posterior = dict.fromkeys(rows, 1.0 / k)
            state = CandidateState(
                candidate_rows=list(rows),
                posterior=posterior,
                observed_cols=[],
                observed_values={},
            )
            # Pick true row uniformly at random from candidates
            true_row = rng.choice(rows)
            # Run acquisition using model
            history = model.run_acquisition(
                df,
                true_row,
                state,
                objective=objective,
                feature_costs=feature_costs,
            )
            steps_used = len(history)
            # Determine whether unique identification succeeded
            # After run_acquisition, state is not updated (passed state unmodified), but
            # we can check the last suggestion's entropy_after to infer uniqueness.
            # Alternatively recompute candidate set by applying history to initial.
            # For simplicity, treat success as steps_used > 0 and entropy_after of last
            # suggestion equals 0.
            unique_identified = False
            if history:
                last = history[-1]
                # if final posterior entropy is ~0, we assume unique
                if last.details.get("entropy_after", 0.0) < 1e-12:
                    unique_identified = True
            else:
                # If no features needed, either we had one candidate or failed.
                unique_identified = k <= 1
            runs.append(
                AcquisitionResult(
                    subset_size=k,
                    steps_used=steps_used,
                    unique_identified=unique_identified,
                    optimal_steps=optimal_steps,
                    cols_used=[s.col for s in history],
                )
            )
        results[k] = runs
    return results
