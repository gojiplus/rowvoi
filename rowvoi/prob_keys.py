"""Probabilistic key and path algorithms.

This module handles the probabilistic case where column values are unknown
and we work with a model to predict expected information gain and coverage.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .core import CandidateState, ColName, RowIndex
from .keys import KeyPath, KeyPathStep
from .policies import MIPolicy
from .session import DisambiguationSession, StopRules

if TYPE_CHECKING:
    from .ml import RowVoiModel


def find_key_probabilistic(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    model: "RowVoiModel",
    *,
    epsilon_posterior: float = 0.05,
    columns: Sequence[ColName] | None = None,
    costs: Mapping[ColName, float] | None = None,
    objective: Literal["mi", "mi_over_cost"] = "mi_over_cost",
    max_steps: int | None = None,
) -> list[ColName]:
    """Find a probabilistic min-key under a model.

    Runs a greedy MI/VOI policy (non-adaptively) until
    max_r p(r | E) >= 1 - epsilon_posterior, then returns
    the set of columns used.

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    rows : Sequence[RowIndex]
        Row indices to distinguish
    model : RowVoiModel
        Trained model for computing expected information
    epsilon_posterior : float, default 0.05
        Target residual uncertainty
    columns : Sequence[ColName], optional
        Columns to consider
    costs : Mapping[ColName, float], optional
        Cost of each column
    objective : str, default "mi_over_cost"
        Objective for column selection
    max_steps : int, optional
        Maximum number of columns to select

    Returns
    -------
    list[ColName]
        Columns selected by the greedy policy
    """
    # Create policy
    policy = MIPolicy(model=model, objective=objective, feature_costs=costs)

    # Create session
    session = DisambiguationSession(
        df=df, candidate_rows=rows, policy=policy, feature_costs=costs
    )

    # Set up stop rules
    stop = StopRules(
        epsilon_posterior=epsilon_posterior,
        max_steps=max_steps,
        target_unique=False,  # We're using epsilon_posterior instead
    )

    # Handle empty rows case
    if not rows:
        return []

    # Run session (pick arbitrary true row for simulation)
    true_row = rows[0]
    steps = session.run(stop, candidate_cols=columns, true_row=true_row)

    # Extract columns used
    return [step.col for step in steps]


def plan_key_path_probabilistic(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    model: "RowVoiModel",
    *,
    objective: Literal[
        "mi", "mi_over_cost", "expected_entropy_reduction"
    ] = "mi_over_cost",
    columns: Sequence[ColName] | None = None,
    costs: Mapping[ColName, float] | None = None,
) -> KeyPath:
    """Build an expected greedy path under the model.

    For each step, compute MI or expected entropy reduction for each column
    given the current posterior (without actually observing values), pick
    the best, and iterate. Returns a KeyPath with expected coverage/cost.

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    rows : Sequence[RowIndex]
        Row indices to distinguish
    model : RowVoiModel
        Trained model for computing expected information
    objective : str, default "mi_over_cost"
        Objective for ordering columns
    columns : Sequence[ColName], optional
        Columns to consider
    costs : Mapping[ColName, float], optional
        Cost of each column

    Returns
    -------
    KeyPath
        Expected path with coverage information
    """
    if not rows:
        return KeyPath(steps=[])

    # Initialize state
    state = CandidateState.uniform(rows)
    selected_cols = []
    steps = []
    cumulative_cost = 0.0

    # Get candidate columns
    if columns is None:
        columns = list(df.columns)
    else:
        columns = list(columns)

    # Compute total pairs for coverage tracking
    n = len(rows)
    total_pairs = n * (n - 1) // 2 if n > 1 else 0

    # Build path greedily
    while len(selected_cols) < len(columns) and not state.is_unique:
        # Find best column among remaining
        best_col = None
        best_score = -float("inf")
        best_mi = 0.0

        for col in columns:
            if col in selected_cols:
                continue

            # Get expected MI from model
            suggestion = model.suggest_next_feature(df, state, candidate_cols=[col])

            if objective == "mi":
                score = (
                    suggestion.expected_voi
                    if suggestion.expected_voi
                    else suggestion.score
                )
            elif objective == "mi_over_cost":
                cost = costs.get(col, 1.0) if costs else 1.0
                voi = (
                    suggestion.expected_voi
                    if suggestion.expected_voi
                    else suggestion.score
                )
                score = voi / cost if cost > 0 else voi
            else:  # expected_entropy_reduction
                score = (
                    state.entropy
                    - suggestion.debug.get("expected_posterior_entropy", state.entropy)
                    if suggestion.debug
                    else suggestion.score
                )

            if score > best_score:
                best_col = col
                best_score = score
                best_mi = (
                    suggestion.expected_voi
                    if suggestion.expected_voi
                    else suggestion.score
                )

        if best_col is None:
            break

        # Add to path
        selected_cols.append(best_col)
        cost = costs.get(best_col, 1.0) if costs else 1.0
        cumulative_cost += cost

        # Estimate coverage (simplified - actual coverage depends on values)
        # For now, use a heuristic based on MI
        estimated_newly_covered = (
            int(best_mi * total_pairs / state.entropy) if state.entropy > 0 else 0
        )
        estimated_cumulative = min(
            len(selected_cols) * total_pairs // len(columns), total_pairs
        )

        step = KeyPathStep(
            col=best_col,
            newly_covered_pairs=estimated_newly_covered,
            cumulative_covered_pairs=estimated_cumulative,
            total_pairs=total_pairs,
            marginal_cost=cost,
            cumulative_cost=cumulative_cost,
        )
        steps.append(step)

        # Update state (simulate average case)
        # This is a simplification - in reality we'd need to average
        # For now, just reduce candidate set proportionally
        reduction_factor = (
            max(0.5, 1.0 - best_mi / state.entropy) if state.entropy > 0 else 0.5
        )
        new_n_candidates = max(1, int(len(state.candidate_rows) * reduction_factor))
        if new_n_candidates < len(state.candidate_rows):
            state = CandidateState.uniform(
                state.candidate_rows[:new_n_candidates],
                observed_cols=set(selected_cols),
            )

    return KeyPath(steps=steps)


def estimate_coverage_probability(
    df: pd.DataFrame,
    rows: Sequence[RowIndex],
    cols: Sequence[ColName],
    model: "RowVoiModel | None" = None,
) -> float:
    """Estimate the probability that cols will distinguish rows.

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    rows : Sequence[RowIndex]
        Row indices to consider
    cols : Sequence[ColName]
        Columns to evaluate
    model : RowVoiModel, optional
        Model for probabilistic estimation.
        If None, uses deterministic coverage.

    Returns
    -------
    float
        Estimated probability of full disambiguation
    """
    if model is None:
        # Deterministic case
        from .keys import pairwise_coverage

        coverage = pairwise_coverage(df, rows, cols)
        return 1.0 if coverage >= 1.0 else 0.0

    # Probabilistic case using model
    # Create initial state
    state = CandidateState.uniform(rows)

    # Simulate observing each column
    for col in cols:
        if len(state.candidate_rows) <= 1:
            return 1.0

        # Get expected reduction from model
        suggestion = model.suggest_next_feature(df, state, candidate_cols=[col])

        # Estimate reduction in candidates
        # This is a rough approximation
        if state.entropy > 0 and suggestion.expected_voi is not None:
            reduction_factor = max(0.3, 1.0 - suggestion.expected_voi / state.entropy)
        else:
            reduction_factor = 0.5

        new_n_candidates = max(1, int(len(state.candidate_rows) * reduction_factor))
        state = CandidateState.uniform(
            state.candidate_rows[:new_n_candidates],
            observed_cols=state.observed_cols | {col},
        )

    # Return probability based on final state
    if len(state.candidate_rows) <= 1:
        return 1.0
    else:
        # Estimate based on how much we've reduced the candidate set
        return (
            1.0 - (len(state.candidate_rows) - 1) / (len(rows) - 1)
            if len(rows) > 1
            else 1.0
        )
