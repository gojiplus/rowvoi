"""Interactive disambiguation sessions.

This module provides tools for running interactive disambiguation sessions
where columns are queried sequentially based on a policy until stopping
criteria are met.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .core import CandidateState, ColName, FeatureSuggestion, RowIndex
from .keys import pairwise_coverage
from .policies import Policy


@dataclass
class StopRules:
    """Conditions for stopping a disambiguation session.

    Attributes
    ----------
    max_steps : int, optional
        Maximum number of columns to query
    cost_budget : float, optional
        Maximum total cost to spend
    epsilon_posterior : float, optional
        Stop when residual_uncertainty <= epsilon
    epsilon_pairs : float, optional
        Stop when unresolved pair fraction <= epsilon
    target_unique : bool, default True
        Stop when state.is_unique is True
    """

    max_steps: int | None = None
    cost_budget: float | None = None
    epsilon_posterior: float | None = None
    epsilon_pairs: float | None = None
    target_unique: bool = True

    def should_stop(
        self,
        state: CandidateState,
        steps: int,
        total_cost: float,
        df: pd.DataFrame | None = None,
    ) -> tuple[bool, str]:
        """Check if any stopping condition is met.

        Parameters
        ----------
        state : CandidateState
            Current state
        steps : int
            Number of steps taken so far
        total_cost : float
            Total cost incurred so far
        df : pd.DataFrame, optional
            Data frame (needed for epsilon_pairs check)

        Returns
        -------
        tuple[bool, str]
            (should_stop, reason) where reason explains why stopping
        """
        # Check max steps
        if self.max_steps is not None and steps >= self.max_steps:
            return True, f"Reached max steps ({self.max_steps})"

        # Check cost budget
        if self.cost_budget is not None and total_cost >= self.cost_budget:
            return True, f"Reached cost budget ({self.cost_budget})"

        # Check uniqueness
        if self.target_unique and state.is_unique:
            return True, "Found unique row"

        # Check posterior epsilon
        if self.epsilon_posterior is not None:
            if state.residual_uncertainty <= self.epsilon_posterior:
                return True, f"Residual uncertainty <= {self.epsilon_posterior}"

        # Check pairwise coverage epsilon
        if self.epsilon_pairs is not None and df is not None:
            coverage = pairwise_coverage(
                df, state.candidate_rows, list(state.observed_cols)
            )
            if coverage >= 1.0 - self.epsilon_pairs:
                return True, f"Pair coverage >= {1.0 - self.epsilon_pairs}"

        return False, ""


@dataclass
class SessionStep:
    """Record of a single step in a disambiguation session.

    Attributes
    ----------
    col : ColName
        Column that was queried
    observed_value : Any
        Value observed for the true row
    suggestion : FeatureSuggestion
        The suggestion that led to this query
    cost : float
        Cost of this query
    cumulative_cost : float
        Total cost up to this point
    entropy_before : float
        Entropy before observing this column
    entropy_after : float
        Entropy after observing this column
    pair_coverage_after : float, optional
        Pairwise coverage after this step
    """

    col: ColName
    observed_value: Any
    suggestion: FeatureSuggestion
    cost: float
    cumulative_cost: float
    entropy_before: float
    entropy_after: float
    pair_coverage_after: float | None = None


class DisambiguationSession:
    """Interactive disambiguation session manager.

    Maintains CandidateState, queries a Policy for the next column,
    and updates with observations.

    Parameters
    ----------
    df : pd.DataFrame
        The data table
    candidate_rows : Sequence[RowIndex]
        Initial candidate row indices
    prior : Mapping[RowIndex, float], optional
        Prior probabilities over candidates
    policy : Policy
        Policy for selecting next column
    feature_costs : Mapping[ColName, float], optional
        Cost of querying each column
    """

    def __init__(
        self,
        df: pd.DataFrame,
        candidate_rows: Sequence[RowIndex],
        *,
        prior: Mapping[RowIndex, float] | None = None,
        policy: Policy,
        feature_costs: Mapping[ColName, float] | None = None,
    ) -> None:
        self.df = df
        self.policy = policy
        self.feature_costs = feature_costs or {}
        self._history: list[SessionStep] = []
        self._cumulative_cost = 0.0

        # Initialize state
        if prior:
            import numpy as np

            posterior = np.array([prior.get(r, 0.0) for r in candidate_rows])
            posterior /= posterior.sum()
            self._state = CandidateState(
                candidate_rows=list(candidate_rows),
                posterior=posterior,
                observed_cols=set(),
                observed_values={},
            )
        else:
            self._state = CandidateState.uniform(candidate_rows)

    @property
    def state(self) -> CandidateState:
        """Current disambiguation state."""
        return self._state

    @property
    def history(self) -> list[SessionStep]:
        """History of all steps taken."""
        return self._history

    @property
    def cumulative_cost(self) -> float:
        """Total cost incurred so far."""
        return self._cumulative_cost

    @property
    def steps_taken(self) -> int:
        """Number of steps taken so far."""
        return len(self._history)

    def next_question(
        self,
        candidate_cols: Sequence[ColName] | None = None,
    ) -> FeatureSuggestion:
        """Ask the policy for the next best column.

        Does NOT update state yet - just returns the suggestion.

        Parameters
        ----------
        candidate_cols : Sequence[ColName], optional
            Columns to consider

        Returns
        -------
        FeatureSuggestion
            Recommendation for next column
        """
        return self.policy.suggest(self.df, self._state, candidate_cols)

    def observe(self, col: ColName, value: Any) -> SessionStep:
        """Incorporate an observation into the state.

        Parameters
        ----------
        col : ColName
            Column that was queried
        value : Any
            Observed value

        Returns
        -------
        SessionStep
            Record of this step
        """
        # Get the suggestion that led to this column (if available)
        # In real use, this would be stored from next_question()
        suggestion = FeatureSuggestion(col=col, score=0.0)

        # Record state before update
        entropy_before = self._state.entropy

        # Update state
        new_state = self._state.filter_candidates(self.df, col, value)

        # Compute metrics
        entropy_after = new_state.entropy
        cost = self.feature_costs.get(col, 1.0)
        self._cumulative_cost += cost

        # Compute pair coverage if we have candidates
        pair_coverage = None
        if len(new_state.candidate_rows) > 1:
            pair_coverage = pairwise_coverage(
                self.df, new_state.candidate_rows, list(new_state.observed_cols)
            )

        # Create step record
        step = SessionStep(
            col=col,
            observed_value=value,
            suggestion=suggestion,
            cost=cost,
            cumulative_cost=self._cumulative_cost,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            pair_coverage_after=pair_coverage,
        )

        # Update internal state
        self._state = new_state
        self._history.append(step)

        return step

    def run(
        self,
        stop: StopRules,
        *,
        candidate_cols: Sequence[ColName] | None = None,
        true_row: RowIndex | None = None,
    ) -> list[SessionStep]:
        """Run an entire session until a stop rule triggers.

        Parameters
        ----------
        stop : StopRules
            Stopping criteria
        candidate_cols : Sequence[ColName], optional
            Columns to consider
        true_row : RowIndex, optional
            The true row index (for simulation).
            If None, picks the highest posterior candidate.

        Returns
        -------
        list[SessionStep]
            Full sequence of steps taken
        """
        # Determine true row for simulation
        if true_row is None:
            # Use highest posterior candidate
            import numpy as np

            idx = np.argmax(self._state.posterior)
            true_row = self._state.candidate_rows[idx]

        # Run session
        while True:
            # Check stopping conditions
            should_stop, reason = stop.should_stop(
                self._state, self.steps_taken, self._cumulative_cost, self.df
            )
            if should_stop:
                break

            # Get next suggestion
            suggestion = self.next_question(candidate_cols)
            if suggestion.col is None:
                # No more columns to query
                break

            # Look up true value
            true_value = self.df.loc[true_row, suggestion.col]

            # Observe and update
            self.observe(suggestion.col, true_value)

        return self._history

    def reset(self, candidate_rows: Sequence[RowIndex] | None = None) -> None:
        """Reset the session to initial state.

        Parameters
        ----------
        candidate_rows : Sequence[RowIndex], optional
            New candidate rows. If None, reset to original candidates.
        """
        if candidate_rows is None:
            candidate_rows = self._state.candidate_rows

        self._state = CandidateState.uniform(candidate_rows)
        self._history = []
        self._cumulative_cost = 0.0
