"""Core types and data structures for rowvoi.

This module defines the fundamental building blocks used throughout the rowvoi package:
- CandidateState: Represents current uncertainty over which row is "the one"
- FeatureSuggestion: A recommendation for which column to query next
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Type aliases
RowIndex = Hashable
ColName = Hashable


@dataclass
class CandidateState:
    """Represents the current uncertainty over which row is "the one".

    Attributes
    ----------
    candidate_rows : Sequence[RowIndex]
        List of row indices under consideration
    posterior : np.ndarray
        Probabilities over candidate_rows, shape (n_candidates,)
        Use uniform if deterministic / no model
    observed_cols : set[ColName]
        Set of columns that have been queried
    observed_values : Mapping[ColName, Any]
        Mapping col -> observed value (may be empty in planning mode)
    """

    candidate_rows: Sequence[RowIndex]
    posterior: np.ndarray
    observed_cols: set[ColName]
    observed_values: Mapping[ColName, Any]

    def __post_init__(self):
        """Validate state consistency."""
        if len(self.candidate_rows) != len(self.posterior):
            raise ValueError(
                f"Length mismatch: {len(self.candidate_rows)} candidates "
                f"but posterior has shape {self.posterior.shape}"
            )
        # Allow empty states (sum=0) or properly normalized states (sum=1)
        if len(self.posterior) > 0 and not np.allclose(self.posterior.sum(), 1.0):
            raise ValueError(f"Posterior must sum to 1, got {self.posterior.sum()}")

    @property
    def entropy(self) -> float:
        """Shannon entropy H(posterior) in bits."""
        # Avoid log(0) by filtering out zero probabilities
        nonzero = self.posterior[self.posterior > 0]
        if len(nonzero) == 0:
            return 0.0
        return -np.sum(nonzero * np.log2(nonzero))

    @property
    def max_posterior(self) -> float:
        """max_r p(r | E)."""
        return np.max(self.posterior) if len(self.posterior) > 0 else 0.0

    @property
    def residual_uncertainty(self) -> float:
        """1 - max_posterior."""
        return 1.0 - self.max_posterior

    @property
    def is_unique(self) -> bool:
        """True if there is a single candidate with posterior ~1."""
        return len(self.candidate_rows) == 1 or (
            len(self.candidate_rows) > 1 and self.max_posterior > 0.99999
        )

    @property
    def unique_row(self) -> RowIndex | None:
        """Return the most probable row if unique, else None."""
        if self.is_unique and len(self.candidate_rows) > 0:
            idx = np.argmax(self.posterior)
            return self.candidate_rows[idx]
        return None

    @classmethod
    def uniform(
        cls,
        candidate_rows: Sequence[RowIndex],
        observed_cols: set[ColName] | None = None,
        observed_values: Mapping[ColName, Any] | None = None,
    ) -> "CandidateState":
        """Create a state with uniform posterior over candidates.

        Parameters
        ----------
        candidate_rows : Sequence[RowIndex]
            The candidate row indices
        observed_cols : set[ColName], optional
            Already observed columns
        observed_values : Mapping[ColName, Any], optional
            Values of observed columns

        Returns
        -------
        CandidateState
            State with uniform posterior distribution
        """
        n = len(candidate_rows)
        posterior = np.ones(n) / n if n > 0 else np.array([])
        return cls(
            candidate_rows=candidate_rows,
            posterior=posterior,
            observed_cols=observed_cols or set(),
            observed_values=observed_values or {},
        )

    def filter_candidates(
        self, df: pd.DataFrame, col: ColName, value: Any
    ) -> "CandidateState":
        """Filter candidates to those matching the observed value.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame containing candidate rows
        col : ColName
            The column that was observed
        value : Any
            The observed value

        Returns
        -------
        CandidateState
            New state with filtered candidates and renormalized posterior
        """
        # Find which candidates match the observed value
        matching_mask = []
        new_candidates = []
        new_posterior_values = []

        for i, row_idx in enumerate(self.candidate_rows):
            if df.iloc[row_idx][col] == value:
                matching_mask.append(True)
                new_candidates.append(row_idx)
                new_posterior_values.append(self.posterior[i])
            else:
                matching_mask.append(False)

        # Renormalize posterior
        if new_posterior_values:
            new_posterior = np.array(new_posterior_values)
            new_posterior /= new_posterior.sum()
        else:
            new_posterior = np.array([])

        # Update observed columns and values
        new_observed_cols = self.observed_cols | {col}
        new_observed_values = dict(self.observed_values)
        new_observed_values[col] = value

        return CandidateState(
            candidate_rows=new_candidates,
            posterior=new_posterior,
            observed_cols=new_observed_cols,
            observed_values=new_observed_values,
        )


@dataclass
class FeatureSuggestion:
    """A recommendation of which column to query next.

    Attributes
    ----------
    col : ColName
        The column name suggested to query next
    score : float
        Raw score used to rank columns (e.g., MI, coverage gain)
    expected_voi : float, optional
        Expected value of information in bits
    marginal_cost : float, optional
        Cost of querying this column
    debug : dict[str, Any], optional
        Additional debugging information
    """

    col: ColName
    score: float
    expected_voi: float | None = None
    marginal_cost: float | None = None
    debug: dict[str, Any] | None = None

    @property
    def cost_adjusted_score(self) -> float:
        """Score divided by cost (if cost is available)."""
        if self.marginal_cost is not None and self.marginal_cost > 0:
            return self.score / self.marginal_cost
        return self.score
