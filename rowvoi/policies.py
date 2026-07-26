"""Policy abstractions for selecting the next best column.

This module defines the Policy protocol and various concrete implementations
for deciding which column to query next during disambiguation.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pandas as pd

from .core import CandidateState, ColName, FeatureSuggestion

if TYPE_CHECKING:
    from .ml import RowVoiModel


def _per_cost(gain: float, cost: float) -> float:
    """Gain per unit cost, defined at cost zero.

    A free column that buys something is infinitely good value and should be
    taken first; a free column that buys nothing is worth nothing. Dividing
    naively gives `inf` for the first and a `nan` for the second, and a `nan`
    silently loses every comparison while emitting a numpy warning.

    Args:
        gain: Information or coverage the column provides.
        cost: Cost of acquiring it. May be zero or negative.

    Returns:
        `gain / cost` when cost is positive; otherwise `inf` for a positive
        gain and 0.0 for none.
    """
    if cost > 0:
        return gain / cost
    return math.inf if gain > 0 else 0.0


class Policy(Protocol):
    """A strategy for picking the next column given the current state."""

    def suggest(
        self,
        df: pd.DataFrame,
        state: CandidateState,
        candidate_cols: Sequence[ColName] | None = None,
    ) -> FeatureSuggestion:
        """Suggest the next best column to query.

        Args:
            df: The data table
            state: Current disambiguation state
            candidate_cols: Columns to consider. If None, consider all columns

        Returns:
            Recommendation for next column to query
        """
        ...


@dataclass
class GreedyCoveragePolicy:
    """Stateless policy: choose column that maximizes pairwise separation.

    Can be used for "next best column" in deterministic mode.

    Attributes:
        costs: Cost of querying each column
        objective: - "pairs": maximize newly covered pairs
            - "entropy": maximize entropy reduction
        weighting: - "uniform": all pairs weighted equally
            - "pair_idf": weight hard-to-separate pairs more
    """

    costs: Mapping[ColName, float] | None = None
    objective: Literal["pairs", "entropy"] = "pairs"
    weighting: Literal["uniform", "pair_idf"] = "uniform"

    def suggest(
        self,
        df: pd.DataFrame,
        state: CandidateState,
        candidate_cols: Sequence[ColName] | None = None,
    ) -> FeatureSuggestion:
        """Suggest column with best coverage gain."""
        rows = state.candidate_rows
        if len(rows) <= 1:
            # No disambiguation needed
            cols = candidate_cols or list(df.columns)
            return FeatureSuggestion(col=cols[0] if len(cols) > 0 else None, score=0.0)

        # Get candidate columns (exclude already observed)
        if candidate_cols is None:
            candidate_cols = [c for c in df.columns if c not in state.observed_cols]
        else:
            candidate_cols = [c for c in candidate_cols if c not in state.observed_cols]

        if not candidate_cols:
            return FeatureSuggestion(col=None, score=0.0)

        # Build pair coverage info
        pairs = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                pairs.append((rows[i], rows[j]))

        # Compute pair weights
        pair_weights = {}
        if self.weighting == "pair_idf":
            for pair in pairs:
                row_i, row_j = pair
                covering_cols = 0
                for col in candidate_cols:
                    if df.iloc[row_i][col] != df.iloc[row_j][col]:
                        covering_cols += 1
                if covering_cols > 0:
                    pair_weights[pair] = 1.0 / covering_cols
                else:
                    pair_weights[pair] = 1.0
        else:
            pair_weights = dict.fromkeys(pairs, 1.0)

        best_col = None
        best_score = -float("inf")
        best_debug = {}

        for col in candidate_cols:
            # Count newly covered pairs
            newly_covered = []
            for row_i, row_j in pairs:
                if df.iloc[row_i][col] != df.iloc[row_j][col]:
                    newly_covered.append((row_i, row_j))

            if not newly_covered and self.objective == "pairs":
                continue

            cost = self.costs.get(col, 1.0) if self.costs else 1.0

            if self.objective == "pairs":
                weighted_gain = sum(pair_weights.get(p, 1.0) for p in newly_covered)
                score = _per_cost(weighted_gain, cost)
            else:  # entropy
                # Compute entropy reduction
                # Group candidates by their value in this column
                value_groups = {}
                for row in rows:
                    val = df.iloc[row][col]
                    if val not in value_groups:
                        value_groups[val] = []
                    value_groups[val].append(row)

                # Current entropy
                h_before = state.entropy

                # Expected entropy after observing this column
                h_after = 0.0
                for _val, group_rows in value_groups.items():
                    # Probability of observing this value
                    p_val = sum(
                        state.posterior[state.candidate_rows.index(r)]
                        for r in group_rows
                        if r in state.candidate_rows
                    )
                    if p_val > 0:
                        # Entropy within this group
                        group_size = len(group_rows)
                        h_group = math.log2(group_size) if group_size > 1 else 0.0
                        h_after += p_val * h_group

                entropy_reduction = h_before - h_after
                score = _per_cost(entropy_reduction, cost)

            if score > best_score:
                best_col = col
                best_score = score
                best_debug = {
                    "newly_covered_pairs": len(newly_covered),
                    "total_pairs": len(pairs),
                    "cost": cost,
                }

        return FeatureSuggestion(
            col=best_col,
            score=best_score,
            marginal_cost=(
                self.costs.get(best_col, 1.0) if self.costs and best_col else 1.0
            ),
            debug=best_debug,
        )


@dataclass
class MIPolicy:
    """Policy that uses mutual information from a RowVoiModel.

    Attributes:
        model: The trained model for computing MI
        objective: - "mi": raw mutual information
            - "mi_over_cost": MI divided by feature cost
        feature_costs: Cost of querying each feature
    """

    model: "RowVoiModel"
    objective: Literal["mi", "mi_over_cost"] = "mi_over_cost"
    feature_costs: Mapping[ColName, float] | None = None

    def suggest(
        self,
        df: pd.DataFrame,
        state: CandidateState,
        candidate_cols: Sequence[ColName] | None = None,
    ) -> FeatureSuggestion:
        """Use the model to suggest the next feature.

        The objective and costs are forwarded to the model, so cost affects
        *which* column is chosen. Rescaling the winner's score afterwards
        would leave the choice itself made on raw mutual information.

        Args:
            df: The data table.
            state: Current disambiguation state.
            candidate_cols: Columns to consider. If None, consider all columns.

        Returns:
            Recommendation for the next column. `col` is None when the model
            has nothing left to suggest.
        """
        objective = self.objective
        costs: dict[ColName, float] | None = None

        if objective == "mi_over_cost":
            if self.feature_costs is None:
                # The model rejects mi_over_cost without costs, and with every
                # column free the two objectives agree anyway.
                objective = "mi"
            else:
                # The model requires a cost for every column it scores, so
                # default unpriced columns rather than letting it raise.
                costs = {c: float(self.feature_costs.get(c, 1.0)) for c in df.columns}

        suggestion = self.model.suggest_next_feature(
            df,
            state,
            candidate_cols=candidate_cols,
            objective=objective,
            feature_costs=costs,
        )
        if suggestion is None:
            return FeatureSuggestion(col=None, score=0.0)
        return suggestion


@dataclass
class CandidateMIPolicy:
    """Policy using local mutual information on candidate set only.

    This policy doesn't require a trained model - it computes MI
    directly from the candidate rows.

    Attributes:
        normalize: Whether to normalize MI by maximum entropy
        costs: Cost of querying each column
    """

    normalize: bool = False
    costs: Mapping[ColName, float] | None = None

    def suggest(
        self,
        df: pd.DataFrame,
        state: CandidateState,
        candidate_cols: Sequence[ColName] | None = None,
    ) -> FeatureSuggestion:
        """Compute MI for each column and suggest the best."""
        rows = state.candidate_rows
        if len(rows) <= 1:
            cols = candidate_cols or list(df.columns)
            return FeatureSuggestion(
                col=cols[0] if len(cols) > 0 else None, score=0.0, expected_voi=0.0
            )

        # Get candidate columns
        if candidate_cols is None:
            candidate_cols = [c for c in df.columns if c not in state.observed_cols]
        else:
            candidate_cols = [c for c in candidate_cols if c not in state.observed_cols]

        if not candidate_cols:
            return FeatureSuggestion(col=None, score=0.0)

        best_col = None
        best_score = -float("inf")
        best_mi = 0.0

        for col in candidate_cols:
            mi = self.compute_mi(df, state, col)
            cost = self.costs.get(col, 1.0) if self.costs else 1.0
            score = _per_cost(mi, cost)

            if score > best_score:
                best_col = col
                best_score = score
                best_mi = mi

        return FeatureSuggestion(
            col=best_col,
            score=best_score,
            expected_voi=best_mi,
            marginal_cost=(
                self.costs.get(best_col, 1.0) if self.costs and best_col else 1.0
            ),
        )

    def compute_mi(
        self, df: pd.DataFrame, state: CandidateState, col: ColName
    ) -> float:
        """Compute conditional mutual information I(R; X_col | E) in bits.

        Model-free: groups candidates by their value in `col` and measures how
        much the posterior's entropy drops in expectation. Public because
        callers often want the full ranking, not just the argmax.
        """
        rows = state.candidate_rows
        if len(rows) <= 1:
            return 0.0

        # Compute prior entropy H(R | E)
        h_prior = state.entropy

        # Group candidate rows by their value in the selected column
        value_groups = {}
        for i, r in enumerate(rows):
            val = df.iloc[r][col]
            if val not in value_groups:
                value_groups[val] = []
            value_groups[val].append(i)

        if len(value_groups) <= 1:
            # All rows have the same value - no information
            return 0.0

        # Compute expected conditional entropy H(R | E, X_col)
        h_conditional = 0.0
        for _val, group_indices in value_groups.items():
            # Probability that X_col takes this value
            p_x = sum(state.posterior[i] for i in group_indices)
            if p_x <= 0:
                continue

            # Posterior over R given X_col = val
            group_posterior = np.array([state.posterior[i] for i in group_indices])
            group_posterior /= group_posterior.sum()

            # Entropy of this group
            h_group = -np.sum(group_posterior * np.log2(group_posterior + 1e-10))
            h_conditional += p_x * h_group

        mi = h_prior - h_conditional

        if self.normalize and h_prior > 0:
            mi /= h_prior

        return mi


@dataclass
class RandomPolicy:
    """Random policy for baseline comparisons.

    Attributes:
        seed: Random seed for reproducibility
    """

    seed: int | None = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)

    def suggest(
        self,
        df: pd.DataFrame,
        state: CandidateState,
        candidate_cols: Sequence[ColName] | None = None,
    ) -> FeatureSuggestion:
        """Select a random column."""
        # Get candidate columns
        if candidate_cols is None:
            candidate_cols = [c for c in df.columns if c not in state.observed_cols]
        else:
            candidate_cols = [c for c in candidate_cols if c not in state.observed_cols]

        if not candidate_cols:
            return FeatureSuggestion(col=None, score=0.0)

        # Index rather than np.random.choice, which coerces the column labels
        # into an array and mangles non-string ones.
        col = candidate_cols[int(np.random.randint(len(candidate_cols)))]
        return FeatureSuggestion(col=col, score=1.0)
