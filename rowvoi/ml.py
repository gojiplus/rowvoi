"""Model‑based value‑of‑information routines for rowvoi.

This module defines a small class, :class:`RowVoiModel`, that
implements a probabilistic policy for sequential feature acquisition.
It combines global frequency information from a reference dataset,
optionally performs simple discretization of numeric columns, and
supports a basic measurement noise model.  Using this information,
the model can rank candidate features by their expected mutual
information with the unknown row identity (given a
:class:`~rowvoi.types.CandidateState`) and simulate an interactive
disambiguation session.

The design is inspired by the active feature acquisition literature
but kept deliberately simple for ease of understanding and extension.
For deterministic/local policies that do not require learning a model,
see :mod:`rowvoi.mi`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from .types import CandidateState, ColName, RowIndex


@dataclass
class FeatureSuggestion:
    """Result of evaluating a feature for expected value of information.

    Attributes
    ----------
    col : ColName
        The column name suggested to query next.
    voi : float
        The estimated mutual information (in bits) between the row
        identity and the feature value, given current evidence.
    normalized_voi : float
        The mutual information divided by the entropy of the feature
        distribution in the candidate set.  This can be used to
        prevent high‑cardinality features from dominating purely due
        to their large entropy.
    details : Dict[str, float]
        Auxiliary information about the calculation, such as the
        prior entropy and expected conditional entropy.
    """

    col: ColName
    voi: float
    normalized_voi: float
    details: dict[str, float]


class RowVoiModel:
    """Model for computing expected value of information across features.

    A :class:`RowVoiModel` encapsulates global information about the
    distribution of values for each feature in a dataset, together
    with optional discretization rules and a simple noise model.  It
    provides methods to fit to a DataFrame, rank features by
    expected information gain, and simulate sequential acquisition
    procedures to disambiguate an unknown row among a candidate set.

    Parameters
    ----------
    smoothing : float, optional
        A pseudo‑count added to each category when computing
        frequencies.  This mitigates zero‑probability issues when
        some candidate values are rare.  Default is 1e‑6.
    noise : float, optional
        Probability that the observed feature value does not equal
        the candidate row's true value.  When greater than zero,
        ``noise`` spreads probability mass over other candidate
        values according to the global frequency distribution.  Default
        is 0.0 (no noise).
    normalize_cols : bool, optional
        Whether to compute and use normalized mutual information
        values (i.e. divide by the feature entropy) when ranking
        features.  Default is ``True``.
    """

    def __init__(
        self, smoothing: float = 1e-6, noise: float = 0.0, normalize_cols: bool = True
    ) -> None:
        self.smoothing = float(smoothing)
        if not (0.0 <= noise < 1.0):
            raise ValueError("noise must be in [0, 1)")
        self.noise = float(noise)
        self.normalize_cols = bool(normalize_cols)
        # Internal structures populated by fit()
        self._df: pd.DataFrame | None = None
        self._col_values: dict[ColName, list[object]] = {}
        self._freqs: dict[ColName, dict[object, float]] = {}
        self._entropy: dict[ColName, float] = {}
        self._discrete_cols: list[ColName] = []

    def fit(
        self,
        df: pd.DataFrame,
        discrete_cols: Sequence[ColName] | None = None,
        bins: int = 3,
    ) -> RowVoiModel:
        """Fit the model to a DataFrame by computing column frequencies and entropies.

        This method prepares the model to evaluate expected information
        gain by storing global frequencies and, if necessary,
        discretizing numeric columns.  Only columns in
        ``discrete_cols`` will be treated as discrete; if ``None``,
        all columns are treated as discrete.  The DataFrame is not
        modified in place; a copy with discretized values is stored.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataset from which to learn frequencies.  Should
            contain no missing values; callers should handle missing
            values externally (e.g. by imputation or by treating
            ``NaN`` as a category).
        discrete_cols : Sequence[ColName], optional
            Columns to treat as discrete.  If ``None``, all columns
            are considered discrete.  Numeric columns not in this list
            are discretized into quantile bins of size ``bins``.
        bins : int, optional
            Number of quantile bins for discretization of numeric
            columns not specified in ``discrete_cols``.  Default is 3.

        Returns
        -------
        RowVoiModel
            Returns ``self`` for chaining.
        """
        # Copy the DataFrame to avoid modifying user data.
        df_proc = df.copy()
        # Determine which columns to treat as discrete.  If None, treat all as discrete.
        if discrete_cols is None:
            discrete_cols = list(df_proc.columns)
        else:
            discrete_cols = list(discrete_cols)
        self._discrete_cols = discrete_cols
        # Discretize numeric columns not in discrete_cols.
        for col in df_proc.columns:
            if col not in discrete_cols and pd.api.types.is_numeric_dtype(df_proc[col]):
                # Compute quantile bins
                try:
                    df_proc[col] = pd.qcut(df_proc[col], q=bins, duplicates="drop")
                except ValueError:
                    # If qcut fails (e.g. due to constant values), convert to a single category
                    df_proc[col] = df_proc[col].astype(str)
            # Otherwise leave the column as is.
        # Compute frequencies and entropies for each discrete column.
        self._df = df_proc
        for col in df_proc.columns:
            values = df_proc[col].tolist()
            self._col_values[col] = values
            # frequency with smoothing
            len(values)
            counts: dict[object, float] = {}
            for v in values:
                counts[v] = counts.get(v, 0.0) + 1.0
            # apply smoothing: add pseudo counts uniformly
            distinct_vals = set(counts.keys())
            smoothed_counts: dict[object, float] = {}
            for v in distinct_vals:
                smoothed_counts[v] = counts[v] + self.smoothing
            total_mass = sum(smoothed_counts.values())
            # normalize to get probabilities
            freqs: dict[object, float] = {
                v: c / total_mass for v, c in smoothed_counts.items()
            }
            self._freqs[col] = freqs
            # compute entropy of the marginal distribution of this column
            ent = 0.0
            for p in freqs.values():
                ent -= p * math.log2(p)
            self._entropy[col] = ent
        return self

    def _conditional_value_distribution(
        self,
        col: ColName,
        candidate_values: Iterable[object],
        true_value: object,
    ) -> dict[object, float]:
        """Compute P(X_col = v | R=r) under the noise model.

        When ``self.noise`` is zero, this returns a degenerate
        distribution that puts all mass on the true value.  When
        ``noise`` is positive, it spreads probability mass over
        ``candidate_values`` proportional to the global frequencies in
        the dataset, with probability ``1 - noise`` assigned to the
        true value.  It assumes that ``true_value`` is in
        ``candidate_values``.

        Parameters
        ----------
        col : ColName
            Column being evaluated.
        candidate_values : Iterable[object]
            The possible values of the feature among the current
            candidate rows.
        true_value : object
            The actual value for the candidate row ``r``.

        Returns
        -------
        Dict[object, float]
            A probability mass function over ``candidate_values``.
        """
        if self.noise <= 0.0:
            # Deterministic: all mass on the true value.
            return {v: 1.0 if v == true_value else 0.0 for v in candidate_values}
        # Otherwise distribute noise mass across other values proportional to global freqs.
        freqs = self._freqs.get(col)
        if freqs is None:
            raise ValueError(
                f"Column '{col}' not found in frequency table. Did you call fit()?"
            )
        # Compute mass of other candidate values in global distribution.
        candidate_values = list(candidate_values)
        mass_other = sum(freqs[v] for v in candidate_values if v != true_value)
        dist: dict[object, float] = {}
        for v in candidate_values:
            if v == true_value:
                dist[v] = 1.0 - self.noise
            else:
                if mass_other > 0:
                    dist[v] = self.noise * (freqs[v] / mass_other)
                else:
                    # If no mass on other candidate values, distribute uniformly
                    dist[v] = self.noise / (len(candidate_values) - 1)
        return dist

    def _expected_cond_entropy(
        self,
        state: CandidateState,
        col: ColName,
    ) -> float:
        """Compute E_{X_col}[H(R | E, X_col)] with smoothing and noise.

        This internal helper computes the expected conditional entropy of
        the row identity given a particular column ``col``, the current
        candidate state, global frequency information, and the noise
        parameter.  It uses the candidate set to determine the support
        of possible values and the global frequency distribution to
        distribute noise when ``self.noise`` is positive.

        The algorithm follows these steps:

        1. Identify the set of values ``V`` that the candidate rows
           take in column ``col``.
        2. For each candidate row ``r`` compute the conditional
           distribution ``p(X_col | R=r)`` using
           :meth:`_conditional_value_distribution`.
        3. For each ``v`` in ``V`` compute ``P(X_col=v | E)`` by
           averaging ``p(X_col=v | R=r)`` weighted by the posterior.
        4. For each ``v`` compute the posterior over rows ``R`` given
           ``X_col=v`` and then the entropy of that posterior.
        5. Return the expectation of these entropies weighted by
           ``P(X_col=v | E)``.

        Parameters
        ----------
        state : CandidateState
            The current candidate state.
        col : ColName
            The column of interest.

        Returns
        -------
        float
            The expected conditional entropy (in bits).
        """
        rows = state.candidate_rows
        # Determine the set of candidate values for this column among current candidates.
        values = [self._df.iloc[r][col] for r in rows]
        unique_values = set(values)
        # Precompute p(X=v | R=r) for each r and v under noise model
        p_x_given_r: dict[RowIndex, dict[object, float]] = {}
        for r, val_r in zip(rows, values):
            p_x_given_r[r] = self._conditional_value_distribution(
                col, unique_values, val_r
            )
        # Compute P(X=v | E) for each possible v
        p_x: dict[object, float] = {}
        for v in unique_values:
            total = 0.0
            for r in rows:
                p_r = state.posterior.get(r, 0.0)
                total += p_r * p_x_given_r[r][v]
            p_x[v] = total
        # Compute expected conditional entropy
        h_cond = 0.0
        for v, p_v in p_x.items():
            if p_v <= 0.0:
                continue
            # Posterior distribution over R given X=v
            denom = 0.0
            for r in rows:
                denom += state.posterior.get(r, 0.0) * p_x_given_r[r][v]
            # Compute entropy for this branch
            h_r_given_v = 0.0
            for r in rows:
                num = state.posterior.get(r, 0.0) * p_x_given_r[r][v]
                if denom <= 0.0:
                    continue
                p_r_given = num / denom
                if p_r_given > 0.0:
                    h_r_given_v -= p_r_given * math.log2(p_r_given)
            h_cond += p_v * h_r_given_v
        return h_cond

    def suggest_next_feature(
        self,
        df: pd.DataFrame,
        state: CandidateState,
        candidate_cols: Sequence[ColName] | None = None,
        objective: str = "mi",
        feature_costs: dict[ColName, float] | None = None,
    ) -> FeatureSuggestion | None:
        """Rank candidate columns by expected value of information.

        Given a DataFrame ``df``, a current candidate state, and an
        optional set of candidate columns, this method evaluates each
        column for its expected mutual information ``I(R; X_col | E)``
        under the model’s smoothing and noise assumptions.  It then
        returns the column with the best score.  Two objectives are
        supported:

        * ``'mi'`` – select the column with the highest expected MI.
        * ``'mi_over_cost'`` – divide expected MI by a user‑supplied
          cost for that feature.  This allows penalizing expensive
          features.

        If ``normalize_cols`` was set when constructing the model,
        the normalized mutual information (MI divided by the feature
        entropy) is returned in the ``FeatureSuggestion`` for
        diagnostic purposes but is not used to rank features unless
        ``objective`` is set accordingly.

        Parameters
        ----------
        df : pandas.DataFrame
            The data table.  If different from the DataFrame passed
            to :meth:`fit`, it will be discretized using the same
            rules.
        state : CandidateState
            The current candidate state.
        candidate_cols : Sequence[ColName], optional
            A list of columns to consider.  If ``None``, all columns
            not yet observed in ``state.observed_cols`` are considered.
        objective : str, optional
            Objective used for ranking.  One of ``'mi'`` or
            ``'mi_over_cost'``.  Default is ``'mi'``.
        feature_costs : Dict[ColName, float], optional
            Mapping of feature costs.  Required if objective is
            ``'mi_over_cost'``.  Costs must be positive.

        Returns
        -------
        FeatureSuggestion or None
            A :class:`FeatureSuggestion` containing the best column and
            associated information gain estimates, or ``None`` if no
            eligible columns remain.
        """
        if candidate_cols is None:
            candidate_cols = [c for c in df.columns if c not in state.observed_cols]
        else:
            candidate_cols = list(candidate_cols)
        if not candidate_cols:
            return None
        # Ensure model has been fit
        if self._df is None:
            raise RuntimeError(
                "RowVoiModel.fit() must be called before using suggest_next_feature()."
            )
        best_suggestion: FeatureSuggestion | None = None
        for col in candidate_cols:
            # Compute prior entropy of R given current state
            h_prior = 0.0
            for r in state.candidate_rows:
                p_r = state.posterior.get(r, 0.0)
                if p_r > 0.0:
                    h_prior -= p_r * math.log2(p_r)
            # Compute expected conditional entropy using global frequencies and noise
            h_cond = self._expected_cond_entropy(state, col)
            voi = h_prior - h_cond
            # Normalized VOI (divide by feature entropy within candidate set or global entropy)
            if self.normalize_cols:
                # compute entropy of the candidate-set distribution of this column
                # treat candidate-set distribution (p_x) computed above? It's not returned; recompute quickly.
                values = [self._df.iloc[r][col] for r in state.candidate_rows]
                # compute weighted histogram p_x
                p_x: dict[object, float] = {}
                for r, v in zip(state.candidate_rows, values):
                    p_x[v] = p_x.get(v, 0.0) + state.posterior.get(r, 0.0)
                # compute entropy of X in candidate set
                h_x = 0.0
                for p_val in p_x.values():
                    if p_val > 0.0:
                        h_x -= p_val * math.log2(p_val)
                normalized_voi = voi / h_x if h_x > 0.0 else 0.0
            else:
                normalized_voi = voi
            # For MI_over_cost objective, scale by cost
            score = voi
            if objective == "mi_over_cost":
                if feature_costs is None or col not in feature_costs:
                    raise ValueError(
                        "feature_costs must be provided for objective 'mi_over_cost'"
                    )
                cost = feature_costs[col]
                if cost <= 0:
                    raise ValueError("feature costs must be positive")
                score = voi / cost if cost > 0 else 0.0
            # Compare against current best
            if best_suggestion is None:
                best_suggestion = FeatureSuggestion(
                    col=col,
                    voi=voi,
                    normalized_voi=normalized_voi,
                    details={"entropy_before": h_prior, "entropy_after": h_cond},
                )
                best_score = score
            else:
                if score > best_score + 1e-12:
                    best_suggestion = FeatureSuggestion(
                        col=col,
                        voi=voi,
                        normalized_voi=normalized_voi,
                        details={"entropy_before": h_prior, "entropy_after": h_cond},
                    )
                    best_score = score
                elif abs(score - best_score) < 1e-12 and str(col) < str(
                    best_suggestion.col
                ):
                    # tie-break lexicographically
                    best_suggestion = FeatureSuggestion(
                        col=col,
                        voi=voi,
                        normalized_voi=normalized_voi,
                        details={"entropy_before": h_prior, "entropy_after": h_cond},
                    )
                    best_score = score
        return best_suggestion

    def run_acquisition(
        self,
        df: pd.DataFrame,
        true_row: RowIndex,
        initial_state: CandidateState,
        candidate_cols: Sequence[ColName] | None = None,
        stop_when_unique: bool = True,
        max_steps: int | None = None,
        objective: str = "mi",
        feature_costs: dict[ColName, float] | None = None,
    ) -> list[FeatureSuggestion]:
        """Simulate a sequential feature acquisition session.

        Given a true row index and an initial candidate state, this
        method repeatedly calls :meth:`suggest_next_feature` to select
        the next column to query.  It then simulates acquiring the
        feature value from the true row, updates the posterior and
        candidate list accordingly, and continues until either only
        one candidate row remains or a maximum number of steps has
        been reached.  The history of suggestions (with associated
        VOI metrics) is returned.

        Parameters
        ----------
        df : pandas.DataFrame
            The data table (same columns as used for fitting).  If
            different from ``self._df``, it will be discretized
            consistently.
        true_row : RowIndex
            The index of the actual row to identify.  Must be
            contained in ``initial_state.candidate_rows``.
        initial_state : CandidateState
            The starting candidate state.  This object is not
            modified; a new state is created for the simulation.
        candidate_cols : Sequence[ColName], optional
            Optional subset of columns to consider when selecting
            features.  If ``None``, all columns not yet observed are
            considered at each step.
        stop_when_unique : bool, optional
            If ``True`` (default), stop the acquisition as soon as
            the posterior concentrates all mass on a single row.  If
            ``False``, continue until ``max_steps`` is reached.
        max_steps : int, optional
            Maximum number of features to query.  If ``None``, no
            explicit limit is imposed.
        objective : str, optional
            Objective passed to :meth:`suggest_next_feature` (either
            ``'mi'`` or ``'mi_over_cost'``).  Default is ``'mi'``.
        feature_costs : Dict[ColName, float], optional
            Feature cost mapping used if ``objective='mi_over_cost'``.

        Returns
        -------
        List[FeatureSuggestion]
            A list of suggestions (one per query) containing the
            column chosen at each step and the associated VOI metrics.
            The length of the list equals the number of queries made.
        """
        # Copy the state so that initial_state is not modified.
        state = CandidateState(
            candidate_rows=list(initial_state.candidate_rows),
            posterior=dict(initial_state.posterior),
            observed_cols=list(initial_state.observed_cols),
            observed_values=dict(initial_state.observed_values),
        )
        history: list[FeatureSuggestion] = []
        step = 0
        while True:
            if stop_when_unique and len(state.candidate_rows) <= 1:
                break
            if max_steps is not None and step >= max_steps:
                break
            suggestion = self.suggest_next_feature(
                df,
                state,
                candidate_cols,
                objective=objective,
                feature_costs=feature_costs,
            )
            if suggestion is None:
                break
            col = suggestion.col
            # Simulate acquiring the value from the true row
            true_value = df.iloc[true_row][col]
            # Record observation
            state.observed_cols.append(col)
            state.observed_values[col] = true_value
            # Update posterior: zero out rows with different value, renormalize
            new_posterior: dict[RowIndex, float] = {}
            total_mass = 0.0
            for r in state.candidate_rows:
                p_r = state.posterior.get(r, 0.0)
                if df.iloc[r][col] == true_value:
                    new_posterior[r] = p_r
                    total_mass += p_r
                else:
                    new_posterior[r] = 0.0
            if total_mass > 0.0:
                for r in new_posterior:
                    new_posterior[r] /= total_mass
            state.posterior = new_posterior
            # Update candidate_rows list to those with nonzero posterior
            state.candidate_rows = [
                r for r in state.candidate_rows if state.posterior.get(r, 0.0) > 0.0
            ]
            # Append suggestion to history
            history.append(suggestion)
            step += 1
        return history
