"""Regression tests for bugs fixed in 0.3.0.

Each test names the specific defect it guards, because in every case the code
path had no coverage at all -- which is why the bug survived.
"""

import pandas as pd
import pytest

from rowvoi import (
    CandidateState,
    MIPolicy,
    RowVoiModel,
    compute_gold_next_column_probabilistic,
)


@pytest.fixture
def df():
    """Four rows where column A splits 2/2 and B identifies uniquely."""
    return pd.DataFrame(
        {
            "cheap": ["x", "x", "y", "y"],
            "pricey": ["p", "q", "r", "s"],
        }
    )


@pytest.fixture
def model(df):
    return RowVoiModel().fit(df)


class TestGoldNextColumnCallable:
    """compute_gold_next_column_probabilistic passed a nonexistent kwarg.

    It called `suggest_next_feature(candidate_features=...)` where the
    parameter is named `candidate_cols`, so *every* call raised TypeError.
    """

    def test_returns_a_column(self, df, model):
        state = CandidateState.uniform([0, 1, 2, 3])
        col = compute_gold_next_column_probabilistic(df, state, model)
        assert col in df.columns

    def test_honors_the_candidate_column_restriction(self, df, model):
        state = CandidateState.uniform([0, 1, 2, 3])
        col = compute_gold_next_column_probabilistic(
            df, state, model, candidate_cols=["cheap"]
        )
        assert col == "cheap"


class TestMIPolicyForwardsObjective:
    """MIPolicy never forwarded its objective/costs to the model.

    It asked the model for the raw-MI argmax and then rescaled that one
    column's score by cost, so cost could never change *which* column was
    chosen -- only the number reported next to it.
    """

    def test_raw_mi_prefers_the_more_informative_column(self, df, model):
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = MIPolicy(model=model, objective="mi")
        # "pricey" separates all four rows; "cheap" only splits them 2/2
        assert policy.suggest(df, state).col == "pricey"

    def test_cost_changes_the_chosen_column(self, df, model):
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = MIPolicy(
            model=model,
            objective="mi_over_cost",
            feature_costs={"cheap": 1.0, "pricey": 100.0},
        )
        # With the informative column priced 100x, the cheap one wins.
        # Before the fix this still returned "pricey".
        assert policy.suggest(df, state).col == "cheap"

    def test_mi_over_cost_without_costs_does_not_raise(self, df, model):
        # The model rejects mi_over_cost when feature_costs is None; the
        # policy must degrade to raw MI rather than propagating that error.
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = MIPolicy(model=model, objective="mi_over_cost")
        assert policy.suggest(df, state).col == "pricey"

    def test_partial_costs_do_not_raise(self, df, model):
        # The model requires a cost for every column it scores; unpriced
        # columns must default rather than raising ValueError.
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = MIPolicy(
            model=model, objective="mi_over_cost", feature_costs={"pricey": 100.0}
        )
        assert policy.suggest(df, state).col == "cheap"

    def test_returns_a_suggestion_when_nothing_is_left(self, df, model):
        # Every column observed: the model returns None and the policy must
        # still honor its `-> FeatureSuggestion` contract.
        state = CandidateState(
            candidate_rows=[0, 1, 2, 3],
            posterior=CandidateState.uniform([0, 1, 2, 3]).posterior,
            observed_cols={"cheap", "pricey"},
            observed_values={},
        )
        suggestion = MIPolicy(model=model).suggest(df, state)
        assert suggestion.col is None
        assert suggestion.score == 0.0
