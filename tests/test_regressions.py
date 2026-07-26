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
    SolverUnavailableError,
    compute_gold_key,
    compute_gold_next_column_probabilistic,
    evaluate_keys,
    find_key,
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


@pytest.fixture
def no_exact_strategies(monkeypatch):
    """Make both exact strategies fail, leaving only greedy workable."""
    real = find_key

    def only_greedy(df, rows, *, strategy="greedy", **kwargs):
        if strategy != "greedy":
            raise SolverUnavailableError(f"{strategy} unavailable in this test")
        return real(df, rows, strategy="greedy", **kwargs)

    monkeypatch.setattr("rowvoi.eval.find_key", only_greedy)


class TestGoldKeyDoesNotFakeOptimality:
    """compute_gold_key used to fall through to greedy and call it optimal.

    Its result is the baseline for KeyEvalResult.optimality_gap. A greedy
    baseline makes that number wrong, and it can go negative -- a method
    appearing to beat the optimum. Failing is the honest outcome.
    """

    def test_returns_the_optimum_when_a_solver_works(self, df):
        # "pricey" separates all four rows on its own, so at unit cost it is
        # the whole optimum.
        assert compute_gold_key(df, [0, 1, 2, 3]) == ["pricey"]

    def test_optimum_follows_cost_not_column_count(self, df):
        # Priced at 100x, the single-column answer is no longer optimal.
        costs = {"cheap": 1.0, "pricey": 100.0}
        key = compute_gold_key(df, [0, 1, 2, 3], costs=costs)
        assert key == ["pricey"], "no cheaper cover exists here"

    def test_raises_rather_than_returning_greedy(self, df, no_exact_strategies):
        with pytest.raises(SolverUnavailableError, match="allow_approximate"):
            compute_gold_key(df, [0, 1, 2, 3])

    def test_allow_approximate_opts_back_in(self, df, no_exact_strategies):
        key = compute_gold_key(df, [0, 1, 2, 3], allow_approximate=True)
        assert key  # greedy still produces a usable key
        assert set(key) <= set(df.columns)

    def test_evaluate_keys_reports_no_gap_instead_of_a_wrong_one(
        self, df, no_exact_strategies
    ):
        # With no trustworthy baseline the gap must be absent, not computed
        # against a greedy stand-in.
        results = evaluate_keys(
            df,
            [[0, 1, 2, 3]],
            methods={"greedy": lambda d, r: find_key(d, r)},
            gold_solver=compute_gold_key,
        )
        assert len(results) == 1
        assert results[0].gold_key is None
        assert results[0].gold_cost is None
        assert results[0].optimality_gap is None

    def test_evaluate_keys_still_reports_a_gap_when_gold_is_real(self, df):
        results = evaluate_keys(
            df,
            [[0, 1, 2, 3]],
            methods={"greedy": lambda d, r: find_key(d, r)},
            gold_solver=compute_gold_key,
        )
        assert results[0].gold_key is not None
        assert results[0].optimality_gap is not None
        # Greedy cannot beat the optimum; a negative gap meant a fake baseline.
        assert results[0].optimality_gap >= 0


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
