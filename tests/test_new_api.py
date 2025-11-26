"""Basic tests for the new rowvoi API."""

import numpy as np
import pandas as pd
import pytest

from rowvoi import (
    CandidateMIPolicy,
    # Core types
    CandidateState,
    # Sessions
    DisambiguationSession,
    FeatureSuggestion,
    # Policies
    GreedyCoveragePolicy,
    # Deterministic methods
    KeyProblem,
    RandomPolicy,
    StopRules,
    evaluate_keys,
    evaluate_policies,
    find_key,
    plan_key_path,
    # Evaluation
    sample_candidate_sets,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3],
            "B": [1, 2, 1, 2, 1],
            "C": [1, 1, 1, 2, 2],
            "D": [1, 2, 3, 4, 5],
        }
    )


class TestCandidateState:
    """Test CandidateState functionality."""

    def test_uniform_creation(self):
        state = CandidateState.uniform([0, 1, 2])
        assert len(state.candidate_rows) == 3
        assert np.allclose(state.posterior.sum(), 1.0)
        assert np.allclose(state.posterior, [1 / 3, 1 / 3, 1 / 3])

    def test_entropy_calculation(self):
        state = CandidateState.uniform([0, 1])
        assert np.isclose(state.entropy, 1.0)  # Binary entropy

        state = CandidateState.uniform([0])
        assert np.isclose(state.entropy, 0.0)  # No uncertainty

    def test_is_unique(self):
        state = CandidateState(
            candidate_rows=[0],
            posterior=np.array([1.0]),
            observed_cols=set(),
            observed_values={},
        )
        assert state.is_unique
        assert state.unique_row == 0

    def test_filter_candidates(self, sample_df):
        state = CandidateState.uniform([0, 1, 2])
        new_state = state.filter_candidates(sample_df, "A", 1)
        assert len(new_state.candidate_rows) == 2  # Rows 0 and 1 have A=1
        assert "A" in new_state.observed_cols
        assert new_state.observed_values["A"] == 1


class TestDeterministicKeys:
    """Test deterministic key finding."""

    def test_find_key_simple(self, sample_df):
        # Rows 0 and 1 differ only on B
        key = find_key(sample_df, [0, 1])
        assert key == ["B"]

    def test_find_key_multiple(self, sample_df):
        # Column D should be sufficient to distinguish all these rows
        key = find_key(sample_df, [0, 1, 2, 3])
        assert len(key) >= 1
        # D distinguishes all rows uniquely, so it should be the minimal key
        assert key == ["D"]

    def test_key_problem_class(self, sample_df):
        problem = KeyProblem(sample_df, [0, 1, 2, 3])

        # Test is_key
        assert not problem.is_key(["A"])
        assert problem.is_key(["D"])

        # Test coverage
        coverage = problem.pairwise_coverage(["A", "B"])
        assert 0 < coverage <= 1.0

    def test_plan_path(self, sample_df):
        path = plan_key_path(sample_df, [0, 1, 2, 3])

        # Path should return columns
        cols = path.columns()
        assert len(cols) > 0

        # Coverage should increase along path
        coverage_curve = path.coverage_curve()
        for i in range(1, len(coverage_curve)):
            assert coverage_curve[i][1] >= coverage_curve[i - 1][1]

    def test_path_with_costs(self, sample_df):
        costs = {"A": 1.0, "B": 2.0, "C": 5.0, "D": 1.0}
        path = plan_key_path(sample_df, [0, 1, 2, 3], costs=costs)

        # Test budget prefix
        budget_3_cols = path.prefix_for_budget(3.0)
        total_cost = sum(costs.get(c, 1.0) for c in budget_3_cols)
        assert total_cost <= 3.0


class TestPolicies:
    """Test policy implementations."""

    def test_greedy_coverage_policy(self, sample_df):
        state = CandidateState.uniform([0, 1, 2])
        policy = GreedyCoveragePolicy()

        suggestion = policy.suggest(sample_df, state)
        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.col in sample_df.columns
        assert suggestion.score >= 0

    def test_candidate_mi_policy(self, sample_df):
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = CandidateMIPolicy()

        suggestion = policy.suggest(sample_df, state)
        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.expected_voi is not None

    def test_random_policy(self, sample_df):
        state = CandidateState.uniform([0, 1, 2])
        policy = RandomPolicy(seed=42)

        suggestion1 = policy.suggest(sample_df, state)
        policy2 = RandomPolicy(seed=42)
        suggestion2 = policy2.suggest(sample_df, state)

        # Same seed should give same result
        assert suggestion1.col == suggestion2.col


class TestSessions:
    """Test interactive sessions."""

    def test_disambiguation_session(self, sample_df):
        policy = GreedyCoveragePolicy()
        session = DisambiguationSession(sample_df, [0, 1, 2], policy=policy)

        # Initial state
        assert len(session.state.candidate_rows) == 3
        assert session.cumulative_cost == 0

        # Get next question
        suggestion = session.next_question()
        assert suggestion.col in sample_df.columns

        # Observe a value
        session.observe("A", 1)
        assert len(session.history) == 1
        assert session.cumulative_cost > 0

    def test_session_with_stop_rules(self, sample_df):
        policy = GreedyCoveragePolicy()
        session = DisambiguationSession(sample_df, [0, 1, 2, 3], policy=policy)

        stop = StopRules(max_steps=2)
        steps = session.run(stop, true_row=0)

        assert len(steps) <= 2

    def test_stop_rules(self, sample_df):
        stop = StopRules(
            max_steps=5, cost_budget=10.0, epsilon_posterior=0.1, target_unique=True
        )

        # Test max steps
        state = CandidateState.uniform([0, 1])
        should_stop, reason = stop.should_stop(state, steps=5, total_cost=0)
        assert should_stop
        assert "max steps" in reason.lower()

        # Test uniqueness
        state = CandidateState(
            candidate_rows=[0],
            posterior=np.array([1.0]),
            observed_cols=set(),
            observed_values={},
        )
        should_stop, reason = stop.should_stop(state, steps=1, total_cost=0)
        assert should_stop
        assert "unique" in reason.lower()


class TestEvaluation:
    """Test evaluation tools."""

    def test_sample_candidate_sets(self, sample_df):
        sets = sample_candidate_sets(
            sample_df, subset_size=3, n_samples=5, random_state=42
        )

        assert len(sets) == 5
        for s in sets:
            assert len(s) == 3
            assert all(0 <= idx < len(sample_df) for idx in s)

    def test_evaluate_keys(self, sample_df):
        candidate_sets = [[0, 1, 2], [1, 2, 3]]

        methods = {
            "greedy": lambda df, rows: find_key(df, rows, strategy="greedy"),
        }

        results = evaluate_keys(sample_df, candidate_sets, methods)

        assert len(results) == 2  # 2 candidate sets
        for result in results:
            assert result.method == "greedy"
            assert len(result.key) > 0
            assert 0 <= result.pair_coverage <= 1.0
            assert result.runtime_sec >= 0

    def test_evaluate_policies(self, sample_df):
        candidate_sets = [[0, 1, 2, 3], [1, 2, 3, 4]]

        policies = {
            "greedy": GreedyCoveragePolicy(),
            "random": RandomPolicy(seed=42),
        }

        stats = evaluate_policies(
            sample_df, candidate_sets, policies, stop=StopRules(max_steps=3)
        )

        assert len(stats) == 2  # 2 policies
        for stat in stats:
            assert stat.mean_steps >= 0
            assert stat.mean_cost >= 0
            assert 0 <= stat.success_rate <= 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
