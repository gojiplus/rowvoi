"""Tests for simulation and evaluation tools (now in eval module)."""

import pandas as pd
import pytest

from rowvoi import (
    CandidateMIPolicy,
    GreedyCoveragePolicy,
    RandomPolicy,
    StopRules,
    benchmark_policy,
    evaluate_keys,
    evaluate_policies,
    find_key,
    sample_candidate_sets,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3, 3],
            "B": [1, 2, 1, 2, 1, 2],
            "C": [1, 1, 1, 2, 2, 2],
            "D": list(range(6)),
        }
    )


class TestSampleCandidateSets:
    """Test candidate set sampling."""

    def test_basic_sampling(self, sample_df):
        sets = sample_candidate_sets(
            sample_df, subset_size=3, n_samples=5, random_state=42
        )

        assert len(sets) == 5
        for s in sets:
            assert len(s) == 3
            assert all(0 <= idx < len(sample_df) for idx in s)
            assert len(set(s)) == 3  # No duplicates

    def test_deterministic_with_seed(self, sample_df):
        sets1 = sample_candidate_sets(
            sample_df, subset_size=3, n_samples=5, random_state=42
        )

        sets2 = sample_candidate_sets(
            sample_df, subset_size=3, n_samples=5, random_state=42
        )

        assert sets1 == sets2

    def test_different_subset_sizes(self, sample_df):
        for size in [1, 2, 3, 4, 5]:
            sets = sample_candidate_sets(
                sample_df, subset_size=size, n_samples=3, random_state=42
            )
            assert all(len(s) == size for s in sets)

    def test_invalid_subset_size(self, sample_df):
        with pytest.raises(ValueError):
            sample_candidate_sets(
                sample_df,
                subset_size=10,  # Larger than DataFrame
                n_samples=1,
            )


class TestEvaluateKeys:
    """Test key evaluation functionality."""

    def test_basic_evaluation(self, sample_df):
        candidate_sets = [[0, 1, 2], [2, 3, 4]]

        methods = {
            "greedy": lambda df, rows: find_key(df, rows, strategy="greedy"),
        }

        results = evaluate_keys(sample_df, candidate_sets, methods)

        assert len(results) == 2  # One result per candidate set
        for result in results:
            assert result.method == "greedy"
            assert isinstance(result.key, list)
            assert result.pair_coverage >= 0
            assert result.runtime_sec >= 0

    def test_with_costs(self, sample_df):
        candidate_sets = [[0, 1, 2, 3]]
        costs = {"A": 1, "B": 2, "C": 5, "D": 10}

        methods = {
            "greedy": lambda df, rows: find_key(df, rows, costs=costs),
        }

        results = evaluate_keys(sample_df, candidate_sets, methods, costs=costs)

        assert len(results) == 1
        result = results[0]
        assert result.key_cost > 0
        assert result.key_cost == sum(costs.get(c, 1) for c in result.key)

    def test_multiple_methods(self, sample_df):
        candidate_sets = [[0, 1, 2]]

        methods = {
            "greedy": lambda df, rows: find_key(df, rows, strategy="greedy"),
            "exact": lambda df, rows: find_key(
                df, rows, strategy="exact", time_limit=0.1
            ),
        }

        results = evaluate_keys(sample_df, candidate_sets, methods)

        assert len(results) == 2  # Two methods
        method_names = [r.method for r in results]
        assert "greedy" in method_names
        assert "exact" in method_names


class TestEvaluatePolicies:
    """Test policy evaluation functionality."""

    def test_basic_policy_evaluation(self, sample_df):
        candidate_sets = [[0, 1, 2, 3], [2, 3, 4, 5]]

        policies = {
            "greedy": GreedyCoveragePolicy(),
            "random": RandomPolicy(seed=42),
        }

        stats = evaluate_policies(
            sample_df,
            candidate_sets,
            policies,
            stop=StopRules(max_steps=3),
            n_repeats=2,
        )

        assert len(stats) == 2  # Two policies
        for stat in stats:
            assert stat.mean_steps >= 0
            assert stat.mean_steps <= 3  # Max steps constraint
            assert stat.mean_cost >= 0
            assert 0 <= stat.success_rate <= 1

    def test_with_feature_costs(self, sample_df):
        candidate_sets = [[0, 1, 2, 3]]
        costs = {"A": 1, "B": 2, "C": 3, "D": 4}

        policies = {
            "greedy": GreedyCoveragePolicy(costs=costs),
        }

        stats = evaluate_policies(
            sample_df,
            candidate_sets,
            policies,
            feature_costs=costs,
            stop=StopRules(cost_budget=5.0),
        )

        assert len(stats) == 1
        assert stats[0].mean_cost <= 5.0  # Budget constraint

    def test_different_stop_rules(self, sample_df):
        candidate_sets = [[0, 1, 2, 3]]

        policies = {
            "mi": CandidateMIPolicy(),
        }

        # Test with uniqueness target
        stats1 = evaluate_policies(
            sample_df, candidate_sets, policies, stop=StopRules(target_unique=True)
        )

        # Test with epsilon posterior
        stats2 = evaluate_policies(
            sample_df, candidate_sets, policies, stop=StopRules(epsilon_posterior=0.1)
        )

        assert len(stats1) == 1
        assert len(stats2) == 1


class TestBenchmarkPolicy:
    """Test policy benchmarking functionality."""

    def test_basic_benchmark(self, sample_df):
        policy = GreedyCoveragePolicy()

        results = benchmark_policy(
            sample_df,
            policy,
            subset_sizes=[2, 3],
            n_samples=3,
            compute_optimal=False,
            random_state=42,
        )

        assert len(results) == 2  # Two subset sizes
        assert 2 in results
        assert 3 in results

        for size, size_results in results.items():
            assert len(size_results) == 3  # n_samples
            for result in size_results:
                assert result.subset_size == size
                assert result.steps_used >= 0
                assert isinstance(result.unique_identified, bool)

    def test_with_optimal_computation(self, sample_df):
        # Use small DataFrame for faster optimal computation
        small_df = sample_df[["A", "B"]].iloc[:4]
        policy = GreedyCoveragePolicy()

        results = benchmark_policy(
            small_df,
            policy,
            subset_sizes=[2],
            n_samples=2,
            compute_optimal=True,
            max_cols_for_exact=2,
            random_state=42,
        )

        for result in results[2]:
            # Optimal steps might be computed
            if result.optimal_steps is not None:
                assert result.optimal_steps >= 0
                assert result.optimal_steps <= result.steps_used

    def test_with_costs(self, sample_df):
        costs = {"A": 1, "B": 2, "C": 3, "D": 4}
        policy = GreedyCoveragePolicy(costs=costs)

        results = benchmark_policy(
            sample_df,
            policy,
            subset_sizes=[3],
            n_samples=2,
            feature_costs=costs,
            random_state=42,
        )

        assert len(results[3]) == 2
        for result in results[3]:
            assert result.cols_used is not None
            assert all(c in sample_df.columns for c in result.cols_used)
