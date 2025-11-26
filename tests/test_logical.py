"""Tests for deterministic key-finding algorithms (now in keys module)."""

import pandas as pd
import pytest

from rowvoi import KeyProblem, find_key, plan_key_path


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing."""
    return pd.DataFrame({"A": [1, 1, 2], "B": [1, 2, 2], "C": [1, 1, 1]})


@pytest.fixture
def complex_df():
    """More complex DataFrame for testing."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
            "Age": [25, 30, 35, 25, 30],
            "City": ["NYC", "LA", "Chicago", "NYC", "LA"],
            "Score": [85, 90, 88, 92, 85],
        }
    )


class TestKeyProblem:
    """Test KeyProblem class functionality."""

    def test_is_key(self, simple_df):
        problem = KeyProblem(simple_df, [0, 1, 2])

        # Single column tests
        assert not problem.is_key(["A"])
        assert not problem.is_key(["B"])
        assert not problem.is_key(["C"])

        # Multiple columns
        assert problem.is_key(["A", "B"])
        assert problem.is_key(["B", "A"])  # Order shouldn't matter for is_key

    def test_is_key_with_epsilon(self, simple_df):
        problem = KeyProblem(simple_df, [0, 1, 2])

        # With epsilon = 0.5, we can leave half the pairs unresolved
        assert problem.is_key(["A"], epsilon_pairs=0.5)

    def test_minimal_key_greedy(self, simple_df):
        problem = KeyProblem(simple_df, [0, 1, 2])
        key = problem.minimal_key(strategy="greedy")

        assert len(key) == 2
        assert set(key) == {"A", "B"}

    def test_minimal_key_exact(self, simple_df):
        problem = KeyProblem(simple_df, [0, 1, 2])
        key = problem.minimal_key(strategy="exact")

        assert len(key) == 2
        assert set(key) == {"A", "B"}

    def test_empty_rows(self, simple_df):
        problem = KeyProblem(simple_df, [])
        key = problem.minimal_key()
        assert key == []

    def test_single_row(self, simple_df):
        problem = KeyProblem(simple_df, [0])
        key = problem.minimal_key()
        assert key == []

    def test_advanced_algorithms(self, complex_df):
        """Test advanced algorithms directly on KeyProblem."""
        problem = KeyProblem(complex_df, [0, 1, 2])

        # Test simulated annealing
        sa_key = problem.minimal_key(strategy="sa", time_limit=0.1)
        assert problem.is_key(sa_key)

        # Test genetic algorithm
        ga_key = problem.minimal_key(strategy="ga", time_limit=0.1)
        assert problem.is_key(ga_key)

        # Test exact algorithm
        exact_key = problem.minimal_key(strategy="exact", time_limit=0.1)
        assert problem.is_key(exact_key)

        # Test hybrid approach
        hybrid_key = problem.minimal_key(strategy="hybrid", time_limit=0.1)
        assert problem.is_key(hybrid_key)

    def test_algorithm_with_epsilon(self, complex_df):
        """Test algorithms with epsilon relaxation."""
        problem = KeyProblem(complex_df, [0, 1, 2, 3])

        for strategy in ["sa", "ga", "exact"]:
            key = problem.minimal_key(
                strategy=strategy, epsilon_pairs=0.2, time_limit=0.1
            )
            # With epsilon, we allow some pairs to remain unresolved
            assert problem.is_key(key, epsilon_pairs=0.2)

    def test_algorithm_edge_cases(self, simple_df):
        """Test algorithms with edge cases."""
        # Very small problem
        problem = KeyProblem(simple_df, [0, 1])

        sa_key = problem.minimal_key(strategy="sa", time_limit=0.05)
        assert problem.is_key(sa_key)

        ga_key = problem.minimal_key(strategy="ga", time_limit=0.05)
        assert problem.is_key(ga_key)

    def test_ilp_algorithm(self, complex_df):
        """Test ILP algorithm if available."""
        problem = KeyProblem(complex_df, [0, 1, 2])

        try:
            ilp_key = problem.minimal_key(strategy="ilp", time_limit=0.1)
            assert problem.is_key(ilp_key)

            # Test with epsilon
            ilp_eps_key = problem.minimal_key(
                strategy="ilp", epsilon_pairs=0.1, time_limit=0.1
            )
            assert problem.is_key(ilp_eps_key, epsilon_pairs=0.1)
        except ImportError:
            pytest.skip("pulp not available for ILP")

    def test_lp_algorithm(self, complex_df):
        """Test LP relaxation algorithm."""
        problem = KeyProblem(complex_df, [0, 1, 2])

        # LP currently falls back to greedy
        lp_key = problem.minimal_key(strategy="lp", time_limit=0.1)
        assert problem.is_key(lp_key)


class TestFindKey:
    """Test the convenience find_key function."""

    def test_basic_usage(self, simple_df):
        key = find_key(simple_df, [0, 1])
        assert key == ["B"]

    def test_with_costs(self, complex_df):
        costs = {"ID": 10, "Name": 1, "Age": 2, "City": 3, "Score": 5}
        key = find_key(complex_df, [0, 1, 2], costs=costs)

        # Should prefer lower cost columns
        total_cost = sum(costs.get(c, 1) for c in key)
        assert total_cost > 0

    def test_different_strategies(self, complex_df):
        rows = [0, 1, 2, 3]

        greedy_key = find_key(complex_df, rows, strategy="greedy")
        assert len(greedy_key) > 0

        # Test that all strategies return valid keys
        for strategy in ["greedy", "sa", "ga", "exact", "hybrid"]:
            key = find_key(complex_df, rows, strategy=strategy, time_limit=0.1)
            problem = KeyProblem(complex_df, rows)
            assert problem.is_key(key)

    def test_ilp_strategy(self, complex_df):
        """Test ILP strategy separately since it may not be available."""
        rows = [0, 1, 2, 3]
        try:
            key = find_key(complex_df, rows, strategy="ilp", time_limit=0.1)
            problem = KeyProblem(complex_df, rows)
            assert problem.is_key(key)
        except ImportError:
            # ILP requires pulp, which may not be available
            pytest.skip("pulp not available for ILP strategy")

    def test_lp_strategy(self, complex_df):
        """Test LP relaxation strategy."""
        rows = [0, 1, 2, 3]
        # LP currently falls back to greedy
        key = find_key(complex_df, rows, strategy="lp", time_limit=0.1)
        problem = KeyProblem(complex_df, rows)
        assert problem.is_key(key)


class TestPlanKeyPath:
    """Test path planning functionality."""

    def test_path_creation(self, simple_df):
        path = plan_key_path(simple_df, [0, 1, 2])

        assert len(path.steps) > 0
        cols = path.columns()
        assert len(cols) == len(path.steps)

    def test_coverage_progression(self, complex_df):
        path = plan_key_path(complex_df, [0, 1, 2, 3])

        # Coverage should be non-decreasing
        prev_coverage = 0
        for step in path.steps:
            assert step.coverage >= prev_coverage
            prev_coverage = step.coverage

    def test_prefix_for_budget(self, complex_df):
        costs = {"ID": 10, "Name": 1, "Age": 2, "City": 3, "Score": 5}
        path = plan_key_path(complex_df, [0, 1, 2], costs=costs)

        # Get columns within budget of 5
        cols = path.prefix_for_budget(5.0)
        actual_cost = sum(costs.get(c, 1) for c in cols)
        assert actual_cost <= 5.0

    def test_prefix_for_epsilon(self, complex_df):
        path = plan_key_path(complex_df, [0, 1, 2, 3])

        # Get columns for 90% coverage
        cols = path.prefix_for_epsilon_pairs(0.1)

        # Verify coverage
        problem = KeyProblem(complex_df, [0, 1, 2, 3])
        coverage = problem.pairwise_coverage(cols)
        assert coverage >= 0.9

    def test_different_objectives(self, complex_df):
        rows = [0, 1, 2, 3]

        path_pairs = plan_key_path(complex_df, rows, objective="pair_coverage")
        path_entropy = plan_key_path(complex_df, rows, objective="entropy")

        # Both should eventually achieve full coverage
        assert path_pairs.steps[-1].coverage == 1.0
        assert path_entropy.steps[-1].coverage == 1.0

    def test_weighting_schemes(self, complex_df):
        rows = [0, 1, 2, 3]

        path_uniform = plan_key_path(complex_df, rows, weighting="uniform")
        path_idf = plan_key_path(complex_df, rows, weighting="pair_idf")

        # Both should be valid paths
        assert len(path_uniform.steps) > 0
        assert len(path_idf.steps) > 0


class TestPairwiseCoverage:
    """Test pairwise coverage calculation."""

    def test_full_coverage(self, simple_df):
        from rowvoi.keys import pairwise_coverage

        # All columns should give full coverage
        coverage = pairwise_coverage(simple_df, [0, 1, 2], ["A", "B"])
        assert coverage == 1.0

    def test_partial_coverage(self, simple_df):
        from rowvoi.keys import pairwise_coverage

        # Single column gives partial coverage
        coverage = pairwise_coverage(simple_df, [0, 1, 2], ["A"])
        assert 0 < coverage < 1.0

    def test_no_coverage(self, simple_df):
        from rowvoi.keys import pairwise_coverage

        # Column C has same value for all rows
        coverage = pairwise_coverage(simple_df, [0, 1, 2], ["C"])
        assert coverage == 0.0

    def test_empty_cases(self, simple_df):
        from rowvoi.keys import pairwise_coverage

        # Empty rows
        coverage = pairwise_coverage(simple_df, [], ["A"])
        assert coverage == 1.0  # No pairs to cover

        # Single row
        coverage = pairwise_coverage(simple_df, [0], ["A"])
        assert coverage == 1.0  # No pairs to cover
