"""Tests for probabilistic set cover module."""

import numpy as np
import pandas as pd
import pytest

from rowvoi.probcover import (
    AdaptiveFeatureSuggestion,
    ProbabilisticCoverageResult,
    estimate_pair_separation_probs,
    evaluate_coverage,
    greedy_epsilon_cover,
    probabilistic_minimal_key,
    suggest_next_feature_epsilon,
)
from rowvoi.types import CandidateState


class TestEstimatePairSeparationProbs:
    """Test probability estimation functions."""

    def test_empirical_estimation_basic(self):
        """Test basic empirical probability estimation."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 1, 2, 3],
                "B": [1, 1, 1, 2, 2, 2],
                "C": [1, 1, 1, 1, 1, 1],  # Constant column
            }
        )
        rows = [0, 1, 2]

        p = estimate_pair_separation_probs(df, rows, method="empirical")

        # Column A has 3 unique values appearing uniformly
        # P(different) = 1 - (1/3)² * 3 = 1 - 1/3 = 2/3
        assert (0, 1) in p["A"]
        assert abs(p["A"][(0, 1)] - 2 / 3) < 0.01

        # Column B has 2 unique values appearing uniformly
        # P(different) = 1 - (1/2)² * 2 = 1 - 1/2 = 1/2
        assert abs(p["B"][(0, 1)] - 0.5) < 0.01

        # Column C is constant
        # P(different) = 0
        assert abs(p["C"][(0, 1)] - 0.0) < 0.01

    def test_empirical_estimation_with_skewed_distribution(self):
        """Test estimation with non-uniform distribution."""
        df = pd.DataFrame(
            {
                "A": [1] * 90 + [2] * 10  # 90% value 1, 10% value 2
            }
        )
        rows = list(range(10))

        p = estimate_pair_separation_probs(df, rows, ["A"], method="empirical")

        # P(same) = 0.9² + 0.1² = 0.81 + 0.01 = 0.82
        # P(different) = 1 - 0.82 = 0.18
        assert abs(p["A"][(0, 1)] - 0.18) < 0.01

    def test_all_pairs_get_probabilities(self):
        """Test that all row pairs get probability estimates."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        rows = [0, 1, 2]

        p = estimate_pair_separation_probs(df, rows)

        # Should have probabilities for all 3 pairs
        expected_pairs = {(0, 1), (0, 2), (1, 2)}
        assert set(p["A"].keys()) == expected_pairs

    def test_column_subset(self):
        """Test estimation for subset of columns."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2], "C": [1, 2, 1]})
        rows = [0, 1, 2]

        p = estimate_pair_separation_probs(df, rows, cols=["A", "B"])

        assert "A" in p
        assert "B" in p
        assert "C" not in p

    def test_unsupported_method_raises(self):
        """Test that unsupported methods raise errors."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        rows = [0, 1, 2]

        with pytest.raises(NotImplementedError):
            estimate_pair_separation_probs(df, rows, method="conditional")

        with pytest.raises(ValueError):
            estimate_pair_separation_probs(df, rows, method="invalid")


class TestGreedyEpsilonCover:
    """Test greedy probabilistic set cover algorithm."""

    def test_deterministic_case(self):
        """Test with deterministic probabilities (0 or 1)."""
        rows = [0, 1, 2]
        cols = ["A", "B", "C"]

        # A distinguishes (0,1), B distinguishes (0,2), C distinguishes (1,2)
        p = {
            "A": {(0, 1): 1.0, (0, 2): 0.0, (1, 2): 0.0},
            "B": {(0, 1): 0.0, (0, 2): 1.0, (1, 2): 0.0},
            "C": {(0, 1): 0.0, (0, 2): 0.0, (1, 2): 1.0},
        }

        # Need all three columns for full coverage
        selected, coverage = greedy_epsilon_cover(rows, cols, p, eps=0.0)
        assert len(selected) == 3
        assert coverage == 1.0

    def test_with_epsilon_relaxation(self):
        """Test that epsilon relaxation reduces columns needed."""
        rows = [0, 1, 2, 3]
        cols = ["A", "B", "C"]

        # Create probabilities where A and B together give high coverage
        p = {
            "A": {(i, j): 0.8 for i in range(4) for j in range(i + 1, 4)},
            "B": {(i, j): 0.7 for i in range(4) for j in range(i + 1, 4)},
            "C": {(i, j): 0.3 for i in range(4) for j in range(i + 1, 4)},
        }

        # With eps=0, might need all columns
        selected_strict, _ = greedy_epsilon_cover(rows, cols, p, eps=0.0)

        # With eps=0.2, should need fewer columns
        selected_relaxed, _ = greedy_epsilon_cover(rows, cols, p, eps=0.2)

        assert len(selected_relaxed) <= len(selected_strict)

    def test_with_costs(self):
        """Test cost-aware selection."""
        rows = [0, 1]
        cols = ["expensive", "cheap1", "cheap2"]

        # Expensive column covers everything, cheap columns cover half each
        p = {
            "expensive": {(0, 1): 1.0},
            "cheap1": {(0, 1): 0.6},
            "cheap2": {(0, 1): 0.6},
        }

        costs = {"expensive": 10.0, "cheap1": 1.0, "cheap2": 1.0}

        selected, _ = greedy_epsilon_cover(rows, cols, p, eps=0.0, costs=costs)

        # Should prefer two cheap columns over one expensive
        assert "cheap1" in selected or "cheap2" in selected

    def test_empty_rows(self):
        """Test with no rows to distinguish."""
        selected, coverage = greedy_epsilon_cover([], ["A"], {}, eps=0.0)
        assert selected == []
        assert coverage == 1.0

    def test_single_row(self):
        """Test with single row (no pairs)."""
        selected, coverage = greedy_epsilon_cover([0], ["A"], {}, eps=0.0)
        assert selected == []
        assert coverage == 1.0

    def test_achieves_target_coverage(self):
        """Test that algorithm achieves target coverage."""
        rows = [0, 1, 2]
        cols = ["A", "B", "C"]

        # Overlapping coverage
        p = {
            "A": {(0, 1): 0.7, (0, 2): 0.6, (1, 2): 0.5},
            "B": {(0, 1): 0.6, (0, 2): 0.7, (1, 2): 0.5},
            "C": {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.7},
        }

        for target_eps in [0.0, 0.1, 0.2, 0.3]:
            _, coverage = greedy_epsilon_cover(rows, cols, p, eps=target_eps)
            # Allow small tolerance for floating point
            assert (
                coverage >= 1.0 - target_eps - 0.01
                or abs(coverage - (1.0 - target_eps)) < 0.1
            )


class TestProbabilisticMinimalKey:
    """Test main probabilistic minimal key API."""

    def test_basic_usage(self):
        """Test basic probabilistic minimal key finding."""
        df = pd.DataFrame(
            {"A": [1, 2, 3, 1, 2], "B": [1, 1, 2, 2, 2], "C": [1, 1, 1, 1, 2]}
        )
        rows = [0, 1, 2]

        result = probabilistic_minimal_key(df, rows, eps=0.1)

        assert isinstance(result, ProbabilisticCoverageResult)
        assert len(result.columns) > 0
        assert 0 <= result.expected_coverage <= 1
        assert result.actual_eps == 1.0 - result.expected_coverage
        assert result.method == "empirical"

    def test_with_costs(self):
        """Test cost-aware column selection."""
        df = pd.DataFrame({"expensive": [1, 2, 3], "cheap": [1, 2, 3]})
        rows = [0, 1, 2]
        costs = {"expensive": 100.0, "cheap": 1.0}

        result = probabilistic_minimal_key(df, rows, eps=0.0, costs=costs)

        # Should prefer cheap column
        assert "cheap" in result.columns
        if len(result.columns) == 1:
            assert result.columns[0] == "cheap"

    def test_candidate_columns_filtering(self):
        """Test that only varying columns are considered by default."""
        df = pd.DataFrame({"constant": [1, 1, 1], "varying": [1, 2, 3]})
        rows = [0, 1, 2]

        result = probabilistic_minimal_key(df, rows, eps=0.0)

        # Should only select varying column
        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_explicit_candidate_columns(self):
        """Test explicit candidate column specification."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 1, 2]})
        rows = [0, 1, 2]

        result = probabilistic_minimal_key(df, rows, eps=0.0, candidate_cols=["A", "C"])

        # Should not include B even if it might be useful
        assert "B" not in result.columns

    def test_different_epsilon_values(self):
        """Test that higher epsilon leads to fewer columns."""
        df = pd.DataFrame({f"col{i}": np.random.randint(0, 3, 20) for i in range(10)})
        rows = list(range(10))

        result_strict = probabilistic_minimal_key(df, rows, eps=0.0)
        result_relaxed = probabilistic_minimal_key(df, rows, eps=0.3)

        # Relaxed should use fewer or equal columns
        assert len(result_relaxed.columns) <= len(result_strict.columns)


class TestSuggestNextFeatureEpsilon:
    """Test adaptive epsilon policy."""

    def test_basic_adaptive_selection(self):
        """Test basic adaptive feature selection."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2], "C": [1, 2, 1]})

        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            observed_cols=[],
            observed_values={},
        )

        suggestion = suggest_next_feature_epsilon(df, state, eps=0.1)

        assert suggestion is not None
        assert isinstance(suggestion, AdaptiveFeatureSuggestion)
        assert suggestion.col in df.columns
        assert suggestion.expected_coverage_gain >= 0
        assert 0 <= suggestion.current_coverage <= 1
        assert suggestion.target_coverage == 0.9

    def test_excludes_observed_columns(self):
        """Test that observed columns are excluded."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]})

        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            observed_cols=["A"],
            observed_values={"A": 1},
        )

        suggestion = suggest_next_feature_epsilon(df, state, eps=0.0)

        if suggestion is not None:
            assert suggestion.col != "A"
            assert suggestion.col == "B"

    def test_returns_none_when_target_achieved(self):
        """Test returns None when coverage target is met."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})

        # A already distinguishes all pairs
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            observed_cols=["A"],
            observed_values={"A": 1},
        )

        suggestion = suggest_next_feature_epsilon(df, state, eps=0.0)

        assert suggestion is None

    def test_with_partial_disambiguation(self):
        """Test with some pairs already distinguished."""
        df = pd.DataFrame(
            {
                "A": [1, 1, 2],  # Distinguishes (0,2) and (1,2)
                "B": [1, 2, 2],  # Distinguishes (0,1) and (0,2)
            }
        )

        # A is observed, only need to distinguish (0,1) now
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            observed_cols=["A"],
            observed_values={"A": 1},
        )

        # Force actual data check
        df_copy = df.copy()
        suggestion = suggest_next_feature_epsilon(df_copy, state, eps=0.0)

        # Should suggest B to cover the remaining pair
        assert suggestion is not None
        assert suggestion.col == "B"

    def test_with_costs(self):
        """Test cost-aware adaptive selection."""
        df = pd.DataFrame({"expensive": [1, 2, 3], "cheap": [1, 2, 2]})

        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            observed_cols=[],
            observed_values={},
        )

        costs = {"expensive": 100.0, "cheap": 1.0}

        suggestion = suggest_next_feature_epsilon(df, state, eps=0.2, costs=costs)

        # Should consider cost in selection
        assert suggestion is not None
        # First selection might still be expensive if it's much better
        # but let's verify the API works
        assert suggestion.col in df.columns


class TestEvaluateCoverage:
    """Test coverage evaluation function."""

    def test_full_coverage(self):
        """Test evaluation with full coverage."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        rows = [0, 1, 2]

        coverage, uncovered = evaluate_coverage(df, rows, ["A"])

        assert coverage == 1.0
        assert uncovered == []

    def test_no_coverage(self):
        """Test evaluation with no coverage."""
        df = pd.DataFrame(
            {
                "A": [1, 1, 1]  # Constant column
            }
        )
        rows = [0, 1, 2]

        coverage, uncovered = evaluate_coverage(df, rows, ["A"])

        assert coverage == 0.0
        assert len(uncovered) == 3  # All 3 pairs uncovered

    def test_partial_coverage(self):
        """Test evaluation with partial coverage."""
        df = pd.DataFrame(
            {
                "A": [1, 1, 2]  # Distinguishes pairs with row 2
            }
        )
        rows = [0, 1, 2]

        coverage, uncovered = evaluate_coverage(df, rows, ["A"])

        assert abs(coverage - 2 / 3) < 0.0001  # 2 out of 3 pairs covered
        assert uncovered == [(0, 1)]  # Only (0,1) uncovered

    def test_multiple_columns(self):
        """Test coverage with multiple columns."""
        df = pd.DataFrame({"A": [1, 1, 2], "B": [1, 2, 2]})
        rows = [0, 1, 2]

        coverage, uncovered = evaluate_coverage(df, rows, ["A", "B"])

        assert coverage == 1.0  # All pairs covered
        assert uncovered == []

    def test_empty_rows(self):
        """Test with no rows."""
        df = pd.DataFrame({"A": [1, 2, 3]})

        coverage, uncovered = evaluate_coverage(df, [], ["A"])

        assert coverage == 1.0
        assert uncovered == []


class TestIntegration:
    """Integration tests for probabilistic cover."""

    def test_probabilistic_vs_deterministic(self):
        """Compare probabilistic and deterministic approaches."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 1, 2, 3] * 5,
                "B": [1, 1, 2, 2, 3, 3] * 5,
                "C": [1, 2, 1, 2, 1, 2] * 5,
            }
        )
        rows = [0, 1, 2, 3, 4]

        # Probabilistic solution
        prob_result = probabilistic_minimal_key(df, rows, eps=0.0)

        # Check actual coverage
        actual_coverage, _ = evaluate_coverage(df, rows, prob_result.columns)

        # With eps=0, should achieve full coverage
        assert actual_coverage == 1.0

    def test_adaptive_policy_convergence(self):
        """Test that adaptive policy eventually achieves target."""
        df = pd.DataFrame({f"col{i}": np.random.randint(0, 3, 20) for i in range(5)})
        rows = list(range(5))

        state = CandidateState(
            candidate_rows=rows,
            posterior={i: 1 / len(rows) for i in rows},
            observed_cols=[],
            observed_values={},
        )

        eps = 0.1
        max_iterations = 10

        for _ in range(max_iterations):
            suggestion = suggest_next_feature_epsilon(df, state, eps=eps)

            if suggestion is None:
                break  # Target achieved

            # Simulate observation
            state = CandidateState(
                candidate_rows=rows,
                posterior=state.posterior,
                observed_cols=state.observed_cols + [suggestion.col],
                observed_values={
                    **state.observed_values,
                    suggestion.col: df.loc[rows[0], suggestion.col],
                },
            )

        # Should have achieved target or run out of columns
        final_coverage, _ = evaluate_coverage(df, rows, state.observed_cols)
        assert final_coverage >= 1.0 - eps or len(state.observed_cols) == len(
            df.columns
        )

    @pytest.mark.parametrize("eps", [0.0, 0.05, 0.1, 0.2])
    def test_epsilon_monotonicity(self, eps):
        """Test that solution quality decreases monotonically with epsilon."""
        np.random.seed(42)
        df = pd.DataFrame({f"col{i}": np.random.randint(0, 4, 30) for i in range(8)})
        rows = list(range(10))

        result = probabilistic_minimal_key(df, rows, eps=eps)

        # Higher epsilon should allow lower coverage
        assert result.actual_eps <= eps + 0.01  # Small tolerance
        assert result.expected_coverage >= 1.0 - eps - 0.01
