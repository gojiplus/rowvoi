"""Tests for core types (now in core module)."""

import numpy as np
import pandas as pd
import pytest

from rowvoi import CandidateState, FeatureSuggestion


class TestCandidateState:
    """Test CandidateState functionality."""

    def test_uniform_creation(self):
        """Test creating uniform candidate state."""
        state = CandidateState.uniform([0, 1, 2])

        assert len(state.candidate_rows) == 3
        assert len(state.posterior) == 3
        assert np.allclose(state.posterior.sum(), 1.0)
        assert np.allclose(state.posterior, 1 / 3)
        assert state.observed_cols == set()
        assert state.observed_values == {}

    def test_custom_creation(self):
        """Test creating candidate state with custom parameters."""
        rows = [0, 1, 2]
        posterior = np.array([0.5, 0.3, 0.2])
        observed_cols = {"A"}
        observed_values = {"A": 1}

        state = CandidateState(
            candidate_rows=rows,
            posterior=posterior,
            observed_cols=observed_cols,
            observed_values=observed_values,
        )

        assert state.candidate_rows == rows
        assert np.allclose(state.posterior, posterior)
        assert state.observed_cols == observed_cols
        assert state.observed_values == observed_values

    def test_validation_length_mismatch(self):
        """Test validation when candidate rows and posterior have different lengths."""
        with pytest.raises(ValueError, match="Length mismatch"):
            CandidateState(
                candidate_rows=[0, 1],
                posterior=np.array([0.5, 0.3, 0.2]),
                observed_cols=set(),
                observed_values={},
            )

    def test_validation_posterior_sum(self):
        """Test validation when posterior doesn't sum to 1."""
        with pytest.raises(ValueError, match="Posterior must sum to 1"):
            CandidateState(
                candidate_rows=[0, 1],
                posterior=np.array([0.3, 0.3]),
                observed_cols=set(),
                observed_values={},
            )

    def test_entropy_calculation(self):
        """Test entropy calculation for different distributions."""
        # Binary entropy
        state = CandidateState.uniform([0, 1])
        assert np.isclose(state.entropy, 1.0)

        # No entropy (deterministic)
        state = CandidateState(
            candidate_rows=[0],
            posterior=np.array([1.0]),
            observed_cols=set(),
            observed_values={},
        )
        assert np.isclose(state.entropy, 0.0)

        # Maximum entropy for 4 candidates
        state = CandidateState.uniform([0, 1, 2, 3])
        assert np.isclose(state.entropy, 2.0)

        # Non-uniform entropy
        state = CandidateState(
            candidate_rows=[0, 1],
            posterior=np.array([0.9, 0.1]),
            observed_cols=set(),
            observed_values={},
        )
        expected_entropy = -(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))
        assert np.isclose(state.entropy, expected_entropy)

    def test_max_posterior_and_residual_uncertainty(self):
        """Test max_posterior and residual_uncertainty properties."""
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior=np.array([0.5, 0.3, 0.2]),
            observed_cols=set(),
            observed_values={},
        )

        assert state.max_posterior == 0.5
        assert state.residual_uncertainty == 0.5

        # Empty state
        state = CandidateState.uniform([])
        assert state.max_posterior == 0.0
        assert state.residual_uncertainty == 1.0

    def test_uniqueness_detection(self):
        """Test is_unique and unique_row properties."""
        # Not unique
        state = CandidateState.uniform([0, 1, 2])
        assert not state.is_unique
        assert state.unique_row is None

        # Unique (very high probability on one candidate)
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior=np.array([0.99999999, 5e-9, 5e-9]),
            observed_cols=set(),
            observed_values={},
        )
        assert state.is_unique
        assert state.unique_row == 0

        # Exactly unique
        state = CandidateState(
            candidate_rows=[5],
            posterior=np.array([1.0]),
            observed_cols=set(),
            observed_values={},
        )
        assert state.is_unique
        assert state.unique_row == 5

    def test_filter_candidates(self):
        """Test filtering candidates based on observed value."""
        df = pd.DataFrame(
            {"A": [1, 1, 2, 2], "B": [1, 2, 1, 2], "C": ["x", "y", "x", "z"]}
        )

        state = CandidateState.uniform([0, 1, 2, 3])

        # Filter by A=1 (should keep rows 0, 1)
        new_state = state.filter_candidates(df, "A", 1)

        assert len(new_state.candidate_rows) == 2
        assert set(new_state.candidate_rows) == {0, 1}
        assert "A" in new_state.observed_cols
        assert new_state.observed_values["A"] == 1
        assert np.allclose(new_state.posterior.sum(), 1.0)
        assert np.allclose(new_state.posterior, 0.5)  # Uniform over 2 candidates

        # Filter again by B=2 (should keep only row 1)
        final_state = new_state.filter_candidates(df, "B", 2)

        assert len(final_state.candidate_rows) == 1
        assert final_state.candidate_rows == [1]
        assert "B" in final_state.observed_cols
        assert final_state.observed_values["B"] == 2
        assert np.allclose(final_state.posterior, [1.0])
        assert final_state.is_unique

    def test_filter_no_matches(self):
        """Test filtering when no candidates match the value."""
        df = pd.DataFrame({"A": [1, 1, 2, 2]})

        state = CandidateState.uniform([0, 1, 2, 3])

        # Filter by A=3 (no matches)
        new_state = state.filter_candidates(df, "A", 3)

        assert len(new_state.candidate_rows) == 0
        assert len(new_state.posterior) == 0
        assert "A" in new_state.observed_cols
        assert new_state.observed_values["A"] == 3

    def test_filter_preserves_non_uniform_posterior(self):
        """Test that filtering correctly renormalizes non-uniform posteriors."""
        df = pd.DataFrame({"A": [1, 1, 2, 2]})

        # Create non-uniform state
        state = CandidateState(
            candidate_rows=[0, 1, 2, 3],
            posterior=np.array([0.4, 0.2, 0.3, 0.1]),
            observed_cols=set(),
            observed_values={},
        )

        # Filter by A=1 (keeps rows 0,1 with original posteriors 0.4, 0.2)
        new_state = state.filter_candidates(df, "A", 1)

        assert len(new_state.candidate_rows) == 2
        assert set(new_state.candidate_rows) == {0, 1}
        # Should renormalize: 0.4 and 0.2 -> 0.4/0.6 and 0.2/0.6 = 2/3 and 1/3
        expected_posterior = np.array([2 / 3, 1 / 3])
        assert np.allclose(new_state.posterior, expected_posterior)
        assert np.allclose(new_state.posterior.sum(), 1.0)

    def test_empty_state_properties(self):
        """Test properties of empty candidate state."""
        state = CandidateState.uniform([])

        assert state.entropy == 0.0
        assert state.max_posterior == 0.0
        assert state.residual_uncertainty == 1.0
        assert not state.is_unique  # Empty state is not unique
        assert state.unique_row is None

    def test_uniform_with_observed_data(self):
        """Test creating uniform state with pre-existing observations."""
        state = CandidateState.uniform(
            [0, 1, 2], observed_cols={"A", "B"}, observed_values={"A": 1, "B": "x"}
        )

        assert state.observed_cols == {"A", "B"}
        assert state.observed_values == {"A": 1, "B": "x"}
        assert np.allclose(state.posterior, 1 / 3)


class TestFeatureSuggestion:
    """Test FeatureSuggestion functionality."""

    def test_basic_creation(self):
        """Test creating feature suggestion with all fields."""
        suggestion = FeatureSuggestion(
            col="A",
            score=1.5,
            expected_voi=1.2,
            marginal_cost=2.0,
            debug={"entropy_before": 2.0, "entropy_after": 0.8},
        )

        assert suggestion.col == "A"
        assert suggestion.score == 1.5
        assert suggestion.expected_voi == 1.2
        assert suggestion.marginal_cost == 2.0
        assert suggestion.debug == {"entropy_before": 2.0, "entropy_after": 0.8}

    def test_minimal_creation(self):
        """Test creating feature suggestion with required fields only."""
        suggestion = FeatureSuggestion(col="B", score=0.8)

        assert suggestion.col == "B"
        assert suggestion.score == 0.8
        assert suggestion.expected_voi is None
        assert suggestion.marginal_cost is None
        assert suggestion.debug is None

    def test_cost_adjusted_score(self):
        """Test cost_adjusted_score property."""
        # With positive cost
        suggestion = FeatureSuggestion(col="A", score=2.0, marginal_cost=2.0)
        assert suggestion.cost_adjusted_score == 1.0

        # Without cost information
        suggestion = FeatureSuggestion(col="A", score=2.0)
        assert suggestion.cost_adjusted_score == 2.0

        # With zero cost (edge case)
        suggestion = FeatureSuggestion(col="A", score=2.0, marginal_cost=0.0)
        assert suggestion.cost_adjusted_score == 2.0

        # With None cost
        suggestion = FeatureSuggestion(col="A", score=2.0, marginal_cost=None)
        assert suggestion.cost_adjusted_score == 2.0

    def test_different_column_types(self):
        """Test feature suggestion with different column name types."""
        # String column
        suggestion1 = FeatureSuggestion(col="column_name", score=1.0)
        assert suggestion1.col == "column_name"

        # Integer column
        suggestion2 = FeatureSuggestion(col=42, score=1.0)
        assert suggestion2.col == 42

        # Tuple column
        suggestion3 = FeatureSuggestion(col=("A", "B"), score=1.0)
        assert suggestion3.col == ("A", "B")

    def test_debug_information(self):
        """Test storing and retrieving debug information."""
        debug_info = {
            "method": "greedy",
            "candidates_considered": 5,
            "time_ms": 12.3,
            "additional_metrics": {"precision": 0.95},
        }

        suggestion = FeatureSuggestion(col="test_col", score=1.5, debug=debug_info)

        assert suggestion.debug == debug_info
        assert suggestion.debug["method"] == "greedy"
        assert suggestion.debug["additional_metrics"]["precision"] == 0.95
