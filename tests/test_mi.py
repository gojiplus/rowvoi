"""Tests for rowvoi.mi module."""

import pandas as pd

from rowvoi.mi import best_feature_by_candidate_mi, candidate_mi
from rowvoi.types import CandidateState


class TestCandidateMI:
    """Tests for candidate_mi function."""

    def test_candidate_mi_basic(self, sample_df, simple_state):
        """Test basic mutual information calculation."""
        mi = candidate_mi(sample_df, simple_state, "B")
        assert isinstance(mi, float)
        assert mi >= 0  # MI is always non-negative

    def test_candidate_mi_perfect_separation(self):
        """Test MI when column perfectly separates candidates."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        state = CandidateState(
            candidate_rows=[0, 1],
            posterior={0: 0.5, 1: 0.5},
            observed_cols=[],
            observed_values={},
        )

        # Column B perfectly separates the rows
        mi_b = candidate_mi(df, state, "B")
        # Should be approximately log2(2) = 1.0 for perfect separation
        assert mi_b > 0.5  # Should be substantial

    def test_candidate_mi_no_separation(self):
        """Test MI when column provides no separation."""
        df = pd.DataFrame(
            {
                "A": [1, 1],  # Same value for both candidates
                "B": [3, 4],
            }
        )
        state = CandidateState(
            candidate_rows=[0, 1],
            posterior={0: 0.5, 1: 0.5},
            observed_cols=[],
            observed_values={},
        )

        # Column A provides no separation
        mi_a = candidate_mi(df, state, "A")
        assert mi_a == 0.0

    def test_candidate_mi_single_candidate(self, sample_df):
        """Test MI with single candidate (should be 0)."""
        state = CandidateState(
            candidate_rows=[0], posterior={0: 1.0}, observed_cols=[], observed_values={}
        )

        mi = candidate_mi(sample_df, state, "B")
        assert mi == 0.0

    def test_candidate_mi_with_observations(self, sample_df):
        """Test MI calculation with prior observations."""
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 0.5, 1: 0.3, 2: 0.2},
            observed_cols=["A"],
            observed_values={"A": 1},
        )

        mi = candidate_mi(sample_df, state, "B")
        assert isinstance(mi, float)
        assert mi >= 0


class TestBestFeatureByCandidateMI:
    """Tests for best_feature_by_candidate_mi function."""

    def test_best_feature_basic(self, sample_df, simple_state):
        """Test basic best feature selection."""
        best_col = best_feature_by_candidate_mi(sample_df, simple_state)
        assert best_col in sample_df.columns
        assert isinstance(best_col, str)

    def test_best_feature_excludes_observed(self, sample_df):
        """Test that observed columns are excluded."""
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 0.5, 1: 0.3, 2: 0.2},
            observed_cols=["A", "B"],
            observed_values={"A": 1, "B": 3},
        )

        best_col = best_feature_by_candidate_mi(sample_df, state)
        assert best_col not in ["A", "B"]
        assert best_col in ["C", "D"]

    def test_best_feature_all_observed(self, sample_df):
        """Test when all columns are already observed."""
        state = CandidateState(
            candidate_rows=[0, 1],
            posterior={0: 0.5, 1: 0.5},
            observed_cols=["A", "B", "C", "D"],
            observed_values={"A": 1, "B": 3, "C": 5, "D": "x"},
        )

        best_col = best_feature_by_candidate_mi(sample_df, state)
        assert best_col is None

    def test_best_feature_single_candidate(self, sample_df):
        """Test with single candidate."""
        state = CandidateState(
            candidate_rows=[0], posterior={0: 1.0}, observed_cols=[], observed_values={}
        )

        # Should return None since no information gain possible
        best_col = best_feature_by_candidate_mi(sample_df, state)
        assert best_col is None

    def test_best_feature_consistency(self, sample_df, simple_state):
        """Test that the function consistently returns the same result."""
        best_col1 = best_feature_by_candidate_mi(sample_df, simple_state)
        best_col2 = best_feature_by_candidate_mi(sample_df, simple_state)
        assert best_col1 == best_col2
