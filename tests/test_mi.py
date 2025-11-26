"""Tests for mutual information calculations (now in policies module)."""

import numpy as np
import pandas as pd
import pytest

from rowvoi import CandidateMIPolicy, CandidateState


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing MI calculations."""
    return pd.DataFrame(
        {
            "A": [1, 1, 2, 2],
            "B": [1, 2, 1, 2],
            "C": [1, 1, 1, 1],
        }
    )


@pytest.fixture
def candidate_state():
    """Create a uniform candidate state."""
    return CandidateState.uniform([0, 1, 2, 3])


class TestCandidateMIPolicy:
    """Test the CandidateMIPolicy for MI calculations."""

    def test_basic_mi_calculation(self, simple_df):
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = CandidateMIPolicy()

        suggestion = policy.suggest(simple_df, state)

        assert suggestion.col in simple_df.columns
        assert suggestion.expected_voi is not None
        assert suggestion.expected_voi >= 0

    def test_perfect_separator(self, simple_df):
        # Column B separates 4 candidates into 2 groups of 2 each
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = CandidateMIPolicy()

        suggestion = policy.suggest(simple_df, state, candidate_cols=["B"])

        # B should have 1 bit of MI: H(Y) - H(Y|B) = 2 - 1 = 1 bit
        assert np.isclose(suggestion.expected_voi, 1.0, rtol=0.1)

    def test_no_information_column(self, simple_df):
        # Column C has same value for all rows
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = CandidateMIPolicy()

        suggestion = policy.suggest(simple_df, state, candidate_cols=["C"])

        # C should have zero MI
        assert suggestion.expected_voi == 0.0

    def test_partial_information(self, simple_df):
        # Column A partially separates rows
        state = CandidateState.uniform([0, 1, 2, 3])
        policy = CandidateMIPolicy()

        suggestion = policy.suggest(simple_df, state, candidate_cols=["A"])

        # A should have some MI but not maximum
        assert 0 < suggestion.expected_voi < 2.0

    def test_normalized_mi(self):
        df = pd.DataFrame(
            {
                "high_card": list(range(10)),
                "low_card": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            }
        )

        state = CandidateState.uniform(list(range(10)))

        # Without normalization
        policy_unnorm = CandidateMIPolicy(normalize=False)
        suggestion_unnorm = policy_unnorm.suggest(df, state)

        # With normalization
        policy_norm = CandidateMIPolicy(normalize=True)
        suggestion_norm = policy_norm.suggest(df, state)

        # Both should suggest high_card (perfect separator)
        assert suggestion_unnorm.col == "high_card"
        assert suggestion_norm.col == "high_card"

    def test_with_costs(self, simple_df):
        state = CandidateState.uniform([0, 1, 2, 3])
        costs = {"A": 10.0, "B": 1.0, "C": 1.0}

        policy = CandidateMIPolicy(costs=costs)
        suggestion = policy.suggest(simple_df, state)

        # Should prefer B (high MI, low cost) over A (medium MI, high cost)
        assert suggestion.col == "B"

    def test_observed_columns_excluded(self, simple_df):
        # Create state with column A already observed
        state = CandidateState.uniform(
            [0, 1, 2, 3], observed_cols={"A"}, observed_values={"A": 1}
        )

        policy = CandidateMIPolicy()
        suggestion = policy.suggest(simple_df, state)

        # Should not suggest A again
        assert suggestion.col != "A"
        assert suggestion.col in ["B", "C"]

    def test_single_candidate(self, simple_df):
        # Single candidate has no uncertainty
        state = CandidateState.uniform([0])
        policy = CandidateMIPolicy()

        suggestion = policy.suggest(simple_df, state)

        # Should return something but MI should be 0
        assert suggestion.col is not None
        assert suggestion.expected_voi == 0.0

    def test_two_candidates(self):
        df = pd.DataFrame(
            {
                "X": [1, 2, 3],
                "Y": [1, 1, 2],
            }
        )

        state = CandidateState.uniform([0, 1])
        policy = CandidateMIPolicy()

        suggestion = policy.suggest(df, state)

        # X perfectly separates rows 0 and 1
        assert suggestion.col == "X"
        assert np.isclose(suggestion.expected_voi, 1.0)  # log2(2) = 1 bit

    def test_non_uniform_posterior(self):
        df = pd.DataFrame(
            {
                "A": [1, 1, 2],
                "B": [1, 2, 3],
            }
        )

        # Create non-uniform posterior
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior=np.array([0.5, 0.3, 0.2]),
            observed_cols=set(),
            observed_values={},
        )

        policy = CandidateMIPolicy()
        suggestion = policy.suggest(df, state)

        # B should have higher MI as it perfectly separates
        assert suggestion.col == "B"
        assert suggestion.expected_voi > 0
