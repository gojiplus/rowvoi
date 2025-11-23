"""Shared test fixtures for rowvoi tests."""

import pandas as pd
import pytest

from rowvoi.types import CandidateState


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1, 1, 2, 3],
            "B": [3, 4, 3, 4],
            "C": [5, 6, 7, 8],
            "D": ["x", "y", "x", "z"],
        }
    )


@pytest.fixture
def simple_state():
    """Return simple candidate state for testing."""
    return CandidateState(
        candidate_rows=[0, 1],
        posterior={0: 0.6, 1: 0.4},
        observed_cols=[],
        observed_values={},
    )


@pytest.fixture
def multi_state():
    """Multi-candidate state for testing."""
    return CandidateState(
        candidate_rows=[0, 1, 2],
        posterior={0: 0.5, 1: 0.3, 2: 0.2},
        observed_cols=["A"],
        observed_values={"A": 1},
    )
