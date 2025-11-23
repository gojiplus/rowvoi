"""Tests for rowvoi.types module."""

from rowvoi.types import CandidateState


class TestCandidateState:
    """Tests for CandidateState dataclass."""

    def test_creation(self):
        """Test basic creation of CandidateState."""
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 0.5, 1: 0.3, 2: 0.2},
            observed_cols=["A", "B"],
            observed_values={"A": 1, "B": "x"},
        )

        assert state.candidate_rows == [0, 1, 2]
        assert state.posterior == {0: 0.5, 1: 0.3, 2: 0.2}
        assert state.observed_cols == ["A", "B"]
        assert state.observed_values == {"A": 1, "B": "x"}

    def test_empty_state(self):
        """Test creation with minimal arguments."""
        state = CandidateState(
            candidate_rows=[], posterior={}, observed_cols=[], observed_values={}
        )

        assert state.candidate_rows == []
        assert state.posterior == {}
        assert state.observed_cols == []
        assert state.observed_values == {}

    def test_equality(self):
        """Test equality comparison of CandidateState objects."""
        state1 = CandidateState(
            candidate_rows=[0, 1],
            posterior={0: 0.6, 1: 0.4},
            observed_cols=["A"],
            observed_values={"A": 1},
        )

        state2 = CandidateState(
            candidate_rows=[0, 1],
            posterior={0: 0.6, 1: 0.4},
            observed_cols=["A"],
            observed_values={"A": 1},
        )

        state3 = CandidateState(
            candidate_rows=[0, 1],
            posterior={0: 0.5, 1: 0.5},
            observed_cols=["A"],
            observed_values={"A": 1},
        )

        assert state1 == state2
        assert state1 != state3
