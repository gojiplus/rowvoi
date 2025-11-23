"""Tests for rowvoi.logical module."""

import pandas as pd

from rowvoi.logical import is_key, minimal_key_exact, minimal_key_greedy


class TestIsKey:
    """Tests for is_key function."""

    def test_is_key_true(self, sample_df):
        """Test when columns form a valid key."""
        # Column B distinguishes rows 0 and 1
        assert is_key(sample_df, ["B"], [0, 1]) is True

        # Columns A and B together distinguish all rows
        assert is_key(sample_df, ["A", "B"], [0, 1, 2, 3]) is True

    def test_is_key_false(self, sample_df):
        """Test when columns don't form a valid key."""
        # Column A doesn't distinguish rows 0 and 1 (both have A=1)
        assert is_key(sample_df, ["A"], [0, 1]) is False

        # Column C doesn't distinguish rows 0 and 2 if they had same value
        # Let's create a case where this fails
        df = sample_df.copy()
        df.loc[0, "C"] = df.loc[2, "C"]  # Make C values same for rows 0 and 2
        assert is_key(df, ["C"], [0, 2]) is False

    def test_is_key_single_row(self, sample_df):
        """Test with single row (should always be True)."""
        assert is_key(sample_df, ["A"], [0]) is True
        assert is_key(sample_df, [], [0]) is True

    def test_is_key_empty_columns(self, sample_df):
        """Test with empty column list."""
        assert is_key(sample_df, [], [0]) is True
        assert is_key(sample_df, [], [0, 1]) is False


class TestMinimalKeyExact:
    """Tests for minimal_key_exact function."""

    def test_minimal_key_exact_simple(self, sample_df):
        """Test finding exact minimal key."""
        # For rows 0 and 1, column B should be sufficient
        key = minimal_key_exact(sample_df, [0, 1])
        assert "B" in key
        assert len(key) == 1

    def test_minimal_key_exact_no_solution(self):
        """Test when no key exists (identical rows)."""
        df = pd.DataFrame({"A": [1, 1], "B": [2, 2]})
        key = minimal_key_exact(df, [0, 1])
        assert key is None

    def test_minimal_key_exact_single_row(self, sample_df):
        """Test with single row."""
        key = minimal_key_exact(sample_df, [0])
        assert key == []


class TestMinimalKeyGreedy:
    """Tests for minimal_key_greedy function."""

    def test_minimal_key_greedy_simple(self, sample_df):
        """Test greedy minimal key finding."""
        key = minimal_key_greedy(sample_df, [0, 1])
        assert isinstance(key, list)
        assert len(key) >= 1
        # Verify it's actually a valid key
        assert is_key(sample_df, key, [0, 1])

    def test_minimal_key_greedy_vs_exact(self, sample_df):
        """Test that greedy gives reasonable results."""
        exact_key = minimal_key_exact(sample_df, [0, 1, 2])
        greedy_key = minimal_key_greedy(sample_df, [0, 1, 2])

        # Both should be valid keys
        if exact_key is not None:
            assert is_key(sample_df, exact_key, [0, 1, 2])
        assert is_key(sample_df, greedy_key, [0, 1, 2])

        # Greedy might not be optimal, but should be reasonable
        if exact_key is not None:
            assert len(greedy_key) <= len(exact_key) + 2  # Allow some slack

    def test_minimal_key_greedy_no_solution(self):
        """Test greedy when no solution exists."""
        df = pd.DataFrame({"A": [1, 1], "B": [2, 2]})
        key = minimal_key_greedy(df, [0, 1])
        # Greedy will return all columns even if no solution exists
        assert key == ["A", "B"]

    def test_minimal_key_greedy_single_row(self, sample_df):
        """Test greedy with single row."""
        key = minimal_key_greedy(sample_df, [0])
        assert key == []
