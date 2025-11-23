"""Tests for rowvoi.simulate module."""

import numpy as np
import pandas as pd
import pytest

from rowvoi.simulate import AcquisitionResult, benchmark_policy, sample_candidate_sets


class TestSampleCandidateSets:
    """Tests for sample_candidate_sets function."""

    def test_sample_basic(self, sample_df):
        """Test basic candidate set sampling."""
        candidate_sets = sample_candidate_sets(sample_df, k=2, n_samples=5)

        assert len(candidate_sets) == 5
        for candidates in candidate_sets:
            assert len(candidates) == 2
            assert all(0 <= idx < len(sample_df) for idx in candidates)
            assert len(set(candidates)) == 2  # No duplicates

    def test_sample_k_larger_than_df(self, sample_df):
        """Test when k is larger than DataFrame size."""
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            sample_candidate_sets(sample_df, k=10, n_samples=5)

    def test_sample_empty_df(self):
        """Test sampling from empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, IndexError)):
            sample_candidate_sets(empty_df, k=2, n_samples=5)

    def test_sample_k_equals_df_size(self, sample_df):
        """Test when k equals DataFrame size."""
        candidate_sets = sample_candidate_sets(sample_df, k=len(sample_df), n_samples=3)

        # All should be the same (all rows)
        for candidates in candidate_sets:
            assert set(candidates) == set(range(len(sample_df)))

    def test_sample_deterministic_with_seed(self, sample_df):
        """Test reproducibility with random seed."""
        # This would require the function to accept a random seed parameter
        # or we'd need to set numpy random seed before calling
        np.random.seed(42)
        sets1 = sample_candidate_sets(sample_df, k=2, n_samples=5)

        np.random.seed(42)
        sets2 = sample_candidate_sets(sample_df, k=2, n_samples=5)

        # Should be identical
        assert sets1 == sets2


class TestAcquisitionResult:
    """Tests for AcquisitionResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = AcquisitionResult(
            mean_queries=3.5,
            median_queries=3,
            total_queries=35,
            success_rate=0.9,
            query_counts=[2, 3, 3, 4, 5],
        )

        assert result.mean_queries == 3.5
        assert result.median_queries == 3
        assert result.total_queries == 35
        assert result.success_rate == 0.9
        assert result.query_counts == [2, 3, 3, 4, 5]


class TestBenchmarkPolicy:
    """Tests for benchmark_policy function."""

    def test_benchmark_basic(self, sample_df):
        """Test basic policy benchmarking."""
        # Create some candidate sets
        candidate_sets = [[0, 1], [1, 2], [0, 3]]

        # Simple policy that always returns first available column
        def simple_policy(df, state):
            available_cols = [
                col for col in df.columns if col not in state.observed_cols
            ]
            return available_cols[0] if available_cols else None

        result = benchmark_policy(sample_df, candidate_sets, simple_policy)

        assert isinstance(result, AcquisitionResult)
        assert isinstance(result.mean_queries, float)
        assert isinstance(result.median_queries, (int, float))
        assert isinstance(result.total_queries, int)
        assert isinstance(result.success_rate, float)
        assert 0 <= result.success_rate <= 1
        assert len(result.query_counts) == len(candidate_sets)

    def test_benchmark_optimal_policy(self, sample_df):
        """Test benchmarking with a policy that should work well."""
        from rowvoi.mi import best_feature_by_candidate_mi

        candidate_sets = [[0, 1], [2, 3]]

        def mi_policy(df, state):
            return best_feature_by_candidate_mi(df, state)

        result = benchmark_policy(sample_df, candidate_sets, mi_policy)

        # MI-based policy should have reasonable performance
        assert result.success_rate > 0
        assert result.mean_queries > 0

    def test_benchmark_failing_policy(self, sample_df):
        """Test benchmarking with a policy that always fails."""
        candidate_sets = [[0, 1], [2, 3]]

        def failing_policy(df, state):
            # Always return None (no suggestion)
            return None

        result = benchmark_policy(sample_df, candidate_sets, failing_policy)

        # Should have low success rate
        assert result.success_rate == 0.0

    def test_benchmark_empty_candidate_sets(self, sample_df):
        """Test benchmarking with empty candidate sets list."""

        def simple_policy(df, state):
            return "A"

        result = benchmark_policy(sample_df, [], simple_policy)

        # Should handle empty input gracefully
        assert result.mean_queries == 0
        assert result.total_queries == 0
        assert result.success_rate == 1.0  # Vacuous success
        assert len(result.query_counts) == 0

    def test_benchmark_single_row_candidates(self, sample_df):
        """Test benchmarking with single-row candidate sets."""
        candidate_sets = [[0], [1], [2]]

        def simple_policy(df, state):
            return "A"

        result = benchmark_policy(sample_df, candidate_sets, simple_policy)

        # Single row candidates should always succeed immediately
        assert result.success_rate == 1.0
        assert all(count == 0 for count in result.query_counts)

    def test_benchmark_statistics_consistency(self, sample_df):
        """Test that benchmark statistics are consistent."""
        candidate_sets = [[0, 1], [1, 2], [2, 3]]

        def simple_policy(df, state):
            available_cols = [
                col for col in df.columns if col not in state.observed_cols
            ]
            return available_cols[0] if available_cols else None

        result = benchmark_policy(sample_df, candidate_sets, simple_policy)

        # Check consistency
        assert result.total_queries == sum(result.query_counts)
        assert (
            abs(result.mean_queries - (result.total_queries / len(candidate_sets)))
            < 1e-10
        )

        sorted_counts = sorted(result.query_counts)
        n = len(sorted_counts)
        if n % 2 == 1:
            expected_median = sorted_counts[n // 2]
        else:
            expected_median = (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2
        assert abs(result.median_queries - expected_median) < 1e-10
