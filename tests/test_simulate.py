"""Tests for rowvoi.simulate module."""

import pandas as pd
import pytest

from rowvoi.ml import RowVoiModel
from rowvoi.simulate import AcquisitionResult, benchmark_policy, sample_candidate_sets


class TestSampleCandidateSets:
    """Tests for sample_candidate_sets function."""

    def test_sample_basic(self, sample_df):
        """Test basic candidate set sampling."""
        candidate_sets = sample_candidate_sets(
            len(sample_df), subset_size=2, n_samples=5
        )

        assert len(candidate_sets) == 5
        for candidates in candidate_sets:
            assert len(candidates) == 2
            assert all(0 <= idx < len(sample_df) for idx in candidates)
            assert len(set(candidates)) == 2  # No duplicates

    def test_sample_k_larger_than_df(self, sample_df):
        """Test when k is larger than DataFrame size."""
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            sample_candidate_sets(len(sample_df), subset_size=10, n_samples=5)

    def test_sample_empty_df(self):
        """Test sampling from empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, IndexError)):
            sample_candidate_sets(len(empty_df), subset_size=2, n_samples=5)

    def test_sample_k_equals_df_size(self, sample_df):
        """Test when k equals DataFrame size."""
        candidate_sets = sample_candidate_sets(
            len(sample_df), subset_size=len(sample_df), n_samples=3
        )

        # All should be the same (all rows)
        for candidates in candidate_sets:
            assert set(candidates) == set(range(len(sample_df)))

    def test_sample_deterministic_with_seed(self, sample_df):
        """Test reproducibility with random seed."""
        # Use the rng parameter for reproducibility
        import random

        rng1 = random.Random(42)
        sets1 = sample_candidate_sets(
            len(sample_df), subset_size=2, n_samples=5, rng=rng1
        )

        rng2 = random.Random(42)
        sets2 = sample_candidate_sets(
            len(sample_df), subset_size=2, n_samples=5, rng=rng2
        )

        # Should be identical
        assert sets1 == sets2


class TestAcquisitionResult:
    """Tests for AcquisitionResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = AcquisitionResult(
            subset_size=3,
            steps_used=2,
            unique_identified=True,
            optimal_steps=2,
            cols_used=["A", "B"],
        )

        assert result.subset_size == 3
        assert result.steps_used == 2
        assert result.unique_identified is True
        assert result.optimal_steps == 2
        assert result.cols_used == ["A", "B"]


class TestBenchmarkPolicy:
    """Tests for benchmark_policy function."""

    def test_benchmark_basic(self, sample_df):
        """Test basic policy benchmarking."""
        # Create and fit a model
        model = RowVoiModel().fit(sample_df)

        # Test with subset sizes
        subset_sizes = [2, 3]

        result = benchmark_policy(sample_df, model, subset_sizes, n_samples=3)

        # Should return dict keyed by subset size
        assert isinstance(result, dict)
        assert set(result.keys()) == set(subset_sizes)

        # Each subset size should have list of AcquisitionResults
        for size in subset_sizes:
            assert isinstance(result[size], list)
            assert len(result[size]) == 3  # n_samples

            for acq_result in result[size]:
                assert isinstance(acq_result, AcquisitionResult)
                assert acq_result.subset_size == size
                assert isinstance(acq_result.steps_used, int)
                assert isinstance(acq_result.unique_identified, bool)
                assert isinstance(acq_result.cols_used, list)

    def test_benchmark_with_fitted_model(self, sample_df):
        """Test benchmarking with a fitted model."""
        # Fit model to sample data
        model = RowVoiModel(noise=0.1).fit(sample_df)

        subset_sizes = [2]
        result = benchmark_policy(sample_df, model, subset_sizes, n_samples=5)

        # Should have reasonable results
        assert len(result[2]) == 5

        # Most simulations should succeed in reasonable steps
        steps_used = [r.steps_used for r in result[2]]
        assert all(steps >= 0 for steps in steps_used)
        assert all(steps <= len(sample_df.columns) for steps in steps_used)

    def test_benchmark_single_row_subsets(self, sample_df):
        """Test benchmarking with single-row subsets."""
        model = RowVoiModel().fit(sample_df)

        subset_sizes = [1]
        result = benchmark_policy(sample_df, model, subset_sizes, n_samples=3)

        # Single-row subsets should need 0 steps
        for acq_result in result[1]:
            assert acq_result.steps_used == 0
            assert acq_result.unique_identified is True

    def test_benchmark_empty_subset_sizes(self, sample_df):
        """Test benchmarking with empty subset sizes list."""
        model = RowVoiModel().fit(sample_df)

        result = benchmark_policy(sample_df, model, [], n_samples=3)

        # Should return empty dict
        assert result == {}

    def test_benchmark_with_costs(self, sample_df):
        """Test benchmarking with feature costs."""
        model = RowVoiModel().fit(sample_df)

        # Assign different costs to features
        feature_costs = dict.fromkeys(sample_df.columns, 1.0)
        feature_costs["A"] = 10.0  # Make A very expensive

        result = benchmark_policy(
            sample_df,
            model,
            [2],
            n_samples=2,
            objective="mi_over_cost",
            feature_costs=feature_costs,
        )

        # Should still work with costs
        assert len(result[2]) == 2
        for acq_result in result[2]:
            assert isinstance(acq_result, AcquisitionResult)

    def test_benchmark_reproducibility(self, sample_df):
        """Test that benchmarks are reproducible with same random seed."""
        import random

        model = RowVoiModel().fit(sample_df)

        # Run with same seed twice
        rng1 = random.Random(42)
        result1 = benchmark_policy(sample_df, model, [2], n_samples=3, rng=rng1)

        rng2 = random.Random(42)
        result2 = benchmark_policy(sample_df, model, [2], n_samples=3, rng=rng2)

        # Results should be identical
        assert len(result1[2]) == len(result2[2])
        for r1, r2 in zip(result1[2], result2[2], strict=True):
            assert r1.steps_used == r2.steps_used
            assert r1.cols_used == r2.cols_used
            assert r1.unique_identified == r2.unique_identified
