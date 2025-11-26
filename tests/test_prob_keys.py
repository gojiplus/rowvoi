"""Tests for probabilistic key and path algorithms."""

import pandas as pd
import pytest

from rowvoi import (
    RowVoiModel,
)
from rowvoi.prob_keys import (
    estimate_coverage_probability,
    find_key_probabilistic,
    plan_key_path_probabilistic,
)


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing."""
    return pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 1, 2], "C": [1, 1, 1, 1]})


@pytest.fixture
def trained_model(simple_df):
    """Create a trained RowVoiModel."""
    model = RowVoiModel()
    model.fit(simple_df)
    return model


class TestFindKeyProbabilistic:
    """Test probabilistic key finding function."""

    def test_basic_usage(self, simple_df, trained_model):
        """Test basic probabilistic key finding."""
        key = find_key_probabilistic(
            simple_df,
            rows=[0, 1, 2, 3],
            model=trained_model,
            epsilon_posterior=0.1,
            max_steps=3,
        )

        assert isinstance(key, list)
        assert len(key) > 0
        assert all(col in simple_df.columns for col in key)

    def test_with_costs(self, simple_df, trained_model):
        """Test probabilistic key finding with costs."""
        costs = {"A": 10.0, "B": 1.0, "C": 1.0}
        key = find_key_probabilistic(
            simple_df,
            rows=[0, 1, 2, 3],
            model=trained_model,
            costs=costs,
            objective="mi_over_cost",
            max_steps=2,
        )

        assert isinstance(key, list)
        # Should prefer lower cost columns
        if "B" in simple_df.columns:
            assert "B" in key or "C" in key  # Should pick cheaper options

    def test_with_mi_objective(self, simple_df, trained_model):
        """Test with pure MI objective."""
        key = find_key_probabilistic(
            simple_df, rows=[0, 1, 2], model=trained_model, objective="mi", max_steps=2
        )

        assert isinstance(key, list)
        assert len(key) >= 0

    def test_with_column_subset(self, simple_df, trained_model):
        """Test with specific column subset."""
        key = find_key_probabilistic(
            simple_df, rows=[0, 1], model=trained_model, columns=["A", "B"], max_steps=1
        )

        assert isinstance(key, list)
        assert all(col in ["A", "B"] for col in key)

    def test_empty_rows(self, simple_df, trained_model):
        """Test with empty row list."""
        key = find_key_probabilistic(
            simple_df, rows=[], model=trained_model, max_steps=1
        )

        assert isinstance(key, list)

    def test_single_row(self, simple_df, trained_model):
        """Test with single row."""
        key = find_key_probabilistic(
            simple_df, rows=[0], model=trained_model, max_steps=1
        )

        assert isinstance(key, list)


class TestPlanKeyPathProbabilistic:
    """Test probabilistic key path planning."""

    def test_basic_path_planning(self, simple_df, trained_model):
        """Test basic path planning."""
        path = plan_key_path_probabilistic(
            simple_df, rows=[0, 1, 2, 3], model=trained_model
        )

        assert hasattr(path, "steps")
        assert len(path.steps) >= 0
        if path.steps:
            assert hasattr(path.steps[0], "col")
            assert hasattr(path.steps[0], "cumulative_cost")

    def test_with_costs(self, simple_df, trained_model):
        """Test path planning with costs."""
        costs = {"A": 5.0, "B": 1.0, "C": 1.0}
        path = plan_key_path_probabilistic(
            simple_df, rows=[0, 1, 2], model=trained_model, costs=costs
        )

        assert hasattr(path, "steps")
        if path.steps:
            # Should have cumulative costs
            for step in path.steps:
                assert step.cumulative_cost >= 0

    def test_different_objectives(self, simple_df, trained_model):
        """Test different objectives."""
        for objective in ["mi", "mi_over_cost", "expected_entropy_reduction"]:
            path = plan_key_path_probabilistic(
                simple_df, rows=[0, 1], model=trained_model, objective=objective
            )
            assert hasattr(path, "steps")

    def test_with_column_subset(self, simple_df, trained_model):
        """Test with specific columns."""
        path = plan_key_path_probabilistic(
            simple_df, rows=[0, 1, 2], model=trained_model, columns=["A", "B"]
        )

        assert hasattr(path, "steps")
        if path.steps:
            for step in path.steps:
                assert step.col in ["A", "B"]

    def test_empty_rows(self, simple_df, trained_model):
        """Test with empty rows."""
        path = plan_key_path_probabilistic(simple_df, rows=[], model=trained_model)

        assert hasattr(path, "steps")
        assert len(path.steps) == 0


class TestEstimateCoverageProbability:
    """Test coverage probability estimation."""

    def test_deterministic_case(self, simple_df):
        """Test deterministic coverage estimation."""
        prob = estimate_coverage_probability(
            simple_df, rows=[0, 1, 2, 3], cols=["A", "B"]
        )

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_probabilistic_case(self, simple_df, trained_model):
        """Test probabilistic coverage estimation with model."""
        prob = estimate_coverage_probability(
            simple_df, rows=[0, 1, 2, 3], cols=["A", "B"], model=trained_model
        )

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_single_column(self, simple_df, trained_model):
        """Test with single column."""
        prob = estimate_coverage_probability(
            simple_df, rows=[0, 1], cols=["B"], model=trained_model
        )

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_no_information_column(self, simple_df, trained_model):
        """Test with uninformative column."""
        prob = estimate_coverage_probability(
            simple_df,
            rows=[0, 1, 2, 3],
            cols=["C"],  # All same value
            model=trained_model,
        )

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_empty_rows(self, simple_df, trained_model):
        """Test with empty rows."""
        prob = estimate_coverage_probability(
            simple_df, rows=[], cols=["A"], model=trained_model
        )

        assert prob == 1.0  # No rows to distinguish

    def test_single_row(self, simple_df, trained_model):
        """Test with single row."""
        prob = estimate_coverage_probability(
            simple_df, rows=[0], cols=["A"], model=trained_model
        )

        assert prob == 1.0  # Single row always covered

    def test_perfect_separator(self, simple_df, trained_model):
        """Test with columns that perfectly separate rows."""
        prob = estimate_coverage_probability(
            simple_df,
            rows=[0, 1],  # These should be distinguishable by B
            cols=["B"],
            model=trained_model,
        )

        assert isinstance(prob, float)
        assert prob > 0.0  # Should have some disambiguation power


class TestModelIntegration:
    """Test integration with RowVoiModel."""

    def test_model_not_fit_error(self, simple_df):
        """Test that functions handle unfitted models appropriately."""
        unfitted_model = RowVoiModel()

        # These should handle unfitted models gracefully
        # by creating initial states that work
        try:
            find_key_probabilistic(
                simple_df, rows=[0, 1], model=unfitted_model, max_steps=1
            )
        except Exception as e:
            # Should get a meaningful error about unfitted model
            assert "fit" in str(e).lower()

    def test_different_model_parameters(self, simple_df):
        """Test with different model configurations."""
        # Model with noise
        noisy_model = RowVoiModel(noise=0.1, smoothing=1e-3)
        noisy_model.fit(simple_df)

        key = find_key_probabilistic(
            simple_df, rows=[0, 1, 2], model=noisy_model, max_steps=2
        )

        assert isinstance(key, list)

        # Model without normalization
        unnorm_model = RowVoiModel(normalize_cols=False)
        unnorm_model.fit(simple_df)

        key2 = find_key_probabilistic(
            simple_df, rows=[0, 1, 2], model=unnorm_model, max_steps=2
        )

        assert isinstance(key2, list)
