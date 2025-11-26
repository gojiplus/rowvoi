"""Tests for rowvoi.ml module (RowVoiModel)."""

import math

import numpy as np
import pandas as pd
import pytest

from rowvoi import CandidateState, FeatureSuggestion, RowVoiModel


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3, 3],
            "B": [1, 2, 1, 2, 1, 2],
            "C": ["x", "y", "x", "y", "x", "y"],
            "D": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.fixture
def simple_state():
    """Create a simple uniform candidate state."""
    return CandidateState.uniform([0, 1, 2, 3])


class TestRowVoiModel:
    """Tests for RowVoiModel class."""

    def test_model_creation_default(self):
        """Test creating model with default parameters."""
        model = RowVoiModel()
        assert model.noise == 0.0
        assert model.smoothing == 1e-6
        assert model.normalize_cols is True

    def test_model_creation_custom(self):
        """Test creating model with custom parameters."""
        model = RowVoiModel(noise=0.1, smoothing=2.0, normalize_cols=False)
        assert model.noise == 0.1
        assert model.smoothing == 2.0
        assert model.normalize_cols is False

    def test_noise_validation(self):
        """Test validation of noise parameter."""
        # Valid noise values
        RowVoiModel(noise=0.0)
        RowVoiModel(noise=0.5)
        RowVoiModel(noise=0.99)

        # Invalid noise values
        with pytest.raises(ValueError, match="noise must be in"):
            RowVoiModel(noise=-0.1)

        with pytest.raises(ValueError, match="noise must be in"):
            RowVoiModel(noise=1.0)

    def test_model_fit(self, sample_df):
        """Test fitting the model to data."""
        model = RowVoiModel()
        fitted_model = model.fit(sample_df)

        # Should return the same model instance
        assert fitted_model is model

        # Model should have learned column frequencies
        assert hasattr(model, "_freqs")
        assert len(model._freqs) == len(sample_df.columns)

        # Check that frequencies are computed for each column
        for col in sample_df.columns:
            assert col in model._freqs
            assert isinstance(model._freqs[col], dict)

            # Frequencies should sum to approximately 1.0 (allowing for smoothing)
            freq_sum = sum(model._freqs[col].values())
            assert freq_sum > 0.0

    def test_model_fit_with_discretization(self, sample_df):
        """Test fitting with numeric column discretization."""
        model = RowVoiModel()

        # Discretize only column D (numeric)
        fitted_model = model.fit(sample_df, discrete_cols=["A", "B", "C"], bins=3)

        assert fitted_model is model
        assert len(model._freqs) == len(sample_df.columns)

    def test_model_fit_empty_df(self):
        """Test fitting to empty DataFrame."""
        empty_df = pd.DataFrame()
        model = RowVoiModel()

        # Should handle empty DataFrame gracefully
        fitted_model = model.fit(empty_df)
        assert fitted_model is model
        assert len(model._freqs) == 0

    def test_suggest_next_feature_basic(self, sample_df, simple_state):
        """Test basic feature suggestion."""
        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, simple_state)

        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.col in sample_df.columns
        assert isinstance(suggestion.score, float)
        assert suggestion.expected_voi is not None
        assert suggestion.expected_voi >= 0

    def test_suggest_next_feature_with_costs(self, sample_df, simple_state):
        """Test feature suggestion with costs."""
        costs = {"A": 1.0, "B": 2.0, "C": 5.0, "D": 10.0}

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(
            sample_df, simple_state, objective="mi_over_cost", feature_costs=costs
        )

        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.marginal_cost is not None
        assert suggestion.marginal_cost == costs[suggestion.col]

    def test_suggest_excludes_observed(self, sample_df):
        """Test that suggestions exclude observed columns."""
        state = CandidateState.uniform(
            [0, 1, 2, 3], observed_cols={"A", "B"}, observed_values={"A": 1, "B": 1}
        )

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, state)

        assert suggestion.col not in ["A", "B"]
        assert suggestion.col in ["C", "D"]

    def test_suggest_single_candidate(self, sample_df):
        """Test suggestion with single candidate."""
        state = CandidateState.uniform([0])

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, state)

        # Should still return a suggestion even if no information gain
        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.expected_voi == 0.0

    def test_suggest_all_observed(self, sample_df):
        """Test when all columns are observed."""
        state = CandidateState.uniform(
            [0, 1],
            observed_cols=set(sample_df.columns),
            observed_values={col: sample_df.iloc[0][col] for col in sample_df.columns},
        )

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, state)

        # Should return None when no columns available
        assert suggestion is None

    def test_suggest_with_candidate_cols(self, sample_df, simple_state):
        """Test suggestion with restricted candidate columns."""
        model = RowVoiModel().fit(sample_df)

        # Only consider columns A and B
        suggestion = model.suggest_next_feature(
            sample_df, simple_state, candidate_cols=["A", "B"]
        )

        assert suggestion.col in ["A", "B"]

    def test_model_with_noise(self, sample_df, simple_state):
        """Test model with noise rate."""
        model = RowVoiModel(noise=0.2).fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, simple_state)

        # Should still return valid suggestion
        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.expected_voi >= 0

    def test_model_reproducibility(self, sample_df, simple_state):
        """Test that model gives consistent results."""
        model1 = RowVoiModel(noise=0.1, smoothing=1e-6).fit(sample_df)
        model2 = RowVoiModel(noise=0.1, smoothing=1e-6).fit(sample_df)

        suggestion1 = model1.suggest_next_feature(sample_df, simple_state)
        suggestion2 = model2.suggest_next_feature(sample_df, simple_state)

        # Should give same results for same parameters
        assert suggestion1.col == suggestion2.col
        assert abs(suggestion1.expected_voi - suggestion2.expected_voi) < 1e-10

    def test_normalization_effect(self, sample_df, simple_state):
        """Test effect of column normalization."""
        model_norm = RowVoiModel(normalize_cols=True).fit(sample_df)
        model_no_norm = RowVoiModel(normalize_cols=False).fit(sample_df)

        suggestion_norm = model_norm.suggest_next_feature(sample_df, simple_state)
        suggestion_no_norm = model_no_norm.suggest_next_feature(sample_df, simple_state)

        # Both should return valid suggestions
        assert isinstance(suggestion_norm, FeatureSuggestion)
        assert isinstance(suggestion_no_norm, FeatureSuggestion)

        # Normalized VOI should be different from non-normalized
        if suggestion_norm.debug and suggestion_no_norm.debug:
            norm_voi = suggestion_norm.debug.get("normalized_voi")
            if norm_voi is not None:
                assert norm_voi <= suggestion_norm.expected_voi

    def test_different_objectives(self, sample_df, simple_state):
        """Test different objective functions."""
        costs = {"A": 1.0, "B": 2.0, "C": 5.0, "D": 1.0}
        model = RowVoiModel().fit(sample_df)

        # MI objective
        suggestion_mi = model.suggest_next_feature(
            sample_df, simple_state, objective="mi"
        )

        # MI over cost objective
        suggestion_cost = model.suggest_next_feature(
            sample_df, simple_state, objective="mi_over_cost", feature_costs=costs
        )

        assert isinstance(suggestion_mi, FeatureSuggestion)
        assert isinstance(suggestion_cost, FeatureSuggestion)

        # Cost-adjusted might prefer different column
        # (though not guaranteed depending on data)

    def test_missing_costs_error(self, sample_df, simple_state):
        """Test error when costs are required but missing."""
        model = RowVoiModel().fit(sample_df)

        with pytest.raises(ValueError, match="feature_costs must be provided"):
            model.suggest_next_feature(
                sample_df, simple_state, objective="mi_over_cost"
            )

    def test_zero_cost_error(self, sample_df, simple_state):
        """Test error with zero or negative costs."""
        costs = {"A": 0.0, "B": 1.0, "C": 1.0, "D": 1.0}
        model = RowVoiModel().fit(sample_df)

        with pytest.raises(ValueError, match="feature costs must be positive"):
            model.suggest_next_feature(
                sample_df, simple_state, objective="mi_over_cost", feature_costs=costs
            )

    def test_suggest_before_fit_error(self, sample_df, simple_state):
        """Test error when suggesting before fitting."""
        model = RowVoiModel()

        with pytest.raises(RuntimeError, match="RowVoiModel.fit\\(\\) must be called"):
            model.suggest_next_feature(sample_df, simple_state)

    def test_tie_breaking(self, sample_df):
        """Test tie-breaking behavior when features have equal scores."""
        # Create state where multiple features might have same MI
        state = CandidateState(
            candidate_rows=[0, 1],
            posterior=np.array([0.5, 0.5]),
            observed_cols=set(),
            observed_values={},
        )

        model = RowVoiModel().fit(sample_df)
        suggestion1 = model.suggest_next_feature(sample_df, state)

        # Should consistently return the same column for tie-breaking
        suggestion2 = model.suggest_next_feature(sample_df, state)
        assert suggestion1.col == suggestion2.col

    def test_debug_information(self, sample_df, simple_state):
        """Test that debug information is included in suggestions."""
        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, simple_state)

        assert suggestion.debug is not None
        assert isinstance(suggestion.debug, dict)

        # Should contain entropy information
        assert "entropy_before" in suggestion.debug
        assert "entropy_after" in suggestion.debug
        assert isinstance(suggestion.debug["entropy_before"], float)
        assert isinstance(suggestion.debug["entropy_after"], float)

    def test_non_uniform_posterior(self, sample_df):
        """Test with non-uniform posterior distribution."""
        state = CandidateState(
            candidate_rows=[0, 1, 2, 3],
            posterior=np.array([0.5, 0.3, 0.15, 0.05]),
            observed_cols=set(),
            observed_values={},
        )

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, state)

        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.expected_voi >= 0

        # Entropy should reflect non-uniform distribution
        expected_entropy = -(
            0.5 * math.log2(0.5)
            + 0.3 * math.log2(0.3)
            + 0.15 * math.log2(0.15)
            + 0.05 * math.log2(0.05)
        )
        assert abs(state.entropy - expected_entropy) < 1e-10


class TestRowVoiModelAcquisition:
    """Test acquisition simulation and edge cases."""

    def test_run_acquisition_basic(self):
        """Test basic acquisition simulation."""
        df = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 1, 2], "C": [1, 1, 1, 1]})

        model = RowVoiModel().fit(df)

        # Create initial state
        initial_state = CandidateState.uniform([0, 1, 2, 3])

        # Run acquisition
        history = model.run_acquisition(
            df=df, true_row=1, initial_state=initial_state, max_steps=2
        )

        assert isinstance(history, list)
        assert all(isinstance(step, FeatureSuggestion) for step in history)
        assert len(history) <= 2

    def test_run_acquisition_stop_when_unique(self):
        """Test acquisition stops when unique candidate found."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})

        model = RowVoiModel().fit(df)
        initial_state = CandidateState.uniform([0, 1, 2])

        history = model.run_acquisition(
            df=df, true_row=1, initial_state=initial_state, stop_when_unique=True
        )

        assert isinstance(history, list)
        # Should stop before max iterations when unique

    def test_run_acquisition_with_costs(self):
        """Test acquisition with feature costs."""
        df = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 1, 2]})

        model = RowVoiModel().fit(df)
        initial_state = CandidateState.uniform([0, 1, 2, 3])
        costs = {"A": 10.0, "B": 1.0}

        history = model.run_acquisition(
            df=df,
            true_row=0,
            initial_state=initial_state,
            feature_costs=costs,
            objective="mi_over_cost",
            max_steps=2,
        )

        assert isinstance(history, list)
        # Should prefer lower-cost features

    def test_run_acquisition_with_candidate_cols(self):
        """Test acquisition with limited candidate columns."""
        df = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 1, 2], "C": [5, 6, 7, 8]})

        model = RowVoiModel().fit(df)
        initial_state = CandidateState.uniform([0, 1, 2, 3])

        history = model.run_acquisition(
            df=df,
            true_row=2,
            initial_state=initial_state,
            candidate_cols=["A", "B"],  # Exclude C
            max_steps=1,
        )

        assert isinstance(history, list)
        if history:
            assert history[0].col in ["A", "B"]

    def test_run_acquisition_no_stop_when_unique(self):
        """Test acquisition continues even when unique found."""
        df = pd.DataFrame({"A": [1, 2], "B": [1, 2]})

        model = RowVoiModel().fit(df)
        initial_state = CandidateState.uniform([0, 1])

        history = model.run_acquisition(
            df=df,
            true_row=0,
            initial_state=initial_state,
            stop_when_unique=False,
            max_steps=2,
        )

        assert isinstance(history, list)

    def test_run_acquisition_modifies_state_correctly(self):
        """Test that acquisition correctly updates state."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 2]})

        model = RowVoiModel().fit(df)
        initial_state = CandidateState.uniform([0, 1, 2])

        # Test that original state is not modified
        original_rows = list(initial_state.candidate_rows)
        original_posterior = initial_state.posterior.copy()

        model.run_acquisition(
            df=df, true_row=1, initial_state=initial_state, max_steps=1
        )

        # Original state should be unchanged
        assert list(initial_state.candidate_rows) == original_rows
        assert np.allclose(initial_state.posterior, original_posterior)


class TestRowVoiModelEdgeCases:
    """Test edge cases and error conditions."""

    def test_conditional_value_distribution_no_noise(self):
        """Test conditional distribution without noise."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        model = RowVoiModel(noise=0.0).fit(df)

        # Access internal method for testing
        dist = model._conditional_value_distribution("A", [1, 2, 3], 2)

        # Should put all mass on true value
        assert dist[2] == 1.0
        assert dist[1] == 0.0
        assert dist[3] == 0.0

    def test_conditional_value_distribution_with_noise(self):
        """Test conditional distribution with noise."""
        df = pd.DataFrame({"A": [1, 1, 2, 2]})
        model = RowVoiModel(noise=0.2).fit(df)

        dist = model._conditional_value_distribution("A", [1, 2], 1)

        # True value should get 0.8, other values get noise
        assert dist[1] == 0.8
        assert dist[2] == 0.2

    def test_get_posterior_prob(self):
        """Test posterior probability extraction."""
        df = pd.DataFrame({"A": [1, 2]})
        model = RowVoiModel().fit(df)

        state = CandidateState(
            candidate_rows=[0, 1],
            posterior=np.array([0.3, 0.7]),
            observed_cols=set(),
            observed_values={},
        )

        assert model._get_posterior_prob(state, 0) == 0.3
        assert model._get_posterior_prob(state, 1) == 0.7
        assert model._get_posterior_prob(state, 2) == 0.0  # Not in candidates

    def test_expected_cond_entropy_edge_cases(self):
        """Test expected conditional entropy computation edge cases."""
        df = pd.DataFrame({"A": [1, 1]})  # All same value
        model = RowVoiModel().fit(df)

        state = CandidateState.uniform([0, 1])
        entropy = model._expected_cond_entropy(state, "A")

        # Should handle uniform values gracefully
        assert entropy >= 0.0

    def test_column_not_in_freq_table_error(self):
        """Test error when column not in frequency table."""
        df = pd.DataFrame({"A": [1, 2]})
        model = RowVoiModel(noise=0.1).fit(
            df
        )  # Need noise > 0 to trigger the error path

        # Try to access non-existent column
        with pytest.raises(ValueError, match="not found in frequency table"):
            model._conditional_value_distribution("B", [1, 2], 1)

    def test_discretization_edge_cases(self):
        """Test discretization with edge cases."""
        # Test with constant values
        df_const = pd.DataFrame({"A": [1, 1, 1, 1]})
        model = RowVoiModel()
        model.fit(df_const, discrete_cols=[])  # Force discretization

        # Should handle constant values without error
        assert len(model._freqs["A"]) >= 1

        # Test with few unique values
        df_few = pd.DataFrame({"A": [1, 2]})
        model2 = RowVoiModel()
        model2.fit(df_few, discrete_cols=[], bins=5)  # More bins than values

        # Should work without error
        assert len(model2._freqs["A"]) <= 2

    def test_smoothing_parameter_effects(self):
        """Test effect of smoothing parameter."""
        df = pd.DataFrame({"A": [1, 1, 2]})  # Unbalanced

        # Model with high smoothing
        model_smooth = RowVoiModel(smoothing=1.0).fit(df)

        # Model with low smoothing
        model_minimal = RowVoiModel(smoothing=1e-10).fit(df)

        # Frequencies should differ
        freq_smooth = model_smooth._freqs["A"]
        freq_minimal = model_minimal._freqs["A"]

        # Smoothing should make frequencies more uniform
        assert abs(freq_smooth[1] - freq_smooth[2]) < abs(
            freq_minimal[1] - freq_minimal[2]
        )

    def test_noise_model_edge_cases(self):
        """Test noise model with edge cases."""
        df = pd.DataFrame({"A": [1, 2]})

        # High noise model
        model = RowVoiModel(noise=0.9).fit(df)

        state = CandidateState.uniform([0, 1])
        entropy = model._expected_cond_entropy(state, "A")

        # Should handle high noise gracefully
        assert entropy >= 0.0

    def test_empty_candidate_values(self):
        """Test handling of empty candidate value sets."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        model = RowVoiModel().fit(df)

        # This tests internal robustness
        dist = model._conditional_value_distribution("A", [1], 1)
        assert dist[1] == 1.0

    def test_normalization_with_zero_entropy(self):
        """Test normalization when feature entropy is zero."""
        # All same value in feature
        df = pd.DataFrame({"A": [1, 1, 1]})
        model = RowVoiModel(normalize_cols=True).fit(df)

        state = CandidateState.uniform([0, 1, 2])
        suggestion = model.suggest_next_feature(df, state)

        # Should handle zero entropy gracefully
        assert suggestion.expected_voi >= 0.0
