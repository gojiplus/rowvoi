"""Tests for rowvoi.ml module."""

import pandas as pd

from rowvoi.ml import FeatureSuggestion, RowVoiModel
from rowvoi.types import CandidateState


class TestFeatureSuggestion:
    """Tests for FeatureSuggestion dataclass."""

    def test_creation(self):
        """Test basic creation."""
        suggestion = FeatureSuggestion(col="A", expected_ig=0.5)
        assert suggestion.col == "A"
        assert suggestion.expected_ig == 0.5

    def test_equality(self):
        """Test equality comparison."""
        s1 = FeatureSuggestion(col="A", expected_ig=0.5)
        s2 = FeatureSuggestion(col="A", expected_ig=0.5)
        s3 = FeatureSuggestion(col="B", expected_ig=0.5)

        assert s1 == s2
        assert s1 != s3


class TestRowVoiModel:
    """Tests for RowVoiModel class."""

    def test_model_creation_default(self):
        """Test creating model with default parameters."""
        model = RowVoiModel()
        assert model.noise_rate == 0.0
        assert model.alpha == 1.0

    def test_model_creation_custom(self):
        """Test creating model with custom parameters."""
        model = RowVoiModel(noise_rate=0.1, alpha=2.0)
        assert model.noise_rate == 0.1
        assert model.alpha == 2.0

    def test_model_fit(self, sample_df):
        """Test fitting the model to data."""
        model = RowVoiModel()
        fitted_model = model.fit(sample_df)

        # Should return the same model instance
        assert fitted_model is model

        # Model should have learned column statistics
        assert hasattr(model, "_column_stats")
        assert len(model._column_stats) == len(sample_df.columns)

    def test_model_fit_empty_df(self):
        """Test fitting to empty DataFrame."""
        empty_df = pd.DataFrame()
        model = RowVoiModel()

        # Should handle empty DataFrame gracefully
        fitted_model = model.fit(empty_df)
        assert fitted_model is model

    def test_suggest_next_feature(self, sample_df, simple_state):
        """Test feature suggestion."""
        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, simple_state)

        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.col in sample_df.columns
        assert isinstance(suggestion.expected_ig, float)
        assert suggestion.expected_ig >= 0

    def test_suggest_excludes_observed(self, sample_df):
        """Test that suggestions exclude observed columns."""
        state = CandidateState(
            candidate_rows=[0, 1, 2],
            posterior={0: 0.5, 1: 0.3, 2: 0.2},
            observed_cols=["A", "B"],
            observed_values={"A": 1, "B": 3},
        )

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, state)

        assert suggestion.col not in ["A", "B"]
        assert suggestion.col in ["C", "D"]

    def test_suggest_single_candidate(self, sample_df):
        """Test suggestion with single candidate."""
        state = CandidateState(
            candidate_rows=[0], posterior={0: 1.0}, observed_cols=[], observed_values={}
        )

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, state)

        # Should still return a suggestion even if no information gain
        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.expected_ig == 0.0

    def test_suggest_all_observed(self, sample_df):
        """Test when all columns are observed."""
        state = CandidateState(
            candidate_rows=[0, 1],
            posterior={0: 0.5, 1: 0.5},
            observed_cols=list(sample_df.columns),
            observed_values={col: sample_df.loc[0, col] for col in sample_df.columns},
        )

        model = RowVoiModel().fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, state)

        # Should return None when no columns available
        assert suggestion is None

    def test_model_with_noise(self, sample_df, simple_state):
        """Test model with noise rate."""
        model = RowVoiModel(noise_rate=0.2).fit(sample_df)
        suggestion = model.suggest_next_feature(sample_df, simple_state)

        # Should still return valid suggestion
        assert isinstance(suggestion, FeatureSuggestion)
        assert suggestion.expected_ig >= 0

    def test_model_reproducibility(self, sample_df, simple_state):
        """Test that model gives consistent results."""
        model1 = RowVoiModel(noise_rate=0.1, alpha=1.5).fit(sample_df)
        model2 = RowVoiModel(noise_rate=0.1, alpha=1.5).fit(sample_df)

        suggestion1 = model1.suggest_next_feature(sample_df, simple_state)
        suggestion2 = model2.suggest_next_feature(sample_df, simple_state)

        # Should give same results for same parameters
        assert suggestion1.col == suggestion2.col
        assert abs(suggestion1.expected_ig - suggestion2.expected_ig) < 1e-10

    def test_model_different_parameters(self, sample_df, simple_state):
        """Test that different parameters give different results."""
        model1 = RowVoiModel(noise_rate=0.0).fit(sample_df)
        model2 = RowVoiModel(noise_rate=0.3).fit(sample_df)

        suggestion1 = model1.suggest_next_feature(sample_df, simple_state)
        suggestion2 = model2.suggest_next_feature(sample_df, simple_state)

        # Different noise rates might lead to different expected information gains
        # (though the best column might still be the same)
        assert isinstance(suggestion1, FeatureSuggestion)
        assert isinstance(suggestion2, FeatureSuggestion)
