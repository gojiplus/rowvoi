#!/usr/bin/env python3
"""📋 Survey and Questionnaire Optimization with RowVoi.

This script demonstrates how to design better surveys and questionnaires by:
1. Minimizing respondent burden (fewer questions)
2. Maximizing information value (better insights)
3. Adapting questions based on previous answers

BUSINESS PROBLEM:
You're designing a market research survey but face these challenges:
- Long surveys have low completion rates
- You need to identify distinct customer segments
- Some questions are expensive/time-consuming to ask
- You want to personalize follow-up questions

SOLUTION:
Use RowVoi to intelligently select which questions to ask each respondent
to maximize information gain while minimizing survey length.

REAL-WORLD APPLICATIONS:
- Market research surveys (customer segmentation)
- Medical questionnaires (symptom assessment)
- Job interviews (candidate evaluation)
- Product feedback forms (feature prioritization)
- User onboarding flows (personalization)

This script demonstrates:
1. Training models on historical data to predict feature usefulness
2. Running interactive disambiguation sessions with different policies
3. Comparing greedy vs model-based vs random selection strategies
4. Testing robustness to measurement noise and distribution shift

🎯 For Use Case 1 (optimizing collection of existing data with known values),
see known_data_setcover_demo.py instead.
"""

import warnings

import numpy as np
import pandas as pd

from rowvoi import (
    CandidateMIPolicy,
    CandidateState,
    DisambiguationSession,
    GreedyCoveragePolicy,
    MIPolicy,
    RandomPolicy,
    RowVoiModel,
    StopRules,
    evaluate_policies,
    sample_candidate_sets,
)

warnings.filterwarnings("ignore")

# Check if we can import sklearn for datasets
try:
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. Using synthetic data instead.")


def load_sample_datasets() -> dict[str, pd.DataFrame]:
    """Load datasets for demonstration."""
    datasets = {}

    if SKLEARN_AVAILABLE:
        # Wine dataset
        wine = load_wine()
        datasets["Wine"] = pd.DataFrame(
            wine.data, columns=[f"chem_{i}" for i in range(wine.data.shape[1])]
        )[:100]  # Limit for faster demo

        # Breast Cancer dataset
        cancer = load_breast_cancer()
        datasets["Cancer"] = pd.DataFrame(
            cancer.data, columns=[f"feature_{i}" for i in range(cancer.data.shape[1])]
        )[:80]

        # Iris dataset
        iris = load_iris()
        datasets["Iris"] = pd.DataFrame(iris.data, columns=iris.feature_names)
    else:
        # Create synthetic datasets
        np.random.seed(42)

        # Dataset with mixed correlations
        datasets["Synthetic_Mixed"] = pd.DataFrame(
            {
                "age_group": np.random.choice(["18-25", "26-40", "41-65", "65+"], 50),
                "income_level": np.random.choice(["low", "medium", "high"], 50),
                "education": np.random.choice(["HS", "College", "Graduate"], 50),
                "location": np.random.choice(["Urban", "Suburban", "Rural"], 50),
                "experience": np.random.randint(0, 30, 50),
                "satisfaction": np.random.choice(["Low", "Medium", "High"], 50),
            }
        )

        # High-dimensional dataset
        datasets["Synthetic_HighD"] = pd.DataFrame(
            {f"var_{i}": np.random.choice(["A", "B", "C"], 30) for i in range(15)}
        )

    return datasets


def discretize_dataset(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Discretize numerical features for better model performance."""
    df_discrete = df.copy()

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"] and df[col].nunique() > n_bins:
            # Use quantile-based binning
            try:
                df_discrete[col] = pd.qcut(
                    df[col], q=n_bins, duplicates="drop", labels=False
                )
            except ValueError:
                # If qcut fails, use regular binning
                df_discrete[col] = pd.cut(df[col], bins=n_bins, labels=False)

    return df_discrete


def create_feature_costs(df: pd.DataFrame) -> dict[str, float]:
    """Create realistic feature costs based on complexity."""
    costs = {}

    for col in df.columns:
        # Base cost on cardinality and name heuristics
        cardinality = df[col].nunique()
        base_cost = 1.0

        # Higher cost for high-cardinality features (harder to measure)
        if cardinality > 10:
            base_cost *= 2.0

        # Add some domain-specific cost heuristics
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ["income", "salary", "financial"]):
            base_cost *= 3.0  # Financial data is sensitive/expensive
        elif any(keyword in col_lower for keyword in ["age", "location", "education"]):
            base_cost *= 1.5  # Demographic data moderately expensive
        elif any(keyword in col_lower for keyword in ["satisfaction", "rating"]):
            base_cost *= 0.8  # Survey data relatively cheap

        # Add some randomness
        costs[col] = base_cost * np.random.uniform(0.7, 1.3)

    return costs


def demonstrate_interactive_session(df: pd.DataFrame, dataset_name: str):
    """Demonstrate an interactive disambiguation session."""
    print(f"\n🎯 Interactive Session Demo: {dataset_name}")
    print(f"   Dataset shape: {df.shape}")

    # Sample a challenging subset
    np.random.seed(42)
    candidate_rows = sorted(
        np.random.choice(len(df), size=min(8, len(df)), replace=False)
    )

    print(f"   Candidate rows: {candidate_rows}")

    # Create feature costs
    costs = create_feature_costs(df)

    # Set up different policies to compare
    policies = {
        "Random": RandomPolicy(seed=42),
        "GreedyCoverage": GreedyCoveragePolicy(costs=costs, objective="entropy"),
        "CandidateMI": CandidateMIPolicy(costs=costs),
    }

    # Add model-based policy if we can train a model
    try:
        model = RowVoiModel(noise=0.1, normalize_cols=True)
        model.fit(df)
        policies["ModelMI"] = MIPolicy(model=model, feature_costs=costs)
    except Exception as e:
        print(f"   ⚠️  Could not train model: {str(e)[:50]}")

    print(f"   Comparing {len(policies)} policies...")

    # Run sessions for each policy
    sessions_results = {}

    for policy_name, policy in policies.items():
        try:
            session = DisambiguationSession(
                df, candidate_rows, policy=policy, feature_costs=costs
            )

            # Set up stop rules
            stop = StopRules(
                max_steps=5,
                cost_budget=15.0,
                epsilon_posterior=0.05,
                target_unique=True,
            )

            # Run session with first candidate as "true" row
            true_row = candidate_rows[0]
            steps = session.run(stop, true_row=true_row)

            sessions_results[policy_name] = {
                "steps": len(steps),
                "final_entropy": session.state.entropy,
                "total_cost": session.cumulative_cost,
                "unique_found": session.state.is_unique,
                "columns_used": [step.col for step in steps],
            }

            print(f"   📊 {policy_name}:")
            print(
                f"      Steps: {len(steps)}, Cost: {session.cumulative_cost:.1f}, "
                f"Final entropy: {session.state.entropy:.2f}"
            )
            print(f"      Columns: {[step.col for step in steps]}")

        except Exception as e:
            print(f"   ❌ {policy_name}: Failed - {str(e)[:50]}")
            sessions_results[policy_name] = {"error": str(e)}

    return sessions_results


def compare_policies_systematic(df: pd.DataFrame, dataset_name: str):
    """Systematic comparison of policies across multiple scenarios."""
    print(f"\n📈 Systematic Policy Comparison: {dataset_name}")

    # Generate multiple candidate sets
    candidate_sets = sample_candidate_sets(
        df, subset_size=min(6, len(df) // 2), n_samples=10, random_state=42
    )

    print(f"   Testing on {len(candidate_sets)} candidate sets...")

    # Create costs
    costs = create_feature_costs(df)

    # Set up policies
    policies = {
        "Random": RandomPolicy(seed=42),
        "GreedyCoverage": GreedyCoveragePolicy(costs=costs, objective="pairs"),
        "CandidateMI": CandidateMIPolicy(costs=costs),
    }

    # Add model-based policy
    try:
        model = RowVoiModel(noise=0.05, normalize_cols=True)
        model.fit(df)
        policies["ModelMI"] = MIPolicy(
            model=model, objective="mi_over_cost", feature_costs=costs
        )
    except Exception as e:
        print(f"   ⚠️  Could not train model for comparison: {str(e)[:50]}")

    # Evaluate policies
    try:
        stats = evaluate_policies(
            df,
            candidate_sets,
            policies,
            feature_costs=costs,
            stop=StopRules(max_steps=4, target_unique=True),
            n_repeats=2,
        )

        print("   Results Summary:")
        for stat in stats:
            print(f"   📊 {stat.name}:")
            print(f"      Avg steps: {stat.mean_steps:.1f} ± {stat.std_steps:.1f}")
            print(f"      Avg cost: {stat.mean_cost:.1f} ± {stat.std_cost:.1f}")
            print(f"      Success rate: {stat.success_rate:.1%}")
            print(f"      Final entropy: {stat.mean_final_entropy:.2f}")

        return stats

    except Exception as e:
        print(f"   ❌ Evaluation failed: {str(e)}")
        return []


def test_model_robustness(df: pd.DataFrame, dataset_name: str):
    """Test robustness of model-based policies to noise and distribution shift."""
    print(f"\n🔬 Model Robustness Testing: {dataset_name}")

    if len(df) < 20:
        print("   ⚠️  Dataset too small for robustness testing")
        return

    # Split data for training and testing
    try:
        if SKLEARN_AVAILABLE:
            train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        else:
            # Simple split without sklearn
            n_train = int(0.7 * len(df))
            train_df = df.iloc[:n_train]
            test_df = df.iloc[n_train:]

        print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

        # Train model on training data
        model = RowVoiModel(noise=0.0, normalize_cols=True)
        model.fit(train_df)

        # Test different noise levels on test data
        noise_levels = [0.0, 0.1, 0.2, 0.3]

        print("   Testing noise robustness:")

        for noise in noise_levels:
            # Create noisy model
            noisy_model = RowVoiModel(noise=noise, normalize_cols=True)
            noisy_model.fit(train_df)

            # Test on a few candidate sets from test data
            n_test_sets = min(3, len(test_df) // 4)
            candidate_sets = []
            for i in range(n_test_sets):
                start_idx = i * 3
                end_idx = min(start_idx + 4, len(test_df))
                if end_idx - start_idx >= 2:
                    candidate_sets.append(list(range(start_idx, end_idx)))

            if candidate_sets:
                policies = {f"Model_noise_{noise}": MIPolicy(model=noisy_model)}

                try:
                    stats = evaluate_policies(
                        test_df,
                        candidate_sets,
                        policies,
                        stop=StopRules(max_steps=3),
                        n_repeats=1,
                    )

                    if stats:
                        stat = stats[0]
                        print(
                            f"     Noise {noise:.1f}: {stat.mean_steps:.1f} steps, "
                            f"success rate {stat.success_rate:.1%}"
                        )

                except Exception:
                    print(f"     Noise {noise:.1f}: Failed")

    except Exception as e:
        print(f"   ❌ Robustness test failed: {str(e)[:50]}")


def analyze_feature_importance(df: pd.DataFrame, dataset_name: str):
    """Analyze which features the model considers most valuable."""
    print(f"\n🔍 Feature Importance Analysis: {dataset_name}")

    try:
        # Train model
        model = RowVoiModel(noise=0.05, normalize_cols=True)
        model.fit(df)

        # Test on a representative candidate set
        n_candidates = min(6, len(df))
        candidate_rows = list(range(0, n_candidates))
        state = CandidateState.uniform(candidate_rows)

        print(f"   Feature rankings for {n_candidates} candidates:")

        # Get suggestions for each feature individually
        feature_scores = {}
        for col in df.columns:
            try:
                suggestion = model.suggest_next_feature(df, state, candidate_cols=[col])
                if suggestion and suggestion.expected_voi is not None:
                    feature_scores[col] = suggestion.expected_voi
            except Exception:
                feature_scores[col] = 0.0

        # Sort features by importance
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )

        for i, (feature, score) in enumerate(sorted_features[:8]):
            print(f"     {i + 1}. {feature}: {score:.3f} bits")

        return sorted_features

    except Exception as e:
        print(f"   ❌ Feature importance analysis failed: {str(e)[:50]}")
        return []


def main():
    """Run comprehensive predictive selection demonstration."""
    print("🔮 ROWVOI PREDICTIVE SELECTION DEMONSTRATION")
    print("=" * 55)
    print("\n🔍 Loading datasets...")

    datasets = load_sample_datasets()
    print(f"   Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    all_results = {}

    for name, df_raw in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"🧪 TESTING DATASET: {name}")
        print("=" * 60)

        # Discretize for better model performance
        df = discretize_dataset(df_raw)

        # Store results for this dataset
        dataset_results = {}

        # Run different types of analysis
        dataset_results["interactive"] = demonstrate_interactive_session(df, name)
        dataset_results["systematic"] = compare_policies_systematic(df, name)
        dataset_results["importance"] = analyze_feature_importance(df, name)

        # Test robustness for larger datasets
        if len(df) > 15:
            test_model_robustness(df, name)

        all_results[name] = dataset_results

    print(f"\n{'=' * 60}")
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n💡 KEY INSIGHTS:")
    print("   • Model-based policies adapt to observed values more effectively")
    print("   • Greedy coverage works well when no historical data available")
    print("   • Feature costs significantly impact selection strategies")
    print("   • Noise tolerance important for real-world deployment")
    print("   • Interactive sessions enable cost-constrained disambiguation")
    print(
        "\n📖 For deterministic optimization with known data, "
        "see known_data_setcover_demo.py"
    )


if __name__ == "__main__":
    main()
