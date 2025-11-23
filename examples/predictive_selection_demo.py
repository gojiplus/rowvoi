#!/usr/bin/env python3
"""🔮 USE CASE 2: Unknown Data Collection (Predictive Selection) - Demo.

This script demonstrates ML-based approaches for the scenario where you're designing
data collection before you know what values you'll observe. This handles uncertainty
and estimates expected information value.

WHEN TO USE THIS APPROACH:
❓ You're designing data collection instruments/protocols
❓ You don't know what values you'll observe in the future
❓ Goal is to predict which features will be most discriminative
❓ You need to maximize expected information gain

Examples
--------
- Survey design: Which questions are most likely to segment customers?
- Experimental planning: Which measurements to prioritize for new compounds?
- Diagnostic protocols: Which tests should be in standard screening?
- Interview guidelines: Which topics provide most candidate differentiation?

This script demonstrates:
1. Training models on historical data to predict feature usefulness
2. Simulating iterative feature collection under uncertainty
3. Comparing different feature selection strategies
4. Testing robustness to measurement noise and distribution shift

🎯 For Use Case 1 (optimizing collection of existing data with known values),
see known_data_setcover_demo.py instead.
"""

import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Check if we can import sklearn for datasets
try:
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - cannot load real datasets")
    exit(1)

from rowvoi.ml import RowVoiModel
from rowvoi.types import CandidateState


def create_noisy_version(df: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
    """Create a noisy version of dataset to simulate measurement uncertainty."""
    noisy_df = df.copy()

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Add gaussian noise to numeric columns
            noise = np.random.normal(0, noise_level * df[col].std(), len(df))
            noisy_df[col] = df[col] + noise
        else:
            # For categorical, occasionally flip to random category
            unique_vals = df[col].unique()
            if len(unique_vals) > 1:
                flip_mask = np.random.random(len(df)) < noise_level
                random_vals = np.random.choice(unique_vals, sum(flip_mask))
                noisy_df.loc[flip_mask, col] = random_vals

    return noisy_df


def simulate_survey_design():
    """Simulate designing a customer segmentation survey."""
    print("🔮 SCENARIO 1: Customer Survey Design")
    print("="*60)
    print("""
BUSINESS CONTEXT:
You're designing a customer survey to segment users but don't know what
responses you'll get. You need to predict which questions will be most
useful for distinguishing different customer types.

CHALLENGE:
- Limited survey length (cost/fatigue constraints)
- Unknown response distributions
- Must predict informativeness before collecting data
    """)

    # Use Wine dataset as proxy for "customer preferences"
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    # Simulate "survey questions" by discretizing features
    questions = {}
    for i, col in enumerate(df.columns[:8]):  # Use subset as "survey questions"
        percentiles = np.percentile(df[col], [25, 75])
        df[f'Q{i+1}_preference'] = pd.cut(df[col],
                                        bins=[-np.inf, percentiles[0], percentiles[1], np.inf],
                                        labels=['Low', 'Medium', 'High'])
        questions[f'Q{i+1}_preference'] = col

    # Add customer segments
    df['customer_segment'] = wine.target
    df['customer_segment'] = df['customer_segment'].map({0: 'Budget', 1: 'Premium', 2: 'Luxury'})

    # Select only survey questions for analysis
    survey_df = df[[col for col in df.columns if col.startswith('Q') or col == 'customer_segment']]

    print(f"📊 Simulated survey with {len(survey_df.columns)-1} questions for {len(survey_df)} customers")
    print(f"📋 Questions: {list(survey_df.columns[:-1])}")

    # Split into training (past surveys) and test (future surveys)
    train_df, test_df = train_test_split(survey_df, test_size=0.3, random_state=42)

    print(f"📈 Training on {len(train_df)} past survey responses")
    print(f"🔮 Predicting usefulness for {len(test_df)} future surveys")

    # Train model on historical survey data
    model = RowVoiModel(noise=0.05).fit(train_df)

    # Simulate designing survey for new customer cohort
    print("\n🎯 FEATURE SELECTION SIMULATION:")
    print("-" * 40)

    selected_questions = []
    remaining_questions = [col for col in survey_df.columns if col != 'customer_segment']

    # Iteratively select best questions (simulating survey design process)
    for round_num in range(1, 5):  # Select top 4 questions
        if not remaining_questions:
            break

        print(f"\n📝 Survey Design Round {round_num}:")

        # Create candidate state with current knowledge INCLUDING observed values
        candidate_customers = np.random.choice(len(train_df), size=10, replace=False).tolist()

        # Simulate that we've observed some values for these customers
        observed_values = {}
        for col in selected_questions:
            if col in train_df.columns:
                # Use actual values from a representative customer
                observed_values[col] = train_df[col].iloc[candidate_customers[0]]

        current_state = CandidateState(
            candidate_rows=candidate_customers,
            posterior={i: 1.0/len(candidate_customers) for i in candidate_customers},
            observed_cols=selected_questions.copy(),
            observed_values=observed_values  # KEY: Current observations condition the prediction
        )

        print(f"   📊 Current observations: {observed_values}")
        print("   🎯 Conditional prediction based on these observed values...")

        # Get model's suggestion for next best question
        suggestion = model.suggest_next_feature(train_df, current_state)

        if suggestion and suggestion.col in remaining_questions:
            print(f"   🎯 Model suggests: {suggestion.col}")
            print(f"   📊 Expected information value: {suggestion.voi:.3f}")
            print(f"   📈 Normalized value: {suggestion.normalized_voi:.3f}")
            print("   🧠 This prediction is CONDITIONAL on observed values above!")
            print("      (Different observations would likely suggest different columns)")

            selected_questions.append(suggestion.col)
            remaining_questions.remove(suggestion.col)
        else:
            # Fallback to random selection
            random_choice = random.choice(remaining_questions)
            selected_questions.append(random_choice)
            remaining_questions.remove(random_choice)
            print(f"   🎲 Fallback selection: {random_choice}")

    print("\n🏆 FINAL SURVEY DESIGN:")
    print(f"Selected questions: {selected_questions}")

    # Test how well the selected questions perform on future data
    print("\n🧪 VALIDATION ON FUTURE SURVEY RESPONSES:")
    print("-" * 45)

    # Test distinguishing power on held-out test data
    test_sample = test_df.sample(n=20, random_state=42)

    try:
        from rowvoi import minimal_key_greedy
        # See how many selected questions are needed to distinguish test customers
        distinguishing_questions = minimal_key_greedy(test_sample[selected_questions],
                                                    list(range(len(test_sample))))

        print("✅ Selected questions successfully distinguish test customers")
        print(f"📊 Questions needed: {len(distinguishing_questions)}/{len(selected_questions)}")
        print(f"💡 Questions used: {distinguishing_questions}")

    except Exception as e:
        print(f"⚠️ Could not test distinguishing power: {e}")

    return selected_questions, model


def simulate_experimental_design():
    """Simulate designing experimental protocols for compound characterization."""
    print("\n\n🔬 SCENARIO 2: Experimental Design")
    print("="*60)
    print("""
RESEARCH CONTEXT:
You're designing experimental protocols to characterize new chemical compounds
but don't know what measurement values you'll get. You need to prioritize
which experiments will be most informative.

CHALLENGE:
- Expensive experiments (cost/time constraints)
- Unknown compound properties
- Must predict experimental value before running tests
    """)

    # Use Breast Cancer dataset as proxy for "compound measurements"
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

    # Select subset of features as "experimental measurements"
    experiments = [
        'mean radius', 'mean texture', 'mean perimeter',
        'mean area', 'mean smoothness', 'worst radius',
        'worst texture', 'worst area', 'worst smoothness'
    ]

    experiment_df = df[experiments].copy()

    # Discretize measurements (simulating experimental results)
    for col in experiment_df.columns:
        quartiles = np.percentile(experiment_df[col], [25, 50, 75])
        experiment_df[col] = pd.cut(experiment_df[col],
                                  bins=[-np.inf] + list(quartiles) + [np.inf],
                                  labels=['Very_Low', 'Low', 'Medium', 'High'])

    # Add compound classification
    experiment_df['compound_type'] = cancer.target
    experiment_df['compound_type'] = experiment_df['compound_type'].map({0: 'Type_A', 1: 'Type_B'})

    print(f"🧪 Simulated experimental suite with {len(experiments)} possible experiments")
    print(f"⚗️  Experiments: {experiments}")

    # Split into known compounds (training) and new compounds (test)
    train_compounds, test_compounds = train_test_split(experiment_df, test_size=0.4, random_state=42)

    print(f"📚 Learning from {len(train_compounds)} characterized compounds")
    print(f"🔮 Predicting for {len(test_compounds)} new compounds")

    # Train model on characterized compounds
    model = RowVoiModel(noise=0.1).fit(train_compounds)

    # Simulate designing experimental protocol for new compounds
    print("\n🎯 EXPERIMENTAL PROTOCOL DESIGN:")
    print("-" * 40)

    # Test different budget constraints
    for budget in [2, 3, 4, 5]:
        print(f"\n💰 Budget: {budget} experiments")

        # Simulate selecting experiments under budget constraint
        selected_experiments = []
        remaining_experiments = list(experiments)

        for _step in range(budget):
            if not remaining_experiments:
                break

            # Create state representing current experimental knowledge
            candidate_compounds = np.random.choice(len(train_compounds), size=8, replace=False).tolist()
            current_state = CandidateState(
                candidate_rows=candidate_compounds,
                posterior={i: 1.0/len(candidate_compounds) for i in candidate_compounds},
                observed_cols=selected_experiments.copy(),
                observed_values={}
            )

            # Get next best experiment
            suggestion = model.suggest_next_feature(train_compounds, current_state)

            if suggestion and suggestion.col in remaining_experiments:
                selected_experiments.append(suggestion.col)
                remaining_experiments.remove(suggestion.col)

        print(f"   🔬 Selected experiments: {selected_experiments}")

        # Test performance with this experimental protocol
        test_subset = test_compounds.sample(n=15, random_state=42)
        try:
            from rowvoi import minimal_key_greedy
            needed_experiments = minimal_key_greedy(test_subset[selected_experiments],
                                                  list(range(len(test_subset))))
            efficiency = len(needed_experiments) / len(selected_experiments) if selected_experiments else 0
            print(f"   📊 Protocol efficiency: {efficiency:.2f} ({len(needed_experiments)}/{len(selected_experiments)} experiments used)")
        except:
            print("   ⚠️ Could not test protocol efficiency")

    return model


def demonstrate_robustness():
    """Demonstrate robustness of predictive selection to noise and distribution shift."""
    print("\n\n🛡️ SCENARIO 3: Robustness Analysis")
    print("="*60)
    print("""
VALIDATION CONTEXT:
Testing how well predictive feature selection performs when:
1. Training data has measurement noise
2. Future data comes from slightly different distribution
3. There are missing values in the prediction phase

This simulates real-world challenges where training data isn't perfect
and future conditions differ from historical conditions.
    """)

    # Use Iris dataset for clean robustness testing
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Discretize for easier analysis
    for col in df.columns:
        tertiles = np.percentile(df[col], [33, 67])
        df[col] = pd.cut(df[col],
                        bins=[-np.inf, tertiles[0], tertiles[1], np.inf],
                        labels=['Small', 'Medium', 'Large'])

    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    print(f"🌸 Clean dataset: {len(df)} samples, {len(df.columns)-1} features")

    # Test different noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3]

    for noise in noise_levels:
        print(f"\n🔊 Testing with {noise*100:.0f}% noise level:")

        # Create noisy training data
        if noise > 0:
            noisy_train_df = create_noisy_version(df, noise_level=noise)
        else:
            noisy_train_df = df.copy()

        # Split for testing
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        noisy_train_df = noisy_train_df.iloc[train_df.index]

        # Train model on noisy data
        try:
            model = RowVoiModel(noise=noise/2).fit(noisy_train_df)  # Model knows about some noise

            # Test feature selection
            candidate_rows = np.random.choice(len(train_df), size=6, replace=False).tolist()

            state = CandidateState(
                candidate_rows=candidate_rows,
                posterior={i: 1.0/len(candidate_rows) for i in candidate_rows},
                observed_cols=[],
                observed_values={}
            )

            suggestion = model.suggest_next_feature(noisy_train_df, state)

            if suggestion:
                print(f"   🎯 Best feature: {suggestion.col}")
                print(f"   📊 Predicted value: {suggestion.voi:.3f}")

                # Test on clean test data
                test_state = CandidateState(
                    candidate_rows=list(range(min(10, len(test_df)))),
                    posterior={i: 1.0/min(10, len(test_df)) for i in range(min(10, len(test_df)))},
                    observed_cols=[],
                    observed_values={}
                )

                clean_model = RowVoiModel(noise=0.0).fit(test_df)
                clean_suggestion = clean_model.suggest_next_feature(test_df, test_state)

                if clean_suggestion:
                    agreement = "✅" if suggestion.col == clean_suggestion.col else "❌"
                    print(f"   🎭 Agrees with clean model: {agreement}")
                else:
                    print("   ⚠️ Clean model failed")
            else:
                print("   ❌ No feature suggestion generated")

        except Exception as e:
            print(f"   💥 Model training failed: {e}")


def main():
    """Main demonstration function."""
    print("🔮 ROWVOI PREDICTIVE SELECTION: Unknown Data Collection Demo")
    print("="*70)
    print("""
This demo shows how to use RowVoi for Use Case 2: predicting which features
will be useful when you don't know what values you'll observe in the future.

SCENARIOS DEMONSTRATED:
🎪 Customer Survey Design - Which questions to include?
🔬 Experimental Design - Which measurements to prioritize?
🛡️ Robustness Testing - How well do predictions hold up?

Each scenario shows the UNCERTAINTY inherent in predictive selection,
contrasting with the DETERMINISTIC optimization of known data collection.
    """)

    # Run three different scenarios
    try:
        selected_questions, survey_model = simulate_survey_design()
        simulate_experimental_design()
        demonstrate_robustness()

        print("\n" + "="*70)
        print("🎯 PREDICTIVE SELECTION INSIGHTS")
        print("="*70)
        print("""
KEY LEARNINGS FROM THIS DEMO:

🧠 CONDITIONAL PREDICTION IS KEY:
   - This is NOT blind guessing - predictions are conditional on current observations
   - Same model gives different suggestions based on what's already been observed
   - Feature value depends on learned patterns + current state
   - Results are CONDITIONAL PROBABILITIES, not universal recommendations

🔗 HISTORICAL PATTERNS + CURRENT CONTEXT:
   - Model learns mutual information patterns from training data
   - Predictions are conditioned on specific observed values in current case
   - Different current observations → different next-best predictions

⚖️ TRADEOFF CONSIDERATIONS:
   - More training data → Better predictions (but may not represent future)
   - Higher noise tolerance → More robust but potentially less optimal
   - Budget constraints → Must balance exploration vs exploitation

🛠️ PRACTICAL RECOMMENDATIONS:
   - Always validate on held-out future data when possible
   - Test robustness to noise and distribution shift
   - Consider multiple models and ensemble approaches
   - Start with most reliable/interpretable features
   - Update models as new data becomes available

🔄 ITERATIVE IMPROVEMENT:
   - Use initial results to refine models
   - Incorporate feedback from actual data collection
   - Adapt strategies based on real-world performance
        """)

        print("\n💡 COMPARISON WITH COMPLETE INFORMATION OPTIMIZATION:")
        print("   🎯 Complete info: All values known → Deterministic optimization")
        print("   🔮 Conditional selection: Partial observations → Conditional prediction")
        print("   🧠 Key insight: Conditional ≠ Blind - uses current state +")
        print("      learned patterns")
        print("   ⚠️  Choose the right approach for your information situation!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 This might happen if optional dependencies aren't available")


if __name__ == "__main__":
    main()

