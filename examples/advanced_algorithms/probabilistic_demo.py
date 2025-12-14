#!/usr/bin/env python3
"""Demo of probabilistic and model-based rowvoi capabilities.

This script demonstrates the probabilistic and model-based features of rowvoi:
1. Model-based value-of-information estimation
2. Probabilistic key finding with uncertainty handling
3. Noise-robust selection policies
4. Cost-aware probabilistic optimization

The probabilistic approach uses trained models to predict expected information
gain when the actual column values are unknown in advance.
"""

import numpy as np
import pandas as pd

from rowvoi import (
    CandidateState,
    DisambiguationSession,
    MIPolicy,
    RowVoiModel,
    StopRules,
    find_key,
    find_key_probabilistic,
    plan_key_path,
    plan_key_path_probabilistic,
)
from rowvoi.keys import pairwise_coverage

print("🎲 PROBABILISTIC & MODEL-BASED ROWVOI DEMO")
print("=" * 60)
print("""
This demo shows how rowvoi handles uncertainty and uses machine learning
models to predict the value of information for sequential column selection
when the actual values are unknown in advance.
""")

# Create a sample dataset with some uncertainty
np.random.seed(42)
df = pd.DataFrame(
    {
        "region": ["North", "South", "East", "West"] * 25,
        "product": ["A", "A", "B", "B", "C", "C", "A", "B"] * 12 + ["C"] * 4,
        "channel": ["Online", "Store", "Online", "Store"] * 25,
        "segment": np.random.choice(["SMB", "Enterprise", "Consumer"], 100),
        "value": np.random.choice(["High", "Medium", "Low"], 100, p=[0.2, 0.5, 0.3]),
        "priority": np.random.choice(["P1", "P2", "P3"], 100, p=[0.1, 0.3, 0.6]),
    }
)

print(f"📊 Dataset: {len(df)} rows × {len(df.columns)} columns")
print(f"   Columns: {list(df.columns)}")
print("")

# Select some rows to disambiguate
rows = [5, 12, 23, 34, 45]
print(f"🎯 Task: Disambiguate rows {rows}")
print("")

# 1. DETERMINISTIC APPROACH (baseline)
print("1️⃣  DETERMINISTIC KEY FINDING")
print("-" * 40)
deterministic = find_key(df, rows, strategy="greedy")
print(f"   Columns needed: {deterministic}")
print(f"   Number of columns: {len(deterministic)}")

# Check coverage
det_coverage = pairwise_coverage(df, rows, deterministic)
print(f"   Actual coverage: {det_coverage:.1%}")
print("")

# 2. MODEL-BASED PROBABILISTIC APPROACH
print("2️⃣  MODEL-BASED PROBABILISTIC SELECTION")
print("-" * 40)
print("   Training model on historical data patterns...")

# Train a model on the dataset
model = RowVoiModel(noise=0.05, normalize_cols=True)
model.fit(df)

# Find probabilistic key with model
try:
    prob_key = find_key_probabilistic(
        df, rows, model, epsilon_posterior=0.05, max_steps=4
    )
    print(f"   Probabilistic key (95% confidence): {prob_key}")
    print(f"   Number of columns: {len(prob_key)}")

    # Check actual coverage of probabilistic key
    prob_coverage = pairwise_coverage(df, rows, prob_key)
    print(f"   Actual coverage: {prob_coverage:.1%}")
    print(f"   Savings: {len(deterministic) - len(prob_key)} fewer columns")
except Exception as e:
    print(f"   ❌ Probabilistic key finding failed: {str(e)[:50]}")
    prob_key = deterministic  # Fallback

print("")

# 3. COST-AWARE PROBABILISTIC SELECTION
print("3️⃣  COST-AWARE PROBABILISTIC SELECTION")
print("-" * 40)
print("   Some columns are more expensive to collect")

costs = {
    "region": 1.0,  # Easy to get
    "product": 1.0,  # Easy to get
    "channel": 2.0,  # Requires lookup
    "segment": 5.0,  # Requires analysis
    "value": 10.0,  # Expensive calculation
    "priority": 3.0,  # Moderate cost
}

try:
    cost_aware_key = find_key_probabilistic(
        df,
        rows,
        model,
        epsilon_posterior=0.1,
        costs=costs,
        objective="mi_over_cost",
        max_steps=4,
    )
    print(f"   Cost-aware columns: {cost_aware_key}")
    total_cost = sum(costs[c] for c in cost_aware_key)
    print(f"   Total cost: {total_cost:.1f}")

    # Compare to non-cost-aware approach
    regular_cost = sum(costs.get(c, 1.0) for c in prob_key)
    print(f"   Regular approach cost: {regular_cost:.1f}")
    if regular_cost > total_cost:
        print(f"   💰 Cost savings: {regular_cost - total_cost:.1f}")
except Exception as e:
    print(f"   ❌ Cost-aware selection failed: {str(e)[:50]}")

print("")

# 4. PROBABILISTIC PATH PLANNING
print("4️⃣  PROBABILISTIC PATH PLANNING")
print("-" * 40)
print("   Planning optimal order for column collection")

try:
    # Plan probabilistic path
    prob_path = plan_key_path_probabilistic(
        df, rows, model, objective="mi_over_cost", costs=costs
    )

    print("   Planned selection order:")
    for i, step in enumerate(prob_path.steps[:4]):
        print(
            f"     {i + 1}. {step.col}: estimated gain={step.newly_covered_pairs} pairs"
        )
        print(
            f"        Cost: {step.marginal_cost:.1f}, "
            f"Cumulative: {step.cumulative_cost:.1f}"
        )

    # Compare to deterministic path
    det_path = plan_key_path(df, rows, costs=costs, objective="pair_coverage")

    print("\n   Deterministic path order:")
    for i, step in enumerate(det_path.steps[:4]):
        print(
            f"     {i + 1}. {step.col}: covers {step.newly_covered_pairs} pairs"
        )
        print(
            f"        Cost: {step.marginal_cost:.1f}, "
            f"Cumulative: {step.cumulative_cost:.1f}"
        )

except Exception as e:
    print(f"   ❌ Path planning failed: {str(e)[:50]}")

print("")

# 5. INTERACTIVE MODEL-BASED DISAMBIGUATION
print("5️⃣  INTERACTIVE MODEL-BASED SESSION")
print("-" * 40)
print("   Simulating adaptive column selection")

try:
    # Create model-based policy
    policy = MIPolicy(model=model, objective="mi_over_cost", feature_costs=costs)

    # Create disambiguation session
    session = DisambiguationSession(df, rows, policy=policy, feature_costs=costs)

    # Set up stop rules
    stop = StopRules(
        max_steps=4, cost_budget=15.0, epsilon_posterior=0.05, target_unique=True
    )

    print(f"   Starting with {len(rows)} candidates")
    print(f"   Initial entropy: {session.state.entropy:.2f} bits")

    # Run session with first row as target
    true_row = rows[0]
    steps = session.run(stop, true_row=true_row)

    print(f"\n   Session completed in {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"     Step {i}: {step.col} = {step.observed_value}")
        print(
            f"       Entropy: {step.entropy_before:.2f} → {step.entropy_after:.2f}"
        )
        total_cost = sum(s.cost for s in steps[:i])
        print(f"       Cost: {step.cost:.1f} (Total: {total_cost:.1f})")

    print("\n   Final state:")
    print(f"     Remaining candidates: {len(session.state.candidate_rows)}")
    print(f"     Final entropy: {session.state.entropy:.2f}")
    print(f"     Unique identification: {session.state.is_unique}")
    print(f"     Total cost: {session.cumulative_cost:.1f}")

except Exception as e:
    print(f"   ❌ Interactive session failed: {str(e)[:50]}")

print("")

# 6. NOISE ROBUSTNESS DEMONSTRATION
print("6️⃣  NOISE ROBUSTNESS TEST")
print("-" * 40)
print("   Testing different noise levels in models")

noise_levels = [0.0, 0.1, 0.2, 0.3]
state = CandidateState.uniform(rows[:4])  # Smaller set for demo

for noise in noise_levels:
    try:
        # Create model with different noise levels
        noisy_model = RowVoiModel(noise=noise, normalize_cols=True)
        noisy_model.fit(df)

        # Get suggestion from model
        suggestion = noisy_model.suggest_next_feature(df, state)

        if suggestion:
            print(
                f"   Noise {noise:.1f}: suggests '{suggestion.col}' "
                f"(VOI: {suggestion.expected_voi:.3f})"
            )
        else:
            print(f"   Noise {noise:.1f}: no suggestion available")

    except Exception as e:
        print(f"   Noise {noise:.1f}: failed - {str(e)[:30]}")

print("")

# 7. SUMMARY COMPARISON
print("📊 SUMMARY COMPARISON")
print("-" * 40)
print(f"{'Approach':<35} {'Columns':<10} {'Coverage':<12}")
print("-" * 57)

det_cov = pairwise_coverage(df, rows, deterministic)
print(f"{'Deterministic':<35} {len(deterministic):<10} {det_cov:>11.1%}")

if "prob_key" in locals() and prob_key != deterministic:
    prob_cov = pairwise_coverage(df, rows, prob_key)
    print(
        f"{'Probabilistic (95% confidence)':<35} {len(prob_key):<10} {prob_cov:>11.1%}"
    )

if "cost_aware_key" in locals():
    cost_cov = pairwise_coverage(df, rows, cost_aware_key)
    cost_total = sum(costs[c] for c in cost_aware_key)
    cost_label = f"Cost-aware (cost={cost_total:.1f})"
    print(f"{cost_label:<35} {len(cost_aware_key):<10} {cost_cov:>11.1%}")

print("")
print("💡 KEY INSIGHTS:")
print("   • Models predict expected information value before observing data")
print("   • Probabilistic methods can reduce column requirements")
print("   • Cost awareness changes selection priorities")
print("   • Noise tolerance important for real-world deployment")
print("   • Interactive sessions adapt based on observed values")
print("   • Trade-off between certainty and efficiency")
