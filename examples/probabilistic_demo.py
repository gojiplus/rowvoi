#!/usr/bin/env python3
"""Demo of probabilistic and ε-relaxed set cover capabilities.

This script demonstrates the new probabilistic set cover algorithms that handle
uncertainty about unseen column values. Instead of requiring deterministic
disambiguation, these algorithms find minimal column sets that achieve
disambiguation with high probability (1-ε).
"""

import numpy as np
import pandas as pd

from rowvoi import (
    evaluate_coverage,
    minimal_key_greedy,
    probabilistic_minimal_key,
    suggest_next_feature_epsilon,
)
from rowvoi.types import CandidateState

print("🎲 PROBABILISTIC SET COVER DEMO")
print("=" * 60)
print("""
This demo shows how probabilistic set cover handles uncertainty
about column values, finding minimal sets that disambiguate rows
with high probability rather than requiring certainty.
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
print()

# Select some rows to disambiguate
rows = [5, 12, 23, 34, 45]
print(f"🎯 Task: Disambiguate rows {rows}")
print()

# 1. DETERMINISTIC APPROACH
print("1️⃣  DETERMINISTIC SET COVER (eps=0)")
print("-" * 40)
deterministic = minimal_key_greedy(df, rows)
print(f"   Columns needed: {deterministic}")
print(f"   Number of columns: {len(deterministic)}")

# Verify coverage
coverage, uncovered = evaluate_coverage(df, rows, deterministic)
print(f"   Actual coverage: {coverage:.1%}")
print()

# 2. PROBABILISTIC WITH SMALL EPSILON
print("2️⃣  PROBABILISTIC SET COVER (eps=0.05)")
print("-" * 40)
print("   Allow 5% expected uncovered pairs")

result_5pct = probabilistic_minimal_key(df, rows, eps=0.05)
print(f"   Columns selected: {result_5pct.columns}")
print(f"   Number of columns: {len(result_5pct.columns)}")
print(f"   Expected coverage: {result_5pct.expected_coverage:.1%}")

# Check actual coverage
actual_cov, _ = evaluate_coverage(df, rows, result_5pct.columns)
print(f"   Actual coverage: {actual_cov:.1%}")
print(f"   Savings: {len(deterministic) - len(result_5pct.columns)} fewer columns")
print()

# 3. PROBABILISTIC WITH LARGER EPSILON
print("3️⃣  PROBABILISTIC SET COVER (eps=0.20)")
print("-" * 40)
print("   Allow 20% expected uncovered pairs")

result_20pct = probabilistic_minimal_key(df, rows, eps=0.20)
print(f"   Columns selected: {result_20pct.columns}")
print(f"   Number of columns: {len(result_20pct.columns)}")
print(f"   Expected coverage: {result_20pct.expected_coverage:.1%}")

actual_cov, uncov_pairs = evaluate_coverage(df, rows, result_20pct.columns)
print(f"   Actual coverage: {actual_cov:.1%}")
print(f"   Savings: {len(deterministic) - len(result_20pct.columns)} fewer columns")
print()

# 4. COST-AWARE SELECTION
print("4️⃣  COST-AWARE PROBABILISTIC SELECTION")
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

result_cost = probabilistic_minimal_key(df, rows, eps=0.1, costs=costs)
print(f"   Columns selected: {result_cost.columns}")
print(f"   Total cost: {sum(costs[c] for c in result_cost.columns):.1f}")
print(f"   Expected coverage: {result_cost.expected_coverage:.1%}")

# Compare to non-cost-aware
result_no_cost = probabilistic_minimal_key(df, rows, eps=0.1)
cost_no_aware = sum(costs[c] for c in result_no_cost.columns)
print(f"   Cost without awareness: {cost_no_aware:.1f}")
print(
    f"   Cost savings: {cost_no_aware - sum(costs[c] for c in result_cost.columns):.1f}"
)
print()

# 5. ADAPTIVE SEQUENTIAL SELECTION
print("5️⃣  ADAPTIVE EPSILON POLICY")
print("-" * 40)
print("   Sequential feature selection with eps=0.1")

# Start with no observations
state = CandidateState(
    candidate_rows=rows,
    posterior={r: 1 / len(rows) for r in rows},
    observed_cols=[],
    observed_values={},
)

selected_sequence = []
for step in range(5):
    suggestion = suggest_next_feature_epsilon(df, state, eps=0.1, costs=costs)

    if suggestion is None:
        print(f"   ✅ Target coverage achieved after {step} features")
        break

    print(f"   Step {step + 1}: Select '{suggestion.col}'")
    print(f"          Current coverage: {suggestion.current_coverage:.1%}")
    print(f"          Expected gain: {suggestion.expected_coverage_gain:.2f} pairs")

    selected_sequence.append(suggestion.col)

    # Update state with observation
    state = CandidateState(
        candidate_rows=rows,
        posterior=state.posterior,
        observed_cols=state.observed_cols + [suggestion.col],
        observed_values={
            **state.observed_values,
            suggestion.col: df.loc[rows[0], suggestion.col],
        },
    )

print(f"\n   Final sequence: {selected_sequence}")
final_cov, _ = evaluate_coverage(df, rows, selected_sequence)
print(f"   Final coverage: {final_cov:.1%}")
print()

# 6. COMPARISON SUMMARY
print("📊 SUMMARY COMPARISON")
print("-" * 40)
print(f"{'Approach':<30} {'Columns':<10} {'Coverage':<12}")
print("-" * 52)

det_cov, _ = evaluate_coverage(df, rows, deterministic)
print(f"{'Deterministic (eps=0)':<30} {len(deterministic):<10} {det_cov:>11.1%}")

r5_cov, _ = evaluate_coverage(df, rows, result_5pct.columns)
print(
    f"{'Probabilistic (eps=0.05)':<30} {len(result_5pct.columns):<10} {r5_cov:>11.1%}"
)

r20_cov, _ = evaluate_coverage(df, rows, result_20pct.columns)
print(
    f"{'Probabilistic (eps=0.20)':<30} {len(result_20pct.columns):<10} {r20_cov:>11.1%}"
)

print(
    f"{'Cost-aware (eps=0.10)':<30} {len(result_cost.columns):<10} {actual_cov:>11.1%}"
)

print(f"{'Adaptive (eps=0.10)':<30} {len(selected_sequence):<10} {final_cov:>11.1%}")

print()
print("💡 KEY INSIGHTS:")
print("   • Relaxing epsilon reduces columns needed")
print("   • Cost awareness changes column selection")
print("   • Adaptive policy responds to observations")
print("   • Trade-off: fewer columns vs coverage guarantee")
