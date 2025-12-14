"""Comprehensive demo of the new rowvoi API.

This example demonstrates all major features of the refactored rowvoi package:
1. Deterministic key finding with various algorithms
2. Path planning with cost optimization
3. Policy-based column selection
4. Interactive disambiguation sessions
5. Probabilistic methods with models
6. Comprehensive evaluation tools
"""

import numpy as np
import pandas as pd

from rowvoi import (
    CandidateMIPolicy,
    # Core types
    CandidateState,
    # Sessions
    DisambiguationSession,
    GreedyCoveragePolicy,
    # Deterministic methods
    RandomPolicy,
    # Model
    RowVoiModel,
    StopRules,
    evaluate_keys,
    evaluate_policies,
    find_key,
    # Probabilistic methods
    find_key_probabilistic,
    get_logger,
    plan_key_path,
    plan_key_path_probabilistic,
    # Evaluation
    sample_candidate_sets,
)

# Set up logging
logger = get_logger(__name__)


def create_sample_data():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    n_rows = 100
    df = pd.DataFrame(
        {
            "age_group": np.random.choice(["18-25", "26-35", "36-50", "50+"], n_rows),
            "education": np.random.choice(["HS", "Bachelor", "Master", "PhD"], n_rows),
            "income": np.random.choice(
                ["<50k", "50-100k", "100-150k", ">150k"], n_rows
            ),
            "location": np.random.choice(["Urban", "Suburban", "Rural"], n_rows),
            "occupation": np.random.choice(
                ["Tech", "Healthcare", "Finance", "Education", "Other"], n_rows
            ),
            "experience": np.random.choice(["0-2", "3-5", "6-10", "10+"], n_rows),
            "department": np.random.choice(
                ["Sales", "Engineering", "HR", "Marketing"], n_rows
            ),
            "satisfaction": np.random.choice(["Low", "Medium", "High"], n_rows),
        }
    )
    return df


def demo_deterministic_keys():
    """Demonstrate deterministic key finding."""
    logger.info("\n" + "=" * 60)
    logger.info("1. DETERMINISTIC KEY FINDING")
    logger.info("=" * 60)

    df = create_sample_data()
    candidate_rows = [0, 5, 10, 15, 20]

    logger.info(f"\nFinding minimal key for rows: {candidate_rows}")
    logger.info(f"Data shape: {df.shape}")

    # Try different algorithms
    algorithms = ["greedy", "exact", "sa", "ga"]
    results = {}

    for algo in algorithms:
        try:
            key = find_key(df, candidate_rows, strategy=algo, time_limit=1.0)
            results[algo] = key
            logger.info(f"\n{algo.upper()} algorithm: {key} (size={len(key)})")
        except Exception as e:
            logger.error(f"\n{algo.upper()} algorithm: Failed - {e}")

    # With costs
    costs = {
        "age_group": 1.0,
        "education": 2.0,
        "income": 5.0,
        "location": 1.0,
        "occupation": 3.0,
        "experience": 2.0,
        "department": 1.0,
        "satisfaction": 1.0,
    }

    logger.info("\n\nWith column costs:")
    key_with_costs = find_key(df, candidate_rows, costs=costs)
    total_cost = sum(costs.get(c, 1.0) for c in key_with_costs)
    logger.info(f"Minimal cost key: {key_with_costs}")
    logger.info(f"Total cost: {total_cost}")


def demo_path_planning():
    """Demonstrate path planning."""
    logger.info("\n" + "=" * 60)
    logger.info("2. PATH PLANNING")
    logger.info("=" * 60)

    df = create_sample_data()
    candidate_rows = [0, 5, 10, 15, 20, 25]

    # Plan path with different objectives
    path_coverage = plan_key_path(df, candidate_rows, objective="pair_coverage")
    path_entropy = plan_key_path(df, candidate_rows, objective="entropy")

    logger.info("\nPath with pair coverage objective:")
    for step in path_coverage.steps[:3]:
        logger.info(
            f"  {step.col}: covers {step.newly_covered_pairs} new pairs "
            f"({step.coverage:.1%} total)"
        )

    logger.info("\nPath with entropy objective:")
    for step in path_entropy.steps[:3]:
        logger.info(
            f"  {step.col}: covers {step.newly_covered_pairs} new pairs "
            f"({step.coverage:.1%} total)"
        )

    # Find prefix for budget
    costs = {
        "age_group": 1,
        "education": 2,
        "income": 5,
        "location": 1,
        "occupation": 3,
        "experience": 2,
        "department": 1,
        "satisfaction": 1,
    }

    path_with_costs = plan_key_path(df, candidate_rows, costs=costs)
    budget_5_cols = path_with_costs.prefix_for_budget(5.0)
    logger.info(f"\nColumns within budget of 5: {budget_5_cols}")

    # Find prefix for epsilon coverage
    epsilon_cols = path_with_costs.prefix_for_epsilon_pairs(0.1)
    logger.info(f"Columns for 90% coverage: {epsilon_cols}")


def demo_policies():
    """Demonstrate different policies."""
    logger.info("\n" + "=" * 60)
    logger.info("3. POLICY-BASED COLUMN SELECTION")
    logger.info("=" * 60)

    df = create_sample_data()
    candidate_rows = [0, 10, 20, 30, 40]
    state = CandidateState.uniform(candidate_rows)

    # Create different policies
    policies = {
        "greedy_coverage": GreedyCoveragePolicy(),
        "greedy_entropy": GreedyCoveragePolicy(objective="entropy"),
        "candidate_mi": CandidateMIPolicy(),
        "random": RandomPolicy(seed=42),
    }

    logger.info("\nPolicy suggestions for next column:")
    for name, policy in policies.items():
        suggestion = policy.suggest(df, state)
        logger.info(f"  {name}: {suggestion.col} (score={suggestion.score:.3f})")


def demo_interactive_session():
    """Demonstrate interactive disambiguation session."""
    logger.info("\n" + "=" * 60)
    logger.info("4. INTERACTIVE DISAMBIGUATION SESSION")
    logger.info("=" * 60)

    df = create_sample_data()
    candidate_rows = [0, 10, 20, 30, 40]

    # Create session with greedy policy
    policy = GreedyCoveragePolicy(objective="entropy")
    session = DisambiguationSession(df, candidate_rows, policy=policy)

    logger.info(f"\nStarting session with {len(candidate_rows)} candidates")
    logger.info(f"Initial entropy: {session.state.entropy:.2f} bits")

    # Run with different stop rules
    stop_rules = StopRules(max_steps=3, epsilon_posterior=0.1, target_unique=True)

    # Simulate session
    true_row = candidate_rows[0]
    steps = session.run(stop_rules, true_row=true_row)

    logger.info(f"\nSession completed in {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        logger.info(f"  Step {i}: {step.col} = {step.observed_value}")
        logger.info(
            f"    Entropy: {step.entropy_before:.2f} -> {step.entropy_after:.2f}"
        )

    logger.info("\nFinal state:")
    logger.info(f"  Remaining candidates: {len(session.state.candidate_rows)}")
    logger.info(f"  Final entropy: {session.state.entropy:.2f}")
    logger.info(f"  Is unique: {session.state.is_unique}")


def demo_probabilistic_methods():
    """Demonstrate probabilistic methods with model."""
    logger.info("\n" + "=" * 60)
    logger.info("5. PROBABILISTIC METHODS")
    logger.info("=" * 60)

    df = create_sample_data()

    # Train a model
    model = RowVoiModel(noise=0.1)
    model.fit(df)

    candidate_rows = [0, 10, 20, 30, 40]

    # Find probabilistic key
    prob_key = find_key_probabilistic(df, candidate_rows, model, epsilon_posterior=0.05)
    logger.info(f"\nProbabilistic key (95% confidence): {prob_key}")

    # Plan probabilistic path
    prob_path = plan_key_path_probabilistic(
        df, candidate_rows, model, objective="mi_over_cost"
    )
    logger.info(f"\nProbabilistic path: {prob_path.columns()[:3]}...")


def demo_evaluation():
    """Demonstrate evaluation tools."""
    logger.info("\n" + "=" * 60)
    logger.info("6. EVALUATION TOOLS")
    logger.info("=" * 60)

    df = create_sample_data()

    # Sample candidate sets
    candidate_sets = sample_candidate_sets(
        df, subset_size=5, n_samples=10, random_state=42
    )
    logger.info(f"\nGenerated {len(candidate_sets)} candidate sets")

    # Define methods to evaluate
    methods = {
        "greedy": lambda df, rows: find_key(df, rows, strategy="greedy"),
        "exact": lambda df, rows: find_key(df, rows, strategy="exact", time_limit=0.5),
    }

    # Evaluate keys
    key_results = evaluate_keys(
        df,
        candidate_sets[:3],  # Just first 3 for speed
        methods,
    )

    logger.info("\nKey evaluation results:")
    for result in key_results:
        logger.info(
            f"  {result.method}: size={len(result.key)}, "
            f"coverage={result.pair_coverage:.1%}, "
            f"time={result.runtime_sec:.3f}s"
        )

    # Define policies to evaluate
    policies = {
        "greedy": GreedyCoveragePolicy(),
        "mi": CandidateMIPolicy(),
        "random": RandomPolicy(seed=42),
    }

    # Evaluate policies
    policy_stats = evaluate_policies(
        df, candidate_sets[:5], policies, stop=StopRules(target_unique=True)
    )

    logger.info("\nPolicy evaluation results:")
    for stat in policy_stats:
        logger.info(f"  {stat.name}:")
        logger.info(f"    Mean steps: {stat.mean_steps:.1f} ± {stat.std_steps:.1f}")
        logger.info(f"    Success rate: {stat.success_rate:.1%}")


def main():
    """Run all demonstrations."""
    logger.info("\n" + "#" * 60)
    logger.info("# ROWVOI NEW API DEMONSTRATION")
    logger.info("#" * 60)

    demo_deterministic_keys()
    demo_path_planning()
    demo_policies()
    demo_interactive_session()
    demo_probabilistic_methods()
    demo_evaluation()

    logger.info("\n" + "#" * 60)
    logger.info("# DEMONSTRATION COMPLETE")
    logger.info("#" * 60)


if __name__ == "__main__":
    main()
