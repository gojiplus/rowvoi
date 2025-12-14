#!/usr/bin/env python3
"""🎯 USE CASE 1: Known Data Collection (Set Cover) - Comprehensive Demo.

This script demonstrates key-finding algorithms for the scenario where you have
existing data with known values and need to find the minimal columns to collect
for distinguishing specific rows. This is deterministic optimization.

WHEN TO USE THIS APPROACH:
✅ You have existing database/dataset with known values
✅ Goal is to minimize data collection/retrieval costs
✅ You know exactly what distinguishing power each column has

Examples
--------
- Medical records: Which fields to retrieve from patient database?
- Database optimization: Which columns to query for customer identification?
- Privacy minimization: Which fields provide ID with minimal exposure?

This script tests algorithms on real datasets from scikit-learn:
1. Wine Quality Dataset - Chemical properties of Portuguese wine
2. Breast Cancer Dataset - Medical tumor characteristics
3. Iris Dataset - Classic botanical measurements
4. Handwritten Digits Dataset - Image recognition features

For each dataset, we:
- Load and preprocess the data (simulating existing database)
- Select challenging subsets of rows to distinguish
- Run all available key-finding algorithms
- Compare performance, solution quality, and timing
- Validate that found columns actually distinguish the target rows
- Provide insights and recommendations

❓ For Use Case 2 (predicting useful features for unknown future data),
see predictive_selection_demo.py instead.
"""

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from rowvoi import KeyProblem, find_key, get_logger, plan_key_path

warnings.filterwarnings("ignore")

# Set up logging
logger = get_logger(__name__)

# Check if we can import sklearn for datasets
try:
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️  scikit-learn not available. Using synthetic data instead.")


def load_sample_datasets() -> dict[str, pd.DataFrame]:
    """Load sample datasets for demonstration."""
    datasets = {}

    if SKLEARN_AVAILABLE:
        # Wine dataset
        wine = load_wine()
        datasets["Wine"] = pd.DataFrame(
            wine.data[:50], columns=[f"feature_{i}" for i in range(wine.data.shape[1])]
        )

        # Breast Cancer dataset
        cancer = load_breast_cancer()
        datasets["Cancer"] = pd.DataFrame(
            cancer.data[:30],
            columns=[f"feature_{i}" for i in range(cancer.data.shape[1])],
        )

        # Iris dataset
        iris = load_iris()
        datasets["Iris"] = pd.DataFrame(iris.data, columns=iris.feature_names)

        # Digits dataset (smaller subset)
        digits = load_digits()
        datasets["Digits"] = pd.DataFrame(
            digits.data[:20],
            columns=[f"pixel_{i}" for i in range(digits.data.shape[1])],
        )
    else:
        # Create synthetic datasets
        np.random.seed(42)
        datasets["Synthetic_A"] = pd.DataFrame(
            {f"col_{i}": np.random.randint(0, 5, 20) for i in range(10)}
        )
        datasets["Synthetic_B"] = pd.DataFrame(
            {f"feat_{i}": np.random.choice(["A", "B", "C", "D"], 15) for i in range(8)}
        )

    return datasets


def discretize_dataset(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Discretize numerical features for better set cover performance."""
    df_discrete = df.copy()

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"] and df[col].nunique() > n_bins:
            # Use quantile-based binning
            df_discrete[col] = pd.qcut(
                df[col], q=n_bins, duplicates="drop", labels=False
            )

    return df_discrete


def generate_challenging_subsets(df: pd.DataFrame) -> list[list[int]]:
    """Generate challenging row subsets for testing algorithms."""
    n_rows = len(df)
    subsets = []

    # Random subsets of different sizes
    np.random.seed(42)
    for size in [3, 5, min(8, n_rows // 2), min(12, n_rows - 2)]:
        if size <= n_rows:
            subset = np.random.choice(n_rows, size=size, replace=False).tolist()
            subsets.append(subset)

    # Edge cases
    if n_rows >= 2:
        subsets.append([0, 1])  # Minimal case
        subsets.append([0, n_rows // 2, n_rows - 1])  # Spread out rows

    return subsets


def benchmark_algorithms(
    df: pd.DataFrame,
    rows: list[int],
    dataset_name: str,
    costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Benchmark all available algorithms on a specific row set."""
    results = {}

    # Available algorithms to test
    algorithms = ["greedy", "exact", "sa", "ga"]

    if len(df.columns) <= 15:  # Only try exact for smaller problems
        algorithms.append("exact")

    problem = KeyProblem(df, rows, costs=costs)
    total_pairs = len(rows) * (len(rows) - 1) // 2

    logger.info(
        f"\n  🎯 Benchmarking {len(algorithms)} algorithms on {len(rows)} rows..."
    )
    logger.info(f"     Total pairs to distinguish: {total_pairs}")

    for algo in algorithms:
        try:
            start_time = time.time()

            if algo == "exact" and len(df.columns) > 10:
                # Skip exact for large problems
                continue

            key = find_key(df, rows, strategy=algo, costs=costs, time_limit=2.0)
            runtime = time.time() - start_time

            # Validate the key
            is_valid = problem.is_key(key)
            coverage = problem.pairwise_coverage(key)
            total_cost = sum(costs.get(c, 1.0) for c in key) if costs else len(key)

            results[algo] = {
                "key": key,
                "size": len(key),
                "cost": total_cost,
                "coverage": coverage,
                "runtime": runtime,
                "valid": is_valid,
                "algorithm": algo,
            }

            status = "✅" if is_valid else "❌"
            if status == "✅":
                logger.info(
                    f"     {status} {algo.upper()}: {len(key)} cols, "
                    f"cost={total_cost:.1f}, {runtime:.3f}s"
                )
            else:
                logger.error(
                    f"     {status} {algo.upper()}: {len(key)} cols, "
                    f"cost={total_cost:.1f}, {runtime:.3f}s"
                )

        except Exception as e:
            logger.error(f"     ❌ {algo.upper()}: Failed ({str(e)[:50]})")
            results[algo] = {"error": str(e)}

    return results


def demonstrate_path_planning(df: pd.DataFrame, rows: list[int], dataset_name: str):
    """Demonstrate path planning functionality."""
    logger.info(f"\n  🛤️  Path Planning for {dataset_name}")

    # Create some example costs
    costs = {col: np.random.uniform(0.5, 3.0) for col in df.columns}

    # Plan different types of paths
    path_coverage = plan_key_path(df, rows, costs=costs, objective="pair_coverage")
    path_entropy = plan_key_path(df, rows, costs=costs, objective="entropy")

    logger.info("     Coverage-optimized path (first 3 steps):")
    for i, step in enumerate(path_coverage.steps[:3]):
        logger.info(
            f"       {i + 1}. {step.col}: +{step.newly_covered_pairs} pairs "
            f"({step.coverage:.0%} total, cost={step.cumulative_cost:.1f})"
        )

    logger.info("     Entropy-optimized path (first 3 steps):")
    for i, step in enumerate(path_entropy.steps[:3]):
        logger.info(
            f"       {i + 1}. {step.col}: +{step.newly_covered_pairs} pairs "
            f"({step.coverage:.0%} total, cost={step.cumulative_cost:.1f})"
        )

    # Demonstrate budget constraints
    budget_cols = path_coverage.prefix_for_budget(5.0)
    epsilon_cols = path_coverage.prefix_for_epsilon_pairs(0.1)

    logger.info(f"     Within budget of 5.0: {budget_cols}")
    logger.info(f"     For 90% coverage: {epsilon_cols}")


def analyze_dataset_properties(df: pd.DataFrame, dataset_name: str):
    """Analyze properties of the dataset that affect set cover performance."""
    logger.info(f"\n📊 Dataset Analysis: {dataset_name}")
    logger.info(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"   Data types: {df.dtypes.value_counts().to_dict()}")

    # Column cardinality analysis
    cardinalities = [df[col].nunique() for col in df.columns]
    logger.info(
        f"   Column cardinalities: min={min(cardinalities)}, "
        f"max={max(cardinalities)}, mean={np.mean(cardinalities):.1f}"
    )

    # Identify potentially problematic columns
    low_card_cols = [col for col in df.columns if df[col].nunique() <= 2]
    high_card_cols = [col for col in df.columns if df[col].nunique() >= len(df) * 0.8]

    if low_card_cols:
        logger.warning(
            f"   ⚠️  Low-cardinality columns (≤2 values): {len(low_card_cols)}"
        )
    if high_card_cols:
        logger.warning(
            f"   ⚠️  High-cardinality columns (≥80% unique): {len(high_card_cols)}"
        )


def main():
    """Run comprehensive set cover demonstration."""
    logger.info("🎯 ROWVOI SET COVER DEMONSTRATION")
    logger.info("=" * 50)
    logger.info("\n🔍 Loading datasets...")

    datasets = load_sample_datasets()
    logger.info(f"   Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    all_results = {}

    for name, df_raw in datasets.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🧪 TESTING DATASET: {name}")
        logger.info("=" * 60)

        # Discretize for better performance
        df = discretize_dataset(df_raw)

        # Analyze dataset
        analyze_dataset_properties(df, name)

        # Generate test cases
        subsets = generate_challenging_subsets(df)

        dataset_results = []

        for i, rows in enumerate(subsets):
            logger.info(f"\n🔬 Test Case {i + 1}: {len(rows)} rows {rows}")

            # Create example costs based on column cardinality
            costs = {
                col: 1.0 / df[col].nunique() + np.random.uniform(0.1, 0.5)
                for col in df.columns
            }

            # Benchmark algorithms
            results = benchmark_algorithms(df, rows, name, costs)
            dataset_results.append({"rows": rows, "results": results})

            # Demonstrate path planning (for first few test cases)
            if i < 2:
                demonstrate_path_planning(df, rows, name)

        all_results[name] = dataset_results

        # Summary for this dataset
        logger.info(f"\n📈 Summary for {name}:")
        successful_results = []
        for test_case in dataset_results:
            for _algo, result in test_case["results"].items():
                if "size" in result and result["valid"]:
                    successful_results.append(result)

        if successful_results:
            avg_size = np.mean([r["size"] for r in successful_results])
            avg_runtime = np.mean([r["runtime"] for r in successful_results])
            logger.info(f"   Average key size: {avg_size:.1f} columns")
            logger.info(f"   Average runtime: {avg_runtime:.3f} seconds")

            # Best algorithm by size
            best_by_size = min(successful_results, key=lambda x: x["size"])
            logger.info(
                f"   Best solution: {best_by_size['size']} columns "
                f"({best_by_size['algorithm']}, cost={best_by_size['cost']:.1f})"
            )

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ DEMONSTRATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\n💡 KEY INSIGHTS:")
    logger.info("   • Greedy algorithm provides good approximation quickly")
    logger.info("   • Exact solutions feasible for small problems (<15 columns)")
    logger.info(
        "   • Metaheuristics (SA, GA) can improve on greedy for larger problems"
    )
    logger.info("   • Column costs significantly impact optimal column selection")
    logger.info(
        "   • Path planning enables budget-constrained and progressive selection"
    )
    logger.info(
        "\n📖 For interactive selection with unknown data, "
        "see predictive_selection_demo.py"
    )


if __name__ == "__main__":
    main()
