#!/usr/bin/env python3
"""Performance benchmarking suite for set cover algorithms.

This script benchmarks different set cover algorithms on synthetic datasets
of varying sizes and characteristics to help users choose appropriate algorithms
for their use cases.
"""

import time
import random
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np

try:
    from rowvoi import minimal_key_advanced, is_key
    from rowvoi.setcover import solve_set_cover, SetCoverAlgorithm
except ImportError:
    print("Please install rowvoi package first: uv sync")
    exit(1)


def generate_synthetic_data(n_rows: int, n_cols: int, sparsity: float = 0.5) -> pd.DataFrame:
    """Generate synthetic dataset for benchmarking.
    
    Parameters
    ----------
    n_rows : int
        Number of rows
    n_cols : int  
        Number of columns
    sparsity : float
        Fraction of unique values (0.5 = medium distinguishability)
        
    Returns
    -------
    pd.DataFrame
        Synthetic dataset
    """
    data = {}
    for i in range(n_cols):
        # Create columns with varying distinguishing power
        if sparsity < 0.3:  # Low distinguishability
            n_unique = max(2, int(n_rows * sparsity))
        elif sparsity > 0.7:  # High distinguishability  
            n_unique = max(n_rows // 2, int(n_rows * sparsity))
        else:  # Medium distinguishability
            n_unique = max(3, int(n_rows * sparsity))
            
        values = [f"val_{i}_{j}" for j in range(n_unique)]
        data[f"col_{i}"] = [random.choice(values) for _ in range(n_rows)]
    
    return pd.DataFrame(data)


def benchmark_algorithm(
    df: pd.DataFrame, 
    rows: List[int], 
    algorithm: str,
    timeout: float = 60.0,
    **kwargs
) -> Dict[str, Any]:
    """Benchmark a single algorithm.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test dataset
    rows : List[int]
        Row indices to distinguish
    algorithm : str
        Algorithm name
    timeout : float
        Maximum time to run
    **kwargs
        Algorithm-specific parameters
        
    Returns
    -------
    Dict[str, Any]
        Benchmark results
    """
    try:
        start_time = time.time()
        
        # Set timeout parameters for algorithms that support it
        if algorithm in ["ilp", "lp"]:
            kwargs.setdefault("time_limit", timeout)
        elif algorithm in ["sa", "ga", "hybrid"]:
            # Estimate reasonable iteration limits based on timeout
            if timeout <= 5.0:
                kwargs.setdefault("max_iterations", 100)
                kwargs.setdefault("max_generations", 50)
            elif timeout <= 30.0:
                kwargs.setdefault("max_iterations", 1000) 
                kwargs.setdefault("max_generations", 200)
            else:
                kwargs.setdefault("max_iterations", 5000)
                kwargs.setdefault("max_generations", 1000)
        
        solution = minimal_key_advanced(df, rows, algorithm=algorithm, random_seed=42, **kwargs)
        end_time = time.time()
        
        runtime = end_time - start_time
        if runtime > timeout:
            return {
                "algorithm": algorithm,
                "success": False,
                "error": "timeout", 
                "runtime": runtime,
                "solution_size": None,
                "valid": None
            }
        
        # Verify solution validity
        valid = is_key(df, rows, solution) if solution else False
        
        return {
            "algorithm": algorithm,
            "success": True,
            "error": None,
            "runtime": runtime,
            "solution_size": len(solution),
            "valid": valid,
            "solution": solution
        }
        
    except Exception as e:
        return {
            "algorithm": algorithm, 
            "success": False,
            "error": str(e),
            "runtime": time.time() - start_time,
            "solution_size": None,
            "valid": None
        }


def run_benchmark_suite(
    n_rows_list: List[int] = [10, 50, 100],
    n_cols_list: List[int] = [5, 10, 20], 
    sparsity_list: List[float] = [0.3, 0.5, 0.8],
    algorithms: List[str] = ["greedy", "exact", "ilp", "sa", "ga"],
    timeout: float = 30.0,
    n_trials: int = 3
) -> List[Dict[str, Any]]:
    """Run comprehensive benchmark suite.
    
    Parameters
    ----------
    n_rows_list : List[int]
        List of row counts to test
    n_cols_list : List[int]  
        List of column counts to test
    sparsity_list : List[float]
        List of sparsity levels to test
    algorithms : List[str]
        List of algorithms to benchmark
    timeout : float
        Per-algorithm timeout in seconds
    n_trials : int
        Number of trials per configuration
        
    Returns
    -------
    List[Dict[str, Any]]
        Detailed benchmark results
    """
    results = []
    total_configs = len(n_rows_list) * len(n_cols_list) * len(sparsity_list)
    config_num = 0
    
    print(f"Running benchmark suite: {total_configs} configurations × {len(algorithms)} algorithms × {n_trials} trials")
    print(f"Timeout per algorithm: {timeout}s")
    print()
    
    for n_rows in n_rows_list:
        for n_cols in n_cols_list:
            for sparsity in sparsity_list:
                config_num += 1
                print(f"[{config_num}/{total_configs}] Testing {n_rows} rows × {n_cols} cols, sparsity={sparsity:.1f}")
                
                # Skip configurations likely to be intractable  
                if n_cols > 25 and "exact" in algorithms:
                    print("  Skipping 'exact' algorithm (too many columns)")
                    algorithms = [alg for alg in algorithms if alg != "exact"]
                
                trial_results = []
                for trial in range(n_trials):
                    # Generate fresh data for each trial
                    df = generate_synthetic_data(n_rows, n_cols, sparsity)
                    rows = list(range(min(n_rows, 20)))  # Limit to first 20 rows for tractability
                    
                    for algorithm in algorithms:
                        print(f"  {algorithm}...", end=" ")
                        result = benchmark_algorithm(df, rows, algorithm, timeout)
                        result.update({
                            "n_rows": n_rows,
                            "n_cols": n_cols, 
                            "sparsity": sparsity,
                            "trial": trial,
                            "config_id": f"{n_rows}x{n_cols}_s{sparsity:.1f}"
                        })
                        trial_results.append(result)
                        
                        # Print quick status
                        if result["success"]:
                            print(f"✓ {result['runtime']:.3f}s (size={result['solution_size']})")
                        else:
                            print(f"✗ {result['error']}")
                
                results.extend(trial_results)
                print()
    
    return results


def analyze_results(results: List[Dict[str, Any]]) -> None:
    """Analyze and display benchmark results.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Benchmark results from run_benchmark_suite
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    print("="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print()
    
    # Overall success rates
    print("SUCCESS RATES BY ALGORITHM:")
    success_rates = df.groupby("algorithm")["success"].agg(["count", "sum", "mean"])
    success_rates.columns = ["total_runs", "successful", "success_rate"]
    success_rates["success_rate"] = success_rates["success_rate"] * 100
    print(success_rates.round(1))
    print()
    
    # Performance statistics for successful runs
    successful = df[df["success"] == True].copy()
    if len(successful) > 0:
        print("PERFORMANCE STATS (SUCCESSFUL RUNS ONLY):")
        perf_stats = successful.groupby("algorithm").agg({
            "runtime": ["mean", "std", "min", "max"],
            "solution_size": ["mean", "std", "min", "max"]
        }).round(3)
        print(perf_stats)
        print()
        
        # Best algorithm by configuration
        print("FASTEST ALGORITHM BY PROBLEM SIZE:")
        config_winners = successful.loc[successful.groupby("config_id")["runtime"].idxmin()]
        winner_counts = config_winners["algorithm"].value_counts()
        print(winner_counts)
        print()
        
        # Quality comparison (solution size)
        print("SOLUTION QUALITY (SMALLER IS BETTER):")
        quality_stats = successful.groupby("algorithm")["solution_size"].agg(["mean", "std"]).round(2)
        print(quality_stats)
        print()
    
    # Error analysis
    errors = df[df["success"] == False]
    if len(errors) > 0:
        print("ERROR BREAKDOWN:")
        error_counts = errors.groupby(["algorithm", "error"]).size().unstack(fill_value=0)
        print(error_counts)
        print()
    
    print("RECOMMENDATIONS:")
    if len(successful) > 0:
        # Find most reliable algorithm
        reliable_alg = success_rates.loc[success_rates["success_rate"].idxmax()].name
        print(f"• Most reliable algorithm: {reliable_alg} ({success_rates.loc[reliable_alg, 'success_rate']:.1f}% success)")
        
        # Find fastest algorithm on average
        avg_times = successful.groupby("algorithm")["runtime"].mean()
        fastest_alg = avg_times.idxmin()
        print(f"• Fastest algorithm on average: {fastest_alg} ({avg_times[fastest_alg]:.3f}s)")
        
        # Find best quality algorithm
        avg_sizes = successful.groupby("algorithm")["solution_size"].mean()
        best_quality_alg = avg_sizes.idxmin()
        print(f"• Best solution quality: {best_quality_alg} (avg size={avg_sizes[best_quality_alg]:.1f})")
    
    print("\nFor detailed results, save to CSV using --save option.")


def main():
    """Main benchmark runner with command line interface."""
    parser = argparse.ArgumentParser(description="Benchmark rowvoi set cover algorithms")
    parser.add_argument("--rows", nargs="+", type=int, default=[10, 50, 100],
                      help="List of row counts to test")
    parser.add_argument("--cols", nargs="+", type=int, default=[5, 10, 20], 
                      help="List of column counts to test")
    parser.add_argument("--sparsity", nargs="+", type=float, default=[0.3, 0.5, 0.8],
                      help="List of sparsity levels (0.0-1.0)")
    parser.add_argument("--algorithms", nargs="+", default=["greedy", "exact", "ilp", "sa", "ga"],
                      choices=["greedy", "exact", "ilp", "sa", "ga", "hybrid", "lp"],
                      help="Algorithms to benchmark")
    parser.add_argument("--timeout", type=float, default=30.0,
                      help="Timeout per algorithm in seconds")
    parser.add_argument("--trials", type=int, default=3,
                      help="Number of trials per configuration") 
    parser.add_argument("--save", type=str, help="Save detailed results to CSV file")
    parser.add_argument("--quick", action="store_true", 
                      help="Run quick benchmark (fewer configurations)")
    
    args = parser.parse_args()
    
    if args.quick:
        # Reduced configuration for quick testing
        results = run_benchmark_suite(
            n_rows_list=[20, 50],
            n_cols_list=[5, 10], 
            sparsity_list=[0.5],
            algorithms=args.algorithms,
            timeout=10.0,
            n_trials=2
        )
    else:
        results = run_benchmark_suite(
            n_rows_list=args.rows,
            n_cols_list=args.cols,
            sparsity_list=args.sparsity, 
            algorithms=args.algorithms,
            timeout=args.timeout,
            n_trials=args.trials
        )
    
    # Analyze results
    analyze_results(results)
    
    # Save detailed results if requested
    if args.save:
        import pandas as pd
        df = pd.DataFrame(results)
        # Remove solution column for cleaner CSV
        df_save = df.drop(columns=["solution"], errors="ignore")
        df_save.to_csv(args.save, index=False)
        print(f"\nDetailed results saved to {args.save}")


if __name__ == "__main__":
    main()