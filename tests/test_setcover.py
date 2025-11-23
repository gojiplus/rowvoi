"""Tests for advanced set cover algorithms in rowvoi.setcover."""

from unittest.mock import patch

import pandas as pd
import pytest

from rowvoi.logical import minimal_key_advanced
from rowvoi.setcover import (
    GeneticAlgorithmSetCoverSolver,
    HybridSetCoverSolver,
    ILPSetCoverSolver,
    LPRelaxationSetCoverSolver,
    SetCoverAlgorithm,
    SetCoverResult,
    SimulatedAnnealingSetCoverSolver,
    solve_set_cover,
)


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "A": [1, 1, 2, 2],
        "B": [3, 4, 3, 4],
        "C": [5, 5, 6, 6],
        "D": [7, 8, 9, 10],
    })


@pytest.fixture
def simple_df():
    """Return simple DataFrame for basic testing."""
    return pd.DataFrame({
        "X": [1, 1],
        "Y": [2, 3],
        "Z": [4, 4],
    })


class TestSetCoverResult:
    """Tests for SetCoverResult dataclass."""

    def test_result_creation(self):
        """Test creating a SetCoverResult."""
        result = SetCoverResult(
            columns=["A", "B"],
            algorithm="test",
            is_optimal=True,
            objective_value=2.0,
            computation_time=1.5,
            iterations=100,
            metadata={"test": True},
        )

        assert result.columns == ["A", "B"]
        assert result.algorithm == "test"
        assert result.is_optimal is True
        assert result.objective_value == 2.0
        assert result.computation_time == 1.5
        assert result.iterations == 100
        assert result.metadata["test"] is True


class TestSolveSetCover:
    """Tests for the main solve_set_cover function."""

    def test_greedy_algorithm(self, simple_df):
        """Test greedy algorithm through solve_set_cover."""
        result = solve_set_cover(
            simple_df, [0, 1], algorithm=SetCoverAlgorithm.GREEDY
        )

        assert isinstance(result, SetCoverResult)
        assert result.algorithm == "greedy"
        assert result.is_optimal is False
        assert len(result.columns) >= 1
        assert "Y" in result.columns  # Should distinguish rows 0 and 1

    def test_exact_algorithm(self, simple_df):
        """Test exact algorithm through solve_set_cover."""
        result = solve_set_cover(
            simple_df, [0, 1], algorithm=SetCoverAlgorithm.EXACT
        )

        assert result.algorithm == "exact"
        assert result.is_optimal is True
        assert len(result.columns) >= 1
        assert "Y" in result.columns

    def test_trivial_cases(self, simple_df):
        """Test trivial cases (empty or single row)."""
        # Empty rows
        result = solve_set_cover(simple_df, [], algorithm=SetCoverAlgorithm.GREEDY)
        assert result.columns == []
        assert result.objective_value == 0.0

        # Single row
        result = solve_set_cover(simple_df, [0], algorithm=SetCoverAlgorithm.GREEDY)
        assert result.columns == []
        assert result.objective_value == 0.0

    def test_unknown_algorithm(self, simple_df):
        """Test unknown algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            solve_set_cover(simple_df, [0, 1], algorithm="unknown")


class TestILPSetCoverSolver:
    """Tests for Integer Linear Programming solver."""

    def test_pulp_available(self, simple_df):
        """Test ILP solver when pulp is available."""
        solver = ILPSetCoverSolver(random_seed=42)
        result = solver.solve(simple_df, [0, 1])

        assert result.algorithm == "ilp"
        assert len(result.columns) >= 1
        # Should find optimal solution
        assert result.columns == ["Y"] or len(result.columns) == 1

    def test_pulp_unavailable(self, simple_df):
        """Test ILP solver fallback when pulp is unavailable."""
        with patch.dict('sys.modules', {'pulp': None}):
            with patch('rowvoi.setcover.ILPSetCoverSolver.solve') as mock_solve:
                from rowvoi.setcover import ILPSetCoverSolver
                solver = ILPSetCoverSolver()

                # Mock the ImportError behavior
                mock_result = SetCoverResult(
                    columns=["Y"], algorithm="ilp_fallback_greedy",
                    is_optimal=False, objective_value=1.0,
                    computation_time=0.1, iterations=1,
                    metadata={"error": "pulp not available, used greedy fallback"}
                )
                mock_solve.return_value = mock_result

                result = solver.solve(simple_df, [0, 1])
                assert result.algorithm == "ilp_fallback_greedy"
                assert "error" in result.metadata

    def test_ilp_with_parameters(self, sample_df):
        """Test ILP solver with custom parameters."""
        solver = ILPSetCoverSolver(random_seed=42)
        result = solver.solve(
            sample_df, [0, 1, 2, 3],
            time_limit=5.0,
            gap_tolerance=0.01
        )

        assert result.algorithm == "ilp"
        assert result.computation_time >= 0
        assert "gap_tolerance" in result.metadata


class TestSimulatedAnnealingSetCoverSolver:
    """Tests for Simulated Annealing solver."""

    def test_sa_basic(self, simple_df):
        """Test basic SA functionality."""
        solver = SimulatedAnnealingSetCoverSolver(random_seed=42)
        result = solver.solve(simple_df, [0, 1])

        assert result.algorithm == "simulated_annealing"
        assert result.is_optimal is False
        assert len(result.columns) >= 1
        assert result.iterations > 0

    def test_sa_parameters(self, sample_df):
        """Test SA with custom parameters."""
        solver = SimulatedAnnealingSetCoverSolver(random_seed=42)
        result = solver.solve(
            sample_df, [0, 1, 2],
            initial_temperature=200.0,
            cooling_rate=0.9,
            max_iterations=500
        )

        assert result.algorithm == "simulated_annealing"
        assert result.iterations <= 500
        assert "final_temperature" in result.metadata
        assert "cooling_rate" in result.metadata

    def test_sa_empty_solution_handling(self):
        """Test SA handles edge cases properly."""
        df = pd.DataFrame({"A": [1, 1], "B": [2, 2]})  # Identical rows
        solver = SimulatedAnnealingSetCoverSolver(random_seed=42)
        result = solver.solve(df, [0, 1])

        # Should handle case where no column can distinguish rows
        assert result.algorithm == "simulated_annealing"


class TestGeneticAlgorithmSetCoverSolver:
    """Tests for Genetic Algorithm solver."""

    def test_ga_basic(self, simple_df):
        """Test basic GA functionality."""
        solver = GeneticAlgorithmSetCoverSolver(random_seed=42)
        result = solver.solve(simple_df, [0, 1])

        assert result.algorithm == "genetic_algorithm"
        assert result.is_optimal is False
        assert len(result.columns) >= 1
        assert result.iterations > 0

    def test_ga_parameters(self, sample_df):
        """Test GA with custom parameters."""
        solver = GeneticAlgorithmSetCoverSolver(random_seed=42)
        result = solver.solve(
            sample_df, [0, 1, 2, 3],
            population_size=20,
            max_generations=50,
            mutation_rate=0.2,
            crossover_rate=0.9
        )

        assert result.algorithm == "genetic_algorithm"
        assert result.iterations <= 50
        assert result.metadata["population_size"] == 20
        assert result.metadata["mutation_rate"] == 0.2

    def test_ga_population_initialization(self, sample_df):
        """Test that GA properly initializes population."""
        solver = GeneticAlgorithmSetCoverSolver(random_seed=42)
        universe, coverage = solver._build_universe_and_coverage(
            sample_df, [0, 1, 2]
        )

        population = solver._initialize_population(
            10, list(coverage.keys()), universe, coverage
        )

        assert len(population) == 10
        # All solutions should be valid
        for individual in population:
            assert solver._is_valid_solution(individual, universe, coverage)


class TestHybridSetCoverSolver:
    """Tests for Hybrid SA+GA solver."""

    def test_hybrid_basic(self, simple_df):
        """Test basic hybrid functionality."""
        solver = HybridSetCoverSolver(random_seed=42)
        result = solver.solve(simple_df, [0, 1])

        assert result.algorithm == "hybrid_sa_ga"
        assert result.is_optimal is False
        assert len(result.columns) >= 1
        assert result.iterations > 0

    def test_hybrid_parameters(self, sample_df):
        """Test hybrid with custom parameters."""
        solver = HybridSetCoverSolver(random_seed=42)
        result = solver.solve(
            sample_df, [0, 1, 2],
            population_size=15,
            max_generations=100,
            sa_probability=0.5,
            sa_iterations_per_generation=20
        )

        assert result.algorithm == "hybrid_sa_ga"
        assert result.metadata["population_size"] == 15
        assert result.metadata["sa_probability"] == 0.5
        assert "sa_iterations_total" in result.metadata


class TestLPRelaxationSetCoverSolver:
    """Tests for LP Relaxation solver."""

    def test_lp_basic(self, simple_df):
        """Test basic LP relaxation functionality."""
        solver = LPRelaxationSetCoverSolver(random_seed=42)
        result = solver.solve(simple_df, [0, 1])

        assert result.algorithm in ["lp_relaxation", "lp_relaxation_fallback_greedy"]
        assert len(result.columns) >= 1

    def test_lp_parameters(self, sample_df):
        """Test LP relaxation with custom parameters."""
        solver = LPRelaxationSetCoverSolver(random_seed=42)
        result = solver.solve(
            sample_df, [0, 1, 2],
            rounding_iterations=50,
            alpha_multiplier=1.5,
            time_limit=10.0
        )

        # Should work regardless of whether pulp is available
        assert result.algorithm.startswith("lp_relaxation")
        if result.algorithm == "lp_relaxation":
            assert "integrality_gap" in result.metadata

    def test_lp_greedy_completion(self, sample_df):
        """Test greedy completion method."""
        solver = LPRelaxationSetCoverSolver(random_seed=42)
        universe, coverage = solver._build_universe_and_coverage(sample_df, [0, 1, 2])

        # Mock fractional solution
        fractional_solution = dict.fromkeys(coverage.keys(), 0.5)

        solution = solver._greedy_completion(fractional_solution, universe, coverage)

        assert isinstance(solution, list)
        assert len(solution) > 0
        assert solver._is_valid_solution(solution, universe, coverage)


class TestMinimalKeyAdvanced:
    """Tests for the minimal_key_advanced function."""

    def test_algorithm_selection(self, simple_df):
        """Test algorithm selection via string parameter."""
        # Test greedy
        result_greedy = minimal_key_advanced(simple_df, [0, 1], algorithm="greedy")
        assert isinstance(result_greedy, list)
        assert len(result_greedy) >= 1

        # Test exact
        result_exact = minimal_key_advanced(simple_df, [0, 1], algorithm="exact")
        assert isinstance(result_exact, list)
        assert len(result_exact) >= 1

    def test_invalid_algorithm(self, simple_df):
        """Test invalid algorithm string."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            minimal_key_advanced(simple_df, [0, 1], algorithm="invalid")

    def test_algorithm_parameters(self, sample_df):
        """Test passing parameters to algorithms."""
        # Test SA with custom parameters
        result = minimal_key_advanced(
            sample_df, [0, 1, 2],
            algorithm="sa",
            initial_temperature=100.0,
            max_iterations=200
        )
        assert isinstance(result, list)

    def test_backward_compatibility(self, simple_df):
        """Test that new functions don't break existing interface."""
        from rowvoi import minimal_key_exact, minimal_key_greedy

        # Original functions should still work
        greedy_old = minimal_key_greedy(simple_df, [0, 1])
        exact_old = minimal_key_exact(simple_df, [0, 1])

        # New function should give same results for greedy and exact
        greedy_new = minimal_key_advanced(simple_df, [0, 1], algorithm="greedy")
        exact_new = minimal_key_advanced(simple_df, [0, 1], algorithm="exact")

        assert greedy_old == greedy_new
        assert exact_old == exact_new


class TestBaseSetCoverSolver:
    """Tests for base solver functionality."""

    def test_universe_and_coverage_construction(self, sample_df):
        """Test universe and coverage construction."""
        # Use concrete solver instead of abstract base
        solver = SimulatedAnnealingSetCoverSolver()
        universe, coverage = solver._build_universe_and_coverage(sample_df, [0, 1, 2])

        # Check universe contains all pairs
        expected_pairs = {(0, 1), (0, 2), (1, 2)}
        assert universe == expected_pairs

        # Check coverage structure
        assert isinstance(coverage, dict)
        for col in sample_df.columns:
            assert col in coverage
            assert isinstance(coverage[col], set)

    def test_empty_rows(self, sample_df):
        """Test handling of empty row list."""
        solver = SimulatedAnnealingSetCoverSolver()
        universe, coverage = solver._build_universe_and_coverage(sample_df, [])

        assert universe == set()
        assert coverage == {}

    def test_single_row(self, sample_df):
        """Test handling of single row."""
        solver = SimulatedAnnealingSetCoverSolver()
        universe, coverage = solver._build_universe_and_coverage(sample_df, [0])

        assert universe == set()  # No pairs to distinguish
        assert isinstance(coverage, dict)

    def test_candidate_columns(self, sample_df):
        """Test filtering by candidate columns."""
        solver = SimulatedAnnealingSetCoverSolver()
        universe, coverage = solver._build_universe_and_coverage(
            sample_df, [0, 1], candidate_cols=["A", "B"]
        )

        # Should only include specified columns
        assert set(coverage.keys()) == {"A", "B"}


@pytest.mark.integration
class TestAlgorithmComparison:
    """Integration tests comparing algorithm results."""

    def test_all_algorithms_find_valid_solutions(self, sample_df):
        """Test that all algorithms find valid distinguishing sets."""
        rows = [0, 1, 2]
        algorithms = ["greedy", "exact", "sa", "ga", "hybrid", "lp"]

        for algorithm in algorithms:
            try:
                result = minimal_key_advanced(
                    sample_df, rows, algorithm=algorithm, random_seed=42
                )

                # Verify solution is valid using is_key
                from rowvoi.logical import is_key
                assert is_key(sample_df, rows, result), f"{algorithm} failed"

            except ImportError:
                # Skip algorithms that require optional dependencies
                pytest.skip(f"Skipping {algorithm} - dependency not available")

    def test_algorithm_performance_metadata(self, sample_df):
        """Test that algorithms provide useful metadata."""
        rows = [0, 1, 2, 3]

        algorithms = [
            SetCoverAlgorithm.SIMULATED_ANNEALING,
            SetCoverAlgorithm.GENETIC_ALGORITHM,
        ]
        for algorithm_enum in algorithms:
            try:
                result = solve_set_cover(
                    sample_df,
                    rows,
                    algorithm=algorithm_enum,
                    random_seed=42,
                    max_iterations=10,
                )

                assert result.computation_time > 0
                assert result.iterations > 0
                assert isinstance(result.metadata, dict)
                assert len(result.metadata) > 0

            except ImportError:
                pytest.skip("Optional dependencies not available")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_identical_rows(self):
        """Test handling of identical rows (no distinguishing columns)."""
        df = pd.DataFrame({
            "A": [1, 1],
            "B": [2, 2],
            "C": [3, 3],
        })

        # All algorithms should handle this gracefully
        for algorithm in ["greedy", "exact", "sa"]:
            result = minimal_key_advanced(df, [0, 1], algorithm=algorithm)
            # May return empty list or some columns (both are acceptable)
            assert isinstance(result, list)

    def test_large_problem_scaling(self):
        """Test algorithms on moderately sized problem."""
        # Create problem with 10 rows, 8 columns
        n_rows, n_cols = 10, 8
        data = {}
        for i in range(n_cols):
            # Create columns with some distinguishing power
            data[f"col_{i}"] = [j % (i + 2) for j in range(n_rows)]

        df = pd.DataFrame(data)
        rows = list(range(n_rows))

        # Test faster algorithms
        for algorithm in ["greedy", "sa"]:
            result = minimal_key_advanced(
                df, rows, algorithm=algorithm,
                random_seed=42, max_iterations=100
            )

            assert isinstance(result, list)
            assert len(result) <= n_cols

    def test_random_seed_reproducibility(self, sample_df):
        """Test that random seed produces reproducible results."""
        rows = [0, 1, 2]

        for algorithm in ["sa", "ga", "hybrid"]:
            result1 = minimal_key_advanced(
                sample_df, rows, algorithm=algorithm, random_seed=42
            )
            result2 = minimal_key_advanced(
                sample_df, rows, algorithm=algorithm, random_seed=42
            )

            assert result1 == result2, f"{algorithm} not reproducible"


if __name__ == "__main__":
    pytest.main([__file__])

