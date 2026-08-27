"""Tests for the generic weighted set cover engine."""

import pytest

from rowvoi.setcover import (
    SetCoverProblem,
    SolverUnavailableError,
    _find_solver,
    coverage_of,
    solve_set_cover,
)

ALL_STRATEGIES = ["greedy", "exact", "ilp", "sa", "ga", "lp", "hybrid"]


def skip_without_solver(strategy: str) -> None:
    """Skip an "ilp" parametrization when no LP solver is installed.

    Previously unnecessary, because _ilp silently returned a greedy result --
    which is precisely why these tests passed while never running ILP.
    """
    if strategy != "ilp":
        return
    pulp = pytest.importorskip("pulp")
    if _find_solver(pulp) is None:
        pytest.skip("no LP solver available")


@pytest.fixture
def weighted_sets():
    """Two cheap sets (cost 1 each) beat one expensive set (cost 4)."""
    return {
        "a": {"p"},
        "b": {"q"},
        "c": {"p", "q"},
    }


@pytest.fixture
def weighted_costs():
    return {"a": 1.0, "b": 1.0, "c": 4.0}


class TestSetCoverProblem:
    """Construction and basic accessors."""

    def test_universe_defaults_to_union(self):
        problem = SetCoverProblem({"a": {1, 2}, "b": {2, 3}})
        assert problem.universe == {1, 2, 3}

    def test_explicit_universe_keeps_uncoverable_elements(self):
        problem = SetCoverProblem({"a": {1}}, universe={1, 2})
        assert problem.universe == {1, 2}
        # Element 2 can never be covered, so coverage tops out at 50%
        assert problem.coverage(["a"]) == 0.5
        assert not problem.is_cover(["a"])
        assert problem.is_cover(["a"], epsilon=0.5)

    def test_cost_defaults_to_one(self):
        problem = SetCoverProblem({"a": {1}, "b": {2}}, costs={"a": 3.0})
        assert problem.cost("a") == 3.0
        assert problem.cost("b") == 1.0
        assert problem.total_cost(["a", "b"]) == 4.0

    def test_covered_is_clipped_to_universe(self):
        problem = SetCoverProblem({"a": {1, 2, 99}}, universe={1, 2})
        assert problem.covered(["a"]) == {1, 2}

    def test_empty_universe_is_trivially_covered(self):
        problem = SetCoverProblem({}, universe=set())
        assert problem.coverage([]) == 1.0
        assert problem.is_cover([])
        assert problem.solve() == []
        assert problem.plan_path().steps == []

    def test_unknown_strategy_raises(self):
        problem = SetCoverProblem({"a": {1}})
        with pytest.raises(ValueError, match="Unknown strategy"):
            problem.solve("nonsense")  # type: ignore[arg-type]


class TestStrategies:
    """All strategies must return a valid, cost-minimal cover."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_every_strategy_finds_a_valid_cover(self, strategy):
        skip_without_solver(strategy)
        problem = SetCoverProblem({"a": {1, 2}, "b": {2, 3}, "c": {3, 4}})
        selection = problem.solve(strategy, time_limit=0.5)
        assert problem.is_cover(selection)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_every_strategy_minimizes_cost_not_cardinality(
        self, strategy, weighted_sets, weighted_costs
    ):
        skip_without_solver(strategy)
        # The single set "c" covers everything but costs 4; "a"+"b" cost 2.
        # A solver that minimizes the number of sets picks the wrong one.
        problem = SetCoverProblem(weighted_sets, costs=weighted_costs)
        selection = problem.solve(strategy, time_limit=0.5)
        assert problem.is_cover(selection)
        assert problem.total_cost(selection) == 2.0
        assert set(selection) == {"a", "b"}

    def test_exact_beats_smallest_cover_when_costs_differ(
        self, weighted_sets, weighted_costs
    ):
        # Guards the specific regression: "exact" used to return the first
        # cover of the smallest size, i.e. ["c"], ignoring cost entirely.
        problem = SetCoverProblem(weighted_sets, costs=weighted_costs)
        assert set(problem.solve("exact")) == {"a", "b"}

    def test_exact_still_returns_smallest_under_uniform_costs(self):
        problem = SetCoverProblem({"a": {1}, "b": {2}, "c": {1, 2}})
        assert problem.solve("exact") == ["c"]

    @pytest.mark.parametrize("strategy", ["greedy", "exact", "ilp"])
    def test_epsilon_permits_a_cheaper_partial_cover(self, strategy):
        skip_without_solver(strategy)
        # "big" covers 3 of 4 elements; "tail" is needed only for the last one.
        problem = SetCoverProblem(
            {"big": {1, 2, 3}, "tail": {4}}, costs={"big": 1.0, "tail": 1.0}
        )
        full = problem.solve(strategy)
        assert set(full) == {"big", "tail"}

        partial = problem.solve(strategy, epsilon=0.25)
        assert set(partial) == {"big"}
        assert problem.coverage(partial) == 0.75

    def test_epsilon_with_uncoverable_element(self):
        # Element 3 is in the universe but in no set. Without epsilon there is
        # no full cover; with epsilon the solver must still not claim it.
        problem = SetCoverProblem({"a": {1}, "b": {2}}, universe={1, 2, 3})
        selection = problem.solve("greedy", epsilon=1 / 3)
        assert problem.coverage(selection) == pytest.approx(2 / 3)
        assert 3 not in problem.covered(selection)

    def test_ilp_epsilon_with_uncoverable_element(self):
        pulp = pytest.importorskip("pulp")
        assert pulp  # silence the unused-import linter
        problem = SetCoverProblem({"a": {1}, "b": {2}}, universe={1, 2, 3})
        selection = problem.solve("ilp", epsilon=1 / 3)
        # The ILP must not count element 3 toward its coverage target
        assert problem.coverage(selection) == pytest.approx(2 / 3)


class TestSolverAvailability:
    """ILP must fail loudly rather than quietly becoming greedy.

    Greedy is an ln(m) approximation and ILP is exact. Returning one where the
    caller asked for the other is a silent correctness change, and it made
    `test_ilp_algorithm` pass for years while never running the ILP code.
    """

    def test_raises_when_pulp_is_missing(self, monkeypatch):
        # Simulate an environment without the optimization extra.
        import builtins

        real_import = builtins.__import__

        def no_pulp(name, *args, **kwargs):
            if name == "pulp":
                raise ImportError("No module named 'pulp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pulp)

        problem = SetCoverProblem({"a": {1}, "b": {2}})
        with pytest.raises(SolverUnavailableError, match="optimization"):
            problem.solve("ilp")

    def test_raises_when_no_solver_is_available(self, monkeypatch):
        pulp = pytest.importorskip("pulp")
        # pulp installed, but nothing it can actually execute. Previously this
        # surfaced as PulpSolverError from deep inside solve().
        monkeypatch.setattr(
            "rowvoi.setcover._find_solver", lambda *a, **k: None, raising=True
        )
        assert pulp  # silence the unused-import linter

        problem = SetCoverProblem({"a": {1}, "b": {2}})
        with pytest.raises(SolverUnavailableError, match="pulp\\[cbc\\]"):
            problem.solve("ilp")

    def test_prefers_the_non_deprecated_solver(self, monkeypatch):
        pulp = pytest.importorskip("pulp")

        # Make COIN_CMD look available and runnable; the search should stop
        # there rather than falling through to the deprecated PULP_CBC_CMD.
        monkeypatch.setattr(
            pulp.COIN_CMD, "available", lambda self: "/usr/bin/cbc", raising=False
        )
        monkeypatch.setattr(
            "rowvoi.setcover._solver_runs", lambda *a, **k: True, raising=True
        )
        assert type(_find_solver(pulp)).__name__ == "COIN_CMD"

    def test_skips_a_solver_that_is_found_but_cannot_run(self, monkeypatch):
        """available() only locates a binary; a wrong-architecture one is killed.

        pulp[cbc] installs a platform-specific CBC. On a mismatched host the
        binary is found and then SIGKILLed, so preferring it on availability
        alone hands back a solver that raises PulpSolverError inside solve().
        """
        pulp = pytest.importorskip("pulp")

        monkeypatch.setattr(
            pulp.COIN_CMD, "available", lambda self: "/usr/bin/cbc", raising=False
        )
        monkeypatch.setattr(
            "rowvoi.setcover._solver_runs",
            lambda _pulp, solver: type(solver).__name__ != "COIN_CMD",
            raising=True,
        )

        found = _find_solver(pulp)

        assert found is not None
        assert type(found).__name__ != "COIN_CMD"

    def test_falls_through_to_the_bundled_solver(self):
        pulp = pytest.importorskip("pulp")
        # On a stock install COIN_CMD.available() is None, so the bundled
        # PULP_CBC_CMD is what actually gets used.
        solver = _find_solver(pulp)
        if solver is None:
            pytest.skip("no LP solver available")
        assert solver.available()

    def test_other_strategies_are_unaffected_by_a_missing_solver(self, monkeypatch):
        # Only "ilp" depends on a solver; the rest must keep working.
        monkeypatch.setattr(
            "rowvoi.setcover._find_solver", lambda *a, **k: None, raising=True
        )
        problem = SetCoverProblem({"a": {1}, "b": {2}})
        for strategy in ("greedy", "exact", "sa", "ga", "lp", "hybrid"):
            assert problem.is_cover(problem.solve(strategy, time_limit=0.2))


class TestCoverPath:
    """Ordered acquisition paths."""

    def test_path_orders_by_gain_per_cost(self):
        problem = SetCoverProblem(
            {"cheap": {1, 2}, "pricey": {3}},
            costs={"cheap": 1.0, "pricey": 5.0},
        )
        path = problem.plan_path()
        assert path.names() == ["cheap", "pricey"]

    def test_coverage_is_monotonic(self):
        problem = SetCoverProblem({"a": {1, 2}, "b": {3}, "c": {4, 5}})
        previous = 0.0
        for step in problem.plan_path().steps:
            assert step.coverage >= previous
            previous = step.coverage
        assert previous == 1.0

    def test_prefix_for_budget(self):
        problem = SetCoverProblem({"a": {1}, "b": {2}}, costs={"a": 10.0, "b": 10.0})
        path = problem.plan_path()
        assert path.prefix_for_budget(5.0) == []
        assert len(path.prefix_for_budget(10.0)) == 1
        assert len(path.prefix_for_budget(25.0)) == 2

    def test_prefix_for_epsilon(self):
        problem = SetCoverProblem({"big": {1, 2, 3}, "tail": {4}})
        path = problem.plan_path()
        assert path.prefix_for_epsilon(0.25) == ["big"]
        assert path.prefix_for_epsilon(0.0) == ["big", "tail"]

    def test_coverage_curve_pairs_cost_with_coverage(self):
        problem = SetCoverProblem({"a": {1}, "b": {2}}, costs={"a": 2.0, "b": 3.0})
        curve = problem.plan_path().coverage_curve()
        assert curve[0] == (2.0, 0.5)
        assert curve[-1] == (5.0, 1.0)

    def test_idf_weighting_prioritizes_scarce_elements(self):
        # "wide" covers two elements, but each is reachable through two other
        # sets as well, so IDF discounts them to 1/3 apiece (total 2/3).
        # "rare" is reachable only through "narrow", so it keeps full weight.
        # Uniform counts raw elements and picks "wide"; IDF picks "narrow".
        problem = SetCoverProblem(
            {
                "wide": {"c1", "c2"},
                "narrow": {"rare"},
                "dup1a": {"c1"},
                "dup1b": {"c1"},
                "dup2a": {"c2"},
                "dup2b": {"c2"},
            }
        )
        uniform = problem.plan_path(weighting="uniform").names()
        idf = problem.plan_path(weighting="idf").names()
        assert uniform[0] == "wide"
        assert idf[0] == "narrow"

    def test_idf_records_weights(self):
        problem = SetCoverProblem({"a": {1}, "b": {2}})
        step = problem.plan_path(weighting="idf").steps[0]
        assert step.newly_covered_weight is not None
        assert step.cumulative_covered_weight is not None

    def test_uniform_leaves_weights_unset(self):
        problem = SetCoverProblem({"a": {1}})
        step = problem.plan_path(weighting="uniform").steps[0]
        assert step.newly_covered_weight is None

    def test_entropy_objective_produces_a_valid_path(self):
        problem = SetCoverProblem({"a": {1, 2}, "b": {3}, "c": {4}})
        path = problem.plan_path(objective="entropy")
        assert set(path.names()) == {"a", "b", "c"}
        assert path.steps[-1].coverage == 1.0

    def test_element_weights_handles_uncoverable_element(self):
        problem = SetCoverProblem({"a": {1}}, universe={1, 2})
        weights = problem.element_weights("idf")
        assert weights[1] == 1.0  # covered by exactly one set
        assert weights[2] == 1.0  # covered by none, falls back to 1.0


class TestConvenienceWrappers:
    """Module-level helpers."""

    def test_solve_set_cover(self):
        selection = solve_set_cover({"a": {1, 2}, "b": {3}})
        assert set(selection) == {"a", "b"}

    def test_coverage_of(self):
        sets = {"a": {1, 2}, "b": {3, 4}}
        assert coverage_of(sets, ["a"]) == 0.5
        assert coverage_of(sets, ["a", "b"]) == 1.0
