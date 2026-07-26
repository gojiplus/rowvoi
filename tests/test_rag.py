"""Tests for the RAG adapters: context selection, clarifying questions, probes."""

import numpy as np
import pandas as pd
import pytest

from rowvoi import CandidateState, StopRules
from rowvoi.rag import (
    Chunk,
    RetrievalSession,
    answer_frame,
    answer_likelihoods,
    extract_and_select,
    next_question,
    observe_answer,
    plan_context_path,
    question_values,
    select_context,
)


@pytest.fixture
def chunks():
    """Two cheap single-claim chunks and one expensive chunk covering both."""
    return [
        Chunk("a", text="The price is $40.", tokens=100),
        Chunk("b", text="It shipped in March.", tokens=100),
        Chunk("c", text="Priced at $40, shipped in March.", tokens=400),
    ]


@pytest.fixture
def claims():
    return ["price", "release_date"]


@pytest.fixture
def support():
    return {"price": {"a", "c"}, "release_date": {"b", "c"}}


@pytest.fixture
def answers():
    """q1 bisects, q2 is useless, q3 identifies uniquely."""
    return {
        "q1": ["a", "a", "b", "b"],
        "q2": ["x", "x", "x", "x"],
        "q3": ["p", "q", "r", "s"],
    }


class TestSelectContext:
    """Minimal sufficient context selection."""

    @pytest.mark.parametrize(
        "strategy", ["greedy", "exact", "sa", "ga", "lp", "hybrid"]
    )
    def test_strategies_agree_on_the_cheap_cover(
        self, strategy, chunks, claims, support
    ):
        selection = select_context(chunks, claims, support, strategy=strategy)
        assert set(selection.chunks) == {"a", "b"}
        assert selection.total_cost == 200.0
        assert selection.coverage == 1.0
        assert selection.missing_claims == set()

    def test_ilp_agrees_too(self, chunks, claims, support):
        pytest.importorskip("pulp")
        selection = select_context(chunks, claims, support, strategy="ilp")
        assert set(selection.chunks) == {"a", "b"}
        assert selection.total_cost == 200.0

    def test_bare_ids_minimize_chunk_count(self, claims, support):
        # Without token costs every chunk costs 1, so the single chunk wins
        selection = select_context(["a", "b", "c"], claims, support)
        assert selection.chunks == ["c"]

    def test_explicit_costs_override_tokens(self, chunks, claims, support):
        selection = select_context(
            chunks, claims, support, costs={"a": 999.0, "b": 999.0, "c": 1.0}
        )
        assert selection.chunks == ["c"]
        assert selection.total_cost == 1.0

    def test_epsilon_claims_drops_a_chunk(self, chunks, claims, support):
        selection = select_context(chunks, claims, support, epsilon_claims=0.5)
        assert len(selection.chunks) == 1
        assert selection.coverage == 0.5
        assert len(selection.missing_claims) == 1
        assert selection.total_cost == 100.0

    def test_unsupported_claim_is_reported_not_hidden(self, chunks):
        # "warranty" is retrieved by nothing; it must surface as missing rather
        # than being dropped from the universe and reported as 100% coverage.
        claims = ["price", "warranty"]
        support = {"price": {"a", "c"}}
        selection = select_context(chunks, claims, support)
        assert selection.coverage == 0.5
        assert selection.missing_claims == {"warranty"}
        assert selection.unsupportable_claims == {"warranty"}

    def test_dataframe_support_matrix(self, chunks, claims):
        frame = pd.DataFrame(
            [[True, False, True], [False, True, True]],
            index=claims,
            columns=["a", "b", "c"],
        )
        selection = select_context(chunks, claims, frame)
        assert set(selection.chunks) == {"a", "b"}

    def test_support_referencing_unknown_claim_raises(self, chunks, claims):
        bad = {"price": {"a"}, "not_a_claim": {"b"}}
        with pytest.raises(ValueError, match="not in `claims`"):
            select_context(chunks, claims, bad)

    def test_support_referencing_unknown_chunk_is_ignored(self, chunks, claims):
        # A judge hallucinating a chunk id must not crash the cover
        support = {"price": {"a", "ghost"}, "release_date": {"b"}}
        selection = select_context(chunks, claims, support)
        assert set(selection.chunks) == {"a", "b"}

    def test_duplicate_chunk_ids_raise(self, claims, support):
        with pytest.raises(ValueError, match="unique"):
            select_context(["a", "a"], claims, support)

    def test_no_claims_selects_nothing(self, chunks):
        selection = select_context(chunks, [], {})
        assert selection.chunks == []
        assert selection.coverage == 1.0


class TestPlanContextPath:
    """Budget-first context ordering."""

    def test_prefix_for_budget(self, chunks, claims, support):
        path = plan_context_path(chunks, claims, support)
        assert path.prefix_for_budget(50) == []
        assert len(path.prefix_for_budget(150)) == 1
        assert len(path.prefix_for_budget(250)) == 2

    def test_coverage_curve_is_monotonic(self, chunks, claims, support):
        curve = plan_context_path(chunks, claims, support).coverage_curve()
        costs = [c for c, _ in curve]
        coverages = [v for _, v in curve]
        assert costs == sorted(costs)
        assert coverages == sorted(coverages)
        assert coverages[-1] == 1.0

    def test_idf_weighting_runs(self, chunks, claims, support):
        path = plan_context_path(chunks, claims, support, weighting="idf")
        assert path.steps[0].newly_covered_weight is not None


class TestExtractAndSelect:
    """The two-LLM-call convenience wrapper."""

    def test_end_to_end_with_stubs(self, chunks):
        class Extractor:
            def extract(self, query):
                return ["price", "release_date"]

        class Judge:
            def judge(self, chunk_pairs, claims):
                assert [cid for cid, _ in chunk_pairs] == ["a", "b", "c"]
                assert all(isinstance(text, str) for _, text in chunk_pairs)
                return {"price": {"a", "c"}, "release_date": {"b", "c"}}

        selection = extract_and_select(
            "how much and when?", chunks, extractor=Extractor(), judge=Judge()
        )
        assert set(selection.chunks) == {"a", "b"}


class TestAnswerFrame:
    """Input coercion for the answer matrix."""

    def test_mapping_form(self, answers):
        frame = answer_frame(answers)
        assert list(frame.columns) == ["q1", "q2", "q3"]
        assert len(frame) == 4

    def test_nested_sequence_form(self):
        rows = [["a", "p"], ["b", "q"]]
        frame = answer_frame(rows, questions=["q1", "q2"])
        assert list(frame.columns) == ["q1", "q2"]
        assert frame.iloc[1]["q2"] == "q"

    def test_nested_sequence_needs_questions(self):
        with pytest.raises(ValueError, match="`questions` is required"):
            answer_frame([["a"], ["b"]])

    def test_ragged_rows_raise(self):
        with pytest.raises(ValueError, match="one per question"):
            answer_frame([["a", "b"], ["c"]], questions=["q1", "q2"])

    def test_dataframe_passthrough_resets_index(self):
        frame = answer_frame(pd.DataFrame({"q": ["a", "b"]}, index=[7, 9]))
        assert list(frame.index) == [0, 1]

    def test_empty_matrix_raises(self):
        with pytest.raises(ValueError, match="no questions"):
            answer_frame(pd.DataFrame())


class TestNextQuestion:
    """Clarifying-question selection by mutual information."""

    def test_uninformative_question_scores_zero(self, answers):
        values = question_values(answers)
        assert values["q2"] == 0.0

    def test_bisecting_question_yields_one_bit(self, answers):
        assert question_values(answers)["q1"] == pytest.approx(1.0)

    def test_identifying_question_yields_two_bits(self, answers):
        assert question_values(answers)["q3"] == pytest.approx(2.0)

    def test_best_question_is_the_most_informative(self, answers):
        assert next_question(answers).col == "q3"

    def test_cost_flips_the_ranking(self, answers):
        # q3 is worth 2 bits but 10x the cost; q1's 1 bit at unit cost wins
        suggestion = next_question(answers, costs={"q3": 10.0})
        assert suggestion.col == "q1"
        assert suggestion.expected_voi == pytest.approx(1.0)

    def test_normalize_reports_fraction_of_uncertainty(self, answers):
        values = question_values(answers, normalize=True)
        assert values["q3"] == pytest.approx(1.0)  # resolves everything
        assert values["q1"] == pytest.approx(0.5)

    def test_prior_skews_information_gain(self, answers):
        # A prior concentrated on the first two candidates makes q1 useless,
        # since both of them answer "a"
        values = question_values(answers, prior=[0.5, 0.5, 0.0, 0.0])
        # compute_mi adds 1e-10 inside the log, so "no information" lands a
        # hair above zero rather than exactly on it
        assert values["q1"] == pytest.approx(0.0, abs=1e-9)
        assert values["q3"] == pytest.approx(1.0)

    def test_observed_questions_are_excluded(self, answers):
        state = CandidateState.uniform(list(range(4)))
        state = observe_answer(state, answers, "q3", "r")
        assert next_question(answers, state=state).col != "q3"

    def test_mismatched_state_length_raises(self, answers):
        state = CandidateState.uniform(list(range(3)))
        with pytest.raises(ValueError, match="3 candidates"):
            next_question(answers, state=state)

    def test_bad_prior_raises(self, answers):
        with pytest.raises(ValueError, match="4 prior weights"):
            question_values(answers, prior=[1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            question_values(answers, prior=[0.0, 0.0, 0.0, 0.0])


class TestAnswerLikelihoods:
    """Soft evidence from a received answer."""

    def test_noiseless_is_a_hard_indicator(self, answers):
        likelihoods = answer_likelihoods(answers, "q1", "b")
        assert list(likelihoods) == [0.0, 0.0, 1.0, 1.0]

    def test_noise_leaves_mass_on_mismatches(self, answers):
        likelihoods = answer_likelihoods(answers, "q1", "b", noise=0.1)
        assert likelihoods[0] == pytest.approx(0.1)  # one other distinct answer
        assert likelihoods[2] == pytest.approx(0.9)

    def test_noise_splits_across_distinct_answers(self, answers):
        # q3 has four distinct answers, so 0.3 is spread over the other three
        likelihoods = answer_likelihoods(answers, "q3", "p", noise=0.3)
        assert likelihoods[0] == pytest.approx(0.7)
        assert likelihoods[1] == pytest.approx(0.1)

    def test_constant_column_is_unaffected_by_noise(self, answers):
        likelihoods = answer_likelihoods(answers, "q2", "x", noise=0.2)
        assert list(likelihoods) == [0.8] * 4

    def test_unknown_question_raises(self, answers):
        with pytest.raises(KeyError, match="nope"):
            answer_likelihoods(answers, "nope", "x")

    @pytest.mark.parametrize("noise", [-0.1, 1.0, 1.5])
    def test_noise_out_of_range_raises(self, answers, noise):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            answer_likelihoods(answers, "q1", "a", noise=noise)


class TestObserveAnswer:
    """Folding an answer into the belief."""

    def test_noiseless_update_matches_a_hard_filter(self, answers):
        frame = answer_frame(answers)
        state = CandidateState.uniform(list(range(4)))

        soft = observe_answer(state, answers, "q1", "b")
        hard = state.filter_candidates(frame, "q1", "b")

        # The soft update keeps all four rows with zeros where the hard filter
        # dropped them; the surviving mass must be identical.
        assert list(hard.candidate_rows) == [2, 3]
        surviving = [soft.posterior[r] for r in hard.candidate_rows]
        assert surviving == pytest.approx(list(hard.posterior))
        assert soft.posterior[0] == 0.0

    def test_noise_keeps_every_candidate_alive(self, answers):
        state = CandidateState.uniform(list(range(4)))
        updated = observe_answer(state, answers, "q1", "b", noise=0.1)
        assert np.all(updated.posterior > 0)
        assert updated.posterior.sum() == pytest.approx(1.0)

    def test_records_the_question_and_answer(self, answers):
        state = CandidateState.uniform(list(range(4)))
        updated = observe_answer(state, answers, "q1", "b")
        assert "q1" in updated.observed_cols
        assert updated.observed_values["q1"] == "b"

    def test_successive_answers_compose(self, answers):
        state = CandidateState.uniform(list(range(4)))
        state = observe_answer(state, answers, "q1", "b")
        state = observe_answer(state, answers, "q3", "r")
        assert state.is_unique
        assert int(np.argmax(state.posterior)) == 2

    def test_impossible_answer_raises(self, answers):
        state = CandidateState.uniform(list(range(4)))
        with pytest.raises(ValueError, match="zero likelihood"):
            observe_answer(state, answers, "q1", "never_predicted")

    def test_impossible_answer_survives_with_noise(self, answers):
        # With noise the same answer is merely surprising, not disqualifying
        state = CandidateState.uniform(list(range(4)))
        updated = observe_answer(state, answers, "q1", "never_predicted", noise=0.1)
        assert updated.posterior == pytest.approx([0.25] * 4)


class TestRetrievalSession:
    """VOI-driven probe sequencing."""

    class ScriptedRunner:
        """Answers probes as though `truth` were the right candidate."""

        def __init__(self, answers, truth):
            self.answers = answers
            self.truth = truth
            self.calls = []

        def run(self, probe):
            self.calls.append(probe)
            return self.answers[probe][self.truth]

    def test_reaches_the_truth_and_stops(self, answers):
        runner = self.ScriptedRunner(answers, truth=2)
        session = RetrievalSession(answers, runner=runner)
        history = session.run(StopRules(epsilon_posterior=0.05))

        assert session.best_candidate == 2
        assert session.state.max_posterior == pytest.approx(1.0)
        # q3 identifies uniquely, so one probe suffices
        assert len(history) == 1
        assert runner.calls == ["q3"]

    def test_cost_changes_which_probes_get_run(self, answers):
        runner = self.ScriptedRunner(answers, truth=2)
        session = RetrievalSession(answers, runner=runner, costs={"q3": 10.0})
        session.run(StopRules(epsilon_posterior=0.05))
        # q1 is now the better bits-per-cost buy, so it goes first
        assert runner.calls[0] == "q1"

    def test_respects_max_steps(self, answers):
        runner = self.ScriptedRunner(answers, truth=0)
        session = RetrievalSession(answers, runner=runner)
        history = session.run(StopRules(max_steps=1, target_unique=False))
        assert len(history) == 1

    def test_respects_cost_budget(self, answers):
        runner = self.ScriptedRunner(answers, truth=0)
        session = RetrievalSession(
            answers, runner=runner, costs={"q1": 5.0, "q2": 5.0, "q3": 5.0}
        )
        session.run(StopRules(cost_budget=6.0, target_unique=False))
        # StopRules checks the budget before each step rather than predicting
        # the next step's cost, so spend stops once the budget is reached --
        # it does not stay strictly under it. Two 5.0 probes run, then it halts.
        assert session.steps_taken == 2
        assert session.cumulative_cost == 10.0

    def test_stops_when_probes_run_out(self, answers):
        # Only an uninformative probe is available, so nothing can resolve
        useless = {"q2": ["x", "x", "x", "x"]}
        runner = self.ScriptedRunner(useless, truth=0)
        session = RetrievalSession(useless, runner=runner)
        history = session.run(StopRules(epsilon_posterior=0.01))
        assert len(history) <= 1
        assert session.state.max_posterior == pytest.approx(0.25)

    def test_epsilon_pairs_stop_rule_uses_the_outcome_matrix(self, answers):
        runner = self.ScriptedRunner(answers, truth=0)
        session = RetrievalSession(answers, runner=runner)
        session.run(StopRules(epsilon_pairs=0.0, target_unique=False))
        assert session.steps_taken >= 1

    def test_manual_driving(self, answers):
        session = RetrievalSession(answers)
        suggestion = session.next_probe()
        assert suggestion.col == "q3"

        step = session.observe("q3", "r", expected_voi=suggestion.expected_voi)
        assert step.probe == "q3"
        assert step.outcome == "r"
        assert step.entropy_before == pytest.approx(2.0)
        assert step.entropy_after == pytest.approx(0.0)
        assert step.realized_gain == pytest.approx(2.0)
        assert step.expected_voi == pytest.approx(2.0)

    def test_explicit_likelihoods_bypass_the_matrix(self, answers):
        session = RetrievalSession(answers)
        session.observe("q1", likelihoods=[0.0, 0.0, 0.0, 1.0])
        assert session.best_candidate == 3
        assert session.state.max_posterior == pytest.approx(1.0)

    def test_prior_is_honored(self, answers):
        session = RetrievalSession(answers, prior=[0.7, 0.1, 0.1, 0.1])
        assert session.best_candidate == 0
        assert session.state.posterior[0] == pytest.approx(0.7)

    def test_bad_prior_raises(self, answers):
        with pytest.raises(ValueError, match="4 prior weights"):
            RetrievalSession(answers, prior=[1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            RetrievalSession(answers, prior=[0.0, 0.0, 0.0, 0.0])

    def test_run_without_a_runner_raises(self, answers):
        session = RetrievalSession(answers)
        with pytest.raises(ValueError, match="needs a ProbeRunner"):
            session.run(StopRules(max_steps=1))

    def test_runner_can_be_supplied_at_run_time(self, answers):
        runner = self.ScriptedRunner(answers, truth=1)
        session = RetrievalSession(answers)
        session.run(StopRules(epsilon_posterior=0.05), runner=runner)
        assert session.best_candidate == 1

    def test_noise_keeps_candidates_alive_across_probes(self, answers):
        runner = self.ScriptedRunner(answers, truth=2)
        session = RetrievalSession(answers, runner=runner, noise=0.05)
        session.run(StopRules(max_steps=3, target_unique=False))
        assert np.all(session.state.posterior > 0)
        assert session.best_candidate == 2

    def test_reset_clears_history(self, answers):
        runner = self.ScriptedRunner(answers, truth=2)
        session = RetrievalSession(answers, runner=runner)
        session.run(StopRules(epsilon_posterior=0.05))
        assert session.steps_taken > 0

        session.reset()
        assert session.steps_taken == 0
        assert session.cumulative_cost == 0.0
        assert session.state.posterior == pytest.approx([0.25] * 4)

        session.reset(prior=[1.0, 0.0, 0.0, 0.0])
        assert session.state.posterior[0] == pytest.approx(1.0)


class TestCandidateStateReweight:
    """The soft-evidence primitive underneath all three adapters."""

    def test_one_hot_matches_a_hard_filter(self):
        frame = pd.DataFrame({"col": ["a", "b", "b"]})
        state = CandidateState.uniform([0, 1, 2])

        soft = state.reweight([0.0, 1.0, 1.0])
        hard = state.filter_candidates(frame, "col", "b")

        assert [soft.posterior[r] for r in hard.candidate_rows] == pytest.approx(
            list(hard.posterior)
        )

    def test_renormalizes(self):
        state = CandidateState.uniform([0, 1, 2])
        updated = state.reweight([0.2, 0.2, 0.6])
        assert updated.posterior.sum() == pytest.approx(1.0)
        assert updated.posterior == pytest.approx([0.2, 0.2, 0.6])

    def test_candidates_are_retained_for_positional_alignment(self):
        state = CandidateState.uniform([0, 1, 2])
        updated = state.reweight([0.0, 1.0, 0.0])
        assert list(updated.candidate_rows) == [0, 1, 2]
        assert list(updated.posterior) == [0.0, 1.0, 0.0]

    def test_accepts_numpy_arrays(self):
        state = CandidateState.uniform([0, 1])
        updated = state.reweight(np.array([1.0, 3.0]))
        assert updated.posterior == pytest.approx([0.25, 0.75])

    def test_records_observation(self):
        state = CandidateState.uniform([0, 1])
        updated = state.reweight([1.0, 1.0], observed_col="c", observed_value=7)
        assert updated.observed_cols == {"c"}
        assert updated.observed_values["c"] == 7

    def test_wrong_length_raises(self):
        state = CandidateState.uniform([0, 1, 2])
        with pytest.raises(ValueError, match="Expected 3 likelihoods"):
            state.reweight([1.0, 1.0])

    def test_negative_likelihood_raises(self):
        state = CandidateState.uniform([0, 1])
        with pytest.raises(ValueError, match="non-negative"):
            state.reweight([-1.0, 1.0])

    def test_zero_total_mass_raises(self):
        state = CandidateState.uniform([0, 1])
        with pytest.raises(ValueError, match="zero likelihood"):
            state.reweight([0.0, 0.0])

    def test_chained_updates_stay_normalized(self):
        state = CandidateState.uniform([0, 1, 2, 3])
        for weights in ([0.9, 0.1, 0.9, 0.1], [0.1, 0.1, 0.9, 0.9]):
            state = state.reweight(weights)
            assert state.posterior.sum() == pytest.approx(1.0)
        assert int(np.argmax(state.posterior)) == 2
