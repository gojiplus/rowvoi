"""Clarifying questions: ask the one question that best splits the candidates.

The RAG analogue of "which column should I observe next". When the retriever
returns k chunks with near-tied scores, the posterior over "which one actually
answers the query" is flat. Stuffing all k into the context or guessing the top
one both discard that ambiguity; asking one good question resolves it.

The mutual information here is computed by
:class:`rowvoi.policies.CandidateMIPolicy` unchanged -- a predicted answer
matrix is structurally the same object as a table of candidate rows.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core import CandidateState, FeatureSuggestion
from ..policies import CandidateMIPolicy
from .protocols import Question


def answer_frame(
    answers: pd.DataFrame | Mapping[Question, Sequence[Any]] | Sequence[Sequence[Any]],
    *,
    questions: Sequence[Question] | None = None,
) -> pd.DataFrame:
    """Coerce a predicted answer matrix into a candidates x questions frame.

    Parameters
    ----------
    answers : DataFrame | Mapping[Question, Sequence] | Sequence[Sequence]
        Three accepted shapes:
        - DataFrame: one row per candidate, one column per question
        - Mapping: question -> per-candidate answers (column-oriented)
        - Nested sequence: `answers[i][q]` (row-oriented); needs `questions`
    questions : Sequence[Question], optional
        Column labels. Required for the nested-sequence form

    Returns
    -------
    pd.DataFrame
        Candidates as positional rows, questions as columns
    """
    if isinstance(answers, pd.DataFrame):
        frame = answers.reset_index(drop=True)
    elif isinstance(answers, Mapping):
        frame = pd.DataFrame({q: list(vals) for q, vals in answers.items()})
    else:
        rows = [list(row) for row in answers]
        if questions is None:
            raise ValueError(
                "`questions` is required when `answers` is a nested sequence"
            )
        if any(len(row) != len(questions) for row in rows):
            raise ValueError(
                f"Every candidate row must have {len(questions)} answers, "
                "one per question"
            )
        frame = pd.DataFrame(rows, columns=pd.Index(list(questions)))

    if frame.shape[1] == 0:
        raise ValueError("Answer matrix has no questions")

    return frame


def _as_state(
    frame: pd.DataFrame, state: CandidateState | None, prior: Sequence[float] | None
) -> CandidateState:
    """Resolve the candidate state to score against."""
    n = len(frame)
    if state is not None:
        if len(state.candidate_rows) != n:
            raise ValueError(
                f"State has {len(state.candidate_rows)} candidates but the answer "
                f"matrix has {n} rows"
            )
        return state
    if prior is not None:
        weights = np.asarray(prior, dtype=float)
        if weights.shape != (n,):
            raise ValueError(f"Expected {n} prior weights, got shape {weights.shape}")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Prior must be non-negative with positive total mass")
        return CandidateState(
            candidate_rows=list(range(n)),
            posterior=weights / weights.sum(),
            observed_cols=set(),
            observed_values={},
        )
    return CandidateState.uniform(list(range(n)))


def question_values(
    answers: pd.DataFrame | Mapping[Question, Sequence[Any]] | Sequence[Sequence[Any]],
    *,
    questions: Sequence[Question] | None = None,
    state: CandidateState | None = None,
    prior: Sequence[float] | None = None,
    normalize: bool = False,
) -> dict[Question, float]:
    """Score every question by expected information gain, in bits.

    Returns the whole ranking rather than just the winner -- useful for
    showing a user their options, or for logging why a question was chosen.

    Parameters
    ----------
    answers : DataFrame | Mapping | nested sequence
        Predicted answer matrix; see :func:`answer_frame`
    questions : Sequence[Question], optional
        Column labels for the nested-sequence form
    state : CandidateState, optional
        Current belief. Defaults to uniform over candidates
    prior : Sequence[float], optional
        Per-candidate prior weights (e.g. retrieval scores), used when `state`
        is not given. Normalized internally
    normalize : bool, default False
        Divide by prior entropy, giving a 0-1 fraction-of-uncertainty-resolved

    Returns
    -------
    dict[Question, float]
        Mutual information per question, in bits (or fractions if normalized)
    """
    frame = answer_frame(answers, questions=questions)
    candidate_state = _as_state(frame, state, prior)
    policy = CandidateMIPolicy(normalize=normalize)
    return {
        col: policy.compute_mi(frame, candidate_state, col) for col in frame.columns
    }


def next_question(
    answers: pd.DataFrame | Mapping[Question, Sequence[Any]] | Sequence[Sequence[Any]],
    *,
    questions: Sequence[Question] | None = None,
    state: CandidateState | None = None,
    prior: Sequence[float] | None = None,
    costs: Mapping[Question, float] | None = None,
    normalize: bool = False,
) -> FeatureSuggestion:
    """Pick the question with the best information gain per unit cost.

    Parameters
    ----------
    answers : DataFrame | Mapping | nested sequence
        Predicted answer matrix; see :func:`answer_frame`
    questions : Sequence[Question], optional
        Column labels for the nested-sequence form
    state : CandidateState, optional
        Current belief. Defaults to uniform over candidates. Questions already
        in `state.observed_cols` are excluded
    prior : Sequence[float], optional
        Per-candidate prior weights, used when `state` is not given
    costs : Mapping[Question, float], optional
        Per-question cost. With users, cost is patience: a yes/no question is
        cheap, "paste your config" is not. Ranking is by MI/cost
    normalize : bool, default False
        Score against normalized MI instead of raw bits

    Returns
    -------
    FeatureSuggestion
        `.col` is the chosen question, `.expected_voi` its information gain in
        bits, `.score` the cost-adjusted ranking value. `.col` is None when no
        question is left to ask

    Examples
    --------
    >>> # q1 splits the four candidates evenly; q2 tells them apart not at all
    >>> answers = {"q1": ["a", "a", "b", "b"], "q2": ["x", "x", "x", "x"]}
    >>> suggestion = next_question(answers)
    >>> suggestion.col
    'q1'
    >>> f"{suggestion.expected_voi:.2f}"
    '1.00'
    """
    frame = answer_frame(answers, questions=questions)
    candidate_state = _as_state(frame, state, prior)
    policy = CandidateMIPolicy(normalize=normalize, costs=costs)
    return policy.suggest(frame, candidate_state)


def answer_likelihoods(
    answers: pd.DataFrame | Mapping[Question, Sequence[Any]] | Sequence[Sequence[Any]],
    question: Question,
    value: Any,
    *,
    questions: Sequence[Question] | None = None,
    noise: float = 0.0,
) -> np.ndarray:
    """P(observed answer | candidate) for one question.

    With `noise=0` this is a hard indicator. With `noise>0` a candidate whose
    predicted answer disagrees keeps a share of the mass, so one surprising
    answer -- a user typo, a shaky prediction -- cannot eliminate the right
    candidate outright.

    Parameters
    ----------
    answers : DataFrame | Mapping | nested sequence
        Predicted answer matrix; see :func:`answer_frame`
    question : Question
        Which question was asked
    value : Any
        The answer actually received
    questions : Sequence[Question], optional
        Column labels for the nested-sequence form
    noise : float, default 0.0
        Probability that a candidate produces an answer other than its
        predicted one, spread evenly over the other observed answers. Must be
        in [0, 1)

    Returns
    -------
    np.ndarray
        Likelihood per candidate, in candidate order
    """
    if not 0.0 <= noise < 1.0:
        raise ValueError(f"noise must be in [0, 1), got {noise}")

    frame = answer_frame(answers, questions=questions)
    if question not in frame.columns:
        raise KeyError(f"Unknown question: {question!r}")

    column = frame[question]
    matches = (column == value).to_numpy()

    n_distinct = column.nunique(dropna=False)
    spread = noise == 0.0 or n_distinct <= 1
    off_value = 0.0 if spread else noise / (n_distinct - 1)

    return np.where(matches, 1.0 - noise, off_value)


def observe_answer(
    state: CandidateState,
    answers: pd.DataFrame | Mapping[Question, Sequence[Any]] | Sequence[Sequence[Any]],
    question: Question,
    value: Any,
    *,
    questions: Sequence[Question] | None = None,
    noise: float = 0.0,
) -> CandidateState:
    """Fold a received answer into the belief over candidates.

    Soft by construction: this reweights via
    :meth:`rowvoi.CandidateState.reweight` rather than filtering, so candidates
    are never dropped and the posterior stays aligned with the answer matrix
    across successive questions.

    Parameters
    ----------
    state : CandidateState
        Belief before the answer
    answers : DataFrame | Mapping | nested sequence
        Predicted answer matrix; see :func:`answer_frame`
    question : Question
        Which question was asked
    value : Any
        The answer actually received
    questions : Sequence[Question], optional
        Column labels for the nested-sequence form
    noise : float, default 0.0
        Per-candidate probability of an off-prediction answer

    Returns
    -------
    CandidateState
        Updated belief, with `question` recorded in `observed_cols`

    Raises
    ------
    ValueError
        If no candidate could have produced `value` (with `noise=0`). That
        means the candidate set is wrong, not merely narrowed -- re-retrieve
        rather than continuing to ask
    """
    likelihoods = answer_likelihoods(
        answers, question, value, questions=questions, noise=noise
    )
    return state.reweight(likelihoods, observed_col=question, observed_value=value)
