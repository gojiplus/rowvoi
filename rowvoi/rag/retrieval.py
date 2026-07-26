"""Adaptive retrieval: choose the next probe by value of information.

Fixed top-k and fixed multi-hop both spend the same budget on every query --
the easy ones and the genuinely ambiguous ones alike. This module treats a
retrieval probe (a sub-query, an index, a reranker, a tool call) the way
:mod:`rowvoi.rag.questions` treats a clarifying question: score it by how much
it is expected to sharpen the posterior, run the best one, stop when the
residual uncertainty is small enough to answer.

The cost here is latency and tokens rather than user patience, which is why
`costs` matters more than in the clarifying-question case: a cheap BM25 probe
and an expensive cross-encoder rerank should not be ranked on raw bits.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core import CandidateState, FeatureSuggestion
from ..session import StopRules
from .protocols import Probe, ProbeRunner
from .questions import answer_frame, answer_likelihoods, next_question


@dataclass
class ProbeStep:
    """Record of a single probe in a retrieval session.

    Attributes
    ----------
    probe : Probe
        The probe that was run
    outcome : Any
        What it returned (None when likelihoods were supplied directly)
    cost : float
        Cost of this probe
    cumulative_cost : float
        Total cost through this step
    entropy_before : float
        Posterior entropy in bits before the update
    entropy_after : float
        Posterior entropy in bits after the update
    expected_voi : float, optional
        Information gain predicted for this probe before running it
    """

    probe: Probe
    outcome: Any
    cost: float
    cumulative_cost: float
    entropy_before: float
    entropy_after: float
    expected_voi: float | None = None

    @property
    def realized_gain(self) -> float:
        """Bits actually resolved, against `expected_voi`'s prediction."""
        return self.entropy_before - self.entropy_after


class RetrievalSession:
    """Run probes against a candidate set until the answer is clear enough.

    Parameters
    ----------
    outcomes : DataFrame | Mapping[Probe, Sequence] | Sequence[Sequence]
        Predicted outcome matrix: `outcomes[i][p]` is what probe `p` would
        return if candidate `i` were the right one. Same shapes as
        :func:`rowvoi.rag.questions.answer_frame`
    runner : ProbeRunner, optional
        Executes probes in :meth:`run`. Not needed for manual
        :meth:`next_probe` / :meth:`observe` driving
    probes : Sequence[Probe], optional
        Column labels, required when `outcomes` is a nested sequence
    prior : Sequence[float], optional
        Per-candidate prior, typically the retrieval scores. Normalized
        internally; defaults to uniform
    costs : Mapping[Probe, float], optional
        Per-probe cost in whatever unit the budget is denominated
    noise : float, default 0.0
        Probability a probe returns something other than predicted. Leave at 0
        only if the outcome predictions are exact; retrieval rarely is
    """

    def __init__(
        self,
        outcomes: pd.DataFrame
        | Mapping[Probe, Sequence[Any]]
        | Sequence[Sequence[Any]],
        *,
        runner: ProbeRunner | None = None,
        probes: Sequence[Probe] | None = None,
        prior: Sequence[float] | None = None,
        costs: Mapping[Probe, float] | None = None,
        noise: float = 0.0,
    ) -> None:
        self.outcomes = answer_frame(outcomes, questions=probes)
        self.runner = runner
        self.costs = dict(costs or {})
        self.noise = noise

        self._history: list[ProbeStep] = []
        self._cumulative_cost = 0.0

        n = len(self.outcomes)
        if prior is not None:
            weights = np.asarray(prior, dtype=float)
            if weights.shape != (n,):
                raise ValueError(
                    f"Expected {n} prior weights, got shape {weights.shape}"
                )
            if np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("Prior must be non-negative with positive total mass")
            self._state = CandidateState(
                candidate_rows=list(range(n)),
                posterior=weights / weights.sum(),
                observed_cols=set(),
                observed_values={},
            )
        else:
            self._state = CandidateState.uniform(list(range(n)))

    @property
    def state(self) -> CandidateState:
        """Current belief over candidates."""
        return self._state

    @property
    def history(self) -> list[ProbeStep]:
        """Probes run so far, in order."""
        return self._history

    @property
    def cumulative_cost(self) -> float:
        """Total cost incurred so far."""
        return self._cumulative_cost

    @property
    def steps_taken(self) -> int:
        """Number of probes run so far."""
        return len(self._history)

    @property
    def best_candidate(self) -> int:
        """Index of the highest-posterior candidate."""
        return int(np.argmax(self._state.posterior))

    def next_probe(self) -> FeatureSuggestion:
        """Score the unrun probes and return the best one.

        Does not run it -- call :meth:`observe` with the result.

        Returns
        -------
        FeatureSuggestion
            `.col` is the probe (None when every probe has been run),
            `.expected_voi` its predicted gain in bits
        """
        return next_question(self.outcomes, state=self._state, costs=self.costs or None)

    def observe(
        self,
        probe: Probe,
        outcome: Any = None,
        *,
        likelihoods: Sequence[float] | None = None,
        expected_voi: float | None = None,
    ) -> ProbeStep:
        """Fold a probe's result into the belief.

        Parameters
        ----------
        probe : Probe
            The probe that was run
        outcome : Any, optional
            What it returned, matched against the predicted outcome matrix.
            Ignored when `likelihoods` is given
        likelihoods : Sequence[float], optional
            P(result | candidate), for callers with their own scoring function
            (a similarity kernel, a reranker's scores). Bypasses the matrix
        expected_voi : float, optional
            Predicted gain, recorded for comparison against what was realized

        Returns
        -------
        ProbeStep
            Record of this step
        """
        entropy_before = self._state.entropy

        if likelihoods is None:
            weights = answer_likelihoods(
                self.outcomes, probe, outcome, noise=self.noise
            )
        else:
            weights = np.asarray(likelihoods, dtype=float)

        self._state = self._state.reweight(
            weights, observed_col=probe, observed_value=outcome
        )

        cost = self.costs.get(probe, 1.0)
        self._cumulative_cost += cost

        step = ProbeStep(
            probe=probe,
            outcome=outcome,
            cost=cost,
            cumulative_cost=self._cumulative_cost,
            entropy_before=entropy_before,
            entropy_after=self._state.entropy,
            expected_voi=expected_voi,
        )
        self._history.append(step)
        return step

    def run(
        self,
        stop: StopRules,
        *,
        runner: ProbeRunner | None = None,
    ) -> list[ProbeStep]:
        """Probe repeatedly until a stop rule fires or probes run out.

        Parameters
        ----------
        stop : StopRules
            Stopping criteria. `epsilon_posterior` is the natural one here --
            "stop when one candidate holds 95% of the mass". `epsilon_pairs`
            also works, measured over the predicted outcome matrix
        runner : ProbeRunner, optional
            Overrides the runner given to the constructor

        Returns
        -------
        list[ProbeStep]
            Every step taken this session, including any from earlier calls

        Raises
        ------
        ValueError
            If no runner is available
        """
        active = runner or self.runner
        if active is None:
            raise ValueError(
                "run() needs a ProbeRunner; pass one here or to the constructor"
            )

        while True:
            should_stop, _ = stop.should_stop(
                self._state, self.steps_taken, self._cumulative_cost, self.outcomes
            )
            if should_stop:
                break

            suggestion = self.next_probe()
            if suggestion.col is None:
                break

            outcome = active.run(suggestion.col)
            self.observe(suggestion.col, outcome, expected_voi=suggestion.expected_voi)

        return self._history

    def reset(self, *, prior: Sequence[float] | None = None) -> None:
        """Clear history and return to the initial (or a new) prior."""
        n = len(self.outcomes)
        if prior is not None:
            weights = np.asarray(prior, dtype=float)
            self._state = CandidateState(
                candidate_rows=list(range(n)),
                posterior=weights / weights.sum(),
                observed_cols=set(),
                observed_values={},
            )
        else:
            self._state = CandidateState.uniform(list(range(n)))
        self._history = []
        self._cumulative_cost = 0.0
