"""Interfaces for the parts of RAG that need a language model.

Nothing in :mod:`rowvoi.rag` imports an LLM SDK. The algorithms take matrices;
these protocols describe the components that *fill* those matrices. Supply your
own, or use :mod:`rowvoi.rag.claude` (needs the ``claude`` extra).
"""

from collections.abc import Hashable, Sequence
from typing import Any, Protocol, runtime_checkable

# Type aliases
ChunkId = Hashable
Claim = Hashable
Question = Hashable
Probe = Hashable


@runtime_checkable
class ClaimExtractor(Protocol):
    """Decomposes a query into the claims an answer must support."""

    def extract(self, query: str) -> list[str]:
        """Return the list of claims an answer to `query` must support."""
        ...


@runtime_checkable
class SupportJudge(Protocol):
    """Decides which chunks support which claims."""

    def judge(
        self,
        chunks: Sequence[tuple[ChunkId, str]],
        claims: Sequence[Claim],
    ) -> dict[Claim, set[ChunkId]]:
        """Map each claim to the set of chunk ids that support it.

        Parameters
        ----------
        chunks : Sequence[tuple[ChunkId, str]]
            (id, text) pairs for the retrieved chunks
        claims : Sequence[Claim]
            Claims to check support for

        Returns
        -------
        dict[Claim, set[ChunkId]]
            Supporting chunks per claim. A claim with no support maps to an
            empty set rather than being omitted.
        """
        ...


@runtime_checkable
class QuestionGenerator(Protocol):
    """Proposes clarifying questions that might separate candidates."""

    def generate(self, candidates: Sequence[str], n: int) -> list[str]:
        """Propose up to `n` clarifying questions for these candidates."""
        ...


@runtime_checkable
class AnswerPredictor(Protocol):
    """Predicts the answer each question gets, per candidate.

    This is what makes value-of-information computable before asking anything:
    the predicted answer matrix says how each question would split the
    candidate set.
    """

    def predict(
        self,
        candidates: Sequence[str],
        questions: Sequence[str],
    ) -> list[list[Any]]:
        """Return an answer matrix `answers[i][q]`.

        `answers[i][q]` is the answer question `questions[q]` would receive if
        `candidates[i]` were the right candidate. Answers are compared for
        equality, so normalize them (lowercase, canonical labels) rather than
        returning free prose.
        """
        ...


@runtime_checkable
class ProbeRunner(Protocol):
    """Executes a retrieval probe and reports what came back."""

    def run(self, probe: Probe) -> Any:
        """Run `probe` and return the observed outcome.

        The outcome is compared against the predicted outcome matrix to build
        a likelihood vector, so it must be drawn from the same value space.
        """
        ...
