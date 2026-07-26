"""Retrieval-augmented generation adapters over rowvoi's core engines.

rowvoi answers two questions about tabular data: which minimal set of columns
distinguishes a set of candidate rows, and which column to observe next. Both
have direct analogues in RAG, and this package is the adapter layer -- the
algorithms themselves are unchanged.

============================  ====================================================
Tabular                       RAG
============================  ====================================================
Set cover over row pairs      :mod:`~rowvoi.rag.context` -- cover claims with chunks
Next column by MI             :mod:`~rowvoi.rag.questions` -- next clarifying question
Sequential acquisition        :mod:`~rowvoi.rag.retrieval` -- next retrieval probe
============================  ====================================================

Everything here is deterministic and depends only on pandas and numpy: the
functions take matrices (which chunks support which claims, what answer each
question gets per candidate) and return selections. Producing those matrices is
what needs a language model, and that boundary is
:mod:`rowvoi.rag.protocols`. For an Anthropic-backed implementation, install
the ``claude`` extra and import :mod:`rowvoi.rag.claude` explicitly -- it is
deliberately not imported here, so ``rowvoi.rag`` never pulls in an LLM SDK.
"""

from .context import (
    Chunk,
    ContextSelection,
    extract_and_select,
    plan_context_path,
    select_context,
)
from .protocols import (
    AnswerPredictor,
    ChunkId,
    Claim,
    ClaimExtractor,
    Probe,
    ProbeRunner,
    Question,
    QuestionGenerator,
    SupportJudge,
)
from .questions import (
    answer_frame,
    answer_likelihoods,
    next_question,
    observe_answer,
    question_values,
)
from .retrieval import ProbeStep, RetrievalSession

__all__ = [
    # Minimal sufficient context
    "Chunk",
    "ContextSelection",
    "select_context",
    "plan_context_path",
    "extract_and_select",
    # Clarifying questions
    "answer_frame",
    "answer_likelihoods",
    "next_question",
    "observe_answer",
    "question_values",
    # Adaptive retrieval
    "RetrievalSession",
    "ProbeStep",
    # Protocols and type aliases
    "AnswerPredictor",
    "ClaimExtractor",
    "ProbeRunner",
    "QuestionGenerator",
    "SupportJudge",
    "ChunkId",
    "Claim",
    "Probe",
    "Question",
]
