"""Anthropic-backed implementations of the :mod:`rowvoi.rag.protocols` interfaces.

Requires the ``claude`` extra::

    pip install "rowvoi[claude]"

This module is deliberately *not* imported by :mod:`rowvoi.rag` -- import it
explicitly. The rest of rowvoi stays on pandas and numpy.

Two design choices worth knowing about:

**One call per matrix, not per cell.** A candidates x questions matrix is filled
in a single request. Per-cell calls would multiply latency by ``n*q`` and resend
the chunk text every time.

**Chunk text is cached.** Chunks go in the ``system`` block behind a
``cache_control`` breakpoint and the varying part (claims, questions) goes in
the user turn. Chunks are the large, stable half of the prompt, so across the
several calls a session makes they are billed once at write price and then at
read price. Check ``usage.cache_read_input_tokens`` to confirm it is working.

**Everything is index-based.** Chunks, claims, and candidates are presented to
the model as numbered lists and it answers with numbers, so nothing depends on
the model echoing an id or a claim string back verbatim.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .protocols import ChunkId, Claim

try:  # pragma: no cover - trivial import guard
    import anthropic
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rowvoi.rag.claude needs the Anthropic SDK. "
        'Install it with: pip install "rowvoi[claude]"'
    ) from exc


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "support": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_index": {"type": "integer"},
                    "chunk_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["claim_index", "chunk_indices"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["support"],
    "additionalProperties": False,
}

_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_ANSWERS_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}


def _numbered(items: Sequence[Any]) -> str:
    """Render a sequence as a numbered list for the prompt."""
    return "\n".join(f"[{i}] {item}" for i, item in enumerate(items))


@dataclass
class _ClaudeComponent:
    """Shared plumbing: client, model, and the structured-output call."""

    client: Any = None
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic()

    def _call(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        cache_system: bool = True,
    ) -> dict[str, Any]:
        """One structured-output request, returning the parsed JSON object.

        Thinking is left at the model default (adaptive on Claude Opus 5) and
        `max_tokens` covers thinking plus response, which is why the default is
        generous rather than sized to the JSON.
        """
        system_block: dict[str, Any] = {"type": "text", "text": system}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}

        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": schema}
        }
        if self.effort is not None:
            output_config["effort"] = self.effort

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[system_block],
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            **self.extra,
        )

        # Safety classifiers can decline; content is empty or partial then.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise RuntimeError(
                f"Claude declined this request (category={category!r}). "
                "The response is empty or partial and cannot be parsed."
            )

        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
        if text is None:
            stop = getattr(response, "stop_reason", None)
            raise RuntimeError(
                f"No text block in the response (stop_reason={stop!r}). "
                "If this is 'max_tokens', raise max_tokens."
            )

        return json.loads(text)


@dataclass
class ClaudeClaimExtractor(_ClaudeComponent):
    """Decompose a query into independently checkable claims.

    Parameters
    ----------
    client : anthropic.Anthropic, optional
        Reuse a configured client. A default one is constructed if omitted
    model : str, default "claude-opus-5"
        Model id
    max_tokens : int, default 16000
        Covers thinking plus response
    effort : str, optional
        "low" through "max"; leave unset for the model default
    max_claims : int, default 12
        Upper bound requested in the prompt
    """

    max_claims: int = 12

    def extract(self, query: str) -> list[str]:
        """Return the claims an answer to `query` must support."""
        system = (
            "You decompose a question into the atomic factual claims that a "
            "complete answer must support.\n\n"
            "Rules:\n"
            "- Each claim must be independently checkable against a source.\n"
            "- Claims must not overlap; two claims satisfied by the same "
            "sentence should be one claim.\n"
            "- Cover the whole question, including implicit parts "
            "(a comparison needs a claim per side).\n"
            "- State claims as short noun phrases describing the needed fact, "
            "not as questions.\n"
            f"- Return at most {self.max_claims} claims."
        )
        payload = self._call(
            system=system,
            user=f"Question:\n{query}",
            schema=_CLAIMS_SCHEMA,
            cache_system=False,
        )
        return list(payload["claims"])


@dataclass
class ClaudeSupportJudge(_ClaudeComponent):
    """Decide which retrieved chunks support which claims."""

    def judge(
        self,
        chunks: Sequence[tuple[ChunkId, str]],
        claims: Sequence[Claim],
    ) -> dict[Claim, set[ChunkId]]:
        """Map each claim to the chunk ids that support it.

        Every claim appears in the result, mapping to an empty set when no
        chunk supports it -- that is a retrieval gap and the caller needs to
        see it rather than have the claim silently disappear.
        """
        chunk_ids = [cid for cid, _ in chunks]
        rendered = "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(chunks))

        system = (
            "You judge which source passages support which claims.\n\n"
            "A passage supports a claim only if it states or directly entails "
            "it. Being on the same topic is not support. A claim may be "
            "supported by several passages, by one, or by none -- return an "
            "empty list when nothing supports it rather than guessing.\n\n"
            "Passages:\n" + rendered
        )
        user = (
            "Claims:\n"
            + _numbered(list(claims))
            + "\n\nFor each claim index, list the passage indices that support it."
        )

        payload = self._call(system=system, user=user, schema=_SUPPORT_SCHEMA)

        support: dict[Claim, set[ChunkId]] = {claim: set() for claim in claims}
        for entry in payload["support"]:
            ci = entry["claim_index"]
            if not 0 <= ci < len(claims):
                continue
            support[claims[ci]] = {
                chunk_ids[k] for k in entry["chunk_indices"] if 0 <= k < len(chunk_ids)
            }
        return support


@dataclass
class ClaudeQuestionGenerator(_ClaudeComponent):
    """Propose clarifying questions that could separate ambiguous candidates."""

    def generate(self, candidates: Sequence[str], n: int) -> list[str]:
        """Propose up to `n` clarifying questions."""
        system = (
            "You write clarifying questions that tell near-identical search "
            "results apart.\n\n"
            "Rules:\n"
            "- A good question splits the candidates into groups; a question "
            "every candidate answers identically is worthless.\n"
            "- Ask about what the user would know without having read the "
            "candidates (their version, their platform, their goal), never "
            "about the candidates' contents.\n"
            "- One short question each, answerable in a few words.\n\n"
            "Candidates:\n" + _numbered(candidates)
        )
        payload = self._call(
            system=system,
            user=f"Propose at most {n} clarifying questions.",
            schema=_QUESTIONS_SCHEMA,
        )
        return list(payload["questions"])[:n]


@dataclass
class ClaudeAnswerPredictor(_ClaudeComponent):
    """Predict the answer each question would get, per candidate.

    This is the matrix that makes value of information computable before
    anything is asked: it says how each question would split the candidates.
    """

    def predict(
        self,
        candidates: Sequence[str],
        questions: Sequence[str],
    ) -> list[list[str]]:
        """Return `answers[i][q]`: the answer if `candidates[i]` were right.

        Raises
        ------
        ValueError
            If the returned matrix is not `len(candidates)` x `len(questions)`
        """
        system = (
            "You predict how a user would answer clarifying questions, "
            "assuming a particular search result is the one they wanted.\n\n"
            "Rules:\n"
            "- Answer as a short canonical label (one or two words, lowercase) "
            "so that answers can be compared for equality. Never write a "
            'sentence. Use "unknown" when the candidate does not determine '
            "the answer.\n"
            "- Identical answers must use identical wording; do not vary "
            'phrasing between rows ("v2" and "version 2" must not both appear).\n'
            "- Return one row per candidate, in order, each row holding one "
            "answer per question, in order.\n\n"
            "Candidates:\n" + _numbered(candidates)
        )
        user = (
            "Questions:\n"
            + _numbered(questions)
            + f"\n\nReturn {len(candidates)} rows of {len(questions)} answers."
        )

        payload = self._call(system=system, user=user, schema=_ANSWERS_SCHEMA)
        rows = [list(row) for row in payload["rows"]]

        if len(rows) != len(candidates) or any(
            len(row) != len(questions) for row in rows
        ):
            shape = f"{len(rows)}x{[len(r) for r in rows]}"
            raise ValueError(
                f"Expected a {len(candidates)}x{len(questions)} answer matrix, "
                f"got {shape}"
            )
        return rows
