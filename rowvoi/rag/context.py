"""Minimal sufficient context: cover the claims with the fewest/cheapest chunks.

The RAG analogue of :func:`rowvoi.find_key`. Where key-finding covers row pairs
with columns, this covers *claims* with *chunks* -- same weighted set cover,
same solvers, cost measured in tokens instead of acquisition effort.

Top-k retrieval optimizes each chunk's relevance independently, so it happily
spends the budget on five chunks that all support the same claim while a sixth
claim goes unsupported. Set cover optimizes the selection jointly.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..setcover import CoverPath, SetCoverProblem, Strategy
from .protocols import ChunkId, Claim


@dataclass
class Chunk:
    """A retrieved chunk.

    Attributes
    ----------
    id : ChunkId
        Stable identifier, used everywhere else in this module
    text : str
        The chunk body
    tokens : int, optional
        Token count, used as the set-cover cost. When omitted, cost falls back
        to an explicit `costs` mapping and then to 1.0 (chunk-count minimization)
    """

    id: ChunkId
    text: str = ""
    tokens: int | None = None


@dataclass
class ContextSelection:
    """The chunks chosen to support a set of claims.

    Attributes
    ----------
    chunks : list[ChunkId]
        Selected chunk ids
    covered_claims : set[Claim]
        Claims supported by the selection
    missing_claims : set[Claim]
        Claims left unsupported -- either traded away via `epsilon_claims`, or
        unsupported by *any* retrieved chunk, which is a retrieval failure
        rather than a selection one
    coverage : float
        Fraction of claims covered
    total_cost : float
        Summed cost (tokens, when supplied) of the selection
    """

    chunks: list[ChunkId] = field(default_factory=list)
    covered_claims: set[Claim] = field(default_factory=set)
    missing_claims: set[Claim] = field(default_factory=set)
    coverage: float = 1.0
    total_cost: float = 0.0

    @property
    def unsupportable_claims(self) -> set[Claim]:
        """Alias for `missing_claims`, read as "no chunk could have helped"."""
        return self.missing_claims


def _normalize_support(
    support: Mapping[Any, Iterable[Any]] | pd.DataFrame,
) -> dict[Claim, set[ChunkId]]:
    """Coerce the support argument into claim -> supporting chunk ids."""
    if isinstance(support, pd.DataFrame):
        # Rows are claims, columns are chunk ids, cells are truthy on support
        return {
            claim: {col for col in support.columns if bool(support.loc[claim, col])}
            for claim in support.index
        }
    return {claim: set(chunk_ids) for claim, chunk_ids in support.items()}


def _resolve_chunks(
    chunks: Sequence[Chunk] | Sequence[Any],
    costs: Mapping[Any, float] | None,
) -> tuple[list[ChunkId], dict[ChunkId, float]]:
    """Return chunk ids in order plus their costs."""
    ids: list[ChunkId] = []
    resolved: dict[ChunkId, float] = {}

    for chunk in chunks:
        if isinstance(chunk, Chunk):
            ids.append(chunk.id)
            if costs and chunk.id in costs:
                resolved[chunk.id] = float(costs[chunk.id])
            elif chunk.tokens is not None:
                resolved[chunk.id] = float(chunk.tokens)
        else:
            ids.append(chunk)
            if costs and chunk in costs:
                resolved[chunk] = float(costs[chunk])

    if len(set(ids)) != len(ids):
        raise ValueError("Chunk ids must be unique")

    return ids, resolved


def _build_problem(
    chunks: Sequence[Chunk] | Sequence[Any],
    claims: Sequence[Claim],
    support: Mapping[Any, Iterable[Any]] | pd.DataFrame,
    costs: Mapping[Any, float] | None,
) -> SetCoverProblem:
    """Build the claims-as-universe, chunks-as-sets cover problem."""
    chunk_ids, resolved_costs = _resolve_chunks(chunks, costs)
    by_claim = _normalize_support(support)

    unknown = set(by_claim) - set(claims)
    if unknown:
        raise ValueError(
            f"support references claims not in `claims`: {sorted(unknown, key=repr)}"
        )

    # Invert: each chunk covers the claims it supports. Claims with no
    # supporting chunk stay in the universe so they surface as missing.
    supports: dict[ChunkId, set[Claim]] = {cid: set() for cid in chunk_ids}
    for claim, supporting in by_claim.items():
        for cid in supporting:
            if cid in supports:
                supports[cid].add(claim)

    return SetCoverProblem(supports, universe=claims, costs=resolved_costs)


def _describe(
    problem: SetCoverProblem, selection: Sequence[ChunkId]
) -> ContextSelection:
    """Package a raw selection with its coverage accounting."""
    covered = problem.covered(selection)
    return ContextSelection(
        chunks=list(selection),
        covered_claims=covered,
        missing_claims=problem.universe - covered,
        coverage=problem.coverage(selection),
        total_cost=problem.total_cost(selection),
    )


def select_context(
    chunks: Sequence[Chunk] | Sequence[Any],
    claims: Sequence[Claim],
    support: Mapping[Any, Iterable[Any]] | pd.DataFrame,
    *,
    costs: Mapping[Any, float] | None = None,
    epsilon_claims: float = 0.0,
    strategy: Strategy = "greedy",
    time_limit: float | None = None,
) -> ContextSelection:
    """Select the cheapest set of chunks that supports the required claims.

    Parameters
    ----------
    chunks : Sequence[Chunk] | Sequence[ChunkId]
        Retrieved chunks. Passing :class:`Chunk` objects lets `tokens` act as
        the cost; passing bare ids minimizes chunk count unless `costs` is given
    claims : Sequence[Claim]
        Claims the answer must support. This is the universe -- a claim absent
        from `support` counts as uncovered rather than being ignored
    support : Mapping[Claim, Iterable[ChunkId]] | pd.DataFrame
        Which chunks support which claims. As a DataFrame: claims on the index,
        chunk ids as columns, truthy cells meaning support
    costs : Mapping[ChunkId, float], optional
        Per-chunk cost, overriding `Chunk.tokens`. Defaults to 1.0
    epsilon_claims : float, default 0.0
        Permit this fraction of claims to go unsupported. Trading 5% of claims
        often halves the context
    strategy : str, default "greedy"
        Any strategy accepted by :meth:`rowvoi.setcover.SetCoverProblem.solve`
    time_limit : float, optional
        Maximum seconds for the solvers that respect one

    Returns
    -------
    ContextSelection
        Selected chunks plus coverage accounting

    Examples
    --------
    >>> chunks = [Chunk("a", tokens=100), Chunk("b", tokens=100),
    ...           Chunk("c", tokens=400)]
    >>> claims = ["price", "release_date"]
    >>> support = {"price": {"a", "c"}, "release_date": {"b", "c"}}
    >>> selection = select_context(chunks, claims, support)
    >>> sorted(selection.chunks)
    ['a', 'b']
    >>> selection.total_cost
    200.0
    """
    problem = _build_problem(chunks, claims, support, costs)
    selection = problem.solve(strategy, epsilon=epsilon_claims, time_limit=time_limit)
    return _describe(problem, selection)


def plan_context_path(
    chunks: Sequence[Chunk] | Sequence[Any],
    claims: Sequence[Claim],
    support: Mapping[Any, Iterable[Any]] | pd.DataFrame,
    *,
    costs: Mapping[Any, float] | None = None,
    weighting: str = "uniform",
) -> CoverPath:
    """Order chunks by marginal claim coverage per token.

    Use this instead of :func:`select_context` when the budget is the binding
    constraint rather than the coverage target: the returned path exposes
    :meth:`~rowvoi.setcover.CoverPath.prefix_for_budget` and
    :meth:`~rowvoi.setcover.CoverPath.coverage_curve`, so you can fill a
    context window and see exactly what the last token bought.

    Parameters
    ----------
    chunks : Sequence[Chunk] | Sequence[ChunkId]
        Retrieved chunks
    claims : Sequence[Claim]
        Claims the answer must support
    support : Mapping[Claim, Iterable[ChunkId]] | pd.DataFrame
        Which chunks support which claims
    costs : Mapping[ChunkId, float], optional
        Per-chunk cost, overriding `Chunk.tokens`
    weighting : str, default "uniform"
        "uniform" weights all claims equally; "idf" upweights claims that few
        chunks support, so scarce evidence is acquired earlier

    Returns
    -------
    CoverPath
        Ordered acquisition path with per-step coverage and cost

    Examples
    --------
    >>> chunks = [Chunk("a", tokens=100), Chunk("b", tokens=100),
    ...           Chunk("c", tokens=400)]
    >>> claims = ["price", "release_date"]
    >>> support = {"price": {"a", "c"}, "release_date": {"b", "c"}}
    >>> path = plan_context_path(chunks, claims, support)
    >>> path.prefix_for_budget(150)
    ['a']
    """
    problem = _build_problem(chunks, claims, support, costs)
    return problem.plan_path(
        objective="coverage",
        weighting="idf" if weighting == "idf" else "uniform",
    )


def extract_and_select(
    query: str,
    chunks: Sequence[Chunk],
    *,
    extractor: Any,
    judge: Any,
    costs: Mapping[Any, float] | None = None,
    epsilon_claims: float = 0.0,
    strategy: Strategy = "greedy",
) -> ContextSelection:
    """End-to-end selection: extract claims, judge support, then cover.

    Convenience wrapper for when you have a
    :class:`~rowvoi.rag.protocols.ClaimExtractor` and
    :class:`~rowvoi.rag.protocols.SupportJudge` (see :mod:`rowvoi.rag.claude`).
    The two LLM calls happen here; the selection itself is deterministic.

    Parameters
    ----------
    query : str
        The user's question
    chunks : Sequence[Chunk]
        Retrieved chunks, with text (the judge needs it)
    extractor : ClaimExtractor
        Turns the query into claims
    judge : SupportJudge
        Decides which chunks support which claims
    costs : Mapping[ChunkId, float], optional
        Per-chunk cost, overriding `Chunk.tokens`
    epsilon_claims : float, default 0.0
        Permit this fraction of claims to go unsupported
    strategy : str, default "greedy"
        Set cover strategy

    Returns
    -------
    ContextSelection
        Selected chunks plus coverage accounting
    """
    claims = extractor.extract(query)
    support = judge.judge([(c.id, c.text) for c in chunks], claims)
    return select_context(
        chunks,
        claims,
        support,
        costs=costs,
        epsilon_claims=epsilon_claims,
        strategy=strategy,
    )
