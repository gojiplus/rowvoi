"""RAG adapters end to end: minimal context, clarifying questions, adaptive probes.

Runs without an API key. The matrices an LLM would normally produce (which
chunks support which claims, what answer each question gets) are hand-written
here so the selection logic is what you actually see.

Run with:
    python examples/rag/rag_pipeline_demo.py
"""

from rowvoi import CandidateState, StopRules
from rowvoi.rag import (
    Chunk,
    RetrievalSession,
    next_question,
    observe_answer,
    plan_context_path,
    question_values,
    select_context,
)

# ---------------------------------------------------------------------------
# 1. Minimal sufficient context
# ---------------------------------------------------------------------------
# The user asked something that needs three facts. The retriever returned six
# chunks, 2,600 tokens in total. Top-k would send all six.

print("=" * 72)
print("1. MINIMAL SUFFICIENT CONTEXT")
print("=" * 72)

chunks = [
    Chunk("doc1#p2", "The Pro plan costs $40 per seat per month.", tokens=180),
    Chunk("doc1#p7", "Billing is annual. The Pro plan costs $40/seat/mo.", tokens=210),
    Chunk("doc2#p1", "v4.2 shipped on March 3rd, 2026.", tokens=150),
    Chunk("doc2#p4", "Release notes for v4.2 (March 3rd) and v4.1.", tokens=320),
    Chunk("doc3#p9", "SSO is included on Pro and above.", tokens=190),
    Chunk("faq#p1", "Pricing, release history, and SSO availability.", tokens=1550),
]

claims = ["pro_plan_price", "v42_release_date", "sso_availability"]

# In production a SupportJudge fills this in; see rowvoi.rag.claude.
support = {
    "pro_plan_price": {"doc1#p2", "doc1#p7", "faq#p1"},
    "v42_release_date": {"doc2#p1", "doc2#p4", "faq#p1"},
    "sso_availability": {"doc3#p9", "faq#p1"},
}

everything = sum(c.tokens or 0 for c in chunks)
print(f"\nRetrieved {len(chunks)} chunks, {everything} tokens total (naive top-k).")

selection = select_context(chunks, claims, support, strategy="exact")
print(f"\nMinimal cover: {sorted(selection.chunks)}")
share = selection.total_cost / everything
print(f"  tokens:   {selection.total_cost:.0f} ({share:.0%} of top-k)")
print(f"  coverage: {selection.coverage:.0%}")

# The one big chunk covers everything by itself, but costs far more. Cost-aware
# set cover prefers three small chunks over one 1,550-token catch-all.
by_count = select_context([c.id for c in chunks], claims, support, strategy="exact")
print(f"\nIgnoring token cost, the smallest cover is {by_count.chunks} -- one chunk,")
print("but 1,550 tokens. Minimizing chunk count is not minimizing context.")

# Trading one claim for a much smaller prompt.
lossy = select_context(chunks, claims, support, epsilon_claims=0.34, strategy="exact")
print(f"\nAllowing one claim to go unsupported: {sorted(lossy.chunks)}")
print(f"  tokens: {lossy.total_cost:.0f}, dropped: {sorted(lossy.missing_claims)}")

# Budget-first: fill a context window and see what the last token bought.
print("\nCoverage per token spent:")
path = plan_context_path(chunks, claims, support)
for step in path.steps:
    print(
        f"  +{step.name:10s} {step.cumulative_cost:5.0f} tokens "
        f"-> {step.coverage:5.0%} of claims"
    )
print(f"\n  Under a 400-token budget: {path.prefix_for_budget(400)}")

# A claim nothing supports is a retrieval failure and must not be hidden.
gap = select_context(chunks, [*claims, "refund_policy"], support)
print(f"\nWith an unretrievable claim added: coverage {gap.coverage:.0%}, ")
print(f"  missing {sorted(gap.missing_claims)} -- no chunk could have helped.")


# ---------------------------------------------------------------------------
# 2. Clarifying questions
# ---------------------------------------------------------------------------
# Four docs match "how do I enable SSO" almost equally well. Rather than guess
# or dump all four into the context, ask the one question that splits them.

print("\n" + "=" * 72)
print("2. CLARIFYING QUESTIONS")
print("=" * 72)

candidates = [
    "SSO setup - Cloud, v4",
    "SSO setup - Cloud, v3",
    "SSO setup - Self-hosted, v4",
    "SSO setup - Self-hosted, v3",
]

# In production an AnswerPredictor fills this in: what each candidate implies
# the answer would be. Rows are candidates, columns are questions.
answers = {
    "Which deployment, cloud or self-hosted?": ["cloud", "cloud", "self", "self"],
    "Which major version?": ["v4", "v3", "v4", "v3"],
    "Are you an admin?": ["yes", "yes", "yes", "yes"],
    "What is your exact build number?": ["4.2.1", "3.9.4", "4.2.1s", "3.9.4s"],
}

print("\nCandidates:")
for i, text in enumerate(candidates):
    print(f"  [{i}] {text}")

print("\nExpected information gain (2.00 bits would identify one outright):")
for question, bits in sorted(question_values(answers).items(), key=lambda kv: -kv[1]):
    print(f"  {bits:5.2f} bits  {question}")

print("\n'Are you an admin?' scores 0.00 -- every candidate answers the same,")
print("so it cannot narrow anything. A relevance-ranked list would still ask it.")

# The build-number question resolves everything at once, but no user knows it
# offhand. Pricing that in flips the choice to a question people can answer.
patience = {
    "Which deployment, cloud or self-hosted?": 1.0,
    "Which major version?": 1.0,
    "Are you an admin?": 1.0,
    "What is your exact build number?": 8.0,
}

free = next_question(answers)
priced = next_question(answers, costs=patience)
print(f"\nIgnoring effort, ask:   {free.col!r} ({free.expected_voi:.2f} bits)")
print(f"Pricing user effort:    {priced.col!r} ({priced.expected_voi:.2f} bits)")

# Walk the session: ask, observe, re-rank against the updated posterior.
#
# Stop on a confidence threshold, not on state.is_unique. With noise > 0 every
# candidate keeps a sliver of probability, and is_unique needs >99.999%, so it
# would keep asking long after the answer is settled. This is the same rule
# StopRules(epsilon_posterior=...) applies in section 3.
CONFIDENCE = 0.95

print("\nInteractive session (true answer: self-hosted, v3):")
state = CandidateState.uniform(list(range(len(candidates))))
truth = 3

while state.max_posterior < CONFIDENCE:
    suggestion = next_question(answers, state=state, costs=patience)
    if suggestion.col is None:
        break
    reply = answers[suggestion.col][truth]
    state = observe_answer(state, answers, suggestion.col, reply, noise=0.02)
    print(f"  ask {suggestion.col!r} -> {reply!r}")
    print(
        f"      posterior {state.posterior.round(2).tolist()}  "
        f"entropy {state.entropy:.2f} bits"
    )

resolved = int(state.posterior.argmax())
print(
    f"\n  Resolved to [{resolved}] {candidates[resolved]} "
    f"at {state.max_posterior:.1%} confidence"
)
print("  Two questions asked; the other two were never needed.")


# ---------------------------------------------------------------------------
# 3. Adaptive retrieval
# ---------------------------------------------------------------------------
# Same machinery, but the "questions" are retrieval probes and the cost is
# latency rather than user patience. Stop as soon as one candidate dominates.

print("\n" + "=" * 72)
print("3. ADAPTIVE RETRIEVAL")
print("=" * 72)

# What each probe would return if a given candidate were the right answer.
probes = {
    "bm25:sso+cloud": ["hit", "hit", "miss", "miss"],
    "metadata:version": ["v4", "v3", "v4", "v3"],
    "rerank:cross-encoder": ["1st", "2nd", "3rd", "4th"],
}
probe_costs = {
    "bm25:sso+cloud": 1.0,
    "metadata:version": 1.0,
    "rerank:cross-encoder": 12.0,  # slow model
}


class ScriptedRunner:
    """Stands in for a real retrieval backend."""

    def __init__(self, truth: int) -> None:
        self.truth = truth

    def run(self, probe):
        """Return what this probe sees, given the true answer."""
        return probes[probe][self.truth]


# Retrieval scores as the prior: candidate 2 looks best going in, wrongly.
prior = [0.20, 0.15, 0.45, 0.20]

leader = prior.index(max(prior))
print(f"\nPrior from retrieval scores: {prior}")
print(f"  leader going in: [{leader}] {candidates[leader]}")
print(f"  (the right answer is [3] {candidates[3]})")

# The stop rule is the cost lever. Run the same probes under two confidence
# targets and watch the expensive reranker come and go.
for target in (0.85, 0.95):
    session = RetrievalSession(
        probes,
        runner=ScriptedRunner(truth=3),
        prior=prior,
        costs=probe_costs,
        noise=0.05,
    )
    history = session.run(StopRules(epsilon_posterior=1 - target, target_unique=False))

    print(f"\n--- stop at {target:.0%} confidence ---")
    for step in history:
        print(
            f"  {step.probe:22s} -> {step.outcome!s:5s}  "
            f"expected {step.expected_voi:4.2f} bits, "
            f"realized {step.realized_gain:4.2f}, "
            f"spent {step.cumulative_cost:4.1f}"
        )
    reranked = any(s.probe == "rerank:cross-encoder" for s in history)
    print(
        f"  -> [{session.best_candidate}] at {session.state.max_posterior:.1%}, "
        f"cost {session.cumulative_cost:.0f}; "
        f"12-cost reranker {'run' if reranked else 'never needed'}"
    )

print("\nBoth runs pick the same answer and both overturn the prior, which")
print("favored candidate 2. The difference is what the last stretch of")
print("confidence costs: the two cheap probes get to ~86% for 2 units, and")
print("the slow reranker buys the rest for 12 more -- 7x the spend for the")
print("tail. Fixed top-k pays that every query; here it is a threshold you set.")
