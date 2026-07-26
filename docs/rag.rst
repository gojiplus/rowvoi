Retrieval-Augmented Generation
==============================

``rowvoi.rag`` applies the same two engines to retrieval. Nothing new is
computed -- set cover and mutual information are reused as they are, over a
different universe.

========================================  ==================================================
Tabular                                   RAG
========================================  ==================================================
Cover row pairs with columns              Cover **claims** with **chunks** (cost = tokens)
Next column by mutual information         Next **clarifying question** (cost = patience)
Sequential acquisition                    Next **retrieval probe** (cost = latency)
========================================  ==================================================

Everything below is deterministic and needs only pandas and numpy. Producing
the matrices is what needs a language model, and that boundary is
:mod:`rowvoi.rag.protocols`.

Minimal sufficient context
--------------------------

Top-k ranks each chunk independently, so it will spend the budget on five
chunks supporting the same claim while a sixth claim goes unsupported. Set
cover optimizes the selection jointly, and because chunk cost is token count
it prefers several small chunks over one large catch-all.

.. testcode::

   from rowvoi.rag import Chunk, select_context

   chunks = [
       Chunk("doc1#p2", "The Pro plan costs $40/seat/mo.", tokens=180),
       Chunk("doc2#p1", "v4.2 shipped March 3rd, 2026.", tokens=150),
       Chunk("faq#p1", "Pricing, releases, and SSO.", tokens=1550),
   ]
   claims = ["pro_plan_price", "v42_release_date"]

   # A SupportJudge fills this in; see "Where the LLM goes" below.
   support = {
       "pro_plan_price": {"doc1#p2", "faq#p1"},
       "v42_release_date": {"doc2#p1", "faq#p1"},
   }

   selection = select_context(chunks, claims, support)
   print(sorted(selection.chunks))
   print(selection.total_cost, "tokens")
   print(f"{selection.coverage:.0%} of claims covered")

.. testoutput::

   ['doc1#p2', 'doc2#p1']
   330.0 tokens
   100% of claims covered

The single ``faq#p1`` chunk covers both claims by itself, but at 1550 tokens.
Minimizing chunk *count* is not minimizing context.

A claim that nothing supports is a retrieval failure, and it is reported
rather than quietly dropped:

.. testcode::

   gap = select_context(chunks, [*claims, "refund_policy"], support)
   print(f"{gap.coverage:.0%}")
   print(sorted(gap.missing_claims))

.. testoutput::

   67%
   ['refund_policy']

When the budget binds rather than the coverage target, ask for the order
instead of the set:

.. testcode::

   from rowvoi.rag import plan_context_path

   path = plan_context_path(chunks, claims, support)
   print(path.prefix_for_budget(200))
   print(path.coverage_curve())

.. testoutput::

   ['doc2#p1']
   [(150.0, 0.5), (330.0, 1.0)]

``epsilon_claims`` trades a fraction of the claims for a smaller prompt.

Clarifying questions
--------------------

When the retrieved set is genuinely ambiguous, ask rather than guess. Given a
matrix of the answer each candidate implies, ``rowvoi`` picks the question
that splits them best per unit of user effort.

.. testcode::

   from rowvoi.rag import next_question, question_values

   answers = {
       "Which deployment?": ["cloud", "cloud", "self", "self"],
       "Which version?": ["v4", "v3", "v4", "v3"],
       "Are you an admin?": ["yes", "yes", "yes", "yes"],
   }

   for question, bits in question_values(answers).items():
       print(f"{bits:.2f} bits  {question}")

.. testoutput::

   1.00 bits  Which deployment?
   1.00 bits  Which version?
   0.00 bits  Are you an admin?

"Are you an admin?" scores exactly zero: every candidate answers it the same
way, so it cannot narrow anything. A relevance-ranked list would still ask it.

Costs express user patience -- a yes/no question is cheap, "paste your config"
is not -- and ranking is by information per unit cost:

.. testcode::

   suggestion = next_question(answers, costs={"Which version?": 5.0})
   print(suggestion.col)

.. testoutput::

   Which deployment?

Folding an answer back in is *soft*: candidates are reweighted, not dropped,
so one surprising answer cannot eliminate the right candidate outright.

.. testcode::

   from rowvoi import CandidateState
   from rowvoi.rag import observe_answer

   state = CandidateState.uniform(range(4))
   state = observe_answer(state, answers, "Which deployment?", "self", noise=0.02)
   print(state.posterior.round(2).tolist())

.. testoutput::

   [0.01, 0.01, 0.49, 0.49]

Adaptive retrieval
------------------

The same machinery, with retrieval probes instead of questions and latency
instead of patience. Stop when the posterior is sharp enough rather than at a
fixed *k*.

.. testcode::

   from rowvoi import StopRules
   from rowvoi.rag import RetrievalSession

   outcomes = {
       "bm25": ["hit", "hit", "miss", "miss"],
       "metadata:version": ["v4", "v3", "v4", "v3"],
       "rerank": ["1st", "2nd", "3rd", "4th"],
   }

   class ScriptedRunner:
       """Stands in for a retrieval backend; candidate 3 is the right one."""

       def run(self, probe):
           return outcomes[probe][3]

   session = RetrievalSession(
       outcomes,
       runner=ScriptedRunner(),
       prior=[0.20, 0.15, 0.45, 0.20],   # retrieval scores, wrongly favoring 2
       costs={"bm25": 1.0, "metadata:version": 1.0, "rerank": 12.0},
       noise=0.05,
   )

   session.run(StopRules(epsilon_posterior=0.15, target_unique=False))
   print("probes:", [step.probe for step in session.history])
   print("best candidate:", session.best_candidate)
   print(f"confidence: {session.state.max_posterior:.0%}")
   print("cost:", session.cumulative_cost)

.. testoutput::

   probes: ['bm25', 'metadata:version']
   best candidate: 3
   confidence: 86%
   cost: 2.0

The prior favored candidate 2; two cheap probes overturned it, and the
expensive reranker was never needed. Demanding more confidence would buy it --
that tradeoff is a threshold you set rather than a *k* you guessed.

Where the LLM goes
------------------

The matrices above -- which chunks support which claims, what answer each
question gets -- are what a model produces.
:mod:`rowvoi.rag.protocols` defines that boundary:
:class:`~rowvoi.rag.protocols.ClaimExtractor`,
:class:`~rowvoi.rag.protocols.SupportJudge`,
:class:`~rowvoi.rag.protocols.QuestionGenerator`,
:class:`~rowvoi.rag.protocols.AnswerPredictor` and
:class:`~rowvoi.rag.protocols.ProbeRunner`.

Implement them yourself, or install the ``claude`` extra and use the bundled
Anthropic-backed versions:

.. code-block:: bash

   pip install "rowvoi[claude]"

.. code-block:: python

   from rowvoi.rag import extract_and_select
   from rowvoi.rag.claude import ClaudeClaimExtractor, ClaudeSupportJudge

   selection = extract_and_select(
       query,
       chunks,
       extractor=ClaudeClaimExtractor(),
       judge=ClaudeSupportJudge(),
   )

``rowvoi.rag.claude`` is never imported by ``rowvoi.rag``, so the core keeps
its pandas-and-numpy-only footprint. It fills each matrix in a single request
and places chunk text behind a prompt-cache breakpoint, since the chunks are
the large stable half of the prompt and the questions are what vary.

A runnable walkthrough of all three capabilities, needing no API key, is in
``examples/rag/rag_pipeline_demo.py``.
