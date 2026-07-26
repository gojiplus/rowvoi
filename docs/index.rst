rowvoi Documentation
====================

Minimal keys and row-wise value-of-information for disambiguating tabular
records -- and, via :doc:`rag`, for retrieval.

Overview
--------

Given a small set of candidate rows you cannot tell apart, ``rowvoi`` answers
two questions:

* Which columns must be observed to distinguish these rows? (a *key* --
  weighted set cover)
* Which single column should I acquire next? (*value of information* --
  mutual information under a budget)

Both generalize past tables. The same set cover picks a minimal set of
retrieved chunks; the same mutual information picks the clarifying question
that best splits an ambiguous result set. See :doc:`rag`.

Installation
------------

.. code-block:: bash

   pip install rowvoi

The core depends only on pandas and numpy. Optional features:

.. code-block:: bash

   pip install "rowvoi[optimization]"  # pulp, for the ILP set-cover strategy
   pip install "rowvoi[claude]"        # anthropic, for the RAG LLM adapters

For development, which uses PEP 735 dependency groups:

.. code-block:: bash

   uv sync --all-groups --all-extras

Quick start
-----------

Finding a minimal key
~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   import pandas as pd
   from rowvoi import find_key

   df = pd.DataFrame({
       "name": ["Alice", "Alice", "Alice", "Bob"],
       "city": ["NYC", "SF", "LA", "NYC"],
       "account_id": [1, 2, 3, 4],
       "plan": ["pro", "pro", "pro", "pro"],
   })

   # Which column separates the first two Alices?
   print(find_key(df, [0, 1]))

.. testoutput::

   ['city']

``name`` cannot separate them and ``plan`` is constant, so ``city`` is the
only column that does any work.

Set cover minimizes *cost*, not column count, and cost defaults to one per
column. Left alone, the account id wins because it separates everyone by
itself:

.. testcode::

   print(find_key(df, [0, 1, 2, 3]))

.. testoutput::

   ['account_id']

Price it as expensive to acquire -- a lookup you would rather avoid -- and
two cheap columns beat it:

.. testcode::

   costs = {"account_id": 50.0, "name": 1.0, "city": 1.0, "plan": 1.0}
   print(find_key(df, [0, 1, 2, 3], costs=costs, strategy="exact"))

.. testoutput::

   ['name', 'city']

Strategies are ``greedy`` (the default), ``exact``, ``ilp`` (needs the
``optimization`` extra), ``sa``, ``ga``, ``lp`` and ``hybrid``.

Asking the next best question
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you would rather acquire columns one at a time, a policy ranks them by
expected information gain and a session tracks the belief:

.. testcode::

   from rowvoi import CandidateMIPolicy, DisambiguationSession

   session = DisambiguationSession(
       df, candidate_rows=[0, 1, 2, 3], policy=CandidateMIPolicy()
   )

   suggestion = session.next_question()
   print(suggestion.col, f"{suggestion.expected_voi:.2f} bits")

.. testoutput::

   account_id 2.00 bits

Two bits resolves four candidates outright. The same costs change the
recommendation to something cheaper that still makes progress:

.. testcode::

   thrifty = DisambiguationSession(
       df, candidate_rows=[0, 1, 2, 3], policy=CandidateMIPolicy(costs=costs)
   )
   suggestion = thrifty.next_question()
   print(suggestion.col, f"{suggestion.expected_voi:.2f} bits")

.. testoutput::

   city 1.50 bits

Observing a value narrows the candidate set:

.. testcode::

   thrifty.observe("city", "NYC")
   print(sorted(thrifty.state.candidate_rows))
   print(f"{thrifty.state.entropy:.2f} bits remaining")

.. testoutput::

   [0, 3]
   1.00 bits remaining

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   rag
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
