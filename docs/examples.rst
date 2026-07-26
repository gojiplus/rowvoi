Examples
========

Every example on this page is executed as part of the documentation build, so
the printed output is what the code actually produces.

Entity resolution
-----------------

You have records that may refer to the same person and need to know which
fields settle it.

.. testcode::

   import pandas as pd
   from rowvoi import KeyProblem

   customers = pd.DataFrame({
       "name":  ["Alice Smith", "Alice Smith", "Bob Jones", "Alice Smith"],
       "city":  ["NYC", "NYC", "NYC", "SF"],
       "email": ["a@x.com", "alice@y.com", "b@x.com", "a@x.com"],
       "tier":  ["gold", "gold", "gold", "gold"],
   })

   problem = KeyProblem(customers, rows=[0, 1, 2, 3])

   print(problem.is_key(["name"]))
   print(problem.is_key(["name", "city"]))
   print(problem.is_key(["email", "city"]))

.. testoutput::

   False
   False
   True

``is_key`` is a method on the problem, not a free function. Partial credit is
available through ``epsilon_pairs``, which asks only that a given fraction of
row pairs be separated:

.. testcode::

   print(f"{problem.pairwise_coverage(['name']):.2f}")
   print(problem.is_key(["name"], epsilon_pairs=0.5))

.. testoutput::

   0.50
   True

Cost-aware acquisition
----------------------

Fields differ in what they cost to obtain, so a minimal key by *count* is not
necessarily a minimal key by *effort*.

.. testcode::

   from rowvoi import find_key, plan_key_path

   costs = {"name": 1.0, "city": 1.0, "email": 20.0, "tier": 1.0}

   print(find_key(customers, [0, 1, 2, 3], strategy="exact"))
   print(find_key(customers, [0, 1, 2, 3], costs=costs, strategy="exact"))

.. testoutput::

   ['city', 'email']
   ['city', 'email']

Here the expensive field is genuinely required -- nothing else separates the
two New York Alices -- so pricing it does not change the answer.
``plan_key_path`` gives the acquisition *order* along with what each step
buys:

.. testcode::

   path = plan_key_path(customers, [0, 1, 2, 3], costs=costs)
   for step in path.steps:
       print(f"{step.col:6s} cost={step.cumulative_cost:5.1f} "
             f"coverage={step.coverage:.0%}")

.. testoutput::

   name   cost=  1.0 coverage=50%
   city   cost=  2.0 coverage=83%
   email  cost= 22.0 coverage=100%

That curve is the useful part: five sixths of the ambiguity goes away for two
units of effort, and the last sixth costs twenty more.

.. testcode::

   print(path.prefix_for_budget(10.0))
   print(path.prefix_for_epsilon_pairs(0.2))

.. testoutput::

   ['name', 'city']
   ['name', 'city']

Running an interactive session
------------------------------

A session drives a policy to completion, either against a real user or, as
here, a simulated one.

.. testcode::

   from rowvoi import CandidateMIPolicy, DisambiguationSession, StopRules

   session = DisambiguationSession(
       customers,
       candidate_rows=[0, 1, 2, 3],
       policy=CandidateMIPolicy(costs=costs),
       feature_costs=costs,
   )

   steps = session.run(StopRules(target_unique=True), true_row=1)
   for step in steps:
       print(f"asked {step.col!r} -> {step.observed_value!r}")
   print("resolved to row", session.state.unique_row)

.. testoutput::

   asked 'name' -> 'Alice Smith'
   asked 'city' -> 'NYC'
   asked 'email' -> 'alice@y.com'
   resolved to row 1

Model-based selection
---------------------

:class:`~rowvoi.RowVoiModel` learns value frequencies from historical data and
carries a noise model, so a disagreeing observation lowers a candidate's
probability instead of eliminating it.

.. testcode::

   from rowvoi import CandidateState, RowVoiModel

   model = RowVoiModel(noise=0.05).fit(customers)
   state = CandidateState.uniform([0, 1, 2, 3])

   suggestion = model.suggest_next_feature(customers, state)
   print(suggestion.col)
   print(f"{suggestion.expected_voi:.2f} bits")

.. testoutput::

   email
   1.17 bits

Benchmarking policies
---------------------

``evaluate_policies`` runs several policies over sampled candidate sets and
reports averages.

.. testcode::

   from rowvoi import (
       GreedyCoveragePolicy,
       RandomPolicy,
       evaluate_policies,
       sample_candidate_sets,
   )

   candidate_sets = sample_candidate_sets(
       customers, subset_size=2, n_samples=5, random_state=0
   )

   stats = evaluate_policies(
       customers,
       candidate_sets,
       policies={
           "coverage": GreedyCoveragePolicy(),
           "mutual_info": CandidateMIPolicy(),
           "random": RandomPolicy(seed=0),
       },
       stop=StopRules(target_unique=True),
   )

   for s in sorted(stats, key=lambda s: s.name):
       print(f"{s.name:12s} mean_steps={s.mean_steps:.2f}")

.. testoutput::

   coverage     mean_steps=1.00
   mutual_info  mean_steps=1.00
   random       mean_steps=1.60

``sample_candidate_sets`` takes ``subset_size`` as a keyword-only argument,
and ``evaluate_policies`` takes a mapping of names to policy objects.
