Examples
========

This section provides detailed examples of using the ``rowvoi`` package for interactive row disambiguation.

Basic Usage
-----------

Finding Minimal Keys
~~~~~~~~~~~~~~~~~~~~~

The most basic task is finding which columns are needed to distinguish between rows:

.. code-block:: python

   import pandas as pd
   from rowvoi import minimal_key_greedy, minimal_key_exact, is_key

   # Create a sample dataset
   df = pd.DataFrame({
       "name": ["Alice", "Bob", "Alice", "Charlie"],
       "age": [25, 30, 25, 35],
       "city": ["NYC", "LA", "SF", "NYC"],
       "salary": [50000, 60000, 55000, 70000]
   })

   # Check if columns form a key for specific rows
   print(is_key(df, ["name"], [0, 1]))      # False (Alice appears twice)
   print(is_key(df, ["name", "city"], [0, 2]))  # True

   # Find minimal distinguishing columns
   print(minimal_key_greedy(df, [0, 2]))    # ['city'] - distinguishes Alice in NYC from Alice in SF
   print(minimal_key_exact(df, [0, 1, 3]))  # ['name'] - distinguishes Alice from Bob and Charlie

Model-Based Feature Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated feature selection using information theory:

.. code-block:: python

   from rowvoi import RowVoiModel, CandidateState

   # Fit a model to the dataset
   model = RowVoiModel().fit(df)

   # Create an initial state with uncertainty between rows 0 and 1
   state = CandidateState(
       candidate_rows=[0, 1],
       posterior={0: 0.6, 1: 0.4},  # Alice is more likely than Bob
       observed_cols=[],
       observed_values={}
   )

   # Get the best feature to query next
   suggestion = model.suggest_next_feature(df, state)
   print(f"Query column: {suggestion.col}")
   print(f"Expected information gain: {suggestion.expected_ig:.3f}")

   # Simulate observing a value
   observed_value = df.loc[0, suggestion.col]  # Assume we observe the true row 0 value
   
   # Update the state
   new_state = CandidateState(
       candidate_rows=state.candidate_rows,
       posterior=state.posterior,
       observed_cols=state.observed_cols + [suggestion.col],
       observed_values={**state.observed_values, suggestion.col: observed_value}
   )

Mutual Information Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate information content of individual features:

.. code-block:: python

   from rowvoi import candidate_mi, best_feature_by_candidate_mi

   # Calculate MI for a specific column
   mi_age = candidate_mi(df, state, "age")
   mi_city = candidate_mi(df, state, "city")
   
   print(f"MI for age: {mi_age:.3f}")
   print(f"MI for city: {mi_city:.3f}")

   # Find the column with highest MI
   best_col = best_feature_by_candidate_mi(df, state)
   print(f"Best column by MI: {best_col}")

Simulation and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate different policies across multiple scenarios:

.. code-block:: python

   from rowvoi import sample_candidate_sets, benchmark_policy

   # Generate random candidate sets for testing
   candidate_sets = sample_candidate_sets(df, k=2, n_samples=50)

   # Define a simple greedy policy
   def greedy_policy(df, state):
       return best_feature_by_candidate_mi(df, state)

   # Define a model-based policy
   def model_policy(df, state):
       model = RowVoiModel().fit(df)
       return model.suggest_next_feature(df, state).col

   # Benchmark the policies
   greedy_results = benchmark_policy(df, candidate_sets, greedy_policy)
   model_results = benchmark_policy(df, candidate_sets, model_policy)

   print(f"Greedy policy - Mean queries: {greedy_results.mean_queries:.2f}")
   print(f"Model policy - Mean queries: {model_results.mean_queries:.2f}")

Advanced Usage
--------------

Custom Noise Models
~~~~~~~~~~~~~~~~~~~

The RowVoiModel supports custom noise and prior specifications:

.. code-block:: python

   # Create a model with custom parameters
   model = RowVoiModel(
       noise_rate=0.1,     # 10% chance of observing wrong value
       alpha=1.0           # Dirichlet prior concentration
   )
   model.fit(df)

   # Use with non-uniform priors
   state = CandidateState(
       candidate_rows=[0, 1, 2],
       posterior={0: 0.5, 1: 0.3, 2: 0.2},  # Non-uniform beliefs
       observed_cols=[],
       observed_values={}
   )

Working with Large Candidate Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For larger candidate sets, consider the computational trade-offs:

.. code-block:: python

   # Create a larger dataset
   large_df = pd.concat([df] * 100, ignore_index=True)
   
   # Sample multiple candidate sets
   large_candidates = sample_candidate_sets(large_df, k=5, n_samples=20)
   
   # Use greedy methods for efficiency
   for candidates in large_candidates[:5]:  # Test first 5
       state = CandidateState(
           candidate_rows=candidates,
           posterior={r: 1.0/len(candidates) for r in candidates},
           observed_cols=[],
           observed_values={}
       )
       
       # Greedy is faster for large candidate sets
       best_col = best_feature_by_candidate_mi(large_df, state)
       print(f"Best column for candidates {candidates}: {best_col}")

Interactive Session Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate a complete interactive disambiguation session:

.. code-block:: python

   def simulate_session(df, true_row, candidate_rows, max_queries=10):
       """Simulate an interactive disambiguation session."""
       state = CandidateState(
           candidate_rows=candidate_rows,
           posterior={r: 1.0/len(candidate_rows) for r in candidate_rows},
           observed_cols=[],
           observed_values={}
       )
       
       model = RowVoiModel().fit(df)
       queries = 0
       
       while len(state.candidate_rows) > 1 and queries < max_queries:
           # Get suggestion
           suggestion = model.suggest_next_feature(df, state)
           
           # Simulate observing the true value
           true_value = df.loc[true_row, suggestion.col]
           
           # Filter candidates based on observation
           new_candidates = [
               r for r in state.candidate_rows 
               if df.loc[r, suggestion.col] == true_value
           ]
           
           # Update state
           state = CandidateState(
               candidate_rows=new_candidates,
               posterior={r: 1.0/len(new_candidates) for r in new_candidates},
               observed_cols=state.observed_cols + [suggestion.col],
               observed_values={**state.observed_values, suggestion.col: true_value}
           )
           
           queries += 1
           print(f"Query {queries}: {suggestion.col} = {true_value}")
           print(f"  Remaining candidates: {new_candidates}")
       
       return queries, state.candidate_rows

   # Run a simulation
   final_queries, final_candidates = simulate_session(df, true_row=0, candidate_rows=[0, 1, 2])
   print(f"Session completed in {final_queries} queries")
   print(f"Final candidates: {final_candidates}")

This example shows how the package can be used to build interactive systems for data disambiguation tasks.