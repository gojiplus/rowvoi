rowvoi Documentation
===================

Interactive disambiguation of rows in a dataset using value-of-information policies.

Overview
--------

The ``rowvoi`` package provides tools for interactively disambiguating rows in a dataset. Given a small set of candidate rows, it helps answer questions such as:

* Which columns (features) must be observed to uniquely distinguish these rows?
* How much information does a given feature provide about which row is correct?
* Under a noise model and frequency priors, which feature should we acquire next to maximize expected reduction in uncertainty?
* How does a greedy feature acquisition policy compare to the optimal minimal key in practice?

Installation
------------

.. code-block:: bash

   pip install rowvoi

For development:

.. code-block:: bash

   uv pip install -e ".[dev,docs]"

Quick Start
-----------

Finding Minimal Keys
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from rowvoi import minimal_key_greedy, minimal_key_exact

   df = pd.DataFrame({
       "A": [1, 1, 2],
       "B": [3, 4, 3],
       "C": [5, 6, 7]
   })

   # Find minimal distinguishing columns for rows 0 and 1
   print(minimal_key_greedy(df, [0, 1]))  # ['B']
   print(minimal_key_exact(df, [0, 1]))   # ['B']

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`