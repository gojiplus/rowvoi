# rowvoi: finding minimal distinguishing columns and the next best feature to query

[![PyPI version](https://img.shields.io/pypi/v/rmcp.svg)](https://pypi.org/project/rowvoi/)
[![CI](https://github.com/gojiplus/rowvoi/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/rowvoi/actions/workflows/ci.yml)
[![Downloads](https://pepy.tech/badge/rowvoi)](https://pepy.tech/project/rowvoi)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://gojiplus.github.io/rowvoi/)

`rowvoi` is a small library for **row disambiguation** in tabular data:

Given a set of candidate rows and a set of columns,

  * which minimal set of columns distinguishes those rows?
  * if some columns are not yet observed, which column should you query next to reduce ambiguity as quickly as possible?

It combines:

  * classical **functional dependencies / minimal keys** (set cover on row pairs),
  * information-theoretic **value-of-information** (mutual information between row identity and a column),
  * and a light model-based layer for instance-specific, adaptive feature acquisition.

----

## Table of Contents

1. [Where this fits in the literature](#1-where-this-fits-in-the-literature)
2. [Business Problems & Use Cases](#2-business-problems--use-cases)
3. [Installation](#3-installation)
4. [Core Concepts](#4-core-concepts)
5. [API Overview](#5-api-overview)
6. [Quick Start](#6-quick-start)
7. [Documentation & Development](#7-documentation--development)

---
-----

## 1. Where this fits in the literature

### 1.1 Deterministic keys and set cover

Suppose you have a subset of rows ($R$) from a DataFrame and want the smallest set of columns ($C$) such that every pair of rows in ($R$) differs on at least one column in ($C$). In database terms, ($C$) is a **key** for ($R$); in combinatorics, this is a **set cover** problem:

  * **Universe ($U$):** all unordered pairs of rows ($\{i,j\}$) in ($R$).
  * **Each column ($c$):** covers the pairs it distinguishes
    $$S_c = \{\{i,j\} \in U : x_{ic} \neq x_{jc}\}$$
  * **Goal:** find a minimum-cost ($C$) such that $$\bigcup_{c\in C} S_c = U$$

Set cover is **NP-hard**; greedy set cover achieves a logarithmic approximation factor and is optimal up to constants under standard complexity assumptions. This is exactly what `minimal_key_greedy` implements for rows and columns.

### 1.2 Mutual information and value-of-information

Now suppose some columns are missing (because you haven't asked the user, run the test, or fetched the field yet). You have a candidate set of rows ($R$), and a belief over which row is "the truth":

  * random variable ($\mathbf{R}$) = row identity,
  * current evidence ($\mathbf{E}$) (already observed columns and their values),
  * candidate column ($\mathbf{X}_j$) you might query next.

The value of querying column ($j$) for disambiguating ($\mathbf{R}$) is the **conditional mutual information**:
$$I(\mathbf{R}; \mathbf{X}_j \mid \mathbf{E}) = H(\mathbf{R} \mid \mathbf{E}) - \mathbb{E}_{\mathbf{X}_j \mid \mathbf{E}}[H(\mathbf{R} \mid \mathbf{E}, \mathbf{X}_j)]$$
where ($H(\cdot)$) is Shannon entropy. Intuitively:

  * $H(\mathbf{R} \mid \mathbf{E})$: how confused you are now about which row is correct;
  * $H(\mathbf{R} \mid \mathbf{E}, \mathbf{X}_j)$: how confused you'll be after seeing ($\mathbf{X}_j$), averaged over the possible values it might take.

Greedy "ask the column with largest $I(\mathbf{R}; \mathbf{X}_j \mid \mathbf{E})$" is the local, data-table version of:

  * **information gain** in decision trees,
  * **active feature acquisition** and **adaptive submodular optimization**,
  * and **Bayesian experimental design / adaptive testing**.

`candidate_mi` and `RowVoiModel.suggest_next_feature` implement this style of policy.

### 1.3 Probabilistic & $\varepsilon$-relaxed disambiguation (conceptual)

In practice:

  * columns might be noisy or locally constant,
  * you might be happy to stop with a small residual ambiguity.

Two complementary notions of "$\varepsilon$-good enough" show up naturally:

1.  **Posterior-based:** stop when the posterior on the most probable row exceeds ($1-\varepsilon$):
    $$1 - \max_r p(r \mid E) \le \varepsilon$$
2.  **Pairwise coverage-based:** stop when the fraction of unresolved row pairs is $\le \varepsilon$.

The underlying math connects to:

  * **probabilistic set cover / chance-constrained covering**, where coverage events are random;
  * **stochastic submodular cover / adaptive submodular optimization**, where you choose columns sequentially and observe their realizations.

`rowvoi`'s deterministic minimal keys and MI-based policies sit on top of this theory; the package is designed so that $\varepsilon$-style stopping rules and probabilistic coverage models can be added without changing the core API.

-----

## 2. Business Problems & Use Cases

Some concrete places where "which columns distinguish these rows?" and "what should I ask next?" show up:

### Entity Resolution & Data Deduplication

  * **Customer Matching:** Merge customer databases by asking/planning which fields (**email, phone, address, DOB**) to verify so that a small set of checks uniquely identifies the right record.
  * **Product Catalog Matching:** When multiple supplier SKUs look similar, decide which attributes (**brand, size, color, GTIN**) to compare to separate otherwise ambiguous items.

### Interactive Data Cleaning

  * **Record Validation:** Data quality issues may leave multiple plausible matches for a record. `rowvoi` can suggest which field to check next (e.g. **postal code vs last name**) to resolve the ambiguity.
  * **Survey Data Linkage:** When linking survey responses to a master frame, choose which demographic questions to ask (or re-ask) to uniquely identify the record.

### Active Learning & Human-in-the-Loop ML

  * **Annotation Prioritization:** Treat each candidate match or cluster of rows as a decision problem and ask humans to label the fields that most reduce confusion.
  * **Costly Feature Selection:** For models that can request expensive features (**lab tests, manual review, external API calls**), use value-of-information to decide which features are worth acquiring.

### Fraud Detection & Investigation

  * **Transaction Investigation:** When several transactions or identities look suspiciously similar, choose which account details or metadata fields to inspect first.
  * **Identity Verification:** Build a short, adaptive script of verification questions that usually identifies a user with a handful of high-information questions.

-----

## 3\. Installation

```bash
pip install rowvoi
```

For development:

```bash
uv pip install -e ".[dev,docs]"
```

-----

## 4. Core Concepts

`rowvoi` is designed around a few simple ideas:

  * **Rows as candidates:** A small set of **row indices** in a DataFrame `df` (e.g. `[12, 45, 101]`) that are plausible matches, models, or strategies.
  * **Columns as questions:** DataFrame columns are **candidate features** you can use to distinguish those rows. Some may already be observed; others may be missing or costly.
  * **Keys / minimal distinguishing sets:** A set of columns is a **key** for the candidate set if no two rows agree on all those columns. Minimal keys are the set-cover side of the problem.
  * **`CandidateState`:** A lightweight object that tracks:
      * which rows are still in play,
      * your current posterior over them,
      * which columns you've already queried and what values you saw.
  * **Policies over columns:** Given a `CandidateState`, a policy (like `RowVoiModel`) suggests the next best column to query using **mutual information / expected entropy reduction**.

-----

## 5. API Overview

### 5.1 Types (`rowvoi.types`)

  * `RowIndex`: Type alias for row indices (usually `int`).
  * `ColName`: Type alias for column labels (any hashable).
  * `CandidateState`:
    ```python
    CandidateState(
        candidate_rows: list[RowIndex],
        posterior: dict[RowIndex, float],
        observed_cols: list[ColName],
        observed_values: dict[ColName, object],
    )
    ```
    Represents the state of an interactive disambiguation session.

### 5.2 Deterministic / Logical Methods (`rowvoi.logical`)

These functions assume all relevant column values are known in `df` and solve the **minimal key / functional dependency** problem.

  * `is_key(df, rows, cols) -> bool`: Check whether the columns in `cols` uniquely distinguish the rows in `rows` (no two rows have identical values on all `cols`).
  * `minimal_key_exact(df, rows, columns=None, max_search_cols=20) -> list[ColName]`: Use exhaustive search / branch-and-bound over columns to find a **provably minimal key** for the specified rows. Exponential in the number of columns considered; `max_search_cols` bounds the search.
  * `minimal_key_greedy(df, rows, columns=None, costs=None) -> list[ColName]`:
      * Greedy **set cover on row pairs**:
          * Universe = all unordered pairs of rows in `rows`,
          * At each step, pick the column that distinguishes the most unresolved pairs (optionally normalized by cost),
          * Stop when all pairs are separated.
      * This is **fast** and comes with the standard logarithmic approximation guarantee for set cover.

### 5.3 Mutual Information (`rowvoi.mi`)

Local, instance-specific mutual information between row identity and a column.

  * `candidate_mi(df, state: CandidateState, col: ColName) -> float`:
      * Compute $I(\mathbf{R}; \mathbf{X}_{\text{col}} \mid \mathbf{E})$ where:
          * $\mathbf{R}$ is "which row from `state.candidate_rows` is true?" with prior/posterior `state.posterior`,
          * $\mathbf{E}$ is the current evidence encoded in `state.observed_cols` and `state.observed_values`.
  * `best_feature_by_candidate_mi(df, state: CandidateState, candidate_cols=None) -> ColName`: Among the unobserved columns (default: all DataFrame columns not in `state.observed_cols`), return the one with **highest mutual information** with row identity.

### 5.4 Model-Based Value of Information (`rowvoi.ml`)

A small model class for reusing global information (e.g. value distributions, noise models) to make feature selection more robust.

  * `RowVoiModel(smoothing=1e-6, noise=0.0, normalize_cols=True)`: A model-based policy for next-feature selection.
  * `fit(df, discrete_cols=None, bins=3) -> RowVoiModel`: Optionally discretizes numeric columns and precomputes marginal distributions / metadata needed for MI approximation and noise handling.
  * `suggest_next_feature(df, state: CandidateState, candidate_cols=None, objective="mi", feature_costs=None) -> FeatureSuggestion`:
      * Suggest the next column to query for this state.
      * `objective="mi"`: maximize expected mutual information ($I(\mathbf{R}; \mathbf{X}_j \mid \mathbf{E})$).
      * `"mi_over_cost"`: maximize MI per unit feature cost.
      * Returns a `FeatureSuggestion` object:
        ```python
        FeatureSuggestion(
            col: ColName,
            voi: float,              # estimated information gain
            normalized_voi: float,   # optionally MI / H(X_j)
            details: dict[str, float]
        )
        ```
    (A separate helper like `run_until_epsilon` can be built on top to run a full session until posterior uncertainty is below a chosen $\varepsilon$.)

### 5.5 Simulation & Benchmarking (`rowvoi.simulate`)

Tools for testing and benchmarking policies on synthetic or real datasets.

  * `sample_candidate_sets(df, k: int, n_samples: int, rng=None) -> list[list[RowIndex]]`: Randomly sample `n_samples` subsets of `k` rows from `df`.
  * `AcquisitionResult`: A small record type storing results for one disambiguation episode (e.g., number of questions asked, whether the correct row was identified, which columns were used).
  * `benchmark_policy(df, candidate_sets, policy, max_steps=10) -> BenchmarkResult`: Run a policy against a list of candidate sets and summarize its behavior. The policy is a callable like:
    ```python
    def policy(df, state: CandidateState) -> ColName:
        ...
    ```
    or, for simpler cases, a function mapping (`df`, `rows`) to a column name.

-----

## 6. Quick Start

### 6.1 Deterministic keys

```python
import pandas as pd
from rowvoi import minimal_key_greedy, minimal_key_exact

df = pd.DataFrame({
    "A": [1, 1, 2],
    "B": [3, 4, 3],
    "C": [5, 6, 7],
})

# Find minimal distinguishing columns for rows 0 and 1
rows = [0, 1]

print(minimal_key_greedy(df, rows))  # e.g. ['B']
print(minimal_key_exact(df, rows))   # ['B']
```

### 6.2 Model-based next-feature selection

```python
from rowvoi import RowVoiModel, CandidateState

# Fit model on historical data
model = RowVoiModel().fit(df)

# Two candidate rows, uniform prior, nothing observed yet
state = CandidateState(
    candidate_rows=[0, 2],
    posterior={0: 0.5, 2: 0.5},
    observed_cols=[],
    observed_values={},
)

suggestion = model.suggest_next_feature(df, state)
print(f"Next feature: {suggestion.col}")
print(f"Estimated VoI: {suggestion.voi:.3f}")
```

### 6.3 Mutual information directly

```python
from rowvoi import candidate_mi, best_feature_by_candidate_mi

mi_A = candidate_mi(df, state, "A")
print(f"MI between row ID and A: {mi_A:.3f}")

best_col = best_feature_by_candidate_mi(df, state)
print(f"Best column by MI: {best_col}")
```

### 6.4 Benchmarking a policy

```python
from rowvoi import sample_candidate_sets, benchmark_policy

# Sample 10 random pairs of rows
candidate_sets = sample_candidate_sets(df, k=2, n_samples=10)

def simple_policy(df, state):
    # use the model to suggest next feature
    return model.suggest_next_feature(df, state).col

results = benchmark_policy(df, candidate_sets, policy=simple_policy)
print(f"Average queries needed: {results.mean_queries:.2f}")
```

-----

## 7. Documentation & Development

Full documentation and examples:

👉 [https://gojiplus.github.io/rowvoi/](https://gojiplus.github.io/rowvoi/)

**Tests**

```bash
make test
```

**Linting & Formatting**

```bash
make lint
make format
```

**Build Docs**

```bash
make docs
```

**Local CI**

```bash
make ci-docker
```

---

## Citation

If you use `rowvoi` in your research, please cite it using:

```bibtex
@software{sood2025rowvoi,
  author       = {Sood, Gaurav},
  title        = {RowVoi: Row-wise Value of Information for Data Collection},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/gojiplus/rowvoi},
  version      = {0.1.0}
}
```

Or use the [CITATION.cff](CITATION.cff) file for automatic citation generation in GitHub.

