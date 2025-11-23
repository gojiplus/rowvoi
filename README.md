# rowvoi

[![PyPI version](https://badge.fury.io/py/rowvoi.svg)](https://badge.fury.io/py/rowvoi)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://gojiplus.github.io/rowvoi/)
[![CI](https://github.com/gojiplus/rowvoi/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/rowvoi/actions/workflows/ci.yml)

Interactive disambiguation of rows in a dataset using value-of-information policies.

## Business Problems & Use Cases

**Entity Resolution & Data Deduplication**
- **Customer Matching**: When merging customer databases, efficiently determine which additional attributes (email, phone, address) to check to uniquely identify duplicate customer records
- **Product Catalog Matching**: In e-commerce, decide which product attributes to compare when matching items across different supplier catalogs

**Interactive Data Cleaning**
- **Record Validation**: When data quality issues create multiple potential matches, systematically determine which fields to verify to confirm the correct record
- **Survey Data Linkage**: Link survey responses to master databases by strategically selecting which demographic questions to ask

**Active Learning & Human-in-the-Loop ML**
- **Annotation Prioritization**: Optimize which data points to label by identifying records that are most informative for disambiguation
- **Feature Selection**: Determine which costly features (lab tests, manual reviews) provide maximum information gain for classification

**Fraud Detection & Investigation**
- **Transaction Investigation**: When multiple transactions could be fraudulent, determine which account details to investigate first
- **Identity Verification**: Efficiently verify user identity by selecting the minimum set of verification questions needed

## Overview

The `rowvoi` package provides tools for interactively disambiguating rows in a dataset. Given a small set of candidate rows, it helps answer questions such as:

- Which columns (features) must be observed to uniquely distinguish these rows?
- How much information does a given feature provide about which row is correct?
- Under a noise model and frequency priors, which feature should we acquire next to maximize expected reduction in uncertainty?
- How does a greedy feature acquisition policy compare to the optimal minimal key in practice?

## Installation

```bash
pip install rowvoi
```

For development:

```bash
uv pip install -e ".[dev,docs]"
```

## Quick Start

### Finding Minimal Keys

```python
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
```

### Model-Based Value of Information

```python
from rowvoi import RowVoiModel, CandidateState

# Fit a model to estimate expected information gain
model = RowVoiModel().fit(df)

# Create initial state with uniform priors
state = CandidateState(
    candidate_rows=[0, 2],
    posterior={0: 0.5, 2: 0.5},
    observed_cols=[],
    observed_values={}
)

# Get feature suggestion
suggestion = model.suggest_next_feature(df, state)
print(f"Next feature to query: {suggestion.col}")
print(f"Expected information gain: {suggestion.expected_ig:.3f}")
```

### Mutual Information for Feature Selection

```python
from rowvoi import candidate_mi, best_feature_by_candidate_mi

# Calculate mutual information for a specific column
mi = candidate_mi(df, state, "A")
print(f"MI for column A: {mi:.3f}")

# Find best feature by mutual information
best_col = best_feature_by_candidate_mi(df, state)
print(f"Best column by MI: {best_col}")
```

### Simulating and Benchmarking

```python
from rowvoi import sample_candidate_sets, benchmark_policy

# Sample candidate sets from the dataframe
candidate_sets = sample_candidate_sets(df, k=2, n_samples=10)

# Benchmark a greedy policy
results = benchmark_policy(
    df,
    candidate_sets,
    policy=lambda df, cs: model.suggest_next_feature(df, cs).col
)
print(f"Average queries needed: {results.mean_queries:.2f}")
```

## API Overview

The package is organized into submodules:

### Core Types (`rowvoi.types`)
- `CandidateState`: State for an interactive disambiguation session
- `RowIndex`, `ColName`: Type aliases for clarity

### Logical Methods (`rowvoi.logical`)
- `is_key()`: Check if columns uniquely identify rows
- `minimal_key_exact()`: Find minimal key using exhaustive search
- `minimal_key_greedy()`: Find minimal key using greedy heuristic

### Mutual Information (`rowvoi.mi`)
- `candidate_mi()`: Calculate MI between column and row identity
- `best_feature_by_candidate_mi()`: Select feature with highest MI

### Machine Learning (`rowvoi.ml`)
- `RowVoiModel`: Model-based value-of-information policy
- `FeatureSuggestion`: Feature recommendation with expected gain

### Simulation (`rowvoi.simulate`)
- `sample_candidate_sets()`: Generate test scenarios
- `benchmark_policy()`: Evaluate feature acquisition policies
- `AcquisitionResult`: Results from policy evaluation

## Documentation

Full documentation is available at [https://gojiplus.github.io/rowvoi/](https://gojiplus.github.io/rowvoi/)

## Development

### Running Tests

```bash
make test
```

### Linting and Formatting

```bash
make lint
make format
```

### Building Documentation

```bash
make docs
```

### Local CI Testing

```bash
make ci-docker
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{rowvoi,
  title = {rowvoi: Interactive Row Disambiguation with Value of Information},
  author = {Gaurav Sood},
  year = {2025},
  url = {https://github.com/gojiplus/rowvoi}
}
```