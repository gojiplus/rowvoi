# rowvoi: Minimal keys and row-wise value-of-information for disambiguating tabular records

[![PyPI version](https://img.shields.io/pypi/v/rowvoi.svg)](https://pypi.org/project/rowvoi/)
[![CI](https://github.com/gojiplus/rowvoi/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/rowvoi/actions/workflows/ci.yml)
[![Downloads](https://pepy.tech/badge/rowvoi)](https://pepy.tech/project/rowvoi)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://gojiplus.github.io/rowvoi/)

**RowVoi** is a Python library for **interactive row disambiguation** in tabular data. It helps you find the minimal set of columns needed to distinguish rows, and suggests which column to query next to reduce ambiguity most efficiently.

## 🎯 Key Features

- **Deterministic Key Finding**: Use set cover algorithms to find minimal distinguishing column sets
- **Interactive Disambiguation**: Step-by-step column selection using value-of-information
- **Multiple Policies**: Greedy coverage, mutual information, model-based, and random selection strategies
- **Cost-Aware Selection**: Support for column acquisition costs and budget constraints
- **Comprehensive Evaluation**: Tools for benchmarking and comparing different strategies

## 📚 Use Cases

### Entity Resolution & Deduplication
- **Customer Matching**: Which fields (email, phone, address) uniquely identify customers?
- **Product Catalogs**: What attributes distinguish similar items across suppliers?

### Interactive Data Collection  
- **Survey Optimization**: Which demographic questions resolve identity ambiguity?
- **Medical Diagnosis**: What tests provide maximum diagnostic information?

### Active Learning & Feature Selection
- **Costly Features**: When API calls or lab tests are expensive, which ones matter most?
- **Human-in-the-Loop**: Guide annotators to the most informative questions

## 🚀 Quick Start

### Installation

```bash
pip install rowvoi
```

### Basic Example

```python
import pandas as pd
from rowvoi import find_key, CandidateState, GreedyCoveragePolicy, DisambiguationSession

# Your data
df = pd.DataFrame({
    'name': ['Alice', 'Alice', 'Bob', 'Bob'],
    'age': [25, 25, 30, 30], 
    'city': ['NYC', 'LA', 'NYC', 'SF'],
    'email': ['a1@x.com', 'a2@x.com', 'b1@x.com', 'b2@x.com']
})

# Find minimal distinguishing columns
rows = [0, 1, 2, 3]
key = find_key(df, rows)  # -> ['email']

# Interactive disambiguation
state = CandidateState.uniform([0, 1])  # Alice records
policy = GreedyCoveragePolicy()
session = DisambiguationSession(df, [0, 1], policy=policy)

# Get next question
suggestion = session.next_question()
print(f"Ask about: {suggestion.col}")  # -> 'city' or 'email'

# Observe an answer and update
step = session.observe('city', 'NYC')  
print(f"Remaining candidates: {session.state.candidate_rows}")  # -> [0]
```

## 📖 API Overview

### Core Types

```python
from rowvoi import CandidateState, FeatureSuggestion

# Track disambiguation state
state = CandidateState(
    candidate_rows=[0, 1, 2],           # Possible rows
    posterior=np.array([0.5, 0.3, 0.2]), # Probabilities  
    observed_cols={'name'},              # Asked columns
    observed_values={'name': 'Alice'}    # Observed values
)

# Column recommendation
suggestion = FeatureSuggestion(
    col='age',                    # Recommended column
    score=1.2,                   # Selection score
    expected_voi=0.8,           # Expected information gain
    marginal_cost=2.0           # Query cost
)
```

### Deterministic Key Finding

```python
from rowvoi import KeyProblem, find_key, plan_key_path

# Find minimal key
key = find_key(df, rows=[0,1,2], strategy="greedy")

# Different algorithms  
key_exact = find_key(df, rows, strategy="exact")       # Optimal (slow)
key_sa = find_key(df, rows, strategy="sa")             # Simulated annealing
key_ga = find_key(df, rows, strategy="ga")             # Genetic algorithm

# Plan acquisition sequence
path = plan_key_path(df, rows, costs={'name': 1, 'email': 5})
print(path.columns())  # -> ['name', 'age', ...]
```

### Interactive Policies

```python
from rowvoi import GreedyCoveragePolicy, CandidateMIPolicy, MIPolicy, RandomPolicy

# Greedy pairwise coverage
policy = GreedyCoveragePolicy(
    costs={'email': 5.0, 'name': 1.0},
    objective='entropy'  # or 'pairs'
)

# Mutual information on candidates only
policy = CandidateMIPolicy(normalize=True)

# Model-based mutual information  
from rowvoi import RowVoiModel
model = RowVoiModel(noise=0.1).fit(df)
policy = MIPolicy(model=model, objective='mi_over_cost')

# Random baseline
policy = RandomPolicy(seed=42)
```

### Interactive Sessions

```python
from rowvoi import DisambiguationSession, StopRules

# Create session
session = DisambiguationSession(
    df, [0,1,2,3], 
    policy=GreedyCoveragePolicy()
)

# Manual interaction
suggestion = session.next_question()
step = session.observe('age', 25)

# Automated session
stop = StopRules(max_steps=5, cost_budget=10.0, target_unique=True)
steps = session.run(stop, true_row=1)  # Simulate answering
print(f"Resolved in {len(steps)} steps")
```

### Evaluation & Benchmarking

```python
from rowvoi import sample_candidate_sets, evaluate_policies, evaluate_keys

# Sample test cases
candidate_sets = sample_candidate_sets(df, subset_size=4, n_samples=20)

# Compare key-finding methods
methods = {
    'greedy': lambda df, rows: find_key(df, rows, strategy='greedy'),
    'exact': lambda df, rows: find_key(df, rows, strategy='exact')
}
key_results = evaluate_keys(df, candidate_sets, methods)

# Compare interactive policies  
policies = {
    'greedy': GreedyCoveragePolicy(),
    'mi': CandidateMIPolicy(),
    'random': RandomPolicy(seed=42)
}
policy_stats = evaluate_policies(df, candidate_sets, policies)
for stat in policy_stats:
    print(f"{stat.name}: {stat.mean_steps:.1f} steps, {stat.success_rate:.1%} success")
```

## 🔬 Advanced Features

### Cost-Aware Selection

```python
# Define column costs
costs = {
    'name': 1.0,      # Cheap: already have
    'age': 2.0,       # Moderate: need to ask  
    'email': 10.0,    # Expensive: need verification
    'ssn': 50.0       # Very expensive: sensitive
}

policy = GreedyCoveragePolicy(costs=costs)
session = DisambiguationSession(df, rows, policy=policy, feature_costs=costs)

# Budget-constrained planning
path = plan_key_path(df, rows, costs=costs)
affordable_cols = path.prefix_for_budget(budget=5.0)
```

### Model-Based Selection

```python
# Train on historical data
model = RowVoiModel(
    noise=0.05,           # Account for measurement noise
    normalize_cols=True   # Normalize feature distributions
).fit(df)

# Use for adaptive selection
policy = MIPolicy(model=model, feature_costs=costs)
suggestion = policy.suggest(df, state)
print(f"Expected VoI: {suggestion.expected_voi:.3f} bits")
```

### Probabilistic Methods

```python
from rowvoi import find_key_probabilistic, plan_key_path_probabilistic

# Account for noise/uncertainty
key = find_key_probabilistic(df, rows, noise_rate=0.1)
path = plan_key_path_probabilistic(df, rows, noise_rate=0.1, costs=costs)
```

## 📊 Complete Examples

### Example 1: Customer Deduplication

```python
import pandas as pd
from rowvoi import find_key, GreedyCoveragePolicy, DisambiguationSession

# Customer database with potential duplicates
customers = pd.DataFrame({
    'first_name': ['John', 'John', 'Jane', 'Jane'],
    'last_name': ['Smith', 'Smith', 'Doe', 'Smith'], 
    'email': ['j1@ex.com', 'j2@ex.com', 'jane@ex.com', 'j3@ex.com'],
    'phone': ['555-0101', '555-0102', '555-0201', '555-0301'],
    'zip_code': ['10001', '10002', '10001', '10001']
})

# Find minimal fields for disambiguation
duplicates = [0, 1]  # Two "John Smith" records
key = find_key(customers, duplicates)
print(f"Minimal distinguishing fields: {key}")

# Interactive disambiguation with costs
costs = {'email': 1, 'phone': 2, 'zip_code': 1, 'first_name': 0, 'last_name': 0}
policy = GreedyCoveragePolicy(costs=costs, objective='entropy')
session = DisambiguationSession(customers, duplicates, policy=policy, feature_costs=costs)

# Simulate resolving the duplicate
suggestion = session.next_question()
print(f"First question: {suggestion.col}")
step = session.observe(suggestion.col, customers.iloc[0][suggestion.col])
print(f"Resolved: {session.state.is_unique}")
```

### Example 2: Survey Optimization

```python
from rowvoi import CandidateMIPolicy, StopRules, evaluate_policies

# Survey response data
survey = pd.DataFrame({
    'age_group': ['18-25', '26-35', '18-25', '36-45', '26-35'],
    'income': ['<50k', '50-100k', '<50k', '>100k', '50-100k'],
    'education': ['HS', 'College', 'HS', 'Graduate', 'College'],
    'location': ['Urban', 'Suburban', 'Rural', 'Urban', 'Suburban']
})

# Compare question-asking strategies
policies = {
    'coverage': GreedyCoveragePolicy(objective='entropy'),
    'mutual_info': CandidateMIPolicy(normalize=True),
    'random': RandomPolicy(seed=42)
}

# Test on random respondent groups
candidate_sets = sample_candidate_sets(survey, subset_size=3, n_samples=50)
stop_rules = StopRules(max_steps=3, target_unique=True)

stats = evaluate_policies(survey, candidate_sets, policies, stop=stop_rules)
for stat in stats:
    print(f"{stat.name}: {stat.mean_steps:.1f} questions, "
          f"{stat.success_rate:.0%} identification rate")
```

## 🧪 Algorithm Details

### Set Cover for Keys

Finding minimal distinguishing columns is NP-hard set cover:
- **Universe**: All pairs of rows that need distinguishing
- **Sets**: Each column covers pairs it separates  
- **Goal**: Minimum cost column set covering all pairs

RowVoi implements:
- **Greedy**: Fast O(nm log m) approximation with ln(m) ratio guarantee
- **Exact**: Branch-and-bound for optimal solutions (small problems)
- **Metaheuristics**: Simulated annealing and genetic algorithms for large problems

### Value of Information

For interactive selection, RowVoi uses mutual information:
```
I(RowID; Column | Observed) = H(RowID | Observed) - E[H(RowID | Observed, Column)]
```

Where:
- `H(RowID | Observed)`: Current uncertainty (entropy) over which row is correct
- `E[H(RowID | Observed, Column)]`: Expected uncertainty after observing the column
- Higher mutual information = more disambiguation value

### Policy Strategies

- **GreedyCoveragePolicy**: Maximize newly distinguished pairs per cost
- **CandidateMIPolicy**: Maximize mutual information on current candidates  
- **MIPolicy**: Use fitted model for robust MI estimation with noise handling
- **RandomPolicy**: Random selection baseline for comparison

## 📝 Development

### Running Tests
```bash
uv run pytest tests/ -v
```

### Code Quality
```bash
uv run ruff check .
uv run mypy .
```

### Building Documentation
```bash
cd docs && make html
```

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 Citation

```bibtex
@software{sood2025rowvoi,
  author       = {Sood, Gaurav},
  title        = {RowVoi: Interactive Row Disambiguation with Value-of-Information},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/gojiplus/rowvoi},
  version      = {0.2.0}
}
```
