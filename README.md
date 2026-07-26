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
- **RAG Adapters**: The same two engines applied to retrieval — minimal sufficient context, clarifying questions, adaptive retrieval ([see below](#-rag-minimal-context-and-clarifying-questions))
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

### Retrieval-Augmented Generation
- **Context Compression**: Which chunks are the minimum needed to support the answer?
- **Ambiguous Queries**: When the top-k are near-tied, what should you ask the user?

## 🚀 Quick Start

### Installation

```bash
pip install rowvoi
```

The core depends only on pandas and numpy. Optional features:

```bash
pip install "rowvoi[optimization]"   # pulp, for the ILP set-cover strategy
pip install "rowvoi[claude]"         # anthropic, for the RAG LLM adapters
```

### Basic Example

```python
import pandas as pd
from rowvoi import find_key, GreedyCoveragePolicy, DisambiguationSession

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

# Interactive disambiguation of the two Alice records
session = DisambiguationSession(df, [0, 1], policy=GreedyCoveragePolicy())

# Get next question
suggestion = session.next_question()
print(f"Ask about: {suggestion.col}")  # -> 'city'

# Observe an answer and update
session.observe('city', 'NYC')
print(f"Remaining candidates: {session.state.candidate_rows}")  # -> [0]
```

`find_key` answers "what would settle this?" in one shot;
`DisambiguationSession` answers it one question at a time, which is what you
want when each answer is expensive to get.

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

A key is a minimum-cost set of columns that separates every pair of candidate
rows. Cost defaults to one per column, so with no costs supplied this
minimizes column count; supply costs and it minimizes effort instead.

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

A policy answers "which column next?" rather than "which set?". They differ in
what they optimize and what they need: coverage and candidate-MI work directly
off the candidate rows, while `MIPolicy` needs a fitted model and in exchange
handles noisy observations.

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

Policies are easy to argue about and easy to measure, so measure them. These
run a policy over sampled candidate sets and report how many questions it
needed and how often it actually resolved the row.

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

## 🔎 RAG: Minimal Context and Clarifying Questions

`rowvoi.rag` applies the same two engines to retrieval. Nothing new is
computed — set cover and mutual information are reused as-is, over a different
universe:

| Tabular                       | RAG                                              |
| ----------------------------- | ------------------------------------------------ |
| Cover row pairs with columns  | Cover **claims** with **chunks** (cost = tokens) |
| Next column by MI             | Next **clarifying question** (cost = patience)   |
| Sequential acquisition        | Next **retrieval probe** (cost = latency)        |

### Minimal sufficient context

Top-k ranks each chunk independently, so it will happily spend the budget on
five chunks supporting the same claim while a sixth claim goes unsupported.
Set cover optimizes the selection jointly, and because chunk cost is token
count, it prefers three small chunks over one large catch-all:

```python
from rowvoi.rag import Chunk, select_context, plan_context_path

chunks = [
    Chunk("doc1#p2", "The Pro plan costs $40/seat/mo.", tokens=180),
    Chunk("doc2#p1", "v4.2 shipped March 3rd, 2026.", tokens=150),
    Chunk("faq#p1",  "Pricing, releases, and SSO.",    tokens=1550),
]
claims = ["pro_plan_price", "v42_release_date"]
support = {                      # a SupportJudge fills this in; see below
    "pro_plan_price":  {"doc1#p2", "faq#p1"},
    "v42_release_date": {"doc2#p1", "faq#p1"},
}

selection = select_context(chunks, claims, support)
selection.chunks        # ['doc2#p1', 'doc1#p2'] — 330 tokens, not 1550
selection.coverage      # 1.0
selection.missing_claims  # set() — a claim nothing supports shows up here

# Or fill a context window and see what the last token bought
path = plan_context_path(chunks, claims, support)
path.prefix_for_budget(200)   # ['doc2#p1']
path.coverage_curve()         # [(150, 0.5), (330, 1.0)]
```

`epsilon_claims=0.05` lets you trade a small fraction of claims for a much
smaller prompt.

### Clarifying questions

When the retrieved set is genuinely ambiguous, ask instead of guessing. Give it
a matrix of the answer each candidate implies, and it picks the question that
splits them best per unit of user effort:

```python
from rowvoi import CandidateState
from rowvoi.rag import next_question, observe_answer, question_values

answers = {                      # an AnswerPredictor fills this in
    "Which deployment?":  ["cloud", "cloud", "self", "self"],
    "Which version?":     ["v4",    "v3",    "v4",   "v3"],
    "Are you an admin?":  ["yes",   "yes",   "yes",  "yes"],
}

# bits of uncertainty each question resolves, out of 2.0 for four candidates
question_values(answers)
# {'Which deployment?': 1.0, 'Which version?': 1.0, 'Are you an admin?': 0.0}

next_question(answers, costs={"Which version?": 5.0}).col  # 'Which deployment?'

state = CandidateState.uniform(range(4))
state = observe_answer(state, answers, "Which deployment?", "self", noise=0.02)
state.posterior            # [0.01, 0.01, 0.49, 0.49]
```

"Are you an admin?" scores exactly 0 bits — every candidate answers it the
same way, so it cannot narrow anything. A relevance-ranked list would still
ask it.

### Adaptive retrieval

Same machinery, with retrieval probes instead of questions and latency instead
of patience. Stop when the posterior is sharp enough rather than at a fixed k:

```python
from rowvoi import StopRules
from rowvoi.rag import RetrievalSession

session = RetrievalSession(
    outcomes,                     # candidates x probes matrix of predicted results
    runner=my_backend,            # anything with .run(probe)
    prior=retrieval_scores,       # retrieval scores become the prior
    costs={"rerank": 12.0},       # expensive probes must earn their place
    noise=0.05,
)
session.run(StopRules(epsilon_posterior=0.05))
session.best_candidate, session.state.max_posterior
```

### Where the LLM goes

Everything above is deterministic and needs only pandas and numpy. Producing
the matrices is what needs a model, and that boundary is `rowvoi.rag.protocols`
(`ClaimExtractor`, `SupportJudge`, `QuestionGenerator`, `AnswerPredictor`,
`ProbeRunner`). Implement them yourself, or use the bundled Claude backing:

```bash
pip install "rowvoi[claude]"
```

```python
from rowvoi.rag import extract_and_select
from rowvoi.rag.claude import ClaudeClaimExtractor, ClaudeSupportJudge

selection = extract_and_select(
    query, chunks,
    extractor=ClaudeClaimExtractor(),
    judge=ClaudeSupportJudge(),
)
```

`rowvoi.rag.claude` is never imported by `rowvoi.rag`, so the core keeps its
pandas-and-numpy-only dependency footprint. It fills each matrix in one request
and puts chunk text behind a prompt-cache breakpoint.

A full runnable walkthrough of all three, with no API key required, is in
[`examples/rag/rag_pipeline_demo.py`](examples/rag/rag_pipeline_demo.py).

## 🔬 Advanced Features

### Cost-Aware Selection

The cheapest key and the smallest key are different objects. When a column is
expensive -- a paid API call, a lab test, a question that annoys the user --
cost belongs in the objective rather than in a comment.

```python
from rowvoi import DisambiguationSession, GreedyCoveragePolicy, plan_key_path

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
from rowvoi import MIPolicy, RowVoiModel

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

When values are uncertain rather than known, work against a fitted model and
a posterior target instead of exact pair coverage. Both functions take the
model as a positional argument:

```python
from rowvoi import RowVoiModel, find_key_probabilistic, plan_key_path_probabilistic

model = RowVoiModel(noise=0.05).fit(df)

# Stop once one candidate holds 90% of the posterior mass
key = find_key_probabilistic(df, rows, model, epsilon_posterior=0.1)

path = plan_key_path_probabilistic(df, rows, model, costs=costs)
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
import pandas as pd
from rowvoi import (
    CandidateMIPolicy,
    GreedyCoveragePolicy,
    RandomPolicy,
    StopRules,
    evaluate_policies,
    sample_candidate_sets,
)

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

rowvoi follows the [py-canon](https://github.com/gojiplus/py-canon) fleet
standard, applied with [preen](https://github.com/gojiplus/preen). Development
dependencies live in PEP 735 dependency groups:

```bash
uv sync --all-groups --all-extras
```

### Running everything CI runs
```bash
make ci          # ruff, pyright, pydoclint, pytest
make check       # preen fleet-conformance check
```

### Individually
```bash
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pydoclint --config pyproject.toml rowvoi/
```

### Building documentation
```bash
make docs        # sphinx-build -W, warnings are errors
```

Every example in the docs is a `testcode` block executed by the docs build, so
`make docs` also verifies that they still produce the output they claim.

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
  version      = {0.3.0}
}
```
