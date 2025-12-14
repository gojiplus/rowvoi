"""rowvoi: finding minimal distinguishing columns and the next best feature to query.

The rowvoi package provides tools for row disambiguation in tabular data:
- Deterministic key finding (minimal set cover on row pairs)
- Probabilistic key finding with models
- Interactive disambiguation sessions
- Policy-based column selection
- Comprehensive evaluation tools
"""

import logging
from importlib import metadata

# Core types and data structures
from .core import CandidateState, FeatureSuggestion

# Evaluation and simulation
from .eval import (
    AcquisitionResult,
    KeyEvalResult,
    PolicyEvalStats,
    benchmark_policy,
    compute_gold_key,
    compute_gold_next_column_probabilistic,
    evaluate_keys,
    evaluate_policies,
    sample_candidate_sets,
)

# Deterministic keys and paths
from .keys import KeyPath, KeyProblem, find_key, plan_key_path

# Machine learning model
from .ml import RowVoiModel

# Policies for column selection
from .policies import (
    CandidateMIPolicy,
    GreedyCoveragePolicy,
    MIPolicy,
    Policy,
    RandomPolicy,
)

# Probabilistic methods
from .prob_keys import find_key_probabilistic, plan_key_path_probabilistic

# Interactive sessions
from .session import DisambiguationSession, StopRules
from .types import ColName, RowIndex

try:
    __version__ = metadata.version("rowvoi")
except metadata.PackageNotFoundError:
    # Fallback for development installs
    __version__ = "0.2.0"

__all__ = [
    # Core types
    "CandidateState",
    "FeatureSuggestion",
    "ColName",
    "RowIndex",
    # Deterministic methods
    "KeyProblem",
    "KeyPath",
    "find_key",
    "plan_key_path",
    # Probabilistic methods
    "find_key_probabilistic",
    "plan_key_path_probabilistic",
    # Policies
    "Policy",
    "GreedyCoveragePolicy",
    "MIPolicy",
    "CandidateMIPolicy",
    "RandomPolicy",
    # Sessions
    "DisambiguationSession",
    "StopRules",
    # Model
    "RowVoiModel",
    # Evaluation
    "sample_candidate_sets",
    "compute_gold_key",
    "compute_gold_next_column_probabilistic",
    "evaluate_keys",
    "evaluate_policies",
    "benchmark_policy",
    "KeyEvalResult",
    "PolicyEvalStats",
    "AcquisitionResult",
    # Logging
    "get_logger",
]


# Configure package-wide logging
def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the rowvoi package.

    Parameters
    ----------
    name : str | None, optional
        Logger name. If None, uses the package name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name is None:
        name = __name__.split(".")[0]
    return logging.getLogger(name)


# Set up default logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
