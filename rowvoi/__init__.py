"""rowvoi: finding minimal distinguishing columns and the next best feature to query.

The rowvoi package provides tools for row disambiguation in tabular data:
- Deterministic key finding (minimal set cover on row pairs)
- Probabilistic key finding with models
- Interactive disambiguation sessions
- Policy-based column selection
- Comprehensive evaluation tools
"""

import importlib.metadata
import logging

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

# Set cover engine
from .setcover import SolverUnavailableError
from .types import ColName, RowIndex

try:
    __version__ = importlib.metadata.version("rowvoi")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    # The version is derived from the git tag at build time, so an uninstalled
    # source tree has no metadata to read. Never hardcode a number here: it
    # would silently disagree with the tag.
    __version__ = "0+unknown"

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
    "SolverUnavailableError",
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

    Args:
        name: Logger name. If None, uses the package name.

    Returns:
        Configured logger instance.
    """
    if name is None:
        name = __name__.split(".")[0]
    return logging.getLogger(name)


# Set up default logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
