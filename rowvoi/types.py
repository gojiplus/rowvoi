"""Type definitions and core data structures for rowvoi.

This module defines the primary dataclasses and type aliases used
throughout the :mod:`rowvoi` package.  In particular it defines
``RowIndex`` and ``ColName`` aliases to clarify interfaces that
operate on pandas DataFrames, and a :class:`CandidateState`
dataclass for maintaining the state of an ongoing feature acquisition
session.

The goal of the library is to assist in the process of
disambiguating an unknown row from a small set of candidates by
sequentially querying feature values.  During such an interactive
procedure you need to keep track of which rows remain plausible,
what their posterior probabilities are, which columns have already
been queried, and what the observed answers were.  The
:class:`CandidateState` dataclass encapsulates this information.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Self

# Type aliases for better code clarity
RowIndex = int
"""Alias for row indices within a pandas DataFrame."""

ColName = Hashable
"""Alias for column identifiers within a pandas DataFrame.

Pandas allows a wide variety of types for column labels (strings,
integers, tuples, etc.), so we use the ``Hashable`` bound to permit
any immutable label type.
"""


@dataclass(slots=True)
class CandidateState:
    """State for an interactive disambiguation session.

    Parameters
    ----------
    candidate_rows : List[RowIndex]
        The list of row indices that are still plausible.  These
        correspond to positions in a pandas DataFrame.  Ordering is
        not important, but it should remain consistent with
        ``posterior``.

    posterior : Dict[RowIndex, float]
        A dictionary mapping each row index in ``candidate_rows`` to
        its posterior probability given the evidence observed so far.
        The probabilities should sum to 1.  In the simplest case (no
        prior information), the posterior is uniform over the
        candidates.

    observed_cols : List[ColName]
        The list of column names that have been queried so far.  This
        is used to avoid re‑querying the same feature and to record
        the order in which information was acquired.

    observed_values : Dict[ColName, object]
        For each queried column, the value returned for the true row.
        After each query the caller should update this dictionary with
        the returned answer.  Only the true row's values should be
        stored here; values for other rows remain in the DataFrame.
    """

    candidate_rows: list[RowIndex]
    posterior: dict[RowIndex, float]
    observed_cols: list[ColName]
    observed_values: dict[ColName, object]

    def copy(self) -> Self:
        """Create a deep copy of the candidate state."""
        return CandidateState(
            candidate_rows=list(self.candidate_rows),
            posterior=dict(self.posterior),
            observed_cols=list(self.observed_cols),
            observed_values=dict(self.observed_values),
        )

    @property
    def entropy(self) -> float:
        """Calculate the entropy of the current posterior distribution."""
        import math

        if not self.candidate_rows:
            return 0.0

        entropy = 0.0
        for row in self.candidate_rows:
            p = self.posterior.get(row, 0.0)
            if p > 0.0:
                entropy -= p * math.log2(p)
        return entropy

    @property
    def is_uniquely_determined(self) -> bool:
        """Check if only one candidate remains with significant probability."""
        return (
            len([r for r in self.candidate_rows if self.posterior.get(r, 0.0) > 1e-10])
            <= 1
        )

    def update_with_observation(self, col: ColName, value: object) -> Self:
        """Create new state with an additional observation.

        This method does not filter candidates - it just records the observation.
        Candidate filtering should be done separately based on the actual data.
        """
        return CandidateState(
            candidate_rows=list(self.candidate_rows),
            posterior=dict(self.posterior),
            observed_cols=self.observed_cols + [col],
            observed_values=self.observed_values | {col: value},
        )
