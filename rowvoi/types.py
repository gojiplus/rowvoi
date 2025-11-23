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

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

RowIndex = int
"""Alias for row indices within a pandas DataFrame."""

ColName = Hashable
"""Alias for column identifiers within a pandas DataFrame.

Pandas allows a wide variety of types for column labels (strings,
integers, tuples, etc.), so we use the ``Hashable`` bound to permit
any immutable label type.
"""


@dataclass
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
