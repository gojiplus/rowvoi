"""Type aliases for rowvoi.

This module defines type aliases used throughout the package for clarity
and consistency. The actual data structures are in core.py.
"""

from collections.abc import Hashable

# Type aliases for row and column references
RowIndex = Hashable
"""Alias for row indices within a pandas DataFrame.

Can be int for positional indexing or any hashable for label-based indexing.
"""

ColName = Hashable
"""Alias for column identifiers within a pandas DataFrame.

Pandas allows various types for column labels (strings, integers, tuples, etc.),
so we use Hashable to permit any immutable label type.
"""
