"""Run the doctests embedded in the RAG modules.

The examples in those docstrings are the first thing a reader tries, so they
are executed rather than trusted.
"""

import doctest

import pytest

from rowvoi.rag import context, questions, retrieval

MODULES = [context, questions, retrieval]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples(module):
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, (
        f"{results.failed} doctest failure(s) in {module.__name__}"
    )
