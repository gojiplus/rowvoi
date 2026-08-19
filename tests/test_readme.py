"""Execute the README's code so its claims cannot quietly become false.

The docs pages are verified by sphinx's doctest builder and module docstrings
by :mod:`tests.test_docstrings`; the README had no such check, which is how it
came to advertise a `noise_rate` parameter that never existed.

Two levels of checking here:

* every fenced ``python`` block must at least parse;
* blocks that are self-contained are executed, and the ones asserting literal
  output have that output checked exactly.
"""

import ast
import io
import pathlib
import re
import textwrap
from contextlib import redirect_stdout

import pytest

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"

# Blocks that deliberately reference names the reader is expected to supply.
# Executing them is not the point; they still have to parse.
PLACEHOLDERS = {
    "rows",
    "costs",
    "state",
    "df",
    "my_backend",
    "retrieval_scores",
    "outcomes",
    "query",
    "chunks",
    "answers",
    "probes",
    "survey",
    "np",
}


def python_blocks() -> list[str]:
    """Return every fenced python block in the README, in document order."""
    return re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)


def test_readme_has_blocks():
    # Guards against the extraction regex silently matching nothing, which
    # would make every other test in this file vacuous.
    assert len(python_blocks()) >= 10


@pytest.mark.parametrize("index", range(len(python_blocks())))
def test_every_block_parses(index):
    ast.parse(python_blocks()[index])


def test_self_contained_blocks_execute():
    """Run the blocks that stand alone; placeholders are recorded, not ignored.

    A block that fails on a placeholder name is skipped. Any other exception
    is a real breakage in documented usage -- that is the case this catches.
    """
    executed, skipped = 0, 0
    for block in python_blocks():
        namespace: dict = {}
        try:
            with redirect_stdout(io.StringIO()):
                exec(block, namespace)  # noqa: S102 - executing our own README
        except ImportError as exc:
            # An optional extra that is not installed. The block is correct;
            # this environment just cannot run it.
            if "rowvoi[" in str(exc):
                skipped += 1
                continue
            raise AssertionError(
                f"README block failed to import:\n{textwrap.indent(block, '    ')}"
            ) from exc
        except NameError as exc:
            missing = str(exc).split("'")[1] if "'" in str(exc) else ""
            if missing in PLACEHOLDERS:
                skipped += 1
                continue
            raise AssertionError(
                f"README block references undefined {missing!r}, which is not a "
                f"documented placeholder:\n{textwrap.indent(block, '    ')}"
            ) from exc
        except Exception as exc:  # pragma: no cover - failure path
            raise AssertionError(
                f"README block raised {type(exc).__name__}: {exc}\n"
                f"{textwrap.indent(block, '    ')}"
            ) from exc
        executed += 1

    # Without a floor this test would pass by skipping everything.
    assert executed >= 4, f"only {executed} blocks ran ({skipped} skipped)"


class TestRagClaims:
    """The RAG section prints specific values; check them exactly."""

    def test_minimal_sufficient_context(self):
        from rowvoi.rag import Chunk, plan_context_path, select_context

        chunks = [
            Chunk("doc1#p2", "The Pro plan costs $40/seat/mo.", tokens=180),
            Chunk("doc2#p1", "v4.2 shipped March 3rd, 2026.", tokens=150),
            Chunk("faq#p1", "Pricing, releases, and SSO.", tokens=1550),
        ]
        claims = ["pro_plan_price", "v42_release_date"]
        support = {
            "pro_plan_price": {"doc1#p2", "faq#p1"},
            "v42_release_date": {"doc2#p1", "faq#p1"},
        }

        selection = select_context(chunks, claims, support)
        assert selection.chunks == ["doc2#p1", "doc1#p2"]
        assert selection.coverage == 1.0
        assert selection.missing_claims == set()

        path = plan_context_path(chunks, claims, support)
        assert path.prefix_for_budget(200) == ["doc2#p1"]
        assert path.coverage_curve() == [(150.0, 0.5), (330.0, 1.0)]

    def test_clarifying_questions(self):
        from rowvoi import CandidateState
        from rowvoi.rag import next_question, observe_answer, question_values

        answers = {
            "Which deployment?": ["cloud", "cloud", "self", "self"],
            "Which version?": ["v4", "v3", "v4", "v3"],
            "Are you an admin?": ["yes", "yes", "yes", "yes"],
        }

        values = question_values(answers)
        assert values["Which deployment?"] == pytest.approx(1.0)
        assert values["Which version?"] == pytest.approx(1.0)
        # The README's central claim: a question every candidate answers
        # identically is worth exactly nothing.
        assert values["Are you an admin?"] == 0.0

        chosen = next_question(answers, costs={"Which version?": 5.0})
        assert chosen.col == "Which deployment?"

        state = CandidateState.uniform(range(4))
        state = observe_answer(state, answers, "Which deployment?", "self", noise=0.02)
        assert state.posterior.round(2).tolist() == [0.01, 0.01, 0.49, 0.49]
