"""Tests for the Anthropic-backed protocol implementations.

Runs against a stub client, so no API key and no network are needed. What is
verified here is the request we build and how we parse the reply -- not the
model's judgement.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("anthropic", reason="needs the 'claude' extra")

from rowvoi.rag.claude import (
    DEFAULT_MODEL,
    ClaudeAnswerPredictor,
    ClaudeClaimExtractor,
    ClaudeQuestionGenerator,
    ClaudeSupportJudge,
)
from rowvoi.rag.protocols import (
    AnswerPredictor,
    ClaimExtractor,
    QuestionGenerator,
    SupportJudge,
)


@dataclass
class StubBlock:
    """One content block in a stubbed response."""

    text: str
    type: str = "text"


@dataclass
class StubResponse:
    """A stubbed Messages API response."""

    content: list[StubBlock]
    stop_reason: str = "end_turn"
    stop_details: Any = None


@dataclass
class StubClient:
    """Records requests and replays a canned JSON payload."""

    payload: Any = None
    stop_reason: str = "end_turn"
    stop_details: Any = None
    blocks: list[StubBlock] | None = None
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.messages = self

    def create(self, **kwargs):
        """Stand in for client.messages.create."""
        self.calls.append(kwargs)
        blocks = (
            self.blocks
            if self.blocks is not None
            else [StubBlock(json.dumps(self.payload))]
        )
        return StubResponse(
            content=blocks,
            stop_reason=self.stop_reason,
            stop_details=self.stop_details,
        )

    @property
    def last(self) -> dict:
        return self.calls[-1]


CHUNKS = [("a", "The price is $40."), ("b", "It shipped in March.")]
CLAIMS = ["price", "release_date"]


class TestProtocolConformance:
    """Each class must actually satisfy the protocol it implements."""

    def test_claim_extractor(self):
        assert isinstance(ClaudeClaimExtractor(client=StubClient()), ClaimExtractor)

    def test_support_judge(self):
        assert isinstance(ClaudeSupportJudge(client=StubClient()), SupportJudge)

    def test_question_generator(self):
        assert isinstance(
            ClaudeQuestionGenerator(client=StubClient()), QuestionGenerator
        )

    def test_answer_predictor(self):
        assert isinstance(ClaudeAnswerPredictor(client=StubClient()), AnswerPredictor)


class TestRequestShape:
    """The request we send to the API."""

    def test_defaults_to_opus_5(self):
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client).extract("q")
        assert client.last["model"] == DEFAULT_MODEL == "claude-opus-5"

    def test_asks_for_structured_output(self):
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client).extract("q")
        fmt = client.last["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["additionalProperties"] is False

    def test_thinking_is_left_at_the_model_default(self):
        # Opus 5 thinks by default; passing budget_tokens would be a 400
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client).extract("q")
        assert "thinking" not in client.last

    def test_no_sampling_parameters(self):
        # temperature/top_p/top_k are rejected on Opus 5
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client).extract("q")
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in client.last

    def test_effort_is_forwarded_when_set(self):
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client, effort="low").extract("q")
        assert client.last["output_config"]["effort"] == "low"

    def test_effort_omitted_by_default(self):
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client).extract("q")
        assert "effort" not in client.last["output_config"]

    def test_max_tokens_leaves_room_for_thinking(self):
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client).extract("q")
        assert client.last["max_tokens"] >= 16000

    def test_extra_kwargs_are_forwarded(self):
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(
            client=client, extra={"metadata": {"user_id": "u1"}}
        ).extract("q")
        assert client.last["metadata"] == {"user_id": "u1"}


class TestPromptCaching:
    """Bulk text must sit behind a cache breakpoint, varying text after it."""

    def test_chunks_are_cached(self):
        client = StubClient(payload={"support": []})
        ClaudeSupportJudge(client=client).judge(CHUNKS, CLAIMS)

        system = client.last["system"][0]
        assert system["cache_control"] == {"type": "ephemeral"}
        # Chunk text is the large stable half and belongs in the cached prefix
        assert "The price is $40." in system["text"]

    def test_claims_stay_outside_the_cached_prefix(self):
        client = StubClient(payload={"support": []})
        ClaudeSupportJudge(client=client).judge(CHUNKS, CLAIMS)

        assert "release_date" not in client.last["system"][0]["text"]
        assert "release_date" in client.last["messages"][0]["content"]

    def test_candidates_are_cached_for_answer_prediction(self):
        client = StubClient(payload={"rows": [["x"], ["y"]]})
        ClaudeAnswerPredictor(client=client).predict(["cand one", "cand two"], ["q?"])

        system = client.last["system"][0]
        assert system["cache_control"] == {"type": "ephemeral"}
        assert "cand one" in system["text"]
        # The question varies per call, so it must come after the breakpoint
        assert "q?" not in system["text"]
        assert "q?" in client.last["messages"][0]["content"]

    def test_short_lived_prompts_are_not_cached(self):
        # Claim extraction has no bulk stable prefix; caching it would only
        # pay the write premium
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client).extract("q")
        assert "cache_control" not in client.last["system"][0]


class TestClaimExtractor:
    """Query decomposition."""

    def test_returns_claims(self):
        client = StubClient(payload={"claims": ["price", "date"]})
        assert ClaudeClaimExtractor(client=client).extract("q") == ["price", "date"]

    def test_max_claims_appears_in_the_prompt(self):
        client = StubClient(payload={"claims": []})
        ClaudeClaimExtractor(client=client, max_claims=5).extract("q")
        assert "at most 5" in client.last["system"][0]["text"]


class TestSupportJudge:
    """Claim/chunk support, mapped back from indices."""

    def test_indices_map_back_to_chunk_ids(self):
        client = StubClient(
            payload={
                "support": [
                    {"claim_index": 0, "chunk_indices": [0]},
                    {"claim_index": 1, "chunk_indices": [1]},
                ]
            }
        )
        support = ClaudeSupportJudge(client=client).judge(CHUNKS, CLAIMS)
        assert support == {"price": {"a"}, "release_date": {"b"}}

    def test_unsupported_claims_map_to_empty_sets(self):
        # A claim the model omits must still appear, so the caller sees the gap
        client = StubClient(
            payload={"support": [{"claim_index": 0, "chunk_indices": [0]}]}
        )
        support = ClaudeSupportJudge(client=client).judge(CHUNKS, CLAIMS)
        assert support["release_date"] == set()
        assert set(support) == set(CLAIMS)

    def test_out_of_range_indices_are_dropped(self):
        client = StubClient(
            payload={
                "support": [
                    {"claim_index": 0, "chunk_indices": [0, 99]},
                    {"claim_index": 42, "chunk_indices": [1]},
                ]
            }
        )
        support = ClaudeSupportJudge(client=client).judge(CHUNKS, CLAIMS)
        assert support["price"] == {"a"}
        assert support["release_date"] == set()


class TestQuestionGenerator:
    """Clarifying-question proposals."""

    def test_returns_questions(self):
        client = StubClient(payload={"questions": ["which version?", "what OS?"]})
        questions = ClaudeQuestionGenerator(client=client).generate(["a", "b"], 5)
        assert questions == ["which version?", "what OS?"]

    def test_truncates_to_n(self):
        client = StubClient(payload={"questions": ["q1", "q2", "q3"]})
        assert len(ClaudeQuestionGenerator(client=client).generate(["a"], 2)) == 2


class TestAnswerPredictor:
    """The predicted answer matrix."""

    def test_returns_the_matrix(self):
        client = StubClient(payload={"rows": [["v1", "mac"], ["v2", "linux"]]})
        rows = ClaudeAnswerPredictor(client=client).predict(["a", "b"], ["q1", "q2"])
        assert rows == [["v1", "mac"], ["v2", "linux"]]

    def test_wrong_row_count_raises(self):
        client = StubClient(payload={"rows": [["v1"]]})
        with pytest.raises(ValueError, match="Expected a 2x1 answer matrix"):
            ClaudeAnswerPredictor(client=client).predict(["a", "b"], ["q1"])

    def test_ragged_matrix_raises(self):
        client = StubClient(payload={"rows": [["v1", "mac"], ["v2"]]})
        with pytest.raises(ValueError, match="Expected a 2x2 answer matrix"):
            ClaudeAnswerPredictor(client=client).predict(["a", "b"], ["q1", "q2"])

    def test_output_feeds_next_question_directly(self):
        from rowvoi.rag import next_question

        client = StubClient(
            payload={"rows": [["v1", "same"], ["v1", "same"], ["v2", "same"]]}
        )
        rows = ClaudeAnswerPredictor(client=client).predict(
            ["a", "b", "c"], ["version?", "useless?"]
        )
        suggestion = next_question(rows, questions=["version?", "useless?"])
        assert suggestion.col == "version?"


class TestFailureModes:
    """Refusals and truncation must not surface as a parse error."""

    def test_refusal_raises_a_clear_error(self):
        @dataclass
        class Details:
            category: str = "cyber"

        client = StubClient(blocks=[], stop_reason="refusal", stop_details=Details())
        with pytest.raises(RuntimeError, match=r"declined.*cyber"):
            ClaudeClaimExtractor(client=client).extract("q")

    def test_missing_text_block_mentions_max_tokens(self):
        client = StubClient(blocks=[], stop_reason="max_tokens")
        with pytest.raises(RuntimeError, match="max_tokens"):
            ClaudeClaimExtractor(client=client).extract("q")

    def test_non_text_blocks_are_skipped(self):
        client = StubClient(
            blocks=[
                StubBlock(text="", type="thinking"),
                StubBlock(text=json.dumps({"claims": ["ok"]})),
            ]
        )
        assert ClaudeClaimExtractor(client=client).extract("q") == ["ok"]
