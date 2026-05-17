from __future__ import annotations

import asyncio
import pathlib
import sys
import time
from typing import Any, AsyncGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import (  # noqa: E402
    APIClient,
    AgentConfig,
    AgentHarness,
    ReadTool,
    StreamingToolExecutor,
    Tool,
    ToolUseBlock,
    summarize_conversation,
)


class StaticClient(APIClient):
    def __init__(self, events: list[dict[str, Any]] | list[list[dict[str, Any]]]):
        self.model = "static"
        self.calls: list[dict[str, Any]] = []
        if events and isinstance(events[0], dict):
            self._event_batches = [events]
        else:
            self._event_batches = events

    async def stream_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        call_index = len(self.calls)
        self.calls.append(
            {"system_prompt": system_prompt, "messages": messages, "tools": tools}
        )
        event_batch = self._event_batches[min(call_index, len(self._event_batches) - 1)]
        for event in event_batch:
            yield event


class DelayedTool(Tool):
    name = "delayed"
    description = "Return after a delay."
    input_schema = {"type": "object", "properties": {}}
    is_concurrency_safe = True

    async def call(
        self,
        input: dict[str, Any],
        abort_signal: asyncio.Event | None,
    ) -> str:
        await asyncio.sleep(0.05)
        return input.get("value", "done")


class ExclusiveTool(DelayedTool):
    name = "exclusive"
    is_concurrency_safe = False


def collect(run: AsyncGenerator[str, None]) -> list[str]:
    async def _collect() -> list[str]:
        return [event async for event in run]

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_collect())
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_agent_returns_text_without_tool_calls() -> None:
    client = StaticClient(
        [
            {"type": "text_delta", "text": "hello"},
            {"type": "message_stop"},
        ]
    )
    harness = AgentHarness(client, AgentConfig(tools=[]))

    events = collect(harness.run("hi"))

    assert any("Final response" in event for event in events)
    assert any("hello" in event for event in events)


def test_agent_executes_tool_and_feeds_result_back() -> None:
    client = StaticClient(
        [
            [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "read_file",
                    "input": {"file_path": "README.md", "start_line": 5, "end_line": 5},
                },
                {"type": "message_stop"},
            ],
            [
                {"type": "text_delta", "text": "done"},
                {"type": "message_stop"},
            ],
        ]
    )
    harness = AgentHarness(client, AgentConfig(max_turns=3, tools=[ReadTool()]))

    events = collect(harness.run("read README"))

    assert any("[Tool call] read_file" in event for event in events)
    assert len(client.calls) == 2
    second_messages = client.calls[1]["messages"]
    assert second_messages[-1]["content"][0]["type"] == "tool_result"
    assert "Minimal Agent Harness" in second_messages[-1]["content"][0]["content"]


def test_read_tool_supports_line_ranges(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = asyncio.run(
        ReadTool().call(
            {"file_path": str(target), "start_line": 2, "end_line": 3},
            None,
        )
    )

    assert result == "two\nthree\n"


def test_safe_tools_run_concurrently() -> None:
    async def scenario() -> tuple[list[tuple[str, str, bool]], float]:
        executor = StreamingToolExecutor([DelayedTool()], asyncio.Event())
        start = time.perf_counter()
        executor.add_tool(ToolUseBlock("a", "delayed", {"value": "a"}))
        executor.add_tool(ToolUseBlock("b", "delayed", {"value": "b"}))
        results = await executor.get_remaining_results()
        elapsed = time.perf_counter() - start
        return results, elapsed

    results, elapsed = asyncio.run(scenario())

    assert [result[1] for result in results] == ["a", "b"]
    assert elapsed < 0.09


def test_exclusive_tools_run_serially() -> None:
    async def scenario() -> tuple[list[tuple[str, str, bool]], float]:
        executor = StreamingToolExecutor([ExclusiveTool()], asyncio.Event())
        start = time.perf_counter()
        executor.add_tool(ToolUseBlock("a", "exclusive", {"value": "a"}))
        executor.add_tool(ToolUseBlock("b", "exclusive", {"value": "b"}))
        results = await executor.get_remaining_results()
        elapsed = time.perf_counter() - start
        return results, elapsed

    results, elapsed = asyncio.run(scenario())

    assert [result[1] for result in results] == ["a", "b"]
    assert elapsed >= 0.09


def test_summarize_conversation_keeps_short_history() -> None:
    client = StaticClient([{"type": "text_delta", "text": "unused"}])
    messages = [{"role": "user", "content": "short"}]

    result = asyncio.run(summarize_conversation(messages, client))

    assert result.summary_text == ""
    assert result.pre_compact_message_count == 1
    assert result.post_compact_message_count == 1
    assert client.calls == []
