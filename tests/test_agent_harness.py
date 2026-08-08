from __future__ import annotations

import asyncio
import copy
import pathlib
import sys
import time
from typing import Any, AsyncGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import (  # noqa: E402
    APIClient,
    AbortCommand,
    AgentCommand,
    AgentConfig,
    AgentEvent,
    AgentHarness,
    AgentSession,
    ApprovalResponse,
    ReadTool,
    RequireApprovalPolicy,
    StreamingToolExecutor,
    Tool,
    ToolCapabilities,
    ToolContext,
    ToolUseBlock,
    WriteTool,
    render_event,
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
            {
                "system_prompt": system_prompt,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        event_batch = self._event_batches[min(call_index, len(self._event_batches) - 1)]
        for event in event_batch:
            yield event


class DelayedTool(Tool):
    name = "delayed"
    description = "Return after a delay."
    input_schema = {"type": "object", "properties": {}}
    capabilities = ToolCapabilities(read_only=True, concurrency_safe=True)

    async def call(
        self,
        input: dict[str, Any],
        context: ToolContext,
    ) -> str:
        await asyncio.sleep(0.05)
        return input.get("value", "done")


class ExclusiveTool(DelayedTool):
    name = "exclusive"
    capabilities = ToolCapabilities(read_only=False, concurrency_safe=False)


def make_context(workspace_root: pathlib.Path | None = None) -> ToolContext:
    return ToolContext(
        abort_signal=asyncio.Event(),
        workspace_root=(workspace_root or ROOT).resolve(),
        session=AgentSession(),
    )


def collect(
    run: AsyncGenerator[AgentEvent, AgentCommand | None],
) -> list[AgentEvent]:
    async def _collect() -> list[AgentEvent]:
        return [event async for event in run]

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_collect())
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_agent_returns_structured_text_events_without_tool_calls() -> None:
    client = StaticClient(
        [
            {"type": "text_delta", "text": "hello"},
            {"type": "message_stop"},
        ]
    )
    harness = AgentHarness(client, AgentConfig(tools=[]))

    events = collect(harness.run("hi"))

    assert any(event.type == "model_delta" and event.data["text"] == "hello" for event in events)
    assert any(event.type == "run_complete" for event in events)
    assert any("Final response" in render_event(event) for event in events)


def test_untraced_event_serialization_keeps_legacy_shape() -> None:
    assert AgentEvent("custom", {"value": 1}).to_dict() == {
        "type": "custom",
        "data": {"value": 1},
    }


def test_trace_fields_reconstruct_one_rollout() -> None:
    client = StaticClient(
        [
            {"type": "text_delta", "text": "done"},
            {"type": "message_stop"},
        ]
    )
    harness = AgentHarness(
        client,
        AgentConfig(tools=[], agent_id="planner"),
    )

    events = collect(
        harness.run(
            "finish the task",
            trace_id="trace-1",
            group_id="group-1",
            rollout_id="rollout-1",
            reward={"team": 1.0},
            provenance={"prompt_id": "prompt-1"},
        )
    )
    records = [event.to_trace_dict() for event in events]

    assert records
    assert {
        (record["trace_id"], record["group_id"], record["rollout_id"])
        for record in records
    } == {("trace-1", "group-1", "rollout-1")}
    assert {record["agent"] for record in records} == {"planner"}
    assert any(
        record["type"] == "model_delta"
        and record["event_type"] == "message"
        for record in records
    )

    terminal = next(record for record in records if record["type"] == "run_complete")
    assert terminal["reward"] == {"team": 1.0}
    assert terminal["provenance"]["prompt_id"] == "prompt-1"
    assert terminal["provenance"]["runtime_event"] == "run_complete"
    assert terminal["provenance"]["reward_source"] == "external"


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
    harness = AgentHarness(
        client,
        AgentConfig(max_turns=3, tools=[ReadTool()], workspace_root=ROOT),
    )

    events = collect(
        harness.run(
            "read README",
            trace_id="trace-tool",
            group_id="group-tool",
            rollout_id="rollout-tool",
        )
    )

    assert any(event.type == "tool_call" and event.data["name"] == "read_file" for event in events)
    trace_records = [event.to_trace_dict() for event in events]
    assert any(
        record["type"] == "tool_call" and record["tool"] == "read_file"
        for record in trace_records
    )
    assert any(
        record["type"] == "tool_result" and record["tool"] == "read_file"
        for record in trace_records
    )
    assert len(client.calls) == 2
    second_messages = client.calls[1]["messages"]
    assert second_messages[-1]["content"][0]["type"] == "tool_result"
    assert "Minimal Agent Harness" in second_messages[-1]["content"][0]["content"]


def test_session_persists_across_runs() -> None:
    client = StaticClient(
        [
            [
                {"type": "text_delta", "text": "first answer"},
                {"type": "message_stop"},
            ],
            [
                {"type": "text_delta", "text": "second answer"},
                {"type": "message_stop"},
            ],
        ]
    )
    session = AgentSession()
    harness = AgentHarness(client, AgentConfig(tools=[]), session=session)

    collect(harness.run("first question"))
    collect(harness.run("follow up"))

    second_call_messages = client.calls[1]["messages"]
    assert second_call_messages == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow up"},
    ]


def test_read_tool_supports_line_ranges(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    async def scenario() -> str:
        return await ReadTool().call(
            {"file_path": "sample.txt", "start_line": 2, "end_line": 3},
            make_context(tmp_path),
        )

    result = asyncio.run(scenario())

    assert result == "two\nthree\n"


def test_workspace_guard_blocks_outside_reads(tmp_path: pathlib.Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    async def scenario() -> str:
        return await ReadTool().call(
            {"file_path": str(outside)},
            make_context(workspace),
        )

    result = asyncio.run(scenario())

    assert "Path outside workspace" in result


def test_safe_tools_run_concurrently() -> None:
    async def scenario() -> tuple[list[tuple[str, str, bool]], float]:
        executor = StreamingToolExecutor([DelayedTool()], make_context())
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
        executor = StreamingToolExecutor([ExclusiveTool()], make_context())
        start = time.perf_counter()
        executor.add_tool(ToolUseBlock("a", "exclusive", {"value": "a"}))
        executor.add_tool(ToolUseBlock("b", "exclusive", {"value": "b"}))
        results = await executor.get_remaining_results()
        elapsed = time.perf_counter() - start
        return results, elapsed

    results, elapsed = asyncio.run(scenario())

    assert [result[1] for result in results] == ["a", "b"]
    assert elapsed >= 0.09


def test_policy_requests_approval_and_denies_without_execution(tmp_path: pathlib.Path) -> None:
    async def scenario() -> list[AgentEvent]:
        client = StaticClient(
            [
                [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "write_file",
                        "input": {"file_path": "created.txt", "content": "content"},
                    },
                    {"type": "message_stop"},
                ],
                [
                    {"type": "text_delta", "text": "saw denial"},
                    {"type": "message_stop"},
                ],
            ]
        )
        harness = AgentHarness(
            client,
            AgentConfig(
                max_turns=3,
                tools=[WriteTool()],
                workspace_root=tmp_path,
                tool_policy=RequireApprovalPolicy(),
            ),
        )

        events: list[AgentEvent] = []
        stream = harness.run("write the file")
        while True:
            event = await stream.__anext__()
            events.append(event)
            if event.type == "approval_requested":
                response = ApprovalResponse(
                    request_id=event.data["request_id"],
                    approved=False,
                    reason="test denied",
                )
                events.append(await stream.asend(response))
                break
        async for event in stream:
            events.append(event)
        return events

    events = asyncio.run(scenario())

    assert any(event.type == "approval_requested" for event in events)
    assert any(
        event.type == "tool_result"
        and event.data["is_error"]
        and "test denied" in event.data["content"]
        for event in events
    )
    assert not (tmp_path / "created.txt").exists()


def test_bad_approval_command_reports_protocol_error(tmp_path: pathlib.Path) -> None:
    async def scenario() -> list[AgentEvent]:
        client = StaticClient(
            [
                [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "write_file",
                        "input": {"file_path": "created.txt", "content": "content"},
                    },
                    {"type": "message_stop"},
                ],
                [
                    {"type": "text_delta", "text": "saw protocol error"},
                    {"type": "message_stop"},
                ],
            ]
        )
        harness = AgentHarness(
            client,
            AgentConfig(
                max_turns=3,
                tools=[WriteTool()],
                workspace_root=tmp_path,
                tool_policy=RequireApprovalPolicy(),
            ),
        )

        events: list[AgentEvent] = []
        stream = harness.run("write the file")
        while True:
            event = await stream.__anext__()
            events.append(event)
            if event.type == "approval_requested":
                events.append(await stream.asend("bad command"))  # type: ignore[arg-type]
                break
        async for event in stream:
            events.append(event)
        return events

    events = asyncio.run(scenario())

    assert any(
        event.type == "tool_result"
        and event.data["is_error"]
        and "Expected ApprovalResponse" in event.data["content"]
        for event in events
    )


def test_hooks_observe_events() -> None:
    seen: list[str] = []

    def hook(event: AgentEvent, session: AgentSession) -> None:
        seen.append(event.type)

    client = StaticClient(
        [
            {"type": "text_delta", "text": "hello"},
            {"type": "message_stop"},
        ]
    )
    harness = AgentHarness(client, AgentConfig(tools=[], hooks=[hook]))

    collect(harness.run("hi"))

    assert "run_started" in seen
    assert "model_delta" in seen
    assert "run_complete" in seen


def test_hook_errors_are_reported_without_crashing_run() -> None:
    def hook(event: AgentEvent, session: AgentSession) -> None:
        raise RuntimeError("hook failed")

    client = StaticClient(
        [
            {"type": "text_delta", "text": "hello"},
            {"type": "message_stop"},
        ]
    )
    harness = AgentHarness(client, AgentConfig(tools=[], hooks=[hook]))

    events = collect(harness.run("hi"))

    assert any(event.type == "run_complete" for event in events)
    assert any(
        event.data.get("hook_errors", [{}])[0].get("error") == "hook failed"
        for event in events
        if event.data.get("hook_errors")
    )


def test_abort_command_only_aborts_active_run() -> None:
    async def scenario() -> tuple[AgentEvent, list[AgentEvent]]:
        client = StaticClient(
            [
                [
                    {"type": "text_delta", "text": "should not run"},
                    {"type": "message_stop"},
                ],
                [
                    {"type": "text_delta", "text": "after abort"},
                    {"type": "message_stop"},
                ],
            ]
        )
        harness = AgentHarness(client, AgentConfig(tools=[]))

        stream = harness.run("stop")
        first = await stream.__anext__()
        assert first.type == "run_started"
        aborted = await stream.asend(AbortCommand())

        later_events = []
        async for event in harness.run("new run"):
            later_events.append(event)
        return aborted, later_events

    aborted, later_events = asyncio.run(scenario())

    assert aborted.type == "run_aborted"
    assert any(
        event.type == "run_complete" and event.data["text"] == "should not run"
        for event in later_events
    )


def test_summarize_conversation_keeps_short_history() -> None:
    client = StaticClient([{"type": "text_delta", "text": "unused"}])
    messages = [{"role": "user", "content": "short"}]

    result = asyncio.run(summarize_conversation(messages, client))

    assert result.summary_text == ""
    assert result.pre_compact_message_count == 1
    assert result.post_compact_message_count == 1
    assert client.calls == []
