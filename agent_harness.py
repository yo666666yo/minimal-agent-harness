"""
Minimal Agent Harness - a single-agent tool-use loop with context summarization.

Educational implementation based on the Claude Code architecture:
  src/query.ts                             - core query loop
  src/QueryEngine.ts                       - per-conversation state wrapper
  src/services/tools/StreamingToolExecutor.ts - concurrent tool execution
  src/services/tools/toolOrchestration.ts  - tool partitioning
  src/services/compact/compact.ts          - conversation summarization
  src/Tool.ts                              - tool type definitions

Key design patterns preserved:
  1. Async-generator query loop
  2. Structured events, rendered separately from runtime state
  3. Bidirectional control commands for approval and abort
  4. Streaming tool execution
  5. Concurrent-safe vs exclusive tool partitioning
  6. Conversation summarization via a separate model call
  7. In-memory session state across user turns

Usage:
  # With a real API key:
  export ANTHROPIC_API_KEY=sk-ant-...
  python agent_harness.py

  # Dry-run / mock mode (no API key needed):
  python agent_harness.py --mock

Dependencies: pip install anthropic
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import pathlib
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Sequence, Union


# ---------------------------------------------------------------------------
# Runtime events, commands, and session state
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """A structured event emitted by the agent runtime."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for hooks or transcripts."""
        return {"type": self.type, "data": self.data}


@dataclass
class ApprovalResponse:
    """Command sent back into run() after an approval_requested event."""

    request_id: str
    approved: bool
    reason: str = ""


@dataclass
class AbortCommand:
    """Command sent back into run() to abort the active turn."""

    reason: str = "Aborted by caller"


AgentCommand = Union[ApprovalResponse, AbortCommand]


@dataclass
class AgentSession:
    """In-memory conversation state shared across run() calls."""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.messages.clear()


Hook = Callable[[AgentEvent, AgentSession], Union[Awaitable[None], None]]


@dataclass
class ToolContext:
    """Runtime context passed to tools."""

    abort_signal: asyncio.Event
    workspace_root: pathlib.Path
    session: AgentSession


def render_event(event: AgentEvent) -> str:
    """Render a structured runtime event as the old human-readable text stream."""
    data = event.data
    if event.type == "run_started":
        return (
            f"[Agent] Starting with {data['tool_count']} tools, "
            f"max {data['max_turns']} turns"
        )
    if event.type == "turn_started":
        return f"\n[Turn {data['turn']}/{data['max_turns']}]"
    if event.type == "context_compacting":
        return f"[Agent] Context at ~{data['estimated_tokens']} tokens - summarizing..."
    if event.type == "context_compacted":
        return (
            f"[Agent] Compressed {data['pre_compact_message_count']} -> "
            f"{data['post_compact_message_count']} messages"
        )
    if event.type == "compaction_failed":
        return f"[Agent] Summarization failed: {data['error']}"
    if event.type == "model_start":
        return "[Agent] Calling model..."
    if event.type == "model_delta":
        return f"[Model] {data['text']}"
    if event.type == "tool_call":
        return (
            f"[Tool call] {data['name']}("
            f"{json.dumps(data['input'], default=str)[:200]})"
        )
    if event.type == "approval_requested":
        return (
            f"[Approval requested] {data['tool_name']}("
            f"{json.dumps(data['input'], default=str)[:200]}): {data['reason']}"
        )
    if event.type == "tool_result":
        tag = "ERROR" if data.get("is_error") else "OK"
        return f"[Tool result/{tag}] {str(data.get('content', ''))[:200]}"
    if event.type == "run_complete":
        return f"\n[Agent] Complete. Final response:\n{data['text']}"
    if event.type == "turn_limit_reached":
        return f"\n[Agent] Reached max turns ({data['max_turns']})"
    if event.type == "run_aborted":
        return f"[Agent] Aborted: {data['reason']}"
    return f"[{event.type}] {json.dumps(data, default=str)}"


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _handle_command(command: AgentCommand | None, abort_event: asyncio.Event) -> None:
    if isinstance(command, AbortCommand):
        abort_event.set()


# ---------------------------------------------------------------------------
# Tool System (cf. src/Tool.ts, src/services/tools/toolExecution.ts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCapabilities:
    """Small capability record used by scheduling and approval policy."""

    read_only: bool = True
    concurrency_safe: bool = True
    requires_approval: bool = False


class Tool(ABC):
    """Base class for all tools (cf. Tool.ts type definitions)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    capabilities: ToolCapabilities

    @abstractmethod
    async def call(self, input: dict[str, Any], context: ToolContext) -> str:
        """Execute the tool. Returns result as a string."""
        ...


def get_tool_capabilities(tool: Tool | None) -> ToolCapabilities:
    """Return a tool's capabilities, with a small fallback for old examples."""
    if tool is None:
        return ToolCapabilities()
    capabilities = getattr(tool, "capabilities", None)
    if isinstance(capabilities, ToolCapabilities):
        return capabilities
    legacy_safe = getattr(tool, "is_concurrency_safe", True)
    return ToolCapabilities(concurrency_safe=bool(legacy_safe))


def _resolve_workspace_path(path: str, workspace_root: pathlib.Path) -> pathlib.Path:
    """Resolve a path and reject anything outside the configured workspace."""
    root = workspace_root.resolve()
    candidate = pathlib.Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path outside workspace: {path}") from exc
    return candidate


# ---------------------------------------------------------------------------
# Concrete tool implementations
# ---------------------------------------------------------------------------


class BashTool(Tool):
    """Shell command execution (cf. src/tools/BashTool/)."""

    name = "bash"
    description = "Execute a bash command and return its output."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
        },
        "required": ["command"],
    }
    capabilities = ToolCapabilities(
        read_only=False,
        concurrency_safe=False,
        requires_approval=True,
    )

    async def call(self, input: dict[str, Any], context: ToolContext) -> str:
        proc = await asyncio.create_subprocess_shell(
            input["command"],
            cwd=str(context.workspace_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        abort_task = asyncio.create_task(context.abort_signal.wait())

        try:
            done, _pending = await asyncio.wait(
                {communicate_task, abort_task},
                timeout=30,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if abort_task in done:
                proc.kill()
                await proc.wait()
                communicate_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await communicate_task
                return "<tool_use_error>Command aborted</tool_use_error>"

            if communicate_task not in done:
                proc.kill()
                await proc.wait()
                communicate_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await communicate_task
                return "<tool_use_error>Command timed out after 30s</tool_use_error>"

            stdout, stderr = communicate_task.result()
        finally:
            abort_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await abort_task

        result = stdout.decode("utf-8", errors="replace")
        if stderr:
            result += "\n[stderr]\n" + stderr.decode("utf-8", errors="replace")
        return result or "(no output)"


class ReadTool(Tool):
    """File reading (cf. src/tools/FileReadTool/)."""

    name = "read_file"
    description = "Read the contents of a file."
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to read"},
            "start_line": {
                "type": "integer",
                "description": "Optional 1-based first line to read",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional 1-based last line to read, inclusive",
            },
        },
        "required": ["file_path"],
    }
    capabilities = ToolCapabilities(read_only=True, concurrency_safe=True)

    async def call(self, input: dict[str, Any], context: ToolContext) -> str:
        try:
            path = _resolve_workspace_path(input["file_path"], context.workspace_root)
            start_line = input.get("start_line")
            end_line = input.get("end_line")
            if start_line is not None or end_line is not None:
                start = int(start_line or 1)
                end = int(end_line) if end_line is not None else None
                if start < 1:
                    return "<tool_use_error>start_line must be >= 1</tool_use_error>"
                if end is not None and end < start:
                    return "<tool_use_error>end_line must be >= start_line</tool_use_error>"

                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                content = "".join(lines[start - 1:end])
                if not content:
                    return "(empty range)"
                if len(content) > 8000:
                    content = content[:8000] + "\n... [truncated at 8000 chars]"
                return content

            with open(path, "r", encoding="utf-8") as f:
                content = f.read(8000)
            if len(content) == 8000:
                content += "\n... [truncated at 8000 chars]"
            return content
        except FileNotFoundError:
            return f"<tool_use_error>File not found: {input['file_path']}</tool_use_error>"
        except Exception as e:
            return f"<tool_use_error>{e}</tool_use_error>"


class WriteTool(Tool):
    """File writing (cf. src/tools/FileWriteTool/)."""

    name = "write_file"
    description = "Write content to a file (overwrites if exists)."
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["file_path", "content"],
    }
    capabilities = ToolCapabilities(
        read_only=False,
        concurrency_safe=False,
        requires_approval=True,
    )

    async def call(self, input: dict[str, Any], context: ToolContext) -> str:
        try:
            path = _resolve_workspace_path(input["file_path"], context.workspace_root)
            os.makedirs(path.parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(input["content"])
            return f"Successfully wrote {len(input['content'])} chars to {path}"
        except Exception as e:
            return f"<tool_use_error>{e}</tool_use_error>"


class GrepTool(Tool):
    """Content search (cf. src/tools/GrepTool/)."""

    name = "grep"
    description = "Search for a pattern in Python files."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Directory to search in (default: '.')"},
        },
        "required": ["pattern"],
    }
    capabilities = ToolCapabilities(read_only=True, concurrency_safe=True)

    async def call(self, input: dict[str, Any], context: ToolContext) -> str:
        import re

        try:
            pattern = re.compile(input["pattern"])
            search_dir = _resolve_workspace_path(
                input.get("path", "."),
                context.workspace_root,
            )
        except Exception as e:
            return f"<tool_use_error>{e}</tool_use_error>"

        results = []
        for filepath in search_dir.rglob("*.py"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if pattern.search(line):
                            results.append(f"{filepath}:{i}: {line.rstrip()}")
                            if len(results) >= 50:
                                break
                if len(results) >= 50:
                    break
            except Exception:
                pass
        return "\n".join(results) if results else "(no matches)"


DEFAULT_TOOLS: list[Tool] = [BashTool(), ReadTool(), WriteTool(), GrepTool()]


# ---------------------------------------------------------------------------
# Message and policy types
# ---------------------------------------------------------------------------


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class ToolRequest:
    block: ToolUseBlock
    tool: Tool | None
    capabilities: ToolCapabilities


@dataclass
class ToolDecision:
    action: str
    reason: str = ""

    @classmethod
    def allow(cls) -> ToolDecision:
        return cls("allow")

    @classmethod
    def deny(cls, reason: str) -> ToolDecision:
        return cls("deny", reason)

    @classmethod
    def request_approval(cls, reason: str) -> ToolDecision:
        return cls("request_approval", reason)


class ToolPolicy(ABC):
    """Policy hook between model-requested tool use and actual execution."""

    @abstractmethod
    async def check(
        self,
        request: ToolRequest,
        context: ToolContext,
    ) -> ToolDecision:
        ...


class AutoAllowPolicy(ToolPolicy):
    """Default teaching policy: preserve the old automatic execution behavior."""

    async def check(
        self,
        request: ToolRequest,
        context: ToolContext,
    ) -> ToolDecision:
        return ToolDecision.allow()


class RequireApprovalPolicy(ToolPolicy):
    """Ask the caller to approve tools whose capabilities require approval."""

    async def check(
        self,
        request: ToolRequest,
        context: ToolContext,
    ) -> ToolDecision:
        if request.tool is None:
            return ToolDecision.deny(f"Unknown tool: {request.block.name}")
        if request.capabilities.requires_approval:
            return ToolDecision.request_approval(
                f"{request.block.name} is marked as requiring approval"
            )
        return ToolDecision.allow()


# ---------------------------------------------------------------------------
# API Client abstraction
# ---------------------------------------------------------------------------


class APIClient(ABC):
    """Abstract API client so we can swap real Anthropic with a mock."""

    @abstractmethod
    async def stream_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Yields normalized streaming events:
          {"type": "text_delta", "text": "..."}
          {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
          {"type": "message_stop"}
        """
        ...


class AnthropicClient(APIClient):
    """Real Anthropic API client (cf. src/services/api/claude.ts)."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.model = model

    async def stream_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            })

        system_parts = []
        if system_prompt:
            system_parts.append({"type": "text", "text": system_prompt})

        async with client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system_parts if system_parts else None,
            tools=anthropic_tools,
            messages=messages,
        ) as stream:
            current_tool_use: dict[str, Any] = {}
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield {"type": "text_delta", "text": event.delta.text}
                    elif event.delta.type == "input_json_delta":
                        current_tool_use.setdefault("input_json", "")
                        current_tool_use["input_json"] += event.delta.partial_json
                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_use = {
                            "id": block.id,
                            "name": block.name,
                            "input_json": "",
                        }
                elif event.type == "content_block_stop":
                    if "id" in current_tool_use and current_tool_use.get("input_json"):
                        try:
                            parsed_input = json.loads(current_tool_use["input_json"])
                        except json.JSONDecodeError:
                            parsed_input = {}
                        yield {
                            "type": "tool_use",
                            "id": current_tool_use["id"],
                            "name": current_tool_use["name"],
                            "input": parsed_input,
                        }
                    current_tool_use = {}
                elif event.type == "message_stop":
                    yield {"type": "message_stop"}
                    break


class MockAPIClient(APIClient):
    """
    Mock API client for dry-run testing.
    Simulates basic LLM responses - the model will use tools if relevant.
    """

    def __init__(self, model: str = "mock"):
        self.model = model
        self._uuid = uuid

    async def stream_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        user_content = ""
        for m in reversed(messages):
            if m["role"] == "user":
                content = m["content"]
                if isinstance(content, str):
                    user_content = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_content += block.get("text", "")
                break

        tool_names = [t["name"] for t in tools]
        lowered = user_content.lower()

        if any(word in lowered for word in ("read", "show", "look")):
            if "read_file" in tool_names:
                import re

                path_match = re.search(r'["\']?([\w./\\-]+\.\w+)["\']?', user_content)
                file_path = path_match.group(1) if path_match else "README.md"
                yield {
                    "type": "tool_use",
                    "id": f"mock_tool_{self._uuid.uuid4().hex[:8]}",
                    "name": "read_file",
                    "input": {"file_path": file_path},
                }
                yield {"type": "message_stop"}
                return

        if any(word in lowered for word in ("search", "find", "grep")):
            if "grep" in tool_names:
                import re

                pattern_match = re.search(r'["\'](.+?)["\']', user_content)
                pattern = pattern_match.group(1) if pattern_match else "TODO"
                yield {
                    "type": "tool_use",
                    "id": f"mock_tool_{self._uuid.uuid4().hex[:8]}",
                    "name": "grep",
                    "input": {"pattern": pattern, "path": "."},
                }
                yield {"type": "message_stop"}
                return

        if "write" in lowered or "create" in lowered:
            if "write_file" in tool_names:
                yield {
                    "type": "tool_use",
                    "id": f"mock_tool_{self._uuid.uuid4().hex[:8]}",
                    "name": "write_file",
                    "input": {"file_path": "mock_output.txt", "content": "mock content"},
                }
                yield {"type": "message_stop"}
                return

        if "run" in lowered or "bash" in lowered or "execute" in lowered:
            if "bash" in tool_names:
                yield {
                    "type": "tool_use",
                    "id": f"mock_tool_{self._uuid.uuid4().hex[:8]}",
                    "name": "bash",
                    "input": {"command": "echo 'mock command executed'"},
                }
                yield {"type": "message_stop"}
                return

        yield {
            "type": "text_delta",
            "text": (
                f"[Mock] I received your message: '{user_content[:200]}'. "
                "I can help with that."
            ),
        }
        yield {"type": "message_stop"}


# ---------------------------------------------------------------------------
# Streaming Tool Executor (cf. src/services/tools/StreamingToolExecutor.ts)
# ---------------------------------------------------------------------------


class StreamingToolExecutor:
    """
    Executes tools concurrently during model streaming.

    Concurrent-safe tools can run in parallel; exclusive tools wait for all
    currently executing tools to complete.
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        context: ToolContext,
    ):
        self._tool_map: dict[str, Tool] = {t.name: t for t in tools}
        self._context = context
        self._tracked: list[dict[str, Any]] = []
        self._has_errored = False

    def add_tool(self, block: ToolUseBlock) -> None:
        """Add a tool to the execution queue (cf. addTool())."""
        tool_def = self._tool_map.get(block.name)
        capabilities = get_tool_capabilities(tool_def)
        self._tracked.append({
            "id": block.id,
            "name": block.name,
            "input": block.input,
            "status": "queued",
            "is_concurrency_safe": capabilities.concurrency_safe,
            "result": None,
        })
        asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """Process the queue respecting concurrency rules."""
        for t in self._tracked:
            if t["status"] != "queued":
                continue
            if self._can_execute(t["is_concurrency_safe"]):
                await self._execute_tool(t)
            elif not t["is_concurrency_safe"]:
                break

    def _can_execute(self, is_safe: bool) -> bool:
        executing = [t for t in self._tracked if t["status"] == "executing"]
        if not executing:
            return True
        return is_safe and all(t["is_concurrency_safe"] for t in executing)

    async def _execute_tool(self, tracked: dict[str, Any]) -> None:
        tracked["status"] = "executing"
        tool_def = self._tool_map.get(tracked["name"])

        if tool_def is None:
            tracked["result"] = (
                f"<tool_use_error>Unknown tool: {tracked['name']}</tool_use_error>",
                True,
            )
            tracked["status"] = "completed"
            await self._process_queue()
            return

        if self._context.abort_signal.is_set():
            tracked["result"] = ("Interrupted by user", True)
            tracked["status"] = "completed"
            await self._process_queue()
            return

        if self._has_errored and tracked["name"] == "bash":
            tracked["result"] = ("Cancelled: parallel tool call errored", True)
            tracked["status"] = "completed"
            await self._process_queue()
            return

        try:
            result = await tool_def.call(tracked["input"], self._context)
            tracked["result"] = (result, False)
        except Exception as e:
            tracked["result"] = (f"<tool_use_error>{e}</tool_use_error>", True)
            if tracked["name"] == "bash":
                self._has_errored = True

        tracked["status"] = "completed"
        await self._process_queue()

    def get_completed_results(self) -> list[tuple[str, str, bool]]:
        """
        Get completed but not-yet-yielded results.

        Returns list of (tool_use_id, content, is_error).
        """
        results = []
        for t in self._tracked:
            if t["status"] == "yielded":
                continue
            if t["status"] == "completed" and t["result"] is not None:
                t["status"] = "yielded"
                content, is_error = t["result"]
                results.append((t["id"], content, is_error))
            elif t["status"] == "executing" and not t["is_concurrency_safe"]:
                break
        return results

    async def get_remaining_results(self) -> list[tuple[str, str, bool]]:
        """Wait for all remaining tools and return results in order."""
        while any(t["status"] in ("queued", "executing") for t in self._tracked):
            await self._process_queue()
            await asyncio.sleep(0.01)

        results = []
        for t in self._tracked:
            if t["status"] == "completed" and t["result"] is not None:
                t["status"] = "yielded"
                content, is_error = t["result"]
                results.append((t["id"], content, is_error))
        return results


# ---------------------------------------------------------------------------
# Context Summarization (cf. src/services/compact/compact.ts)
# ---------------------------------------------------------------------------


@dataclass
class CompactionResult:
    """Result of conversation summarization."""

    summary_text: str
    pre_compact_message_count: int
    post_compact_message_count: int


async def summarize_conversation(
    messages: list[dict[str, Any]],
    api_client: APIClient,
    max_summary_input_tokens: int = 8000,
) -> CompactionResult:
    """
    Summarize conversation history to save context window space.

    Strategy:
      1. Keep the last few messages
      2. Summarize everything before that into a single user message
      3. Use that summary as the compact conversation prefix
    """
    if len(messages) <= 6:
        return CompactionResult(
            summary_text="",
            pre_compact_message_count=len(messages),
            post_compact_message_count=len(messages),
        )

    keep_count = min(4, len(messages) - 2)
    messages_to_summarize = messages[:-keep_count]
    summary_input = _format_messages_for_summary(messages_to_summarize)

    if len(summary_input) > max_summary_input_tokens * 4:
        summary_input = summary_input[-(max_summary_input_tokens * 4):]

    summary_parts: list[str] = []
    async for event in api_client.stream_message(
        system_prompt=(
            "You are a conversation summarizer. Create a concise summary of "
            "the conversation below. Focus on: what the user asked for, what "
            "tools were used, key findings, and the current state of the task. "
            "Write in plain paragraphs."
        ),
        messages=[{"role": "user", "content": summary_input}],
        tools=[],
    ):
        if event["type"] == "text_delta":
            summary_parts.append(event["text"])

    summary = "".join(summary_parts).strip()

    return CompactionResult(
        summary_text=summary,
        pre_compact_message_count=len(messages),
        post_compact_message_count=1 + keep_count,
    )


def _format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Format messages into a readable text block for summarization."""
    lines = ["## Conversation History to Summarize\n"]
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        parts.append(
                            f"[Tool call: {block.get('name', '?')}"
                            f"({json.dumps(block.get('input', {}))})]"
                        )
                    elif block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        parts.append(f"[Tool result: {str(result_content)[:500]}]")
                else:
                    parts.append(str(block))
            content = " ".join(parts)
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Query Loop (cf. src/query.ts queryLoop)
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""

    max_turns: int = 20
    auto_compact_threshold_tokens: int = 4000
    system_prompt: str = ""
    workspace_root: str | pathlib.Path = "."
    tools: list[Tool] = field(default_factory=lambda: DEFAULT_TOOLS.copy())
    tool_policy: ToolPolicy = field(default_factory=AutoAllowPolicy)
    hooks: list[Hook] = field(default_factory=list)


class AgentHarness:
    """
    The main agent harness - a single-agent tool-use loop.

    Usage:
        harness = AgentHarness(api_client, config)
        async for event in harness.run("Your task here"):
            print(render_event(event))
    """

    def __init__(
        self,
        api_client: APIClient,
        config: AgentConfig | None = None,
        session: AgentSession | None = None,
    ):
        self.api = api_client
        self.config = config or AgentConfig()
        self.session = session or AgentSession()
        self.workspace_root = pathlib.Path(self.config.workspace_root).resolve()
        self._active_abort_event: asyncio.Event | None = None

    def abort(self) -> None:
        """Signal the active run to stop, if one is running."""
        if self._active_abort_event is not None:
            self._active_abort_event.set()

    async def run(
        self,
        user_message: str,
    ) -> AsyncGenerator[AgentEvent, AgentCommand | None]:
        """
        Run the agent loop for one user message.

        The generator yields AgentEvent objects. Callers can send AgentCommand
        objects back with asend(), currently ApprovalResponse or AbortCommand.
        """
        abort_event = asyncio.Event()
        self._active_abort_event = abort_event
        context = ToolContext(
            abort_signal=abort_event,
            workspace_root=self.workspace_root,
            session=self.session,
        )
        self.session.messages.append({"role": "user", "content": user_message})
        tool_schemas = self._build_tool_schemas()
        turn = 0
        turn_limit = self.config.max_turns

        try:
            command = yield await self._emit(
                AgentEvent(
                    "run_started",
                    {"tool_count": len(self.config.tools), "max_turns": turn_limit},
                )
            )
            _handle_command(command, abort_event)
            if abort_event.is_set():
                command = yield await self._emit(
                    AgentEvent("run_aborted", {"reason": "Aborted before first turn"})
                )
                _handle_command(command, abort_event)
                return

            while True:
                turn += 1
                command = yield await self._emit(
                    AgentEvent(
                        "turn_started",
                        {"turn": turn, "max_turns": turn_limit},
                    )
                )
                _handle_command(command, abort_event)

                if abort_event.is_set():
                    command = yield await self._emit(
                        AgentEvent("run_aborted", {"reason": "Aborted by user"})
                    )
                    _handle_command(command, abort_event)
                    return

                estimated_tokens = sum(
                    len(json.dumps(m, default=str)) // 4
                    for m in self.session.messages
                )
                if estimated_tokens > self.config.auto_compact_threshold_tokens:
                    command = yield await self._emit(
                        AgentEvent(
                            "context_compacting",
                            {"estimated_tokens": estimated_tokens},
                        )
                    )
                    _handle_command(command, abort_event)
                    if abort_event.is_set():
                        command = yield await self._emit(
                            AgentEvent("run_aborted", {"reason": "Aborted by user"})
                        )
                        _handle_command(command, abort_event)
                        return

                    try:
                        compact_result = await summarize_conversation(
                            self.session.messages,
                            self.api,
                        )
                        if compact_result.summary_text:
                            keep_count = max(
                                1,
                                len(self.session.messages)
                                - compact_result.post_compact_message_count
                                + 1,
                            )
                            kept = (
                                self.session.messages[-keep_count:]
                                if keep_count < len(self.session.messages)
                                else self.session.messages[-4:]
                            )
                            self.session.messages = [
                                {
                                    "role": "user",
                                    "content": (
                                        "[Earlier conversation summary]\n"
                                        f"{compact_result.summary_text}"
                                    ),
                                },
                                *kept,
                            ]
                            context.session = self.session
                            command = yield await self._emit(
                                AgentEvent(
                                    "context_compacted",
                                    {
                                        "pre_compact_message_count": (
                                            compact_result.pre_compact_message_count
                                        ),
                                        "post_compact_message_count": (
                                            len(self.session.messages)
                                        ),
                                    },
                                )
                            )
                            _handle_command(command, abort_event)
                    except Exception as e:
                        command = yield await self._emit(
                            AgentEvent("compaction_failed", {"error": str(e)})
                        )
                        _handle_command(command, abort_event)

                if abort_event.is_set():
                    command = yield await self._emit(
                        AgentEvent("run_aborted", {"reason": "Aborted by user"})
                    )
                    _handle_command(command, abort_event)
                    return

                command = yield await self._emit(AgentEvent("model_start"))
                _handle_command(command, abort_event)
                if abort_event.is_set():
                    command = yield await self._emit(
                        AgentEvent("run_aborted", {"reason": "Aborted by user"})
                    )
                    _handle_command(command, abort_event)
                    return

                assistant_text: list[str] = []
                tool_use_blocks: list[ToolUseBlock] = []
                all_tool_results: list[tuple[str, str, bool]] = []
                streaming_executor = StreamingToolExecutor(self.config.tools, context)

                async for event in self.api.stream_message(
                    system_prompt=self.config.system_prompt,
                    messages=self.session.messages,
                    tools=tool_schemas,
                ):
                    if abort_event.is_set():
                        break

                    if event["type"] == "text_delta":
                        assistant_text.append(event["text"])
                        command = yield await self._emit(
                            AgentEvent("model_delta", {"text": event["text"]})
                        )
                        _handle_command(command, abort_event)

                    elif event["type"] == "tool_use":
                        block = ToolUseBlock(
                            id=event["id"],
                            name=event["name"],
                            input=event["input"],
                        )
                        tool_use_blocks.append(block)
                        command = yield await self._emit(
                            AgentEvent(
                                "tool_call",
                                {
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                },
                            )
                        )
                        _handle_command(command, abort_event)
                        if abort_event.is_set():
                            break

                        policy_result = await self._check_tool_policy(block, context)
                        if policy_result.action == "deny":
                            denied_result = (
                                f"<tool_use_error>Tool denied: "
                                f"{policy_result.reason}</tool_use_error>"
                            )
                            all_tool_results.append((block.id, denied_result, True))
                            command = yield await self._emit_tool_result(
                                block.id,
                                denied_result,
                                True,
                            )
                            _handle_command(command, abort_event)

                        elif policy_result.action == "request_approval":
                            request_id = f"approval_{uuid.uuid4().hex[:8]}"
                            command = yield await self._emit(
                                AgentEvent(
                                    "approval_requested",
                                    {
                                        "request_id": request_id,
                                        "tool_use_id": block.id,
                                        "tool_name": block.name,
                                        "input": block.input,
                                        "reason": policy_result.reason,
                                    },
                                )
                            )
                            _handle_command(command, abort_event)
                            if abort_event.is_set():
                                break

                            approval = command
                            if (
                                isinstance(approval, ApprovalResponse)
                                and approval.request_id == request_id
                                and approval.approved
                            ):
                                streaming_executor.add_tool(block)
                            else:
                                if isinstance(approval, ApprovalResponse):
                                    if approval.request_id != request_id:
                                        reason = (
                                            "Approval response request_id mismatch: "
                                            f"expected {request_id}, got "
                                            f"{approval.request_id}"
                                        )
                                    else:
                                        reason = (
                                            approval.reason
                                            or "User denied approval"
                                        )
                                elif approval is None:
                                    reason = "Approval response missing"
                                else:
                                    reason = (
                                        "Expected ApprovalResponse, got "
                                        f"{type(approval).__name__}"
                                    )
                                denied_result = (
                                    f"<tool_use_error>Tool denied: "
                                    f"{reason}</tool_use_error>"
                                )
                                all_tool_results.append((block.id, denied_result, True))
                                command = yield await self._emit_tool_result(
                                    block.id,
                                    denied_result,
                                    True,
                                )
                                _handle_command(command, abort_event)

                        else:
                            streaming_executor.add_tool(block)

                        for tid, content, is_error in (
                            streaming_executor.get_completed_results()
                        ):
                            all_tool_results.append((tid, content, is_error))
                            command = yield await self._emit_tool_result(
                                tid,
                                content,
                                is_error,
                            )
                            _handle_command(command, abort_event)

                    elif event["type"] == "message_stop":
                        break

                if abort_event.is_set():
                    command = yield await self._emit(
                        AgentEvent("run_aborted", {"reason": "Aborted by user"})
                    )
                    _handle_command(command, abort_event)
                    return

                if not tool_use_blocks:
                    full_text = "".join(assistant_text)
                    if full_text.strip():
                        self.session.messages.append({
                            "role": "assistant",
                            "content": full_text,
                        })
                        command = yield await self._emit(
                            AgentEvent("run_complete", {"text": full_text})
                        )
                        _handle_command(command, abort_event)
                    return

                remaining = await streaming_executor.get_remaining_results()
                for tid, content, is_error in remaining:
                    all_tool_results.append((tid, content, is_error))
                    command = yield await self._emit_tool_result(
                        tid,
                        content,
                        is_error,
                    )
                    _handle_command(command, abort_event)

                assistant_content: list[dict[str, Any]] = []
                if assistant_text:
                    assistant_content.append({
                        "type": "text",
                        "text": "".join(assistant_text),
                    })
                for block in tool_use_blocks:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                self.session.messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                tool_result_content: list[dict[str, Any]] = []
                for tid, content, is_error in all_tool_results:
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": content,
                        "is_error": is_error,
                    })
                if tool_result_content:
                    self.session.messages.append({
                        "role": "user",
                        "content": tool_result_content,
                    })

                if turn >= turn_limit:
                    command = yield await self._emit(
                        AgentEvent(
                            "turn_limit_reached",
                            {"max_turns": turn_limit},
                        )
                    )
                    _handle_command(command, abort_event)
                    return
        finally:
            self._active_abort_event = None

    async def _check_tool_policy(
        self,
        block: ToolUseBlock,
        context: ToolContext,
    ) -> ToolDecision:
        tool = next((t for t in self.config.tools if t.name == block.name), None)
        request = ToolRequest(
            block=block,
            tool=tool,
            capabilities=get_tool_capabilities(tool),
        )
        return await self.config.tool_policy.check(request, context)

    async def _emit_tool_result(
        self,
        tool_use_id: str,
        content: str,
        is_error: bool,
    ) -> AgentEvent:
        return await self._emit(
            AgentEvent(
                "tool_result",
                {
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": is_error,
                },
            )
        )

    async def _emit(self, event: AgentEvent) -> AgentEvent:
        hook_errors = []
        for hook in self.config.hooks:
            try:
                await _maybe_await(hook(event, self.session))
            except Exception as e:
                hook_errors.append({
                    "hook": getattr(hook, "__name__", hook.__class__.__name__),
                    "error": str(e),
                })
        if hook_errors:
            event.data = {**event.data, "hook_errors": hook_errors}
        return event

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert Tool objects to provider tool schemas."""
        schemas = []
        for tool in self.config.tools:
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
        return schemas


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = """You are a helpful coding assistant with access to tools.
You can read files, write files, search code, and run shell commands.
When you need information, use a tool. When you're done, respond with plain text.
Be concise."""


async def _run_interactive_turn(harness: AgentHarness, user_input: str) -> None:
    stream = harness.run(user_input)
    command: AgentCommand | None = None

    while True:
        try:
            if command is None:
                event = await stream.__anext__()
            else:
                event = await stream.asend(command)
                command = None
        except StopAsyncIteration:
            return

        print(render_event(event), end="", flush=True)

        if event.type == "approval_requested":
            answer = input("\nApprove this tool call? [y/N] ").strip().lower()
            approved = answer in ("y", "yes")
            reason = "" if approved else "User denied approval"
            command = ApprovalResponse(
                request_id=event.data["request_id"],
                approved=approved,
                reason=reason,
            )


async def interactive_loop(api_client: APIClient) -> None:
    """Simple interactive REPL for testing the agent."""
    config = AgentConfig(
        max_turns=10,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tool_policy=RequireApprovalPolicy(),
    )
    harness = AgentHarness(api_client, config)

    print("=" * 60)
    print("Minimal Agent Harness - interactive mode")
    print(f"Model: {api_client.model}")
    print(f"Workspace root: {harness.workspace_root}")
    print(f"Available tools: {[t.name for t in config.tools]}")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        await _run_interactive_turn(harness, user_input)


async def main() -> None:
    """Parse CLI args and run."""
    if "--mock" in sys.argv:
        api_client = MockAPIClient()
        print("[Using mock API client - no real API calls]")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set. Use --mock for dry-run mode.")
            print("  export ANTHROPIC_API_KEY=sk-ant-...")
            print("  or: python agent_harness.py --mock")
            sys.exit(1)
        api_client = AnthropicClient(api_key=api_key)

    await interactive_loop(api_client)


if __name__ == "__main__":
    asyncio.run(main())
