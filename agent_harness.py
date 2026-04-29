"""
Minimal Agent Harness — a single-agent tool-use loop with context summarization.

Educational implementation based on the Claude Code architecture:
  src/query.ts           — core query loop (the while-True async generator)
  src/QueryEngine.ts     — per-conversation state wrapper
  src/services/tools/StreamingToolExecutor.ts — concurrent tool execution
  src/services/tools/toolOrchestration.ts     — tool partitioning
  src/services/compact/compact.ts             — conversation summarization
  src/Tool.ts            — tool type definitions

Key design patterns preserved:
  1. Async-generator query loop (query.ts:241, the while-true loop)
  2. Streaming tool execution (tools start during model streaming)
  3. Concurrent-safe vs exclusive tool partitioning (toolOrchestration.ts:91)
  4. Conversation summarization via a separate model call (compact.ts:387)
  5. Turn limit and abort as termination signals

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
import json
import os
import sys
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Tool System (cf. src/Tool.ts, src/services/tools/toolExecution.ts)
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Base class for all tools (cf. Tool.ts type definitions)."""

    name: str
    description: str
    # JSON Schema for the tool's input parameters
    input_schema: dict[str, Any]
    # Whether this tool can run concurrently with other concurrent-safe tools
    is_concurrency_safe: bool = True

    @abstractmethod
    async def call(self, input: dict[str, Any], abort_signal: asyncio.Event | None) -> str:
        """Execute the tool. Returns result as a string."""
        ...


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
    is_concurrency_safe = False  # Write tool — exclusive access

    async def call(self, input: dict[str, Any], abort_signal: asyncio.Event | None) -> str:
        proc = await asyncio.create_subprocess_shell(
            input["command"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return f"<tool_use_error>Command timed out after 30s</tool_use_error>"
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
        },
        "required": ["file_path"],
    }
    is_concurrency_safe = True  # Read-only

    async def call(self, input: dict[str, Any], abort_signal: asyncio.Event | None) -> str:
        path = input["file_path"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(8000)
            if len(content) == 8000:
                content += "\n... [truncated at 8000 chars]"
            return content
        except FileNotFoundError:
            return f"<tool_use_error>File not found: {path}</tool_use_error>"
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
    is_concurrency_safe = False  # Write tool — exclusive access

    async def call(self, input: dict[str, Any], abort_signal: asyncio.Event | None) -> str:
        path = input["file_path"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(input["content"])
        return f"Successfully wrote {len(input['content'])} chars to {path}"


class GrepTool(Tool):
    """Content search (cf. src/tools/GrepTool/)."""

    name = "grep"
    description = "Search for a pattern in files."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Directory to search in (default: '.')"},
        },
        "required": ["pattern"],
    }
    is_concurrency_safe = True  # Read-only

    async def call(self, input: dict[str, Any], abort_signal: asyncio.Event | None) -> str:
        import glob
        import re

        pattern = re.compile(input["pattern"])
        search_dir = input.get("path", ".")
        results = []
        for filepath in glob.glob(f"{search_dir}/**/*.py", recursive=True):
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


# Tool registry
DEFAULT_TOOLS: list[Tool] = [BashTool(), ReadTool(), WriteTool(), GrepTool()]


# ---------------------------------------------------------------------------
# Message types (cf. src/types/message.ts)
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
        Yields streaming events. Each event is a dict:
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

        # Convert OpenAI-style tool defs to Anthropic format
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
    Simulates basic LLM responses — the model will use tools if they're relevant.
    """

    def __init__(self, model: str = "mock"):
        self.model = model
        import uuid

        self._uuid = uuid

    async def stream_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Extract the user's last message
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
                        elif isinstance(block, dict) and block.get("type") == "tool_result":
                            pass  # Skip tool results when looking for user intent
                break

        tool_names = [t["name"] for t in tools]

        # Crude routing: pick tools based on keywords
        if "read" in user_content.lower() or "show" in user_content.lower() or "look" in user_content.lower():
            if "read_file" in tool_names:
                # Extract a path hint
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

        if "search" in user_content.lower() or "find" in user_content.lower() or "grep" in user_content.lower():
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

        if "write" in user_content.lower() or "create" in user_content.lower():
            if "write_file" in tool_names:
                yield {
                    "type": "tool_use",
                    "id": f"mock_tool_{self._uuid.uuid4().hex[:8]}",
                    "name": "write_file",
                    "input": {"file_path": "/tmp/mock_output.txt", "content": "mock content"},
                }
                yield {"type": "message_stop"}
                return

        if "run" in user_content.lower() or "bash" in user_content.lower() or "execute" in user_content.lower():
            if "bash" in tool_names:
                yield {
                    "type": "tool_use",
                    "id": f"mock_tool_{self._uuid.uuid4().hex[:8]}",
                    "name": "bash",
                    "input": {"command": "echo 'mock command executed'"},
                }
                yield {"type": "message_stop"}
                return

        # Default: text response
        yield {"type": "text_delta", "text": f"[Mock] I received your message: '{user_content[:200]}'. I can help with that."}
        yield {"type": "message_stop"}


# ---------------------------------------------------------------------------
# Streaming Tool Executor (cf. src/services/tools/StreamingToolExecutor.ts)
# ---------------------------------------------------------------------------

class StreamingToolExecutor:
    """
    Executes tools concurrently during model streaming.

    Key design (StreamingToolExecutor.ts):
      - Concurrent-safe tools run in parallel
      - Non-concurrent tools run serially
      - Tools are queued, executed, and results buffered in order
      - On abort, synthetic error results are generated for pending tools
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        abort_event: asyncio.Event,
    ):
        self._tool_map: dict[str, Tool] = {t.name: t for t in tools}
        self._abort_event = abort_event
        self._sibling_abort = asyncio.Event()  # Bash error cascading

        # Tracked tools (cf. StreamingToolExecutor.ts:40)
        self._tracked: list[dict[str, Any]] = []
        self._has_errored = False

    def add_tool(self, block: ToolUseBlock) -> None:
        """Add a tool to the execution queue (cf. addTool())."""
        tool_def = self._tool_map.get(block.name)
        is_safe = tool_def.is_concurrency_safe if tool_def else True
        self._tracked.append({
            "id": block.id,
            "name": block.name,
            "input": block.input,
            "status": "queued",
            "is_concurrency_safe": is_safe,
            "result": None,
        })
        # Kick off processing
        asyncio.ensure_future(self._process_queue())

    async def _process_queue(self) -> None:
        """Process the queue respecting concurrency rules."""
        for t in self._tracked:
            if t["status"] != "queued":
                continue
            if self._can_execute(t["is_concurrency_safe"]):
                await self._execute_tool(t)
            elif not t["is_concurrency_safe"]:
                # Non-concurrent tool blocks — stop processing
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

        # Check for abort/error before executing
        if self._abort_event.is_set():
            tracked["result"] = ("Interrupted by user", True)
            tracked["status"] = "completed"
            await self._process_queue()
            return

        if self._has_errored and tracked["name"] == "bash":
            # Bash sibling cancellation (StreamingToolExecutor.ts:357-363)
            tracked["result"] = ("Cancelled: parallel tool call errored", True)
            tracked["status"] = "completed"
            await self._process_queue()
            return

        try:
            result = await tool_def.call(tracked["input"], self._sibling_abort)
            tracked["result"] = (result, False)
        except Exception as e:
            tracked["result"] = (f"<tool_use_error>{e}</tool_use_error>", True)
            if tracked["name"] == "bash":
                # Bash error cascades to siblings (StreamingToolExecutor.ts:357)
                self._has_errored = True
                self._sibling_abort.set()

        tracked["status"] = "completed"
        await self._process_queue()

    def get_completed_results(self) -> list[tuple[str, str, bool]]:
        """
        Get completed but not-yet-yielded results (non-blocking).
        Returns list of (tool_use_id, content, is_error).
        Maintains order for non-concurrent tools.
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
                # Block on non-concurrent tools — must yield in order
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
    """Result of conversation summarization (cf. CompactionResult in compact.ts:299)."""
    summary_text: str
    pre_compact_message_count: int
    post_compact_message_count: int


async def summarize_conversation(
    messages: list[dict[str, Any]],
    api_client: APIClient,
    max_summary_input_tokens: int = 8000,
) -> CompactionResult:
    """
    Summarize the conversation history to save context window space.

    Strategy (cf. compactConversation() in compact.ts:387):
      1. Keep the last ~4 messages (recent context is most valuable)
      2. Summarize everything before that into a single user message
      3. The summary becomes the new "prefix" of the conversation

    In the real Claude Code, this uses a forked sub-agent that reuses the
    main conversation's prompt cache. Here we make a simple standalone API call.
    """
    if len(messages) <= 6:
        # Not enough to summarize
        return CompactionResult(
            summary_text="",
            pre_compact_message_count=len(messages),
            post_compact_message_count=len(messages),
        )

    # Split: keep the last 4 messages, summarize the rest
    keep_count = min(4, len(messages) - 2)
    messages_to_summarize = messages[:-keep_count]
    messages_to_keep = messages[-keep_count:]

    # Build a text representation of what we're summarizing
    summary_input = _format_messages_for_summary(messages_to_summarize)

    # Truncate if too long (rough estimate: 4 chars ≈ 1 token)
    if len(summary_input) > max_summary_input_tokens * 4:
        summary_input = summary_input[-(max_summary_input_tokens * 4):]

    # Call the model to generate a summary (no tools needed)
    summary_parts: list[str] = []
    async for event in api_client.stream_message(
        system_prompt="You are a conversation summarizer. Create a concise summary of the conversation below. "
                       "Focus on: what the user asked for, what tools were used, key findings, "
                       "and the current state of the task. Write in plain paragraphs.",
        messages=[{"role": "user", "content": summary_input}],
        tools=[],  # No tools — pure text summary
    ):
        if event["type"] == "text_delta":
            summary_parts.append(event["text"])

    summary = "".join(summary_parts).strip()

    return CompactionResult(
        summary_text=summary,
        pre_compact_message_count=len(messages),
        post_compact_message_count=1 + len(messages_to_keep),
    )


def _format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Format messages into a readable text block for summarization."""
    lines = ["## Conversation History to Summarize\n"]
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if isinstance(content, list):
            # Multi-block content (text + tool_use + tool_result)
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        parts.append(f"[Tool call: {block.get('name', '?')}({json.dumps(block.get('input', {}))})]")
                    elif block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        parts.append(f"[Tool result: {str(result_content)[:500]}]")
                else:
                    parts.append(str(block))
            content = " ".join(parts)
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Query Loop (cf. src/query.ts:241 queryLoop)
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""
    max_turns: int = 20
    max_budget_usd: float = 5.0
    auto_compact_threshold_tokens: int = 4000  # Summarize when context > this
    system_prompt: str = ""
    tools: list[Tool] = field(default_factory=lambda: DEFAULT_TOOLS.copy())


class AgentHarness:
    """
    The main agent harness — a single-agent tool-use loop.

    Usage:
        harness = AgentHarness(api_client, config)
        async for event in harness.run("Your task here"):
            print(event)
    """

    def __init__(self, api_client: APIClient, config: AgentConfig | None = None):
        self.api = api_client
        self.config = config or AgentConfig()
        self._abort_event = asyncio.Event()

    def abort(self) -> None:
        """Signal the agent to stop (cf. abortController.abort())."""
        self._abort_event.set()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Run the agent loop for one user message.

        This is equivalent to QueryEngine.submitMessage() → queryLoop().
        Each yield is a human-readable event string.
        """
        # Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]

        # Build tool schemas for the API
        tool_schemas = self._build_tool_schemas()

        # The query loop (cf. query.ts:306 while-true)
        turn = 0
        turn_limit = self.config.max_turns

        yield f"[Agent] Starting with {len(self.config.tools)} tools, max {turn_limit} turns"

        while True:
            turn += 1
            yield f"\n[Turn {turn}/{turn_limit}]"

            # ---- Auto-compact: summarize if over threshold ----
            estimated_tokens = sum(
                len(json.dumps(m, default=str)) // 4 for m in messages
            )
            if estimated_tokens > self.config.auto_compact_threshold_tokens:
                yield f"[Agent] Context at ~{estimated_tokens} tokens — summarizing..."
                try:
                    compact_result = await summarize_conversation(messages, self.api)
                    if compact_result.summary_text:
                        # Replace old messages with summary + last few kept messages
                        keep_count = max(1, len(messages) - compact_result.post_compact_message_count + 1)
                        kept = messages[-keep_count:] if keep_count < len(messages) else messages[-4:]
                        messages = [
                            {
                                "role": "user",
                                "content": f"[Earlier conversation summary]\n{compact_result.summary_text}",
                            },
                            *kept,
                        ]
                        yield f"[Agent] Compressed {compact_result.pre_compact_message_count} → {len(messages)} messages"
                except Exception as e:
                    yield f"[Agent] Summarization failed: {e}"

            # ---- Check abort ----
            if self._abort_event.is_set():
                yield "[Agent] Aborted by user"
                return

            # ---- Call the model (streaming) ----
            yield "[Agent] Calling model..."

            assistant_text: list[str] = []
            tool_use_blocks: list[ToolUseBlock] = []
            all_tool_results: list[tuple[str, str, bool]] = []  # Collect ALL results

            # Streaming tool executor (cf. StreamingToolExecutor)
            streaming_executor = StreamingToolExecutor(
                self.config.tools, self._abort_event
            )

            async for event in self.api.stream_message(
                system_prompt=self.config.system_prompt,
                messages=messages,
                tools=tool_schemas,
            ):
                if event["type"] == "text_delta":
                    assistant_text.append(event["text"])
                    yield f"[Model] {event['text']}"

                elif event["type"] == "tool_use":
                    block = ToolUseBlock(
                        id=event["id"],
                        name=event["name"],
                        input=event["input"],
                    )
                    tool_use_blocks.append(block)
                    yield f"[Tool call] {event['name']}({json.dumps(event['input'], default=str)[:200]})"

                    # Start executing immediately during streaming (cf. StreamingToolExecutor.addTool())
                    streaming_executor.add_tool(block)

                    # Yield completed results as they arrive (non-blocking)
                    for tid, content, is_error in streaming_executor.get_completed_results():
                        all_tool_results.append((tid, content, is_error))
                        tag = "ERROR" if is_error else "OK"
                        yield f"[Tool result/{tag}] {content[:200]}"

                elif event["type"] == "message_stop":
                    break

            # ---- No tool calls → turn is done ----
            if not tool_use_blocks:
                full_text = "".join(assistant_text)
                if full_text.strip():
                    yield f"\n[Agent] Complete. Final response:\n{full_text}"
                return

            # ---- Wait for remaining tool results ----
            remaining = await streaming_executor.get_remaining_results()
            for tid, content, is_error in remaining:
                all_tool_results.append((tid, content, is_error))
                tag = "ERROR" if is_error else "OK"
                yield f"[Tool result/{tag}] {content[:200]}"

            # ---- Build the next message batch (cf. query.ts:1715) ----
            # Append the assistant message (with tool_use blocks)
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
            messages.append({"role": "assistant", "content": assistant_content})

            # Append tool results as a user message (cf. query.ts:1716)
            tool_result_content: list[dict[str, Any]] = []
            for tid, content, is_error in all_tool_results:
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": content,
                    "is_error": is_error,
                })
            if tool_result_content:
                messages.append({"role": "user", "content": tool_result_content})

            # ---- Check turn limit ----
            if turn >= turn_limit:
                yield f"\n[Agent] Reached max turns ({turn_limit})"
                return

            # ---- Check abort ----
            if self._abort_event.is_set():
                yield "[Agent] Aborted by user"
                return

            # ---- Continue the loop (next iteration starts at model call) ----

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert Tool objects to Anthropic API tool schemas."""
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


async def interactive_loop(api_client: APIClient) -> None:
    """Simple interactive REPL for testing the agent."""
    config = AgentConfig(
        max_turns=10,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    harness = AgentHarness(api_client, config)

    print("=" * 60)
    print("Minimal Agent Harness — interactive mode")
    print(f"Model: {api_client.model}")
    print(f"Available tools: {[t.name for t in config.tools]}")
    print("Type 'quit' to exit, 'abort' to stop the current turn.")
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
        if user_input.lower() == "abort":
            harness.abort()
            continue

        async for event in harness.run(user_input):
            print(event, end="", flush=True)


async def main() -> None:
    """Parse CLI args and run."""
    if "--mock" in sys.argv:
        api_client = MockAPIClient()
        print("[Using mock API client — no real API calls]")
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
