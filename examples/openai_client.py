"""Example OpenAI provider adapter for minimal-agent-harness.

This keeps OpenAI-specific event parsing outside ``agent_harness.py`` so the core
loop stays small. The Responses API event names may evolve; treat this as a
starting point and keep the adapter covered by your own integration tests.

Install:
    pip install openai

Run:
    export OPENAI_API_KEY=sk-...
    python examples/openai_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
from typing import Any, AsyncGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import (  # noqa: E402
    APIClient,
    AgentConfig,
    AgentHarness,
    DEFAULT_SYSTEM_PROMPT,
    render_event,
)


class OpenAIResponsesClient(APIClient):
    """Adapter that normalizes OpenAI Responses API streams for AgentHarness."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.model = model

    async def stream_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        input_messages = self._to_openai_input(system_prompt, messages)
        openai_tools = [self._to_openai_tool(tool) for tool in tools]

        stream = await client.responses.create(
            model=self.model,
            input=input_messages,
            tools=openai_tools,
            stream=True,
        )

        partial_tool_calls: dict[str, dict[str, Any]] = {}

        async for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                yield {"type": "text_delta", "text": event.delta}

            elif event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    item_id = getattr(item, "id", "")
                    call_id = getattr(item, "call_id", None) or item_id
                    partial_tool_calls[item_id] = {
                        "id": call_id,
                        "item_id": item_id,
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", "") or "",
                    }

            elif event_type == "response.function_call_arguments.delta":
                item_id = getattr(event, "item_id", "")
                partial_tool_calls.setdefault(
                    item_id,
                    {
                        "id": item_id,
                        "item_id": item_id,
                        "name": getattr(event, "name", ""),
                        "arguments": "",
                    },
                )
                partial_tool_calls[item_id]["arguments"] += getattr(event, "delta", "")

            elif event_type == "response.function_call_arguments.done":
                item_id = getattr(event, "item_id", "")
                call = partial_tool_calls.get(item_id, {})
                raw_arguments = getattr(event, "arguments", None) or call.get("arguments", "{}")
                try:
                    parsed_input = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    parsed_input = {}

                yield {
                    "type": "tool_use",
                    "id": call.get("id", item_id),
                    "name": call.get("name", getattr(event, "name", "")),
                    "input": parsed_input,
                }

            elif event_type == "response.completed":
                yield {"type": "message_stop"}
                break

    def _to_openai_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }

    def _to_openai_input(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})

        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    text_parts.append(
                        f"[tool result {block.get('tool_use_id')}]\n{block.get('content', '')}"
                    )
                elif block.get("type") == "tool_use":
                    text_parts.append(
                        f"[tool call {block.get('name')}]\n"
                        f"{json.dumps(block.get('input', {}), default=str)}"
                    )

            converted.append({"role": role, "content": "\n\n".join(text_parts)})

        return converted


async def main() -> None:
    harness = AgentHarness(
        OpenAIResponsesClient(),
        AgentConfig(max_turns=5, system_prompt=DEFAULT_SYSTEM_PROMPT),
    )

    async for event in harness.run("Search for 'AgentHarness' in this repository"):
        print(render_event(event))


if __name__ == "__main__":
    asyncio.run(main())
