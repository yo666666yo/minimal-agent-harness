"""Example: add a small domain-specific tool to the harness.

Run:
    python examples/custom_tool.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from typing import Any, AsyncGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import (  # noqa: E402
    APIClient,
    AgentConfig,
    AgentHarness,
    DEFAULT_TOOLS,
    Tool,
    ToolCapabilities,
    ToolContext,
    render_event,
)


class TicketLookupTool(Tool):
    name = "ticket_lookup"
    description = "Look up an internal ticket by ID."
    input_schema = {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}},
        "required": ["ticket_id"],
    }
    capabilities = ToolCapabilities(read_only=True, concurrency_safe=True)

    async def call(
        self,
        input: dict[str, Any],
        context: ToolContext,
    ) -> str:
        ticket_id = input["ticket_id"]
        return f"{ticket_id}: Fix flaky checkout tests. Owner: platform. Status: open."


class TicketDemoClient(APIClient):
    """Tiny scripted client that demonstrates the custom tool path."""

    model = "ticket-demo"

    async def stream_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        if not any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in message["content"]
            )
            for message in messages
        ):
            yield {
                "type": "tool_use",
                "id": "ticket_lookup_1",
                "name": "ticket_lookup",
                "input": {"ticket_id": "PLAT-123"},
            }
            yield {"type": "message_stop"}
            return

        yield {
            "type": "text_delta",
            "text": "Ticket PLAT-123 is open and owned by platform.",
        }
        yield {"type": "message_stop"}


async def main() -> None:
    config = AgentConfig(tools=DEFAULT_TOOLS + [TicketLookupTool()])
    harness = AgentHarness(TicketDemoClient(), config)

    async for event in harness.run("Look up ticket PLAT-123"):
        print(render_event(event))


if __name__ == "__main__":
    asyncio.run(main())
