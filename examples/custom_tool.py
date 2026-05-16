"""Example: add a small domain-specific tool to the harness.

Run:
    python examples/custom_tool.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import AgentConfig, AgentHarness, DEFAULT_TOOLS, MockAPIClient, Tool  # noqa: E402


class TicketLookupTool(Tool):
    name = "ticket_lookup"
    description = "Look up an internal ticket by ID."
    input_schema = {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}},
        "required": ["ticket_id"],
    }
    is_concurrency_safe = True

    async def call(
        self,
        input: dict[str, Any],
        abort_signal: asyncio.Event | None,
    ) -> str:
        ticket_id = input["ticket_id"]
        return f"{ticket_id}: Fix flaky checkout tests. Owner: platform. Status: open."


async def main() -> None:
    config = AgentConfig(tools=DEFAULT_TOOLS + [TicketLookupTool()])
    harness = AgentHarness(MockAPIClient(), config)

    async for event in harness.run("Search for 'StreamingToolExecutor'"):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
