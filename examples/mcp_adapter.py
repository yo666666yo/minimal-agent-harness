"""Example: adapt MCP-style tools to the harness Tool interface.

This file intentionally avoids a concrete MCP SDK dependency so it stays
runnable. Replace ``FakeMCPToolSession`` with your MCP client session that can
list tools and call a named tool.

Run:
    python examples/mcp_adapter.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, AsyncGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import (  # noqa: E402
    APIClient,
    AgentConfig,
    AgentHarness,
    Tool,
    ToolCapabilities,
    ToolContext,
    render_event,
)


@dataclass
class MCPToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = True


class MCPToolAdapter(Tool):
    """Wrap one MCP server tool as an AgentHarness Tool."""

    def __init__(self, session: FakeMCPToolSession, spec: MCPToolSpec):
        self._session = session
        self.name = spec.name
        self.description = spec.description
        self.input_schema = spec.input_schema
        self.capabilities = ToolCapabilities(
            read_only=spec.read_only,
            concurrency_safe=spec.read_only,
            requires_approval=not spec.read_only,
        )

    async def call(self, input: dict[str, Any], context: ToolContext) -> str:
        return await self._session.call_tool(self.name, input)


class FakeMCPToolSession:
    """Small stand-in for an MCP client session."""

    async def list_tools(self) -> list[MCPToolSpec]:
        return [
            MCPToolSpec(
                name="docs_lookup",
                description="Look up a page in the internal docs server.",
                input_schema={
                    "type": "object",
                    "properties": {"slug": {"type": "string"}},
                    "required": ["slug"],
                },
                read_only=True,
            )
        ]

    async def call_tool(self, name: str, input: dict[str, Any]) -> str:
        if name != "docs_lookup":
            return f"<tool_use_error>Unknown MCP tool: {name}</tool_use_error>"
        return f"Docs page {input['slug']}: MCP tools can be adapted into Tool objects."


class DocsDemoClient(APIClient):
    """Scripted model that asks for the adapted MCP tool once."""

    model = "mcp-demo"

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
                "id": "docs_lookup_1",
                "name": "docs_lookup",
                "input": {"slug": "agent-runtime"},
            }
            yield {"type": "message_stop"}
            return

        yield {
            "type": "text_delta",
            "text": "The MCP docs lookup tool returned the runtime page.",
        }
        yield {"type": "message_stop"}


async def load_mcp_tools(session: FakeMCPToolSession) -> list[Tool]:
    return [MCPToolAdapter(session, spec) for spec in await session.list_tools()]


async def main() -> None:
    mcp_session = FakeMCPToolSession()
    tools = await load_mcp_tools(mcp_session)
    harness = AgentHarness(DocsDemoClient(), AgentConfig(tools=tools))

    async for event in harness.run("Look up the agent runtime docs"):
        print(render_event(event))


if __name__ == "__main__":
    asyncio.run(main())
