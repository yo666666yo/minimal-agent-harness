"""Example: drive the bidirectional approval protocol.

Run:
    python examples/approval_policy.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import tempfile
from typing import Any, AsyncGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import (  # noqa: E402
    APIClient,
    AgentCommand,
    AgentConfig,
    AgentEvent,
    AgentHarness,
    ApprovalResponse,
    RequireApprovalPolicy,
    render_event,
)


async def drive_with_auto_denial(harness: AgentHarness, prompt: str) -> None:
    """Deny requested approvals to show the command path without writing files."""
    stream = harness.run(prompt)
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

        print(render_event(event))
        command = approval_command_for(event)


def approval_command_for(event: AgentEvent) -> AgentCommand | None:
    if event.type != "approval_requested":
        return None
    return ApprovalResponse(
        request_id=event.data["request_id"],
        approved=False,
        reason="Denied by approval_policy.py demo",
    )


class ApprovalDemoClient(APIClient):
    """Scripted model that asks to write a file, then reports the policy result."""

    model = "approval-demo"

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
                "id": "write_file_1",
                "name": "write_file",
                "input": {"file_path": "demo.txt", "content": "created by agent"},
            }
            yield {"type": "message_stop"}
            return

        yield {
            "type": "text_delta",
            "text": "The write_file request was denied, so no file was created.",
        }
        yield {"type": "message_stop"}


async def main() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        harness = AgentHarness(
            ApprovalDemoClient(),
            AgentConfig(
                tool_policy=RequireApprovalPolicy(),
                workspace_root=workspace,
            ),
        )
        await drive_with_auto_denial(harness, "write a small file")


if __name__ == "__main__":
    asyncio.run(main())
