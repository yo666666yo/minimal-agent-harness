"""Example: write structured AgentEvent objects to a JSONL transcript.

Run:
    python examples/jsonl_transcript.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_harness import (  # noqa: E402
    AgentConfig,
    AgentEvent,
    AgentHarness,
    AgentSession,
    MockAPIClient,
    render_event,
)


class JsonlTranscriptHook:
    """Hook that appends every event to a JSONL file."""

    def __init__(self, path: pathlib.Path):
        self.path = path

    def __call__(self, event: AgentEvent, session: AgentSession) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")


async def main() -> None:
    transcript_path = pathlib.Path(tempfile.gettempdir()) / "agent_harness_transcript.jsonl"
    transcript_path.unlink(missing_ok=True)

    hook = JsonlTranscriptHook(transcript_path)
    harness = AgentHarness(
        MockAPIClient(),
        AgentConfig(hooks=[hook]),
    )

    async for event in harness.run("Say hello in one sentence"):
        print(render_event(event))

    print(f"\nTranscript written to {transcript_path}")


if __name__ == "__main__":
    asyncio.run(main())
