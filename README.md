# minimal-agent-harness

> A runnable, single-file reference implementation of the core loop behind coding agents.

[![Stars](https://img.shields.io/github/stars/yo666666yo/minimal-agent-harness?style=flat-square&color=yellow)](https://github.com/yo666666yo/minimal-agent-harness/stargazers)
[![CI](https://img.shields.io/github/actions/workflow/status/yo666666yo/minimal-agent-harness/ci.yml?style=flat-square)](https://github.com/yo666666yo/minimal-agent-harness/actions)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)

Most agent frameworks hide the interesting part behind abstractions. This repo keeps the
runtime small enough to read in one sitting while still showing the mechanics that matter:

- **Streaming model loop**: text and tool events are yielded as they arrive.
- **Tool execution during streaming**: tools start as soon as the model emits a call.
- **Safe parallelism**: read-only tools can overlap; write/shell tools run exclusively.
- **Auto-compact**: long conversations are summarized instead of blindly truncated.
- **Provider boundary**: the agent loop talks to a tiny `APIClient` interface.

This is a learning harness, not a production framework. Use it to understand how coding
agents work, then lift the patterns into your own stack.

## Quick Start

```bash
git clone https://github.com/yo666666yo/minimal-agent-harness.git
cd minimal-agent-harness
pip install anthropic
python agent_harness.py --mock
```

Mock mode needs no API key and still exercises the loop:

```text
> search for "Tool" in the codebase
[Agent] Starting with 4 tools, max 10 turns
[Turn 1/10]
[Agent] Calling model...
[Tool call] grep({"pattern": "Tool", "path": "."})
[Tool result/OK] ./agent_harness.py:...
```

To use Anthropic:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python agent_harness.py
```

## Why This Exists

If you want to build agents, you eventually need to understand the runtime:

1. How does the app keep streaming while tools are running?
2. How are tool calls represented in conversation history?
3. Which tools can run concurrently, and which must lock the workspace?
4. What happens when the context window fills up?
5. Where does provider-specific API code stop and agent logic begin?

`agent_harness.py` answers those questions directly in Python.

## Core Concepts

| Concept | Where to look | What it demonstrates |
|---|---|---|
| Tool interface | `Tool` | Name, description, JSON schema, async execution |
| Provider adapter | `APIClient`, `AnthropicClient`, `MockAPIClient` | Normalize model events into `text_delta`, `tool_use`, `message_stop` |
| Streaming executor | `StreamingToolExecutor` | Start tools while model streaming continues |
| Concurrency policy | `is_concurrency_safe` | Parallel reads, exclusive writes |
| Context management | `summarize_conversation()` | Replace old history with a compact summary |
| Agent loop | `AgentHarness.run()` | The async generator that ties everything together |

## Architecture

```text
User input
  |
  v
AgentHarness.run()
  |
  +--> estimate context size
  |      |
  |      +--> summarize older messages if needed
  |
  +--> stream model events
  |      |
  |      +--> text_delta ---------------> yield to caller
  |      |
  |      +--> tool_use -----------------> StreamingToolExecutor.add_tool()
  |                                       |
  |                                       +--> safe tools run in parallel
  |                                       +--> exclusive tools run serially
  |
  +--> append assistant tool calls and user tool results
  |
  +--> continue until no tool calls or max_turns is reached
```

## Built-In Tools

| Tool | Safe to run in parallel? | Purpose |
|---|---:|---|
| `read_file` | Yes | Read full files or optional line ranges |
| `grep` | Yes | Search Python files in a directory |
| `write_file` | No | Write a file, serially |
| `bash` | No | Run a shell command with a timeout |

The tools are intentionally small. The point is to make the control flow visible.

## Programmatic Usage

```python
import asyncio

from agent_harness import AgentConfig, AgentHarness, MockAPIClient


async def main():
    harness = AgentHarness(MockAPIClient(), AgentConfig(max_turns=5))

    async for event in harness.run("Search for 'StreamingToolExecutor'"):
        print(event)


asyncio.run(main())
```

## Add a Custom Tool

```python
from agent_harness import AgentConfig, AgentHarness, MockAPIClient, Tool, DEFAULT_TOOLS


class TicketLookupTool(Tool):
    name = "ticket_lookup"
    description = "Look up an internal ticket by ID."
    input_schema = {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}},
        "required": ["ticket_id"],
    }
    is_concurrency_safe = True

    async def call(self, input, abort_signal):
        return f"Ticket {input['ticket_id']}: example result"


config = AgentConfig(tools=DEFAULT_TOOLS + [TicketLookupTool()])
harness = AgentHarness(MockAPIClient(), config)
```

More examples:

- [`examples/custom_tool.py`](examples/custom_tool.py): add a small domain tool.
- [`examples/openai_client.py`](examples/openai_client.py): adapt the harness to OpenAI's Responses API.

## Project Layout

```text
agent_harness.py              # the readable core
examples/
  custom_tool.py              # custom tool example
  openai_client.py            # optional OpenAI provider adapter
tests/
  test_agent_harness.py       # focused regression tests
.github/workflows/ci.yml      # pytest + ruff
```

## When to Use This

| You want to... | Good fit? |
|---|---|
| Understand how coding-agent loops work | Yes |
| Teach tool calling, streaming, and context compaction | Yes |
| Prototype a narrow custom agent runtime | Yes |
| Replace LangChain, AutoGen, or a production agent platform | No |
| Run untrusted shell commands safely | No |

## Roadmap

- Provider examples: OpenAI, Gemini, LiteLLM.
- Better tool sandbox examples.
- More precise token accounting.
- A terminal recording for the README.
- Small walkthrough docs for each subsystem.

## Contributing

Contributions that keep the code readable are welcome. Good first issues include:

- Add a provider adapter under `examples/`.
- Improve a built-in tool without hiding the control flow.
- Add a focused regression test.
- Clarify README or tutorial sections.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow.

## Security

This project includes a `bash` tool for teaching purposes. It executes local shell commands.
Do not expose it to untrusted users or run it against sensitive workspaces. See
[`SECURITY.md`](SECURITY.md) for details.

## License

MIT. See [`LICENSE`](LICENSE).
