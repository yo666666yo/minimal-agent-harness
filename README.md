<p align="center">
  <img src="assets/banner.svg" alt="Minimal Agent Harness banner" width="100%">
</p>

<h1 align="center">Minimal Agent Harness</h1>

<p align="center">
  <strong>The coding-agent runtime, distilled into one readable Python file.</strong>
</p>

<p align="center">
  <a href="https://github.com/yo666666yo/minimal-agent-harness/stargazers"><img src="https://img.shields.io/github/stars/yo666666yo/minimal-agent-harness?style=for-the-badge&color=F0C866" alt="GitHub stars"></a>
  <a href="https://github.com/yo666666yo/minimal-agent-harness/actions"><img src="https://img.shields.io/github/actions/workflow/status/yo666666yo/minimal-agent-harness/ci.yml?style=for-the-badge" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2E8B7A?style=for-the-badge" alt="MIT license"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-356EA4?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
</p>

Most agent repos show you the product. This repo shows you the runtime.

`minimal-agent-harness` is a compact, runnable reference implementation of the loop behind coding agents: stream model events, start tools while the model is still streaming, feed tool results back into the conversation, compact long context, and keep going until the task is done.

It is intentionally small enough to read in one sitting, but complete enough to teach the control flow that production agent systems hide behind layers of framework code.

---

## What It Teaches

**A real agent loop.** `AgentHarness.run()` is an async generator that yields model text, tool calls, and tool results as they happen. It is the control surface a CLI, TUI, web UI, or IDE plugin can consume.

**Streaming tool execution.** Tool calls begin as soon as the provider stream emits them. The harness does not wait for the full model message before starting work.

**Safe concurrency.** Read-only tools can run in parallel. Shell and write tools are exclusive, so workspace-changing operations do not trample each other.

**Context compaction.** Long histories are summarized into a compact prefix instead of being blindly truncated.

**Provider isolation.** Anthropic, OpenAI, mocks, or your own endpoint fit behind the same tiny `APIClient` event interface.

---

## Quick Install

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

For a real Anthropic model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python agent_harness.py
```

---

## The Runtime In One Picture

<p align="center">
  <img src="assets/runtime-flow.svg" alt="Agent runtime flow diagram" width="100%">
</p>

---

## Core Pieces

| Piece | File | Why it matters |
|---|---|---|
| Tool interface | `Tool` | Defines name, description, JSON schema, and async execution |
| Provider adapter | `APIClient` | Normalizes model streams into `text_delta`, `tool_use`, and `message_stop` |
| Mock model | `MockAPIClient` | Lets you run the harness without an API key |
| Streaming executor | `StreamingToolExecutor` | Starts tools during model streaming |
| Context compacting | `summarize_conversation()` | Preserves task state when history gets long |
| Agent loop | `AgentHarness.run()` | The readable heart of the project |

---

## Built-In Tools

| Tool | Parallel? | Purpose |
|---|---:|---|
| `read_file` | Yes | Read full files or line ranges |
| `grep` | Yes | Search Python files |
| `write_file` | No | Write files with exclusive access |
| `bash` | No | Run shell commands with a timeout |

The tools are deliberately simple. The value is the orchestration pattern.

---

## Use It As A Library

```python
import asyncio

from agent_harness import AgentConfig, AgentHarness, MockAPIClient


async def main():
    harness = AgentHarness(MockAPIClient(), AgentConfig(max_turns=5))

    async for event in harness.run("Search for 'StreamingToolExecutor'"):
        print(event)


asyncio.run(main())
```

---

## Extend It

```python
from agent_harness import AgentConfig, DEFAULT_TOOLS, MockAPIClient, Tool, AgentHarness


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

Examples:

- [`examples/custom_tool.py`](examples/custom_tool.py) - add a small domain tool.
- [`examples/openai_client.py`](examples/openai_client.py) - adapt the harness to OpenAI's Responses API.

---

## Project Map

```text
agent_harness.py              # the readable core runtime
examples/
  custom_tool.py              # custom tool example
  openai_client.py            # optional OpenAI provider adapter
tests/
  test_agent_harness.py       # focused regression tests
.github/workflows/ci.yml      # pytest + ruff
```

---

## Good Fit / Bad Fit

| Use case | Fit |
|---|---|
| Learn how coding-agent loops work | Excellent |
| Teach streaming tool use and compaction | Excellent |
| Prototype a narrow custom agent runtime | Good |
| Replace LangChain, AutoGen, or a production platform | Poor |
| Run untrusted shell commands safely | Poor |

---

## Roadmap

- Provider examples for Gemini and LiteLLM.
- Better sandbox examples for shell and file tools.
- A terminal recording for the README.
- More precise token accounting.
- Walkthrough docs for the executor, compacting, and provider adapter.

---

## Contributing

Contributions are welcome when they keep the core easy to read. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md), run `pytest -q`, and keep provider-specific complexity in `examples/` unless it belongs in the runtime itself.

## Security

This harness includes local filesystem and shell tools for teaching purposes. It is not a sandbox. Do not expose the built-in tools to untrusted users. See [`SECURITY.md`](SECURITY.md).

## License

MIT - see [`LICENSE`](LICENSE).
