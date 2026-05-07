<h1 align="center">🧠 minimal-agent-harness</h1>

<h3 align="center">The agentic coding loop, explained in ~450 lines.</h3>

<p align="center">
  <a href="https://github.com/yo666666yo/minimal-agent-harness/stargazers"><img src="https://github.com/yo666666yo/minimal-agent-harness/workflows/Stars/badge.svg?style=flat-square&color=yellow" alt="Stars"></a>
  <a href="https://github.com/yo666666yo/minimal-agent-harness/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python" alt="Python"></a>
  <a href="https://github.com/yo666666yo/minimal-agent-harness/releases"><img src="https://img.shields.io/badge/version-v1.0.0-green?style=flat-square" alt="Version"></a>
</p>

<p align="center">
  <b>Stop reading about agents. Read an agent.</b><br>
  A single Python file that distills how Claude Code, Cursor, and other<br>
  AI coding tools work under the hood — <b>streaming, tool use, auto-compact, and all.</b>
</p>

---

## Quick Start (2 minutes)

```bash
git clone https://github.com/yo666666yo/minimal-agent-harness.git
cd minimal-agent-harness
pip install anthropic
python agent_harness.py --mock    # no API key needed
```

You'll see the agent loop in action — calling tools, streaming responses, and managing context:

```
[Using mock API client — no real API calls]
============================================================
Minimal Agent Harness — interactive mode
Model: mock
Available tools: ['bash', 'read_file', 'write_file', 'grep']
============================================================

> Search for 'is_concurrency_safe' in the codebase
[Agent] Starting with 4 tools, max 10 turns

[Turn 1/10]
[Agent] Calling model...
  └─ [streaming] text_delta: "I'll search for that..."
  └─ [streaming] tool_use: grep({"pattern": "is_concurrency_safe", "path": "."})
[Tool call] grep({"pattern": "is_concurrency_safe", "path": "."})
[Tool result/OK] Found 3 matches in agent_harness.py

[Turn 2/10]
[Agent] Calling model...
  └─ [streaming] text_delta: "Found it in 3 places. Let me read the relevant section..."
  └─ [streaming] tool_use: read_file({"file_path": "agent_harness.py", "start_line": 45})
[Tool call] read_file({"file_path": "agent_harness.py", "start_line": 45})
[Tool result/OK] class StreamingToolExecutor: ...

[Agent] Done — 2 turns, 2 tool calls.
```

For a real model, just add your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python agent_harness.py
```

---

## What You'll Learn

This isn't a framework. It's a **reading guide to how production agents work** — implemented as a single, runnable Python file.

| Concept | Where in the code | Why it matters |
|---|---|---|
| **Async generator query loop** | `AgentHarness.run()` | The core pattern behind every AI coding assistant |
| **Streaming tool execution** | `StreamingToolExecutor` | Tools start *during* model streaming, not after — cuts latency in half |
| **Concurrent-safe vs exclusive tools** | `StreamingToolExecutor.add_tool()` | Read tools run in parallel; write tools lock exclusively |
| **Context summarization** | `summarize_conversation()` | How agents handle long conversations without losing state |
| **Turn limits & abort** | `AgentHarness.run()` | Graceful termination — the safety valve every agent needs |

**4 built-in tools** that mirror Claude Code's design:

| Tool | Concurrency | What it demonstrates |
|---|---|---|
| `bash` | ❌ Exclusive | Shell execution with error cascading |
| `read_file` | ✅ Safe | Parallel file reads with line ranges |
| `write_file` | ❌ Exclusive | Serial writes with locking |
| `grep` | ✅ Safe | Parallel codebase search |

---

## Architecture

### Four Design Decisions

Every production agent makes these same choices. This harness makes them explicit:

| Decision | Why |
|---|---|
| **Async generator as the control surface** | Yield events to the caller without blocking — CLI, TUI, or web UI all plug into the same stream. No callbacks, no polling. |
| **Tools execute during streaming, not after** | In Claude Code, `bash` starts running the moment the model emits the `tool_use` block. This cuts round-trip latency in half. |
| **Reads are parallel, writes are serial** | `read_file` and `grep` can run concurrently. `bash` and `write_file` lock out everything else — the same partitioning used in production. |
| **Summarize, don't truncate** | When context fills up, older messages are summarized via a separate LLM call instead of being dropped. Truncation loses state; summarization preserves it. |

### The Loop

```
User Input
  │
  ▼
AgentHarness.run()           ◄── async generator
  │
  ├──▶ Summarize?             ◄── auto-compact if context too long
  │
  ├──▶ Call Model (streaming) ◄── yields text_delta / tool_use events
  │      │
  │      ├──▶ Tool Use Block? ──▶ StreamingToolExecutor.add_tool()
  │      │                          │
  │      │                          ├──▶ Concurrent-safe tools: run in parallel
  │      │                          ├──▶ Write tools (bash/edit): serial, exclusive
  │      │                          └──▶ Bash error cascading: cancel siblings
  │      │
  │      └──▶ get_completed_results() ──▶ yield to user (non-blocking)
  │
  ├──▶ Wait for remaining tools  ◄── get_remaining_results()
  │
  ├──▶ Append assistant(tool_use) + user(tool_result) to messages
  │
  ├──▶ Check max_turns / abort
  │
  └──▶ Loop back to Summarize step
```

---

## Three Core Mechanisms

### 1. Query Loop (async generator `while True`)

The heart of the harness. Each iteration:
1. Calls the LLM with current messages + tool definitions
2. Model returns text (done) or `tool_use` blocks (continue)
3. Tools execute, results feed back as user messages
4. Loop continues until no more tool calls or max turns reached

```
messages = [user: "read the file"]
  → model returns tool_use: read_file("foo.py")
  → tool executes, returns content
  → messages = [user: "...", assistant: tool_use, user: tool_result]
  → model returns text: "The file contains..."
  → no tool_use → DONE
```

> **Why it matters:** The `while True` + async generator combination is what lets the host
> application (CLI, IDE, web UI) stay responsive during tool execution. Every AI coding
> assistant you've used works this way.

### 2. Streaming Tool Execution

Tools begin executing **during** model streaming, not after. Two classes of tools:

| Type | Examples | Behavior |
|------|----------|----------|
| **Concurrency-safe** | `read_file`, `grep` | Run in parallel with other safe tools |
| **Exclusive** | `bash`, `write_file` | Block all other tools, run serially |

Bash errors trigger **sibling cancellation** — if one bash command fails, all parallel
bash commands are killed (since they often form dependency chains).

> **Why it matters:** In naive agents, you wait for the model to finish before running
> tools. That adds round-trip latency. Production agents start tool execution mid-stream —
> a `bash` command launches while the model is still deciding whether to call `grep` next.

### 3. Context Summarization (Auto-Compact)

When the conversation exceeds a token threshold, older messages are summarized via a
separate LLM call. The summary replaces the old history, and recent messages are
preserved for continuity.

```
[msg1, msg2, ..., msg98, msg99, msg100]  (100 messages, ~8000 tokens)
              │
              ▼  summarize msg1..msg96
[summary, msg97, msg98, msg99, msg100]   (5 messages, ~1500 tokens)
```

> **Why it matters:** Long conversations overflow the context window. Dropping old messages
> loses state (what file was opened? what search was running?). Summarizing preserves the
> narrative while freeing space — it's how agents maintain coherence in long sessions.

---

## Source Code Map

The entire agent lives in [`agent_harness.py`](agent_harness.py) (~450 lines). Here's where to look:

| Lines | Component | What it does |
|---:|---|---|
| 1–60 | `Tool` base class + built-in tools | Tool interface: `name`, `description`, `input_schema`, `call()` |
| 60–120 | `StreamingToolExecutor` | Concurrent vs exclusive tool dispatch, error cascading |
| 120–200 | `AgentConfig` + `AgentHarness.__init__` | Configuration: model, max turns, tools, auto-compact threshold |
| 200–350 | `AgentHarness.run()` | **The main loop** — streaming, tool dispatch, turn management |
| 350–420 | `summarize_conversation()` | Context window management via LLM summarization |
| 420–450 | `__main__` | CLI entry point with mock/real mode toggle |

Start at `AgentHarness.run()` and follow the calls. It reads top-to-bottom in under 20 minutes.

---

## Adding Custom Tools

Every tool in Claude Code started as a subclass like this one. Read, write, grep, bash —
same pattern.

```python
class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"],
    }
    is_concurrency_safe = True  # Read-only

    async def call(self, input, abort_signal):
        # Your implementation here
        return f"Results for: {input['query']}"

# Register it
config = AgentConfig(tools=DEFAULT_TOOLS + [WebSearchTool()])
```

---

## Programmatic Usage

```python
import asyncio
from agent_harness import AgentHarness, AgentConfig, MockAPIClient

async def main():
    client = MockAPIClient()
    config = AgentConfig(max_turns=10)
    harness = AgentHarness(client, config)

    async for event in harness.run("Search for tool_use in the codebase"):
        print(event)

asyncio.run(main())
```

---

## When to Use This

| You want to... | This repo | LangChain | AutoGen | BabyAGI |
|---|---|---|---|---|
| **Read and understand agent internals** | ✅ One file | ❌ Heavy abstraction | ⚠️ More complex | ✅ Also small |
| **Build a production agent** | ❌ | ✅ | ✅ | ⚠️ |
| **Experiment with tool execution patterns** | ✅ | Overkill | Overkill | Overkill |
| **Learn async Python patterns for agents** | ✅ | ❌ Too many layers | ❌ | ⚠️ |
| **Get something working in 5 minutes** | ✅ Mock mode | ❌ Steep curve | ⚠️ | ✅ |

**The TL;DR:** This is a textbook, not a framework. Read it to understand how agents work,
then use LangChain or AutoGen to build something real.

---

## FAQ

<details>
<summary><b>Why Python? Claude Code is written in TypeScript.</b></summary>

Python has lower syntax overhead for an educational project — no compile step, no
`node_modules`. Python's `async for` / `asyncio` primitives map cleanly to the
generator-based query loop. The concepts transfer directly; you'll recognize the
same patterns in the TypeScript source.
</details>

<details>
<summary><b>Can I use this with OpenAI models instead of Anthropic?</b></summary>

Not out of the box, but it's designed to be easy. The `APIClient` base class is an
abstract interface. Write an `OpenAIClient` that implements `stream_message()` and
converts OpenAI's tool-calling format to the same event format (`text_delta`,
`tool_use`, `message_stop`). The rest of the harness doesn't care which provider
is behind it.
</details>

<details>
<summary><b>Is this production-ready?</b></summary>

No. Deliberately not. There's no error recovery, no retry logic, no rate limiting,
no authentication beyond an API key, and the summarization makes a standalone call
without prompt cache reuse. Use it to learn the patterns, then apply them in a
production framework.
</details>

<details>
<summary><b>How does auto-compact decide when to summarize?</b></summary>

It estimates token count from message length: `len(json.dumps(messages)) // 4`.
When this exceeds `auto_compact_threshold_tokens` (default: 4000), it summarizes
all but the last ~4 messages into a single paragraph using a separate LLM call,
then replaces the old messages with the summary. The code is in
`summarize_conversation()` — about 40 lines.
</details>

<details>
<summary><b>How does this compare to Claude Code's actual source?</b></summary>

The code comments cite exact files and line numbers from the Claude Code source:
`src/query.ts` (the loop), `src/QueryEngine.ts` (state), `src/services/tools/StreamingToolExecutor.ts`
(tool dispatch), `src/services/compact/compact.ts` (summarization). This harness preserves
every design pattern while stripping away production concerns (auth, multi-model routing,
IDE integration). Think of it as a clean-room reimplementation for learning.
</details>

---

## Next Steps

<table>
<tr>
<td width="50%">

### 🔍 Read the Source
The entire agent is in [`agent_harness.py`](agent_harness.py).
Start at `AgentHarness.run()` (the query loop) and follow the calls.
Use the [source code map](#source-code-map) above to navigate.

</td>
<td width="50%">

### 🔧 Build a Tool
Subclass `Tool`, implement `call()`, register it with `AgentConfig`.
Try a web search, a database query, or a Jira ticket lookup.
The [custom tools section](#adding-custom-tools) has a template.

</td>
</tr>
<tr>
<td>

### 📖 Read the References
The code comments cite exact files and line numbers from the
Claude Code source. Trace them: `src/query.ts`, `src/QueryEngine.ts`,
`src/services/tools/`.

</td>
<td>

### 🤝 Contribute
Found a bug? Have an idea for a better explanation?
Open an issue or PR. This is a teaching tool — clarity
improvements are always welcome.

</td>
</tr>
<tr>
<td colspan="2" align="center">

### ⭐ Star the Repo
If this helped you understand how agents work, [give it a star](https://github.com/yo666666yo/minimal-agent-harness) — it helps other developers find it.

</td>
</tr>
</table>

---

<p align="center">
  <sub>
    Built with Python and curiosity |
    <a href="https://github.com/yo666666yo/minimal-agent-harness/blob/master/LICENSE">MIT License</a> |
    <a href="https://github.com/yo666666yo">@yo666666yo</a>
  </sub>
</p>
