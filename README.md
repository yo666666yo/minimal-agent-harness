<p align="center">
  <pre>
 ███ ███ ███ ███ ███ █   █ ███ █   █ ███ █   █ ███ ███ 
  █  █ █  █   █   █ █  ██ ██  █  ██  █ █ █ ██  █  █  
  █  ███  ███ █   ███  █ █ █  █  █ █ █ █   █ █ █ ███ ███ 
  █  █ █  █   █   █ █  █   █  █  █  ██ █   ██  █ █    █ 
  █  █ █  ███ ███ █ █  █   █ ███ █   █ ███ █   █ ███ ███ 
  </pre>
</p>

<h3 align="center">The agentic coding loop, explained in ~450 lines.</h3>

<p align="center">
  <a href="https://github.com/yo666666yo/minimal-agent-harness/stargazers"><img src="https://img.shields.io/github/stars/yo666666yo/minimal-agent-harness?style=flat-square&color=yellow" alt="Stars"></a>
  <a href="https://github.com/yo666666yo/minimal-agent-harness/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python" alt="Python"></a>
  <a href="https://github.com/yo666666yo/minimal-agent-harness/releases"><img src="https://img.shields.io/badge/version-v1.0.0-green?style=flat-square" alt="Version"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/quick_start-2_minutes-orange?style=flat-square" alt="Quick Start"></a>
</p>

<p align="center">
  <b>Stop reading about agents. Read an agent.</b><br>
  A single Python file that distills how Claude Code, Cursor, and other<br>
  AI coding tools work under the hood — <b>streaming, tool use, auto-compact, and all.</b>
</p>

---

## Demo

<p align="center">
  <i>A 15-second terminal demo belongs here. Coming soon.<br>
  Record with <a href="https://github.com/charmbracelet/vhs">vhs</a> or <a href="https://asciinema.org">asciinema</a>.</i>
</p>

```bash
# Try it yourself right now — mock mode, no API key needed:
python agent_harness.py --mock
> Search the codebase for 'is_concurrency_safe'
```

---

## Why This Exists

I use Claude Code every day. It reads files, writes files, runs bash, and searches my
codebase — hundreds of times per session. One day I wondered: **what does that loop
actually look like under the hood?**

Turns out, the core architecture is elegant and small. But to understand it, I had to
trace through thousands of lines of TypeScript across dozens of files. This project
is the result of that digging — a single Python file that preserves every important
design pattern while stripping away everything inessential.

**You'll find this useful if:**

- You use AI coding tools daily and want to understand how they work internally
- You're building your own agent and want a reference implementation to study
- You've read _about_ agents but never _read_ an agent's source code
- You want to experiment with tool design, streaming execution, or context management

**This is NOT:**

- A production framework (use [LangChain](https://github.com/langchain-ai/langchain) or [AutoGen](https://github.com/microsoft/autogen))
- An autonomous agent benchmark (see [BabyAGI](https://github.com/yoheinakajima/babyagi) or [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT))
- A multi-agent orchestration system
- Something you should wrap in Docker and deploy to prod

---

## Architecture

### Four Design Decisions

The entire harness flows from four choices, each borrowed from production agents:

| Decision | Why |
|---|---|
| **Async generator as the control surface** | Yield events to the caller without blocking — CLI, TUI, or web UI all plug into the same stream. No callbacks, no polling. |
| **Tools execute during streaming, not after** | In Claude Code, `bash` starts running the moment the model emits the tool_use block. This cuts round-trip latency in half. |
| **Reads are parallel, writes are serial** | `read_file` and `grep` can run concurrently. `bash` and `write_file` lock out everything else — the same partitioning used in production. |
| **Summarize, don't truncate** | When context fills up, older messages are summarized via a separate LLM call instead of being dropped. Truncation loses state; summarization preserves it. |

### The Loop

```
User Input
  |
  v
AgentHarness.run()           <-- async generator
  |
  +--> Summarize?             <-- auto-compact if context too long
  |
  +--> Call Model (streaming) <-- yields text_delta / tool_use events
  |      |
  |      +--> Tool Use Block? --> StreamingToolExecutor.add_tool()
  |      |                          |
  |      |                          +--> Concurrent-safe tools: run in parallel
  |      |                          +--> Write tools (bash/edit): serial, exclusive
  |      |                          +--> Bash error cascading: cancel siblings
  |      |
  |      +--> get_completed_results() --> yield to user (non-blocking)
  |
  +--> Wait for remaining tools  <-- get_remaining_results()
  |
  +--> Append assistant(tool_use) + user(tool_result) to messages
  |
  +--> Check max_turns / abort
  |
  +--> Loop back to Summarize step
```

---

## Quick Start

### Step 1 — Clone and run (no API key needed)

```bash
git clone https://github.com/yo666666yo/minimal-agent-harness.git
cd minimal-agent-harness
pip install anthropic
python agent_harness.py --mock
```

You'll see:

```
[Using mock API client — no real API calls]
============================================================
Minimal Agent Harness — interactive mode
Model: mock
Available tools: ['bash', 'read_file', 'write_file', 'grep']
Type 'quit' to exit, 'abort' to stop the current turn.
============================================================

> Read the README file
[Agent] Starting with 4 tools, max 10 turns

[Turn 1/10]
[Agent] Calling model...
[Tool call] read_file({"file_path": "README.md"})
[Tool result/OK] # Minimal Agent Harness ...
```

### Step 2 — Use a real model (optional)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python agent_harness.py
```

### Programmatic usage

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

## Three Core Mechanisms

### 1. Query Loop (async generator `while True`)

The heart of the harness. Each iteration:
1. Calls the LLM with current messages + tool definitions
2. Model returns text (done) or tool_use blocks (continue)
3. Tools execute, results feed back as user messages
4. Loop continues until no more tool calls or max turns reached

```
messages = [user: "read the file"]
  -> model returns tool_use: read_file("foo.py")
  -> tool executes, returns content
  -> messages = [user: "...", assistant: tool_use, user: tool_result]
  -> model returns text: "The file contains..."
  -> no tool_use -> DONE
```

> **Why it matters:** The `while True` + async generator combination is what lets the host
> application (CLI, IDE, web UI) stay responsive during tool execution. Every AI coding
> assistant you've used works this way.

### 2. Streaming Tool Execution

Tools begin executing **during** model streaming, not after. Two classes of tools:

| Type | Examples | Behavior |
|------|----------|----------|
| **Concurrency-safe** | read_file, grep | Run in parallel with other safe tools |
| **Exclusive** | bash, write_file | Block all other tools, run serially |

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
              |
              v  summarize msg1..msg96
[summary, msg97, msg98, msg99, msg100]   (5 messages, ~1500 tokens)
```

> **Why it matters:** Long conversations overflow the context window. Dropping old messages
> loses state (what file was opened? what search was running?). Summarizing preserves the
> narrative while freeing space — it's how agents maintain coherence in long sessions.

---

## When to Use This

| You want to... | This repo | LangChain | AutoGen | BabyAGI |
|---|---|---|---|---|
| **Read and understand agent internals** | Yes — it's one file | No — heavy abstraction | Maybe — more complex | Yes — also small |
| **Build a production agent** | No | Yes | Yes | Maybe |
| **Experiment with tool execution patterns** | Yes | Overkill | Overkill | Overkill |
| **Learn async Python patterns for agents** | Yes | No — too many layers | No | Maybe |
| **Get something working in 5 minutes** | Yes — mock mode | No — steep curve | Maybe | Yes |

**The TL;DR:** This is a textbook, not a framework. Read it to understand how agents work,
then use LangChain or AutoGen to build something real.

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

---

## Next Steps

<table>
<tr>
<td width="50%">

### Read the Source
The entire agent is in [`agent_harness.py`](agent_harness.py).
Start at `AgentHarness.run()` (the query loop) and follow the calls.
It reads top-to-bottom in under 20 minutes.

</td>
<td width="50%">

### Build a Tool
Subclass `Tool`, implement `call()`, register it with `AgentConfig`.
Try a web search, a database query, or a Jira ticket lookup.
The [custom tools section](#adding-custom-tools) has a template.

</td>
</tr>
<tr>
<td>

### Read the References
The code comments cite exact files and line numbers from the
Claude Code source. Trace them: `src/query.ts`, `src/QueryEngine.ts`,
`src/services/tools/`.

</td>
<td>

### Contribute
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
