# Minimal Agent Harness

A **minimal, educational** implementation of an LLM agent harness in Python — a single-agent tool-use loop with streaming tool execution and context summarization. Built for beginners to understand how agentic coding assistants work under the hood.

## Architecture Overview

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

### 2. Streaming Tool Execution

Tools begin executing **during** model streaming, not after. Two classes of tools:

| Type | Examples | Behavior |
|------|----------|----------|
| **Concurrency-safe** | read_file, grep | Run in parallel with other safe tools |
| **Exclusive** | bash, write_file | Block all other tools, run serially |

Bash errors trigger **sibling cancellation** — if one bash command fails, all parallel bash commands are killed (since they often form dependency chains).

### 3. Context Summarization (Auto-Compact)

When the conversation exceeds a token threshold, older messages are summarized via a separate LLM call. The summary replaces the old history, and recent messages are preserved for continuity.

```
[msg1, msg2, ..., msg98, msg99, msg100]  (100 messages, ~8000 tokens)
              |
              v  summarize msg1..msg96
[summary, msg97, msg98, msg99, msg100]   (5 messages, ~1500 tokens)
```

## Design Patterns From Production Agents

This implementation distills patterns from production agentic coding assistants:

| Pattern | Implementation | Purpose |
|---------|---------------|---------|
| Async generator loop | `async for event in harness.run()` | Stream events to UI without blocking |
| Tool partitioning | `is_concurrency_safe` flag | Read-only tools parallelize, writes serialize |
| Sibling abort | `_sibling_abort` event | Bash error cascading (mkdir fails -> rest cancelled) |
| Content replacement budget | token threshold check | Keep context window manageable |
| Turn limit | `max_turns` config | Safety valve against infinite loops |
| Abort signal | `asyncio.Event` | User Ctrl+C graceful shutdown |

## Quick Start

```bash
# Install dependencies
pip install anthropic

# Mock mode (no API key needed — simulates tool calling)
python agent_harness.py --mock

# Real API mode
export ANTHROPIC_API_KEY=sk-ant-...
python agent_harness.py
```

### Example session (mock mode)

```
> read the file agent_harness.py
[Agent] Starting with 4 tools, max 10 turns
[Turn 1/10]
[Agent] Calling model...
[Tool call] read_file({"file_path": "agent_harness.py"})
[Tool result/OK] """
Minimal Agent Harness — a single-agent tool-use loop...

> search for 'Tool' in the codebase
[Turn 1/10]
[Agent] Calling model...
[Tool call] grep({"pattern": "Tool", "path": "."})
[Tool result/OK] ./agent_harness.py:7: Tool System...
```

### Programmatic usage

```python
import asyncio
from agent_harness import AgentHarness, AgentConfig, MockAPIClient

async def main():
    client = MockAPIClient()
    config = AgentConfig(max_turns=10)
    harness = AgentHarness(client, config)

    async for event in harness.run("read README.md"):
        print(event)

asyncio.run(main())
```

## Files

- `agent_harness.py` — Full implementation (~450 lines): Tool base class, concrete tools (Bash/Read/Write/Grep), streaming executor, query loop, context summarization, mock API client
- `requirements.txt` — Python dependencies

## Adding Custom Tools

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
