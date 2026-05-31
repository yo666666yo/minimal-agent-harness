# Changelog

## v2.0.0 - 2026-05-31

- Added structured `AgentEvent` output with `render_event()` for display text.
- Added bidirectional `AgentCommand` support for tool approvals and aborts.
- Added in-memory `AgentSession`, `ToolContext`, `ToolCapabilities`, `ToolPolicy`,
  hooks, workspace guards, and per-run abort semantics.
- Updated README and examples for the new event/control API.
- Added examples for approval policy, JSONL transcript hooks, and MCP-style tool
  adapters.
- Breaking: `AgentHarness.run()` now yields `AgentEvent` objects instead of
  strings.
- Breaking: custom tools now implement `call(input, context)` instead of
  `call(input, abort_signal)`.

## v1.0.0 - 2026-05-04

- Reworked the README around the project's core value: a readable agent loop.
- Added examples for custom tools and an OpenAI provider adapter.
- Added focused tests for the agent loop, file reading, tool concurrency, and compaction.
- Added GitHub Actions CI with pytest and ruff.
- Added contribution and security documentation.
