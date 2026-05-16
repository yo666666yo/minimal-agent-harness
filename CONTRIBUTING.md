# Contributing

Thanks for helping improve `minimal-agent-harness`.

The main rule is to keep the core readable. This repository is a reference
implementation, so clarity is more important than feature volume.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest -q
ruff check .
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## Good Contributions

- Focused tests for the agent loop, tool executor, or context compaction.
- Provider adapters under `examples/`, not inside the core file.
- Small tool improvements that do not hide the control flow.
- Documentation that helps a reader understand agent internals faster.

## Pull Request Checklist

- Keep `agent_harness.py` easy to read top-to-bottom.
- Add or update tests when behavior changes.
- Run `pytest -q`.
- Run `ruff check .`.
- Avoid adding heavyweight framework dependencies to the core harness.
