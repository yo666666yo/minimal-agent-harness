# Security Policy

This project is an educational agent harness. It is not a sandbox.

## Important Notes

- `BashTool` executes local shell commands.
- `ReadTool` and `WriteTool` access the local filesystem.
- Tool inputs are assumed to be trusted.
- The harness does not isolate processes, users, networks, or directories.

Do not expose the built-in tools to untrusted users. Do not run the harness against
sensitive workspaces unless you have reviewed and restricted the tools.

## Reporting a Vulnerability

Please open a private security advisory on GitHub if available. If that is not
available, open an issue with minimal reproduction details and avoid including
secrets or exploit payloads.
