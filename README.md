# paperqa-mcp

MCP server wrapping [paperqa v5](https://github.com/Future-House/paper-qa) for semantic Q&A over local PDF papers.

## Tools

- `ask_question(question)` — Answer a research question from indexed PDFs, with citations.
- `list_papers()` — List PDFs in the paper directory and report index status.

## Setup

Requires Python 3.11+ and `uv`.

### Environment variables

| Variable | Purpose |
|---|---|
| `PAPER_DIR` | Absolute path to directory containing PDFs |
| `ANTHROPIC_API_KEY` | LLM authentication (Haiku) |
| `OPENROUTER_API_KEY` | Embedding authentication (text-embedding-3-small) |

### Running

```bash
uv run --python 3.11 --project /path/to/paperqa-mcp paperqa-mcp
```

### MCP configuration (`.mcp.json`)

```json
"paperqa": {
  "command": "uv",
  "args": [
    "run",
    "--python", "3.11",
    "--project", "/path/to/paperqa-mcp",
    "paperqa-mcp"
  ],
  "env": {
    "PAPER_DIR": "/path/to/paper/references"
  }
}
```

`ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` are inherited from the shell environment.
