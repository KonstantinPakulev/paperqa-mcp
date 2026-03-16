# paperqa-mcp

MCP server wrapping [paperqa v5](https://github.com/Future-House/paper-qa) for semantic Q&A over local PDF papers.

## Tools

- `ask_question(question)` — Answer a research question from indexed PDFs, with citations.
- `list_papers()` — List PDFs in the paper directory and report index status.

## Setup

Requires Python 3.11+ and `uv`.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `PAPER_DIR` | Absolute path to directory containing PDFs | *Required* |
| `PAPERQA_LLM` | Main LLM model | `claude-haiku-4-5-20251001` |
| `PAPERQA_SUMMARY_LLM` | Summary LLM model | Same as `PAPERQA_LLM` |
| `PAPERQA_AGENT_LLM` | Agent LLM model | Same as `PAPERQA_LLM` |
| `PAPERQA_EMBEDDING` | Embedding model | `openrouter/openai/text-embedding-3-small` |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude models) | *Required for Claude* |
| `OPENROUTER_API_KEY` | OpenRouter API key (for OpenRouter LLMs and embeddings) | *Required for OpenRouter* |
| `ZHIPUAI_API_KEY` | Zhipu AI API key (for z.ai/GLM models) | *Required for z.ai models* |

### Supported models

#### Anthropic Claude models
- `claude-haiku-4-5-20251001` (default)
- `claude-opus-4-6`
- `claude-sonnet-4-6`

#### z.ai (Zhipu AI) models
- `openai/glm-4` - Latest, complex tasks
- `openai/glm-4.7` - Enhanced coding, long tasks
- `openai/glm-4.7-FlashX` - Lightweight, fast
- `openai/glm-4.6` - Coding specialist (uses coding API endpoint)
- `openai/glm-4.5-air` - Cost-effective

**Note**: z.ai models require `ZHIPUAI_API_KEY` to be set. The server automatically uses the appropriate API endpoint (`/api/paas/v4/` or `/api/coding/paas/v4/`) based on the model.

#### OpenRouter models
- `openrouter/free` - OpenRouter free-model router
- `openrouter/<model-id>` - Any direct OpenRouter model ID, including `:free` variants

**Note**: OpenRouter models require `OPENROUTER_API_KEY`. Chat-model support is built into this MCP wrapper, but PaperQA depends on agent/tool-calling behavior and some free OpenRouter routes may still be incompatible with the current `paperqa`/LiteLLM stack. Treat free variants as experimental and test them before making them your default.

### Running

```bash
uv run --python 3.11 --project /path/to/paperqa-mcp paperqa-mcp
```

### MCP configuration (`.mcp.json`)

#### Default configuration (Claude Haiku)

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

#### Using z.ai GLM-4.6 (coding specialist)

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
    "PAPER_DIR": "/path/to/paper/references",
    "PAPERQA_LLM": "openai/glm-4.6",
    "PAPERQA_SUMMARY_LLM": "openai/glm-4.6",
    "PAPERQA_AGENT_LLM": "openai/glm-4.6"
  }
}
```

#### Using OpenRouter free models

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
    "PAPER_DIR": "/path/to/paper/references",
    "PAPERQA_LLM": "openrouter/free",
    "PAPERQA_SUMMARY_LLM": "openrouter/free",
    "PAPERQA_AGENT_LLM": "openrouter/free",
    "PAPERQA_EMBEDDING": "openrouter/openai/text-embedding-3-small"
  }
}
```

If `ask_question()` fails with OpenRouter-specific `tool_choice`, `tool use`, or provider-routing errors, switch to a different OpenRouter model or fall back to a non-free model/provider for the agent LLM.

#### Hybrid configuration (different LLMs for different tasks)

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
    "PAPER_DIR": "/path/to/paper/references",
    "PAPERQA_LLM": "openai/glm-4",
    "PAPERQA_SUMMARY_LLM": "claude-haiku-4-5-20251001",
    "PAPERQA_AGENT_LLM": "openai/glm-4.6"
  }
}
```

### API key management

**Important**: The `.mcp.json` file does NOT support shell variable expansion (e.g., `$API_KEY`). API keys must be inherited from the shell environment.

#### Setting API keys

You can set API keys in one of these ways:

1. **Shell profile** (recommended for persistent setup):
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   export ZHIPUAI_API_KEY="your-zhipuai-key"
   ```

2. **Project `.env` file** (for Docker-based workflows):
   ```bash
   # Add to .env (gitignored)
   ANTHROPIC_API_KEY=your-anthropic-key
   OPENROUTER_API_KEY=your-openrouter-key
   ZHIPUAI_API_KEY=your-zhipuai-key
   ```

3. **Session environment** (temporary):
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-key" && claude-code
   ```

The MCP server inherits these environment variables automatically when launched.
