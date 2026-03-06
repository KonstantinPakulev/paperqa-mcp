"""PaperQA MCP Server — Q&A over local PDF papers."""

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from paperqa import Settings, ask
from paperqa.agents.tools import get_directory_index
from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

mcp = FastMCP("paperqa-mcp")


def _get_settings() -> Settings:
    paper_dir = Path(os.environ["PAPER_DIR"]).resolve()

    # Read model configuration from environment variables with defaults
    llm = os.environ.get("PAPERQA_LLM", "claude-haiku-4-5-20251001")
    summary_llm = os.environ.get("PAPERQA_SUMMARY_LLM", llm)
    agent_llm = os.environ.get("PAPERQA_AGENT_LLM", llm)
    embedding = os.environ.get("PAPERQA_EMBEDDING", "openrouter/openai/text-embedding-3-small")

    # Configure z.ai models if used
    llm_config = _build_llm_config(llm)
    summary_llm_config = _build_llm_config(summary_llm)
    agent_llm_config = _build_llm_config(agent_llm)
    embedding_config = _build_embedding_config(embedding)

    return Settings(
        llm=llm,
        llm_config=llm_config,
        summary_llm=summary_llm,
        summary_llm_config=summary_llm_config,
        embedding=embedding,
        embedding_config=embedding_config,
        agent=AgentSettings(
            agent_llm=agent_llm,
            agent_llm_config=agent_llm_config,
            index=IndexSettings(
                paper_directory=str(paper_dir),
                index_directory=str(paper_dir / ".pqa"),
            ),
        ),
        parsing=ParsingSettings(multimodal=False),
    )


def _build_llm_config(model: str) -> dict | None:
    """Build LiteLLM config for z.ai models."""
    if not model.startswith("openai/glm"):
        return None

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError(
            "ZHIPUAI_API_KEY environment variable required for z.ai models. "
            "Set ZHIPUAI_API_KEY in your environment or .mcp.json."
        )

    # Determine API base based on model type
    if "glm-4.6" in model:  # Coding specialist
        api_base = "https://open.bigmodel.cn/api/coding/paas/v4/"
    else:
        api_base = "https://open.bigmodel.cn/api/paas/v4/"

    return {
        "model_list": [
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": api_key,
                    "api_base": api_base,
                },
            }
        ]
    }


def _build_embedding_config(model: str) -> dict | None:
    """Build LiteLLM config for OpenRouter embeddings."""
    if not model.startswith("openrouter/"):
        return None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable required for OpenRouter models. "
            "Set OPENROUTER_API_KEY in your environment."
        )

    return {
        "model_list": [
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": api_key,
                    "api_base": "https://openrouter.ai/api/v1",
                },
            }
        ]
    }


@mcp.tool()
async def ask_question(question: str) -> dict[str, Any]:
    """Ask a question answered from the local paper library."""
    try:
        settings = _get_settings()
        await get_directory_index(settings=settings)
        result = await ask(question, settings=settings)
        return {
            "question": result.session.question,
            "answer": result.session.formatted_answer,
        }
    except Exception as e:
        return {"error": f"Failed to answer question: {e!s}"}


@mcp.tool()
async def list_papers() -> dict[str, Any]:
    """List PDF files in the paper directory and index status."""
    try:
        settings = _get_settings()
        paper_dir = Path(settings.agent.index.paper_directory)
        pdf_files = [f.name for f in sorted(paper_dir.glob("*.pdf"))]

        indexed_docs: int | str
        try:
            index = await get_directory_index(settings=settings)
            indexed_docs = len(await index.index_files)
        except Exception:
            indexed_docs = "unknown"

        return {
            "paper_directory": str(paper_dir),
            "pdf_files": pdf_files,
            "total_pdfs": len(pdf_files),
            "indexed_docs": indexed_docs,
        }
    except Exception as e:
        return {"error": f"Failed to list papers: {e!s}"}


def main() -> None:
    """Entry point for paperqa-mcp."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
