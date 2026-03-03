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
    return Settings(
        llm="claude-haiku-4-5-20251001",
        summary_llm="claude-haiku-4-5-20251001",
        embedding="openrouter/openai/text-embedding-3-small",
        agent=AgentSettings(
            agent_llm="claude-haiku-4-5-20251001",
            index=IndexSettings(
                paper_directory=str(paper_dir),
                index_directory=str(paper_dir / ".pqa"),
            ),
        ),
        parsing=ParsingSettings(multimodal=False),
    )


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
