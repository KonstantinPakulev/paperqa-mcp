"""PaperQA MCP Server — Q&A over local PDF papers."""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from paperqa import Settings, ask
from paperqa.agents.search import FAILED_DOCUMENT_ADD_ID
from paperqa.agents.tools import get_directory_index
from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings

mcp = FastMCP("paperqa-mcp")


def _patch_litellm_for_minimax() -> None:
    """Wrap litellm.acompletion to fix MiniMax's non-standard tool call format.

    MiniMax returns tool calls as JSON inside `content` with `tool_calls=None`.
    This patch moves them into the proper `tool_calls` field before aviary sees them.
    """
    import litellm
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    _original = litellm.acompletion

    async def _patched(*args: Any, **kwargs: Any) -> Any:
        response = await _original(*args, **kwargs)
        for choice in response.choices:
            msg = choice.message
            if choice.finish_reason == "tool_calls" and not msg.tool_calls and msg.content:
                clean = re.sub(r"<think>.*?</think>\s*", "", msg.content, flags=re.DOTALL).strip()
                try:
                    data = json.loads(clean)
                    if isinstance(data, list) and data and "name" in data[0]:
                        msg.tool_calls = [
                            ChatCompletionMessageToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                type="function",
                                function=Function(
                                    name=tc["name"],
                                    arguments=json.dumps(tc.get("parameters", tc.get("arguments", {}))),
                                ),
                            )
                            for tc in data
                        ]
                        msg.content = None
                except (json.JSONDecodeError, KeyError):
                    pass
        return response

    litellm.acompletion = _patched


_patch_litellm_for_minimax()


def _format_exception(error: BaseException) -> str:
    messages: list[str] = []
    seen: set[str] = set()

    def collect(current: BaseException) -> None:
        if isinstance(current, BaseExceptionGroup):
            for child in current.exceptions:
                collect(child)

        cause = current.__cause__
        if cause is not None:
            collect(cause)
        elif not current.__suppress_context__ and current.__context__ is not None:
            collect(current.__context__)

        if isinstance(current, BaseExceptionGroup):
            return

        message = f"{type(current).__name__}: {current!s}"
        if message not in seen:
            seen.add(message)
            messages.append(message)

    collect(error)

    if not messages:
        return str(error)

    return " | ".join(messages)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _register_custom_models() -> None:
    """Register custom provider models in litellm's global model cost registry."""
    if not os.environ.get("PAPERQA_BASE_URL"):
        return
    import litellm

    llm = os.environ.get("PAPERQA_LLM", "")
    agent_llm = os.environ.get("PAPERQA_AGENT_LLM", llm)
    embedding = os.environ.get("PAPERQA_EMBEDDING", "")
    model_limits = {
        llm: {"max_input_tokens": 262144, "max_output_tokens": 8192, "max_tokens": 8192},
        agent_llm: {"max_input_tokens": 262144, "max_output_tokens": 8192, "max_tokens": 8192},
        embedding: {"max_input_tokens": 8192, "max_output_tokens": 0, "max_tokens": 8192},
    }
    for model, limits in model_limits.items():
        if model and model not in litellm.model_cost:
            litellm.model_cost[model] = {
                **limits,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }


def _get_settings() -> Settings:
    _register_custom_models()
    paper_dir = Path(os.environ["PAPER_DIR"]).resolve()
    index_dir = Path(os.environ.get("PAPERQA_INDEX_DIR", paper_dir / ".pqa")).resolve()

    # Read model configuration from environment variables with defaults
    llm = os.environ.get("PAPERQA_LLM", "claude-haiku-4-5-20251001")
    summary_llm = os.environ.get("PAPERQA_SUMMARY_LLM", llm)
    agent_llm = os.environ.get("PAPERQA_AGENT_LLM", llm)
    embedding = os.environ.get("PAPERQA_EMBEDDING", "openrouter/openai/text-embedding-3-small")
    chunk_chars = int(os.environ.get("PAPERQA_CHUNK_CHARS", 5000))

    # Configure custom LiteLLM providers if used
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
                index_directory=str(index_dir),
                concurrency=1,
            ),
        ),
        parsing=ParsingSettings(multimodal=False, reader_config={"chunk_chars": chunk_chars, "overlap": 250}),
    )


def _build_llm_config(model: str) -> dict | None:
    """Build LiteLLM config for non-default LLM providers."""
    if model.startswith("openai/") and os.environ.get("PAPERQA_BASE_URL"):
        api_key = os.environ.get("SKOLTECH_API_KEY")
        if not api_key:
            raise ValueError("SKOLTECH_API_KEY required when PAPERQA_BASE_URL is set.")
        litellm_params: dict[str, Any] = {
            "model": model,
            "api_base": os.environ["PAPERQA_BASE_URL"],
            "api_key": api_key,
            "timeout": 300,
        }
        extra_body_raw = os.environ.get("PAPERQA_LLM_EXTRA_BODY")
        if extra_body_raw:
            litellm_params["extra_body"] = json.loads(extra_body_raw)
        return {
            "model_list": [
                {
                    "model_name": model,
                    "litellm_params": litellm_params,
                    "model_info": {
                        "max_input_tokens": 262144,
                        "max_output_tokens": 8192,
                    },
                }
            ]
        }

    if model.startswith("openai/glm"):
        api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError(
                "ZHIPUAI_API_KEY environment variable required for z.ai models. "
                "Set ZHIPUAI_API_KEY in your environment or .mcp.json."
            )

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

    if model.startswith("openrouter/"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable required for OpenRouter models. "
                "Set OPENROUTER_API_KEY in your environment."
            )

        # LiteLLM has native OpenRouter support for chat models, so avoid forcing
        # a custom model_list route here. That preserves router model IDs such as
        # `openrouter/free` and lets LiteLLM select the correct provider behavior.
        return None

    return None


def _build_embedding_config(model: str) -> dict | None:
    """Build LiteLLM config for OpenRouter embeddings."""
    if model.startswith("openai/") and os.environ.get("PAPERQA_BASE_URL"):
        api_key = os.environ.get("SKOLTECH_API_KEY")
        if not api_key:
            raise ValueError("SKOLTECH_API_KEY required when PAPERQA_BASE_URL is set.")
        return {
            "model_list": [
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": model,
                        "api_base": os.environ["PAPERQA_BASE_URL"],
                        "api_key": api_key,
                    },
                }
            ],
        }

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
    except Exception as e:
        return {"error": f"Failed to initialize PaperQA settings: {_format_exception(e)}"}

    try:
        index = await get_directory_index(settings=settings, build=False)
    except RuntimeError:
        return {"error": "Index not built yet. Call index_papers first."}
    except Exception as e:
        return {"error": f"Failed to load index: {_format_exception(e)}"}

    index_files = await index.index_files
    failed_files = sorted(
        Path(file_path).name
        for file_path, file_hash in index_files.items()
        if file_hash == FAILED_DOCUMENT_ADD_ID
    )

    if failed_files:
        return {
            "error": (
                "Failed to answer question because the paper index is incomplete. "
                f"Failed files: {failed_files}. "
                "PaperQA caches failed files in the index, so fix the underlying "
                "provider/indexing error and rebuild the index before retrying."
            )
        }

    try:
        result = await ask(question, settings=settings)
    except Exception as e:
        return {"error": f"Failed to answer question: {_format_exception(e)}"}

    return {
        "question": result.session.question,
        "answer": _strip_thinking(result.session.formatted_answer),
    }


@mcp.tool()
async def index_papers() -> dict[str, Any]:
    """Index any new PDFs in the paper directory into the search index.

    Must be called after adding new papers before ask_question can find them.
    Requires the LLM provider to be available (used for citation extraction).
    """
    try:
        settings = _get_settings()
    except Exception as e:
        return {"error": f"Failed to initialize PaperQA settings: {_format_exception(e)}"}

    try:
        index = await get_directory_index(settings=settings)
    except Exception as e:
        return {"error": f"Failed to index papers: {_format_exception(e)}"}

    index_files = await index.index_files
    failed_files = sorted(
        Path(file_path).name
        for file_path, file_hash in index_files.items()
        if file_hash == FAILED_DOCUMENT_ADD_ID
    )
    indexed_count = len(index_files) - len(failed_files)

    result: dict[str, Any] = {"indexed_docs": indexed_count}
    if failed_files:
        result["failed_files"] = failed_files
        result["warning"] = (
            "Some files failed to index. Fix the underlying error and call "
            "index_papers again. Use scripts/paperqa/reindex_paper.py to retry a specific file."
        )
    return result


@mcp.tool()
async def list_papers() -> dict[str, Any]:
    """List PDF files in the paper directory and index status."""
    try:
        settings = _get_settings()
        paper_dir = Path(settings.agent.index.paper_directory)
        pdf_files = [f.name for f in sorted(paper_dir.glob("*.pdf"))]

        indexed_docs: int | str
        try:
            index = await get_directory_index(settings=settings, build=False)
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
        return {"error": f"Failed to list papers: {_format_exception(e)}"}


def main() -> None:
    """Entry point for paperqa-mcp."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
