from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.answering import (
    AnswerPipeline,
    ContextPacker,
    MockLLMProvider,
    OpenAICompatibleProvider,
    PromptBuilder,
)
from app.retrieval.schemas import RetrievalResponse


def main() -> int:
    args = _parse_args()
    retrieval_response = RetrievalResponse.model_validate_json(
        args.retrieval.read_text(encoding="utf-8")
    )
    provider = _build_provider(args)
    pipeline = AnswerPipeline(
        llm_provider=provider,
        context_packer=ContextPacker(
            top_contexts=args.top_contexts,
            max_context_chars=args.max_context_chars,
            max_chars_per_context=args.max_chars_per_context,
        ),
        prompt_builder=PromptBuilder(language=args.language),
        include_prompt=args.include_prompt,
    )
    result = pipeline.answer(
        retrieval_response,
        query=args.query or retrieval_response.query,
    )

    output_json = result.to_json()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output_json)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_to_markdown(result), encoding="utf-8")
        print(f"wrote {args.markdown_output}")
    return 0


def _build_provider(args: argparse.Namespace) -> object:
    if args.llm_provider == "mock":
        return MockLLMProvider(args.mock_response)
    if args.llm_provider == "openai-compatible":
        api_key = args.api_key
        if api_key is None and args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
        return OpenAICompatibleProvider(
            base_url=args.base_url,
            model_name=args.model,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    raise ValueError(f"Unsupported LLM provider: {args.llm_provider}")


def _to_markdown(result: object) -> str:
    answer = result.answer
    citations = "\n".join(
        f"- {citation.label}: `{citation.context_id}` / `{citation.source_chunk_id}`"
        for citation in result.citations
    )
    warnings = "\n".join(f"- {warning}" for warning in result.warnings)
    return (
        f"# Answer\n\n{answer}\n\n"
        f"## Citations\n\n{citations or '- None'}\n\n"
        f"## Metadata\n\n"
        f"- confidence: `{result.confidence.value}`\n"
        f"- provider: `{result.provider}`\n"
        f"- model: `{result.model}`\n"
        f"- warnings:\n{warnings or '- None'}\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cited answer from a retrieval response JSON."
    )
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--query", help="Override the query stored in retrieval JSON.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--top-contexts", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-chars-per-context", type=int, default=3000)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--include-prompt", action="store_true")

    parser.add_argument(
        "--llm-provider",
        choices=["openai-compatible", "mock"],
        default="openai-compatible",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--mock-response",
        default="这是一个 mock answer。[Context 1]",
        help="Used only with --llm-provider mock.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
