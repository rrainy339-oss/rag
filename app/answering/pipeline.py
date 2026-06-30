from __future__ import annotations

import re

from app.answering.context_packer import ContextPacker, PackedContext
from app.answering.prompts import PromptBuilder
from app.answering.providers import LLMProvider
from app.answering.schemas import AnswerCitation, AnswerResult, ConfidenceLevel
from app.retrieval.schemas import RetrievalResponse


class AnswerPipeline:
    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        context_packer: ContextPacker | None = None,
        prompt_builder: PromptBuilder | None = None,
        include_prompt: bool = False,
        no_answer_text: str = "检索到的资料不足以回答该问题。",
    ) -> None:
        self.llm_provider = llm_provider
        self.context_packer = context_packer or ContextPacker()
        self.prompt_builder = prompt_builder or PromptBuilder(no_answer_text=no_answer_text)
        self.include_prompt = include_prompt
        self.no_answer_text = no_answer_text

    def answer(
        self,
        retrieval_response: RetrievalResponse,
        *,
        query: str | None = None,
    ) -> AnswerResult:
        resolved_query = query or retrieval_response.query
        contexts = self.context_packer.pack(retrieval_response)
        if not contexts:
            return AnswerResult(
                query=resolved_query,
                answer=self.no_answer_text,
                citations=[],
                confidence=ConfidenceLevel.LOW,
                used_context_ids=[],
                warnings=["no_contexts"],
                provider=self.llm_provider.provider_name,
                model=self.llm_provider.model_name,
                prompt=None,
            )

        prompt = self.prompt_builder.build(query=resolved_query, contexts=contexts)
        llm_response = self.llm_provider.generate(prompt)
        cited_contexts = _extract_cited_contexts(llm_response.text, contexts)
        warnings: list[str] = []
        if not cited_contexts:
            warnings.append("answer_has_no_context_citations")

        answer_text = llm_response.text or self.no_answer_text
        confidence = _confidence(answer_text, cited_contexts, self.no_answer_text)
        citations = [_to_citation(context) for context in cited_contexts]
        return AnswerResult(
            query=resolved_query,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            used_context_ids=[context.context_id for context in cited_contexts],
            warnings=warnings,
            provider=llm_response.provider,
            model=llm_response.model,
            prompt=prompt if self.include_prompt else None,
            generation_metadata=llm_response.metadata,
        )


def _extract_cited_contexts(
    answer: str,
    contexts: list[PackedContext],
) -> list[PackedContext]:
    by_number = {index: context for index, context in enumerate(contexts, start=1)}
    cited_numbers: list[int] = []
    patterns = [
        r"\[Context\s*(\d+)\]",
        r"\[上下文\s*(\d+)\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
            cited_numbers.append(int(match.group(1)))

    cited: list[PackedContext] = []
    seen: set[int] = set()
    for number in cited_numbers:
        if number in seen:
            continue
        context = by_number.get(number)
        if context is None:
            continue
        cited.append(context)
        seen.add(number)
    return cited


def _to_citation(context: PackedContext) -> AnswerCitation:
    return AnswerCitation(
        context_id=context.context_id,
        source_chunk_id=context.source_chunk_id,
        parent_chunk_id=context.parent_chunk_id,
        label=context.label,
        quote=_quote(context.text),
        section_path=context.section_path,
        score=context.score,
        metadata=context.metadata,
    )


def _quote(text: str, max_chars: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12].rstrip() + " ...[truncated]"


def _confidence(
    answer: str,
    cited_contexts: list[PackedContext],
    no_answer_text: str,
) -> ConfidenceLevel:
    if no_answer_text in answer:
        return ConfidenceLevel.LOW
    if len(cited_contexts) >= 2:
        return ConfidenceLevel.HIGH
    if len(cited_contexts) == 1:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
