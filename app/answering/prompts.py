from __future__ import annotations

from app.answering.context_packer import PackedContext


class PromptBuilder:
    def __init__(
        self,
        *,
        language: str = "zh",
        no_answer_text: str = "检索到的资料不足以回答该问题。",
    ) -> None:
        self.language = language
        self.no_answer_text = no_answer_text

    def build(self, *, query: str, contexts: list[PackedContext]) -> str:
        context_block = "\n\n".join(_format_context(context) for context in contexts)
        answer_language = "中文" if self.language == "zh" else self.language
        return (
            "你是一个严谨的文档问答助手。\n"
            "请只基于给定的 Context 回答用户问题，不要使用外部知识。\n"
            f"请使用{answer_language}回答。\n"
            "如果 Context 中没有足够依据，请直接回答："
            f"{self.no_answer_text}\n"
            "回答中的每个关键结论后必须标注引用，格式如 [Context 1]。\n"
            "不要编造不存在的分类、数字、章节、结论或引用。\n\n"
            "用户问题：\n"
            f"{query}\n\n"
            "可用 Context：\n"
            f"{context_block}\n\n"
            "请给出最终答案："
        )


def _format_context(context: PackedContext) -> str:
    section = " > ".join(context.section_path) if context.section_path else "N/A"
    return (
        f"[{context.label}]\n"
        f"context_id: {context.context_id}\n"
        f"source_chunk_id: {context.source_chunk_id}\n"
        f"section: {section}\n"
        f"score: {context.score:.6f}\n"
        "text:\n"
        f"{context.text}"
    )
