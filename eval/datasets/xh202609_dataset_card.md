# XH-202609 RAG Evaluation Dataset

## Summary

This dataset evaluates RAG behavior on the PDF:

`asset/first-test.pdf`

The PDF is the same file as:

`C:\Users\Admin\Downloads\XH-202609_具备自主决策能力的通用网络安全智能体.pdf`

The dataset is stored in:

`eval/datasets/xh202609_rag_eval.json`

It contains 32 questions:

- 30 answerable questions.
- 2 unanswerable/refusal questions.
- Coverage across facts, lists, summaries, timeline questions, table/rubric questions, multi-evidence questions, contacts, awards, and company profile.

## Why This Shape

The goal is not only to check whether the final answer looks right. Each item includes enough structure to score:

- retrieval quality: did the system retrieve the right chunk or section?
- answer quality: did it cover the expected answer points?
- citation quality: did citations support the key claims?
- reliability: did it avoid hallucination and refuse when the document lacks evidence?

## Source And Chunk Mapping

The dataset references the current project chunk ids from:

`artifacts/documents/77293340-41cd-494e-92ca-82b6c327d8ee/chunks/XH-202609_具备自主决策能力的通用网络安全智能体.chunks.json`

If the PDF is parsed or chunked again, chunk ids may change. In that case:

1. Keep `page`, `section`, and `quote` as stable evidence.
2. Regenerate `gold_context_ids` and `gold_source_chunk_ids`.
3. Keep the question ids stable so trend reports remain comparable.

## Scoring

Use the rubric:

`eval/rubrics/rag_score_v1.md`

Default answerable question weights:

```text
retrieval: 30
answer: 45
citation: 15
reliability: 10
```

Default unanswerable question weights:

```text
retrieval: 10
answer: 45
citation: 0
reliability: 45
```

## Suggested Report Sections

A final report generated from this dataset should include:

- overall score out of 100.
- score by question type.
- score by difficulty.
- retrieval metrics: Recall@3, Recall@5, Recall@10, MRR@10, nDCG@10.
- answer metrics: answer point coverage, unsupported claim rate, refusal accuracy.
- citation metrics: citation precision, citation recall, citation support rate.
- failure reason distribution.
- worst 10 questions with retrieved contexts, answer, expected points, and judge rationale.

## Recommended Baseline Run

Use the current project defaults first:

```text
retriever: qdrant_hybrid
embedding: BGE-M3 dense+sparse
reranker: noop
final_top_k: 5
answer_top_contexts: 5
```

Then compare one variable at a time:

- reranker: noop vs bge/qwen.
- final_top_k: 5 vs 10.
- answer_top_contexts: 5 vs 8.
- chunking parameters if re-indexing.
- prompt version after fixing any Chinese prompt encoding issues.

