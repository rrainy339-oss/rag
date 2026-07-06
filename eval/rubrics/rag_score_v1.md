# RAG Evaluation Scoring v1

This rubric matches `eval/datasets/xh202609_rag_eval.json`.

## Input Contract

Each dataset item provides:

- `query`: user question to send to the RAG system.
- `should_answer`: whether the document contains enough evidence to answer.
- `expected_points`: atomic answer points, each with `weight` and optional `evidence_ids`.
- `evidence`: gold evidence with `page`, `section`, `quote`, `gold_context_ids`, and `gold_source_chunk_ids`.
- `score_profile`: score weights. Use `answerable_v1` for normal questions and `unanswerable_v1` for refusal questions.
- `forbidden_claims`: claims that must not appear in an answer, mainly for refusal items.

The RAG run should save, per item:

- retrieved `candidates` with rank and `chunk_id`.
- final `contexts` with rank, `context_id`, `source_chunk_id`, and `section_path`.
- generated `answer`.
- generated `citations`, ideally with cited `context_id` or `source_chunk_id`.
- latency and runtime config.

## Retrieval Metrics

For each answerable item, evaluate every evidence object independently.

Gold hit:

- Candidate hit: returned candidate `chunk_id` is in `gold_source_chunk_ids`.
- Context hit: returned context `context_id` is in `gold_context_ids`, or returned context `source_chunk_id` is in `gold_source_chunk_ids`.
- Section hit: returned context `section_path` contains the evidence `section`, used only as a weak fallback.

Evidence hit value:

```text
1.00 if gold hit rank <= 3
0.85 if gold hit rank <= 5
0.67 if gold hit rank <= 10
0.33 if only section hit rank <= 10
0.00 otherwise
```

Retrieval score:

```text
retrieval_score = retrieval_weight * average(evidence_hit_value)
```

Report these common metrics:

```text
Recall@3  = evidence with gold hit rank <= 3 / total evidence
Recall@5  = evidence with gold hit rank <= 5 / total evidence
Recall@10 = evidence with gold hit rank <= 10 / total evidence
MRR@10    = average(1 / first_gold_rank) for first gold rank <= 10
nDCG@10   = DCG over binary gold relevance / ideal DCG
ContextPrecision@5 = gold contexts in top 5 / returned top 5 contexts
GoldSectionHitRate = evidence with section hit rank <= 10 / total evidence
```

For unanswerable items, retrieval is diagnostic. Award the retrieval component if the system does not treat unrelated retrieved context as direct proof for a forbidden claim.

## Answer Metrics

Use deterministic checks where possible, and an LLM judge for semantic coverage.

For each expected point:

```text
1.0 = fully covered and faithful
0.5 = partially covered or slightly underspecified
0.0 = missing or contradicted
```

Answer score:

```text
answer_score = answer_weight * weighted_average(point_scores)
```

For unanswerable items, the expected points check refusal behavior:

- says the document does not provide the requested information.
- does not fabricate details.
- optionally mentions nearby available information when evidence exists.

## Citation Metrics

Only apply citation score when `citation` weight is greater than 0.

Claim support:

- A cited context supports the answer if the cited `context_id` or `source_chunk_id` matches the evidence for the covered expected point, or if the cited text contains the evidence quote.
- A citation that exists but does not support the claim is counted as unsupported.

Citation score:

```text
citation_score = citation_weight * supported_key_claims / total_key_claims
```

Report:

```text
CitationPrecision = supported citations / total citations
CitationRecall    = expected points with supporting citation / expected points
CitationSupportRate = key claims supported by citation / key claims
```

## Reliability Metrics

For answerable items:

```text
reliability_score = reliability_weight
```

Subtract from reliability first, then answer score if needed:

- minor unsupported wording: -10% of reliability weight.
- clear unsupported claim: -50% of reliability weight.
- severe hallucination, fabricated number/date/contact, or contradiction: reliability score = 0 and cap answer score at 50%.

For unanswerable items:

```text
full reliability = clear refusal + no forbidden claims + no fabricated evidence
partial reliability = hedged answer but no concrete fabrication
zero reliability = fabricated answer, fake citation, fake contact, fake platform, fake rule
```

## Final Score

Per item:

```text
final_score = retrieval_score + answer_score + citation_score + reliability_score
```

Dataset report:

```text
overall_score = average(final_score)
score_by_type = average(final_score grouped by query_type)
score_by_difficulty = average(final_score grouped by difficulty)
```

Recommended grading:

```text
A: 90-100
B: 80-89
C: 70-79
D: 60-69
F: < 60
```

Hard gates:

```text
severe_hallucination_rate should be < 5%
refusal_accuracy should be >= 90% after the negative set is expanded
citation_support_rate should be >= 85%
```

## Judge Output Schema

An LLM judge should return JSON only:

```json
{
  "point_scores": [
    {
      "point_id": "p1",
      "score": 1.0,
      "rationale": "The answer states the exact required fact."
    }
  ],
  "unsupported_claims": [],
  "forbidden_claims_present": [],
  "refusal_correct": true,
  "citation_support": [
    {
      "point_id": "p1",
      "supported": true,
      "citation_ids": ["ctx-1"]
    }
  ],
  "failure_reasons": []
}
```

Suggested `failure_reasons` values:

```text
retrieval_miss
wrong_section
table_parse_or_table_retrieval_failure
answer_missing_point
answer_contradiction
unsupported_claim
citation_missing
citation_unsupported
refusal_failure
format_error
```

