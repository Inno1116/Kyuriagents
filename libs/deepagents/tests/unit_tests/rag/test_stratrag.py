from __future__ import annotations

import json

from deepagents.rag import (
    HybridSearchConfig,
    StratRAGExampleResult,
    build_in_memory_stratrag_retriever,
    evaluate_stratrag_retriever,
    load_stratrag_jsonl,
    ndcg_at_k,
    parse_stratrag_row,
    recall_at_k,
    score_stratrag_example,
    stratrag_chunks,
)


def _row() -> dict[str, object]:
    return {
        "id": "val_000001",
        "query": "Which city links Alpha Museum and Beta River?",
        "reference_answer": "Paris",
        "doc_pool": [
            {"doc_id": "doc-alpha", "text": "Alpha Museum is located in Paris.", "source": "Alpha Museum"},
            {"doc_id": "doc-beta", "text": "Beta River flows through Paris.", "source": "Beta River"},
            {"doc_id": "doc-noise", "text": "Gamma Stadium is in Berlin.", "source": "Gamma Stadium"},
        ],
        "gold_doc_indices": [0, 1],
        "metadata": {"split": "val", "question_type": "bridge"},
        "created_at": "2026-01-01T00:00:00Z",
        "provenance": {"base": "hotpot_qa(distractor)", "seed": 42},
    }


def _embed(text: str) -> tuple[float, ...]:
    lowered = text.lower()
    return (
        1.0 if "paris" in lowered else 0.0,
        1.0 if "museum" in lowered else 0.0,
        1.0 if "river" in lowered else 0.0,
    )


def test_load_stratrag_jsonl_parses_local_file(tmp_path) -> None:
    path = tmp_path / "val.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    examples = load_stratrag_jsonl(path)

    assert examples[0].example_id == "val_000001"
    assert examples[0].question_type == "bridge"
    assert examples[0].gold_doc_indices == (0, 1)
    assert len(examples[0].doc_pool) == 3


def test_parse_stratrag_row_accepts_dataset_multi_hop_label() -> None:
    row = _row()
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    metadata["question_type"] = "multi-hop"

    example = parse_stratrag_row(row)

    assert example.question_type == "multi-hop"


def test_parse_stratrag_row_shuffle_recomputes_gold_indices() -> None:
    example = parse_stratrag_row(_row(), shuffle_docs=True)

    gold_sources = {example.doc_pool[index].source for index in example.gold_doc_indices}

    assert gold_sources == {"Alpha Museum", "Beta River"}


def test_stratrag_chunks_scope_each_example_as_knowledge_base() -> None:
    example = parse_stratrag_row(_row())

    chunks = stratrag_chunks([example])

    assert chunks[0].metadata.kb_id == example.kb_id
    assert chunks[0].metadata.source_type == "stratrag"
    assert chunks[0].metadata.visibility == "public"
    assert chunks[0].metadata.tags == ("stratrag", "val", "bridge", "gold")
    assert chunks[2].metadata.tags[-1] == "distractor"


def test_stratrag_metrics_handle_two_gold_documents() -> None:
    example = parse_stratrag_row(_row())
    gold = example.gold_doc_ids()
    result = score_stratrag_example(example, [gold[1], "wrong", gold[0]])

    assert result.recall_at_1 == 0.5
    assert result.recall_at_2 == 0.5
    assert result.recall_at_5 == 1.0
    assert result.mrr == 1.0
    assert round(result.ndcg_at_5, 4) == round(ndcg_at_k([gold[1], "wrong", gold[0]], gold, k=5), 4)
    assert recall_at_k([gold[1], "wrong", gold[0]], gold, k=3) == 1.0


def test_evaluate_stratrag_retriever_uses_per_example_scope() -> None:
    example = parse_stratrag_row(_row())
    retriever = build_in_memory_stratrag_retriever(
        [example],
        embed_text=_embed,
        config=HybridSearchConfig(top_k=5, vector_candidates=5, keyword_candidates=5, rerank_candidates=5),
    )

    evaluation = evaluate_stratrag_retriever([example], retriever, top_k=5)

    assert evaluation.overall.count == 1
    assert evaluation.overall.recall_at_5 == 1.0
    assert evaluation.by_question_type["bridge"].count == 1
    assert isinstance(evaluation.examples[0], StratRAGExampleResult)
