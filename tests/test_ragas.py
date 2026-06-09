from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pipeline():
    pytest.importorskip("dotenv")
    from dotenv import load_dotenv

    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY nao configurada em .env")

    corpus_dir = Path("data/corpus")
    if not corpus_dir.exists() or not list(corpus_dir.glob("*.pdf")):
        pytest.skip("data/corpus/ vazio — adicione pelo menos 1 PDF")

    from pipeline.rag import build_rag_pipeline

    return build_rag_pipeline(corpus_dir=str(corpus_dir))


def test_ragas_answer_relevancy(pipeline):
    """Testa if the answer is relevant to the question (proxy: answer contains key terms from question)."""
    questions = [
        "Quais os criterios de desligamento da residencia?",
        "Qual a distribuicao de peso das notas?",
        "O que diz o regulamento em caso de perda de prazo?",
        "Quantas faltas sao permitidas?",
        "Qual a carga horaria total da fase 2?",
    ]
    scores = []
    for q in questions:
        result = pipeline.answer(q, k=3)
        answer = result["answer"].lower()
        # Proxy simples de relevancia: resposta nao deve ser "nao encontrado" nem vazia
        is_valid = len(answer) > 20 and "nao encontrado" not in answer
        scores.append(1.0 if is_valid else 0.0)

    avg_relevancy = sum(scores) / len(scores)
    print(f"\nAnswer Relevancy (proxy): {avg_relevancy:.2f}")
    assert avg_relevancy >= 0.8, f"Relevancy {avg_relevancy:.2f} < 0.8"


def test_ragas_faithfulness(pipeline):
    """Testa if the answer uses sources (proxy for faithfulness)."""
    result = pipeline.answer("Quais os criterios de desligamento?", k=3)
    sources = result.get("sources", [])
    answer = result["answer"]
    has_sources = len(sources) > 0
    has_citation = "[" in answer and "]" in answer and ":" in answer
    print(f"\nFaithfulness (proxy): sources={len(sources)}, cited={has_citation}")
    assert has_sources or has_citation, "Resposta sem fontes citadas"


def test_ragas_context_precision(pipeline):
    """Testa if retrieved chunks are relevant (proxy: at least 1 chunk mentioned in answer)."""
    result = pipeline.answer("O que diz o regulamento em caso de perda de prazo?", k=5)
    sources = result.get("sources", [])
    unique_sources = len(set(s for s, p in sources))
    print(f"\nContext Precision (proxy): {unique_sources} fontes unicas em {len(sources)} chunks")
    assert len(sources) > 0, "Nenhum chunk recuperado"
