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


def test_pipeline_indexa_chunks(pipeline):
    assert pipeline.collection.count() > 0


def test_retrieve_top_k(pipeline):
    hits = pipeline.retrieve("qual o prazo de entrega do projeto", k=3)
    assert isinstance(hits, list)
    assert len(hits) <= 3
    if hits:
        h = hits[0]
        assert "text" in h
        assert "source" in h
        assert "distance" in h


def test_answer_retorna_resposta_com_fonte(pipeline):
    result = pipeline.answer("Quais os criterios de desligamento da residencia?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert "sources" in result
    assert isinstance(result["sources"], list)


def test_semantic_cache():
    from pipeline.cache import SemanticCache
    cache = SemanticCache(threshold=0.8)
    cache.put("Qual o prazo de entrega?", "30 dias uteis")
    hit = cache.get("Qual o prazo para entregar?")
    assert hit is not None
    assert "30 dias" in hit


def test_exact_cache():
    from pipeline.cache import ExactCache
    cache = ExactCache()
    cache.put("teste", "resposta")
    assert cache.get("teste") == "resposta"
    assert cache.get("outra") is None


def test_ferramenta_consultar_prazos_entregas():
    from pipeline.tools import consultar_prazos_entregas
    resultado = consultar_prazos_entregas()
    assert isinstance(resultado, str)
    assert len(resultado) > 0



def test_routing_simple():
    from pipeline.routing import classify_complexity
    decision = classify_complexity("Qual o horario da aula?")
    assert decision.complexity == "simple"


def test_routing_complex():
    from pipeline.routing import classify_complexity
    decision = classify_complexity("Explique os criterios de desligamento por falta")
    assert decision.complexity == "complex"
