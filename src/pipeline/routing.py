from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str
    reason: str


_COMPLEX_KEYWORDS = [
    "explique", "detalhe", "compare", "analise", "projete",
    "desligamento", "reprovacao", "conformidade", "regulamento",
    "critério", "justifique", "fundamente",
    "qual a diferenca", "o que diz", "como funciona",
]

_ENTITY_PATTERN = [
    "artigo", "paragrafo", "inciso", "clausula",
    "prazo", "carga horaria", "frequencia", "nota",
    "avaliacao", "certificado", "reprovado", "aprovado",
]


def classify_complexity(query: str) -> RouteDecision:
    cheap_model = os.environ.get("CHEAP_MODEL", "llama-3.3-70b-versatile")
    premium_model = os.environ.get("PREMIUM_MODEL", "llama-3.3-70b-versatile")

    q_lower = query.lower().strip()
    n_complex = sum(1 for kw in _COMPLEX_KEYWORDS if kw in q_lower)
    n_entities = sum(1 for kw in _ENTITY_PATTERN if kw in q_lower)

    if n_complex >= 1 or n_entities >= 2:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Palavras-chave complexas ({n_complex}) ou entidades regulatorias ({n_entities}) detectadas",
        )

    if len(q_lower) > 100 and "?" in q_lower:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Query longa ({len(q_lower)} chars) com interrogacao",
        )

    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="Query curta sem termos regulatorios — roteada para modelo barato",
    )


def make_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Configure GROQ_API_KEY no .env")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
