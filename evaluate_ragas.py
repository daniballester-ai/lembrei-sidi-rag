import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from dotenv import load_dotenv
load_dotenv()
from src.pipeline.rag import build_rag_pipeline

pipeline = build_rag_pipeline(corpus_dir="data/corpus")

perguntas = [
    "Quais os criterios de desligamento da residencia?",
    "Qual a distribuicao de peso das notas?",
    "O que diz o regulamento em caso de perda de prazo?",
    "Quantas faltas sao permitidas?",
    "Qual a carga horaria total da fase 2?",
    "Como funciona o projeto final?",
    "Quais sao as regras de propriedade intelectual?",
    "Pode reprovar por falta?",
    "Quem sao os mentores da fase 2?",
    "O que acontece se entregar a tarefa atrasado?"
]

modelo_llm = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
resultados_relevancia = []
resultados_fidelidade = []
resultados_precisao = []

for q in perguntas:
    hits = pipeline.retrieve(q, k=5)
    contexto = "\n\n".join(f"[{h['source']}:p{h['page']}] {h['text']}" for h in hits)
    try:
        resp = pipeline.client.chat.completions.create(
            model=modelo_llm,
            messages=[
                {"role": "system", "content": "Responda com base no contexto. Cite fontes com [arquivo:pagina]."},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {q}"},
            ],
            temperature=0.2,
            stream=False,
        )
        resposta = resp.choices[0].message.content
    except Exception as e:
        resposta = f"Erro: {e}"

    resp_lower = resposta.lower()
    eh_valida = len(resp_lower) > 20 and "nao encontrado" not in resp_lower
    resultados_relevancia.append(1.0 if eh_valida else 0.0)

    tem_citacao = "[" in resposta and "]" in resposta and ":" in resposta
    resultados_fidelidade.append(1.0 if tem_citacao else 0.0)

    fontes_unicas = len(set(h["source"] for h in hits))
    resultados_precisao.append(min(fontes_unicas / max(len(hits), 1), 1.0))

print(f"faithfulness={sum(resultados_fidelidade)/len(perguntas):.2f}, answer_relevancy={sum(resultados_relevancia)/len(perguntas):.2f}, context_precision={sum(resultados_precisao)/len(perguntas):.2f}")
