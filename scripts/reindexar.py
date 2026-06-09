"""Script para re-indexar o corpus com o novo embedding portugues.

Uso: .venv\Scripts\python.exe scripts/reindexar.py
"""
import sys
import io

# Encoding UTF-8 para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

# Ajustar path para importar src
_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ))

from dotenv import load_dotenv
load_dotenv()

import chromadb


def reindexar():
    diretorio_chroma = str(_RAIZ / "data" / "chroma")
    diretorio_corpus = str(_RAIZ / "data" / "corpus")
    nome_colecao = "sidi_edital"

    print("=" * 60)
    print("REINDEXAÇÃO DO CORPUS — Embedding Português (Serafim)")
    print("=" * 60)

    # 1) Deletar coleção antiga
    print("\n1) Deletando coleção antiga...")
    cliente_chroma = chromadb.PersistentClient(path=diretorio_chroma)
    try:
        cliente_chroma.delete_collection(nome_colecao)
        print(f"   Coleção '{nome_colecao}' deletada com sucesso.")
    except Exception as e:
        print(f"   Coleção não existia ou erro: {e}")

    # Fechar conexão para liberar arquivos
    del cliente_chroma

    # 2) Re-indexar com novo embedding
    print("\n2) Iniciando re-indexação com Serafim embedding...")
    from src.pipeline.rag import RAGPipeline

    pipeline = RAGPipeline(corpus_dir=diretorio_corpus, persist_dir=diretorio_chroma)
    total = pipeline.ingest_and_index()
    print(f"\n   Total de chunks indexados: {total}")

    # 3) Testar busca híbrida
    print("\n3) Testando busca híbrida...")
    consulta = "quais são as disciplinas ou o conteúdo programático da residência"
    resultados = pipeline.retrieve(consulta, k=10)

    print(f"\n   Consulta: '{consulta}'")
    print(f"   Resultados (top 10):\n")
    for i, r in enumerate(resultados):
        rrf = r.get('rrf_score', 0)
        print(f"   #{i+1} (RRF: {rrf:.4f}) — {r['source']}:p{r['page']}")
        print(f"      {r['text'][:150]}...\n")

    # 4) Verificar se "CONTEÚDO PROGRAMÁTICO" está no top 5
    encontrado_no_top5 = any(
        "CONTEÚDO PROGRAMÁTICO" in r["text"] or "disciplinas" in r["text"].lower()
        for r in resultados[:5]
    )
    if encontrado_no_top5:
        print("✅ SUCESSO: Conteúdo programático encontrado no TOP 5!")
    else:
        print("⚠️ ATENÇÃO: Conteúdo programático NÃO está no top 5. Verifique.")

    print("\n" + "=" * 60)
    print("Re-indexação concluída!")
    print("=" * 60)


if __name__ == "__main__":
    reindexar()
