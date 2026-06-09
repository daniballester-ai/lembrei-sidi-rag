from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ---------------------------------------------------------------------------
# Indice BM25 para busca por palavras-chave
# ---------------------------------------------------------------------------

def _tokenizar(texto: str) -> list[str]:
    """Tokenizacao simples para BM25 — lowercase + split por nao-alfanumerico."""
    return re.findall(r"\w+", texto.lower())


class IndiceBM25:
    """Indice BM25 in-memory sobre os chunks do corpus."""

    def __init__(self) -> None:
        self._corpus_tokenizado: list[list[str]] = []
        self._ids: list[str] = []
        self._documentos: list[str] = []
        self._metadados: list[dict] = []
        self._indice: BM25Okapi | None = None

    def construir(
        self,
        ids: list[str],
        documentos: list[str],
        metadados: list[dict],
    ) -> None:
        """Constroi o indice BM25 a partir dos chunks."""
        self._ids = ids
        self._documentos = documentos
        self._metadados = metadados
        self._corpus_tokenizado = [_tokenizar(doc) for doc in documentos]
        self._indice = BM25Okapi(self._corpus_tokenizado)

    def buscar(self, consulta: str, k: int = 15) -> list[dict]:
        """Retorna os top-k resultados BM25."""
        if self._indice is None or not self._ids:
            return []

        tokens_consulta = _tokenizar(consulta)
        pontuacoes = self._indice.get_scores(tokens_consulta)

        # Indices ordenados por score decrescente
        indices_ordenados = np.argsort(pontuacoes)[::-1][:k]

        resultados = []
        for idx in indices_ordenados:
            if pontuacoes[idx] > 0:
                metadados_doc = self._metadados[idx] if (idx < len(self._metadados) and self._metadados[idx] is not None) else None
                fonte = metadados_doc.get("source", "Unknown") if metadados_doc else "Unknown"
                pagina = metadados_doc.get("page", 1) if metadados_doc else 1
                resultados.append({
                    "id": self._ids[idx],
                    "text": self._documentos[idx],
                    "source": fonte,
                    "page": pagina,
                    "bm25_score": float(pontuacoes[idx]),
                })
        return resultados


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def fusao_rrf(
    listas_ranqueadas: list[list[dict]],
    k_rrf: int = 60,
    top_k: int = 12,
) -> list[dict]:
    """Combina multiplas listas ranqueadas via Reciprocal Rank Fusion.

    Para cada documento d:  score_rrf(d) = SUM(1 / (k_rrf + rank_i(d)))
    """
    pontuacoes: dict[str, float] = {}
    documentos_por_id: dict[str, dict] = {}

    for lista in listas_ranqueadas:
        for rank, doc in enumerate(lista):
            doc_id = doc.get("id", f"{doc['source']}_p{doc['page']}")
            pontuacoes[doc_id] = pontuacoes.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            if doc_id not in documentos_por_id:
                documentos_por_id[doc_id] = doc

    # Ordenar por score RRF decrescente
    ids_ordenados = sorted(pontuacoes, key=lambda x: pontuacoes[x], reverse=True)

    resultados = []
    for doc_id in ids_ordenados[:top_k]:
        doc = documentos_por_id[doc_id].copy()
        doc["rrf_score"] = pontuacoes[doc_id]
        resultados.append(doc)

    return resultados


# ---------------------------------------------------------------------------
# Pipeline RAG principal
# ---------------------------------------------------------------------------

def _criar_cliente() -> OpenAI:
    chave_api = os.environ.get("GROQ_API_KEY")
    if not chave_api:
        raise RuntimeError("Configure GROQ_API_KEY no .env")
    return OpenAI(api_key=chave_api, base_url=GROQ_BASE_URL)


class RAGPipeline:
    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "sidi_edital",
        llm_model: str | None = None,
    ) -> None:
        self.client = _criar_cliente()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

        self.funcao_embedding = DefaultEmbeddingFunction()

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self._chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self._chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.funcao_embedding
        )

        # Limpar registros antigos do banco de prazos (deadlines.json) no RAG
        try:
            self.collection.delete(where={"source": "deadlines.json"})
        except Exception as erro_limpeza:
            print(f"Aviso: erro ao limpar chunks legados do ChromaDB: {erro_limpeza}")


        # Indice BM25 — construido junto com ingest ou reconstruido do Chroma
        self.indice_bm25 = IndiceBM25()
        self._reconstruir_bm25_do_chroma()

    def _reconstruir_bm25_do_chroma(self) -> None:
        """Reconstroi indice BM25 a partir dos documentos ja indexados no Chroma."""
        contagem = self.collection.count()
        if contagem == 0:
            return

        dados = self.collection.get(include=["documents", "metadatas"])
        self.indice_bm25.construir(
            ids=dados["ids"],
            documentos=dados["documents"],
            metadados=dados["metadatas"],
        )
        print(f"  Indice BM25 reconstruido com {contagem} chunks")

    def _limpar_collection(self) -> None:
        try:
            self._chroma.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._chroma.get_or_create_collection(
            name=self.collection_name, embedding_function=self.funcao_embedding
        )

    @staticmethod
    def _enriquecer_datas(texto: str) -> str:
        """Converte datas numericas (MM/DD/YY) para incluir o nome do mes,
        melhorando a busca semantica por nome de meses (ex: 'junho')."""
        meses = {
            "1": "Janeiro", "2": "Fevereiro", "3": "Marco",
            "4": "Abril", "5": "Maio", "6": "Junho",
            "7": "Julho", "8": "Agosto", "9": "Setembro",
            "10": "Outubro", "11": "Novembro", "12": "Dezembro",
        }
        def _substituir_data(match):
            mes_num = match.group(1)
            mes_nome = meses.get(mes_num, "")
            if mes_nome:
                return f"{match.group(0)} ({mes_nome})"
            return match.group(0)
        return re.sub(r"\b(1[0-2]|[1-9])/\d{1,2}/\d{2,4}\b", _substituir_data, texto)

    @staticmethod
    def _limpar_texto_pdf(texto: str) -> str:
        texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
        texto = re.sub(r" {2,}", " ", texto)
        # Remover URLs longas (ex: SharePoint) que poluem embeddings
        texto = re.sub(r"https?://\S{30,}", "", texto)
        # Remover metadados de lista SharePoint (ex: "Elemento sites/...")
        texto = re.sub(r"Elemento\s+sites/\S+", "", texto)
        # Remover "Item Type Path" repetido
        texto = re.sub(r"\s*Item Type Path\s*", " ", texto)
        # Limpar espaços extras gerados pelas remoções
        texto = re.sub(r" {2,}", " ", texto)
        return texto.strip()

    @staticmethod
    def _dividir_por_secoes(texto: str, max_chars: int = 800) -> list[str]:
        # Tentar split por seções do edital (ex: "1. DO OBJETO")
        secao_re = re.compile(r"(?=\d+\.\s+[A-ZÀ-Ú])")
        partes = secao_re.split(texto)

        # Se não encontrou seções do edital, tentar formato da Grade (ex: "01-Aula Magna")
        if len(partes) <= 1:
            grade_re = re.compile(r"(?=\d{2,3}-[A-ZÀ-Úa-zà-ú])")
            partes = grade_re.split(texto)

        # Fallback: se o texto ainda é uma parte única e muito grande, usar splitter recursivo
        if len(partes) <= 1 and len(texto) > max_chars:
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chars,
                chunk_overlap=120,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            return fallback_splitter.split_text(texto)

        resultado = []
        bloco_atual = ""

        for parte in partes:
            parte = parte.strip()
            if not parte:
                continue
            
            # Adicionar contexto se for da Grade (reconhecido pelo regex)
            grade_match = re.match(r"^\d{2,3}-[A-ZÀ-Úa-zà-ú]", parte)
            if grade_match:
                parte = f"Disciplina da Residência: {parte}"

            # Se a parte sozinha for maior que max_chars, precisamos splittar ela
            if len(parte) > max_chars:
                if bloco_atual:
                    resultado.append(bloco_atual)
                    bloco_atual = ""
                
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chars,
                    chunk_overlap=120,
                    separators=[". ", ".\n", "\n\n", "\n", " ", ""],
                )
                resultado.extend(sub_splitter.split_text(parte))
            # Se juntar com o bloco atual não estourar o limite, agrupa
            elif len(bloco_atual) + len(parte) <= max_chars:
                bloco_atual += ("\n" if bloco_atual else "") + parte
            # Se estourar o limite, salva o bloco atual e começa um novo
            else:
                resultado.append(bloco_atual)
                bloco_atual = parte

        if bloco_atual:
            resultado.append(bloco_atual)

        return resultado

    def ingest_and_index(self) -> int:
        """Extrai, chunka e indexa os PDFs do corpus no ChromaDB + BM25."""
        self._limpar_collection()
        chunks: list[dict] = []
        for caminho_pdf in self.corpus_dir.glob("*.pdf"):
            leitor = PdfReader(str(caminho_pdf))
            total_paginas = len(leitor.pages)
            pdf_chunks = 0
            for num_pagina, pagina in enumerate(leitor.pages, start=1):
                texto = pagina.extract_text()
                if texto and texto.strip():
                    texto_processado = self._enriquecer_datas(self._limpar_texto_pdf(texto.strip()))
                    textos_chunk = self._dividir_por_secoes(texto_processado, max_chars=800)
                    for i, texto_chunk in enumerate(textos_chunk):
                        chunk_id = f"{caminho_pdf.name}_p{num_pagina}_c{i}"
                        chunks.append({
                            "id": chunk_id,
                            "text": texto_chunk,
                            "source": caminho_pdf.name,
                            "page": num_pagina,
                        })
                    pdf_chunks += len(textos_chunk)
            print(f"  [{caminho_pdf.name}] {total_paginas} paginas, {pdf_chunks} chunks")
        print(f"  Total chunks de PDF: {len(chunks)}")

        if chunks:
            # Indexar no ChromaDB (embedding semantico)
            self.collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
            )

            # Construir indice BM25 (busca por keywords)
            self.indice_bm25.construir(
                ids=[c["id"] for c in chunks],
                documentos=[c["text"] for c in chunks],
                metadados=[{"source": c["source"], "page": c["page"]} for c in chunks],
            )

        return self.collection.count()


    def retrieve(self, query: str, k: int = 12) -> list[dict]:
        """Busca hibrida: combina semantica (ChromaDB) + BM25 via RRF."""

        # 1) Busca semantica via ChromaDB
        k_parcial = min(k * 2, self.collection.count() or 1)
        resultados_chroma = self.collection.query(query_texts=[query], n_results=k_parcial)
        hits_semanticos: list[dict] = []
        if resultados_chroma and resultados_chroma.get("ids") and resultados_chroma["ids"] and resultados_chroma["ids"][0]:
            for i in range(len(resultados_chroma["ids"][0])):
                metadados_doc = None
                docs = resultados_chroma.get("documents")
                metas = resultados_chroma.get("metadatas")
                dists = resultados_chroma.get("distances")

                if metas and metas[0] and metas[0][i] is not None:
                    metadados_doc = metas[0][i]

                fonte = metadados_doc.get("source", "Unknown") if metadados_doc else "Unknown"
                pagina = metadados_doc.get("page", 1) if metadados_doc else 1

                texto = ""
                if docs and docs[0] and docs[0][i] is not None:
                    texto = docs[0][i]

                distancia = 0.0
                if dists and dists[0] and dists[0][i] is not None:
                    distancia = dists[0][i]

                hits_semanticos.append({
                    "id": resultados_chroma["ids"][0][i],
                    "text": texto,
                    "source": fonte,
                    "page": pagina,
                    "distance": distancia,
                })

        # 2) Busca BM25 (keywords)
        hits_bm25 = self.indice_bm25.buscar(query, k=k_parcial)

        # 3) Fusao RRF
        resultados_fusionados = fusao_rrf(
            listas_ranqueadas=[hits_semanticos, hits_bm25],
            k_rrf=60,
            top_k=k,
        )

        return resultados_fusionados

    def answer(self, question: str, k: int = 12) -> dict:
        """Responde a pergunta usando busca hibrida + LLM."""
        try:
            hits = self.retrieve(question, k=k)
        except Exception:
            hits = []
        partes_contexto = []
        fontes = []
        for h in hits:
            if not isinstance(h, dict):
                continue
            fonte = h.get("source") or "Unknown"
            pagina = h.get("page") or 1
            texto = h.get("text") or ""
            partes_contexto.append(f"[{fonte}:p{pagina}] {texto}")
            fontes.append((fonte, pagina))
        contexto = "\n\n".join(partes_contexto)

        try:
            resposta = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Contexto do edital:\n{contexto}\n\nPergunta do usuario: {question}"},
                ],
                temperature=0.3,
            )
            return {"answer": resposta.choices[0].message.content, "sources": fontes}
        except Exception as e:
            return {"answer": f"Erro ao consultar LLM: {e}", "sources": fontes}


SYSTEM_PROMPT = """Voce e um assistente especializado no edital da residencia SiDi. Responda APENAS com base no contexto abaixo.
Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".
Sempre cite a fonte usando o formato [arquivo:pagina]."""


def build_rag_pipeline(corpus_dir: str = "data/corpus", persist_dir: str | None = None) -> RAGPipeline:
    if persist_dir is None:
        persist_dir = str(Path(corpus_dir).resolve().parent.parent / "data" / "chroma")
    pipeline = RAGPipeline(corpus_dir=corpus_dir, persist_dir=persist_dir)
    print("Indexando corpus...")
    pipeline.ingest_and_index()
    return pipeline
