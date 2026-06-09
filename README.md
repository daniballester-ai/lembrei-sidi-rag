# a🏛️ Lembrei, SiDi! — Insight Bot

**Assistente LLM-powered para consulta ao edital e gestão de prazos da residência SiDi**

🔗 **App:** [https://lembrei-sidi.streamlit.app/](https://lembrei-sidi.streamlit.app/)

> 👩‍💻 **Desenvolvedora:** Dani Ballester
>
> 🎓 **Disciplina:** Mod4 / PPI — Desenvolvendo Software com IA Generativa
>
> 🏢 *Residência em IA — SiDi* (TIC 44 — Fase 2)

---

## 📺 Vídeo de Apresentação

[![Assista ao vídeo de apresentação do projeto](https://img.youtube.com/vi/04hv6ZrclxI/maxresdefault.jpg)](https://youtu.be/04hv6ZrclxI)

---

## Problema

Residentes de tecnologia do SiDi enfrentam dificuldades para centralizar, interpretar e acompanhar a alta densidade de informações contidas no **Edital da Fase 2** e na **Grade Curricular**. Critérios de avaliação, regras de conformidade acadêmica, prazos de entrega e formato de envio são informações dispersas que geram dúvidas recorrentes e prazos perdidos.

## Solução

Um chatbot RAG-based com function-calling que:

1. **Responde dúvidas sobre o edital** com citação de fonte (página exata)
2. **Consulta prazos cadastrados** via function-calling (`consultar_prazos_entregas`)
3. **Gerencia tarefas** com cadastro, edição, exclusão e indicadores de urgência
4. **Reduz custos** com cache semântico + model routing cheap-first
5. **Avalia qualidade** com métricas proxy de RAGAS integradas na UI

---

## Arquitetura

```
sidi-insight-bot/
├── data/
│   ├── corpus/                    # PDFs do edital + grade curricular
│   │   ├── PPI TIC 44 - EDITAL DE SELEÇÃO - FASE 2.pdf
│   │   └── Grade_Fase2_TIC44.xlsx.PDF
│   ├── deadlines.json             # Banco NoSQL de prazos (admin)
│   └── chroma/                    # Vector store persistido (embeddings Serafim)
├── src/
│   ├── pipeline/
│   │   ├── rag.py                 # RAG: chunking → ChromaDB + BM25 → RRF → generate
│   │   ├── tools.py               # Function-calling: deadlines + email + cadastro
│   │   ├── cache.py               # Exact cache (SHA256) + Semantic cache (cosine ≥0.93)
│   │   └── routing.py             # Cheap-first routing: llama3-70b vs llama3-70b
│   ├── ui/
│   │   └── streamlit_app.py       # Streamlit: Admin + Aluno + Chat + RAGAS Eval
│   ├── auth.py                    # Autenticação SQLite (admin/student)
│   └── observability/
│       └── trace.py               # Logs estruturados + tracking P95
├── tests/
│   ├── test_smoke.py              # Smoke tests do pipeline
│   └── test_ragas.py              # Avaliação com RAGAS (proxy)
├── scripts/
│   └── reindexar.py               # Script de re-indexação do corpus
└── images/                        # Assets visuais do SiDi
```

### Fluxo de Requisição

```
Usuário → Autenticação → Exact Cache → Semantic Cache → Router → LLM + RAG + Tools → Resposta
            (SQLite)    (SHA256)     (cosine ≥0.93)  (cheap-first)       + Cache
```

### Stack Tecnológica

| Componente    | Tecnologia                                                          |
| ------------- | ------------------------------------------------------------------- |
| LLM (premium) | Groq —`llama-3.3-70b-versatile`                                  |
| LLM (cheap)   | Groq —`llama-3.1-8b-instant`                                     |
| Embedding     | **Serafim** (`PORTULAN/serafim-100m-portuguese-ir`)         |
| Vector Store  | **ChromaDB** (persistente local)                              |
| Search        | **Híbrida**: Semântica (ChromaDB) + BM25 + RRF              |
| UI            | **Streamlit** (dark theme)                                    |
| Auth          | **SQLite** (SHA256) — admin + student roles                  |
| Cache         | **SHA256 exact** + **Cosine semantic** (threshold 0.93) |
| Observability | Logs JSON estruturados + Latência P95                              |

---

## Mapeamento dos Requisitos do Professor

| #  | Requisito (projeto-portfolio.md)                                       | Status | Implementação                                                                           |
| -- | ---------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------- |
| 1  | **Corpus textual ≥ 10 páginas**                                | ✅     | 2 PDFs: Edital Fase 2 (~30 pág) + Grade Curricular                                       |
| 2  | **≥ 3 perguntas que se beneficiam de RAG**                      | ✅     | 5 perguntas no `test_ragas.py` e aba "RAGAS Eval" na UI                                 |
| 3  | **≥ 1 tool customizada do domínio**                            | ✅     | `consultar_prazos_entregas()`, `schedule_deadline_email()`, `cadastrar_atividade()` |
| 4  | **Deploy em URL pública**                                       | ✅     | Streamlit Cloud — [https://lembrei-sidi.streamlit.app/](https://lembrei-sidi.streamlit.app/) |
| 5  | **README profissional**                                          | ✅     | Este README                                                                               |
| 6  | **≥ 1 medida de redução de custo**                            | ✅     | Cache exact + semântico + routing cheap-first                                            |
| 7  | **Pipeline RAG ponta-a-ponta**                                   | ✅     | `rag.py`: ingest → chunking → embedding → retrieve → generate                       |
| 8  | **Chunking (800/100)**                                           | ✅     | `_dividir_por_secoes()` com RecursiveCharacterTextSplitter e overlap 120                |
| 9  | **Embedding em português**                                      | ✅     | Serafim (SentenceTransformer) — modelo específico pt-BR                                 |
| 10 | **Busca híbrida**                                               | ✅     | Semântica (ChromaDB) + BM25 (keyword) + RRF fusion                                       |
| 11 | **Function-calling (tool-use)**                                  | ✅     | OpenAI schema em `TOOLS` + `TOOL_REGISTRY` roteado via LLM                            |
| 12 | **Cache semântico**                                             | ✅     | `SemanticCache` com cosine similarity ≥ 0.93                                           |
| 13 | **Model routing cheap-first**                                    | ✅     | `Routing` por palavras-chave — queries simples vs complexas                            |
| 14 | **Avaliação RAGAS**                                            | ✅     | `test_ragas.py` + UI interativa com Answer Relevancy, Faithfulness, Context Precision   |
| 15 | **Smoke tests**                                                  | ✅     | `test_smoke.py` — 7 testes (pipeline, cache, routing, tools)                           |
| 16 | **README com arquitetura + setup + custo + decisões + limites** | ✅     | README completo                                                                           |
| 17 | **Autenticação (admin + aluno)**                               | ✅     | `auth.py` com SQLite — admin gerencia tarefas, aluno consulta                          |
| 18 | **Gestão de prazos com urgência**                              | ✅     | Cores 🟢🟡🔴🟣 por dias restantes                                                         |
| 19 | **Observabilidade (P95)**                                        | ✅     | `trace.py` — LatencyTracker com janela deslizante de 20 requests                       |
| 20 | **Vídeo de apresentação (≤3 min)**                           | ✅     | [Link do YouTube](https://youtu.be/04hv6ZrclxI)                                              |

---

## Setup

### 1. Pré-requisitos

- Python ≥ 3.10
- **Groq API key** (gratuita em [console.groq.com](https://console.groq.com))
- Opcional: **Resend API key** para envio de e-mails

### 2. Instalação

```bash
cd sidi-insight-bot

# Criar ambiente virtual
python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Instalar dependências
pip install -e .
pip install -e ".[dev]"
```

### 3. Configurar .env

```bash
cp .env.example .env  # ou crie manualmente
```

Conteúdo do `.env`:

```env
GROQ_API_KEY=sua_chave_aqui
LLM_MODEL=llama-3.3-70b-versatile
CHEAP_MODEL=llama-3.3-70b-versatile
PREMIUM_MODEL=llama-3.3-70b-versatile
RESEND_API_KEY=opcional
```

### 4. Indexar o corpus

```bash
streamlit run src/ui/streamlit_app.py
```

Na primeira execução, o pipeline indexa automaticamente os PDFs em `data/corpus/`. Para reindexar:

```bash
python scripts/reindexar.py
```

### 5. Credenciais padrão

| Papel | E-mail                       | Senha    |
| ----- | ---------------------------- | -------- |
| Admin | admin@sidi.org               | admin123 |
| Aluno | (criar via tela de registro) | —       |

*Solicitar para a desenvolvedora cadastro do tipo admin*

---

## Métricas Observadas

| Métrica                     | Target  | Como é medido                                          |
| ---------------------------- | ------- | ------------------------------------------------------- |
| Cache Hit Rate (exact)       | ≥10%   | SHA256 — contagem no sidebar do Streamlit              |
| Cache Hit Rate (semântico)  | ≥20%   | Cosine similarity ≥ 0.93 — contagem no sidebar        |
| Redução de custo (routing) | ≥50%   | Heurística: queries simples vs complexas               |
| Latência P95                | <3000ms | LatencyTracker nos últimos 20 requests                 |
| Answer Relevancy (RAGAS)     | ≥0.85  | Proxy: resposta > 20 chars e não "não encontrado"     |
| Faithfulness (RAGAS)         | ≥0.90  | Proxy: resposta contém citação `[arquivo:página]` |
| Context Precision (RAGAS)    | ≥0.80  | Proxy: fontes únicas nos chunks recuperados            |

---

## Custo por Requisição

| Modelo                      | Custo (input)                               | Custo (output)      | Gatilho |
| --------------------------- | ------------------------------------------- | ------------------- | ------- |
| `llama-3.1-8b-instant`    | ~$0.05/1M tokens¹     | ~$0.08/1M tokens¹ | Consultas simples   |         |
| `llama-3.3-70b-versatile` | ~$0.59/1M tokens¹     | ~$0.79/1M tokens¹ | Consultas complexas |         |

¹ Preços Groq — [groq.com/pricing](https://groq.com/pricing)

Com cache semântico + routing cheap-first (8B para simples, 70B para complexas), a estimativa é de redução ≥50% no custo total.

---

## Decisões de Design

| Decisão                            | Alternativa          | Por quê                                                          |
| ----------------------------------- | -------------------- | ----------------------------------------------------------------- |
| **ChromaDB local**            | Vector DB na nuvem   | Zero custo operacional, deploy 1-click no Streamlit Cloud         |
| **Groq + LLaMA 3.3**          | Gemini / OpenAI pago | Grátis para uso acadêmico, alta velocidade (token/s)            |
| **Serafim embedding**         | all-MiniLM-L6-v2     | Modelo específico para português (SentenceTransformer)          |
| **Busca híbrida (BM25+RRF)** | Só semântica       | Aumenta recall em consultas com termos técnicos do edital        |
| **Heurística de routing**    | Classifier ML        | Simples o suficiente para o escopo; ML adicionaria complexidade   |
| **Cache semântico (cosine)** | ANN index            | Threshold explícito é mais debuggável para MVP                 |
| **deadlines.json (NoSQL)**    | SQLite / PostgreSQL  | Zero dependência externa, versionável no Git                    |
| **Auth SQLite local**         | Auth0 / Firebase     | Sem custo, sem dependência de rede, adequado para MVP acadêmico |

---

## Limites

- **PDFs escaneados**: não suporta OCR — o corpus precisa ser PDF com texto selecionável
- **Cache**: volátil por sessão do Streamlit (não persiste entre reinicializações)
- **Routing heurístico**: não evolui com o uso; para produção, substituir por classifier
- **Autenticação**: senha fixa SHA256 em SQLite local (adequado para MVP acadêmico)
- **Proxy RAGAS**: métricas são proxies (não usam a biblioteca RAGAS oficial por limitação de dependências)

---

## Testes

```bash
pytest tests/ -v
```

| Teste                                         | O que valida                                         |
| --------------------------------------------- | ---------------------------------------------------- |
| `test_pipeline_indexa_chunks`               | Pipeline indexa chunks no ChromaDB                   |
| `test_retrieve_top_k`                       | Busca retorna top-k resultados com estrutura correta |
| `test_answer_retorna_resposta_com_fonte`    | Resposta gera saída com answer + sources            |
| `test_semantic_cache`                       | Cache semântico retorna hit para query similar      |
| `test_exact_cache`                          | Cache exato SHA256 funciona                          |
| `test_ferramenta_consultar_prazos_entregas` | Tool de prazos retorna string                        |
| `test_routing_simple/complex`               | Classificador de complexidade funciona               |

---

## 👩‍💻 Desenvolvedora

- **Dani Ballester** — Residente em IA no SiDi
- GitHub: [@daniballester-ai](https://github.com/daniballester-ai)
- Disciplina: Desenvolvendo Software com IA Generativa (Mod4 / PPI)
- Instrutor: Prof. Nicksson

---

## Licença 📄

MIT
