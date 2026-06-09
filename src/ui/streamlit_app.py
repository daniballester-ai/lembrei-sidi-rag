from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
_ASSETS = _ROOT / "images"
sys.path.insert(0, str(_ROOT))

import importlib

load_dotenv()

import streamlit as st

from src.observability.trace import trace, log_event, get_latency_tracker
from src.pipeline.cache import ExactCache, SemanticCache
from src.pipeline.rag import build_rag_pipeline
from src.pipeline.routing import classify_complexity
import src.pipeline.tools as sidi_tools
importlib.reload(sidi_tools)
from src.pipeline.tools import TOOLS, TOOL_REGISTRY, run_tool_call, load_deadlines, save_deadlines, delete_deadline
from src.auth import authenticate, create_user
import uuid

st.set_page_config(
    page_title="Lembrei, SiDi!",
    page_icon=str(_ASSETS / "sidi_2.png"),
    layout="wide",
)

# Injetar CSS premium para estilização do menu lateral (Sidebar) e fontes base
st.markdown(
    """
    <style>
    /* Estilização da Sidebar (Tema Dark Streamlit + Streamly Red Accent) */
    /* Fundo padrao do Streamlit mantido igual ao do Streamly */
    
    /* Efeito de brilho neon vermelho na logo da sidebar (estilo Streamly) */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        border-radius: 20px !important;
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.6) !important;
        border: 2px solid rgba(255, 75, 75, 0.4) !important;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out !important;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] {
        text-align: center !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] img:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 30px rgba(255, 75, 75, 0.8) !important;
    }
    
    /* Cores de texto específicas da Sidebar (evitando seletores curinga) */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] div.stSubheader {
        color: #fafafa !important;
    }
    
    /* Títulos da Sidebar */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }

    /* Cards de Métricas na Sidebar para evitar scrollbars com hover em vermelho */
    [data-testid="stSidebar"] div[data-testid="metric-container"] {
        background-color: #1e2025 !important;
        border: 1px solid #283048 !important;
        border-radius: 10px !important;
        padding: 6px 10px !important;
        margin-bottom: 6px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] div[data-testid="metric-container"]:hover {
        transform: translateY(-1px) !important;
        border-color: #ff4b4b !important; /* brilho vermelho */
        box-shadow: 0 6px 10px -3px rgba(255, 75, 75, 0.25) !important;
    }
    [data-testid="stSidebar"] div[data-testid="metric-container"] label,
    [data-testid="stSidebar"] div[data-testid="metric-container"] label p {
        color: #a3a3a3 !important;
        font-size: 0.55rem !important;
        font-weight: 500 !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
        color: #a3a3a3 !important;
        font-size: 0.55rem !important;
        font-weight: 500 !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.1 !important;
        overflow: visible !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"] > *,
    [data-testid="stSidebar"] [data-testid="stMetricValue"] > * > * {
        color: #ffffff !important;
        font-size: 0.35rem !important;
        font-weight: 700 !important;
        word-break: break-word !important;
        line-height: 1.2 !important;
        overflow: visible !important;
    }

    /* Botões da Sidebar (exceto botão de collapse) com hover em vermelho */
    [data-testid="stSidebar"] button:not([data-testid="stSidebarCollapseButton"]), 
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1e2025 !important;
        color: #fafafa !important;
        border: 1px solid #283048 !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] button:not([data-testid="stSidebarCollapseButton"]):hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #ff4b4b !important;
        border-color: #ff3333 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3) !important;
    }
    [data-testid="stSidebar"] button:not([data-testid="stSidebarCollapseButton"]):active,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:active {
        transform: translateY(0px) !important;
    }



    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] div.stCaption,
    [data-testid="stSidebar"] div.stCaption p {
        text-align: center !important;
        display: block !important;
        width: 100% !important;
    }
    </style>
    <style>
    /* Fixar chat_input no final da aba */
    .stChatInput {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 100 !important;
        background-color: #1e2025 !important;
    }
    [data-testid="stTabContent"] {
        display: flex;
        flex-direction: column;
    }
    [data-testid="stTabContent"] > div:last-child {
        margin-top: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_logo, col_title = st.columns([1, 12])
with col_logo:
    st.image(str(_ASSETS / "sidi.png"), width=120)
with col_title:
    st.title("Lembrei, SiDi!")
    st.caption("Assistente inteligente do edital da residência SiDi — consulte regras, prazos e agende alertas")

@st.cache_resource
def get_rag_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))

@st.cache_resource
def get_exact_cache():
    return ExactCache()

@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


@st.cache_data
def obter_disciplinas_grade():
    """Extrai todas as disciplinas numeradas do PDF da grade da residência."""
    caminho_pdf = _ROOT / "data" / "corpus" / "Grade_Fase2_TIC44.xlsx.PDF"
    if not caminho_pdf.exists():
        # Fallback caso o arquivo não exista
        return ["AWS", "Engenharia de Dados", "NoSQL", "Aula Magna"]
    
    from pypdf import PdfReader
    import re
    
    leitor = PdfReader(str(caminho_pdf))
    disciplinas_encontradas = set()
    
    # Regex para capturar tudo de "XX-..." até a data (ex: 3/27/26 10:00)
    padrao_disciplina = re.compile(r"(\d{2,3}-.*?)\s+\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}")

    for pagina in leitor.pages:
        texto_pagina = pagina.extract_text()
        if texto_pagina:
            ocorrencias = padrao_disciplina.findall(texto_pagina)
            for ocorrencia in ocorrencias:
                texto_limpo = ocorrencia.strip()
                texto_limpo = re.sub(r"\s+", " ", texto_limpo)
                if texto_limpo:
                    disciplinas_encontradas.add(texto_limpo)
                    
    if not disciplinas_encontradas:
        return ["AWS", "Engenharia de Dados", "NoSQL", "Aula Magna"]
        
    # Ordenar numericamente com base na numeração inicial (ex: "01", "02")
    disciplinas_ordenadas = sorted(list(disciplinas_encontradas), key=lambda d: int(d.split('-')[0]))
    return disciplinas_ordenadas

with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_rag_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()

with st.sidebar:
    st.markdown(
        f'<div style="display: flex; justify-content: center; width: 100%;">'
        f'<img src="data:image/png;base64,{__import__("base64").b64encode(open(str(_ASSETS / "sidi.png"), "rb").read()).decode()}" '
        f'width="150" style="border-radius: 20px; box-shadow: 0 0 20px rgba(255,75,75,0.6); border: 2px solid rgba(255,75,75,0.4);">'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.header("Lembrei, SiDi!")
    st.caption("v0.1.0")

if "user" not in st.session_state:
    col_vazia1, col_login, col_vazia2 = st.columns([1, 2, 1])
    with col_login:
        st.subheader("Autenticação")
        tab_login, tab_register = st.tabs(["Login", "Registrar"])
        with tab_login:
            with st.form("login_form"):
                email_login = st.text_input("E-mail")
                senha_login = st.text_input("Senha", type="password")
                if st.form_submit_button("Entrar"):
                    user = authenticate(email_login, senha_login)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    else:
                        st.error("Credenciais inválidas.")
        with tab_register:
            with st.form("register_form"):
                email_reg = st.text_input("E-mail institucional")
                senha_reg = st.text_input("Senha", type="password")
                if st.form_submit_button("Registrar"):
                    if create_user(email_reg, senha_reg):
                        st.success("Registrado com sucesso! Faça login.")
                    else:
                        st.error("E-mail já cadastrado.")
    st.stop()

# Sessão logada
is_admin = st.session_state["user"]["role"] == "admin"
tipo_usuario = "Administrador" if is_admin else "Aluno"
st.sidebar.markdown(f"**Logado como:**\n`{st.session_state['user']['email']}` ({tipo_usuario})")
if st.sidebar.button("Sair"):
    del st.session_state["user"]
    st.rerun()

import re as _re
from datetime import datetime as _datetime, date as _date


def _parse_data_limite(data_limite_str: str) -> _datetime | None:
    formatos = [
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y",
    ]
    for fmt in formatos:
        try:
            return _datetime.strptime(str(data_limite_str).strip(), fmt)
        except ValueError:
            continue
    return None


def calcular_urgencia_tarefa(data_limite_str: str) -> str:
    """Retorna ícone colorido de urgência com base nos dias restantes até o prazo."""
    prazo_dt = _parse_data_limite(data_limite_str)
    if prazo_dt is None:
        return "⚪"
    dias_restantes = (prazo_dt - _datetime.now()).total_seconds() / 86400
    if dias_restantes > 2:
        return "🟢"
    elif dias_restantes > 1:
        return "🟡"
    elif dias_restantes >= 0:
        return "🔴"
    else:
        return "🟣"


def _tarefa_expirada(data_limite_str: str) -> bool:
    prazo_dt = _parse_data_limite(data_limite_str)
    if prazo_dt is None:
        return False
    return (prazo_dt - _datetime.now()).total_seconds() / 86400 < 0


if is_admin:
    tab_tarefas_admin, tab_chat, tab_admin, tab_ragas = st.tabs(["📋 Minhas Tarefas", "💬 Chatbot", "⚙️ Painel de Administrador", "📈 RAGAS Eval"])
    
    if "tarefa_em_edicao" not in st.session_state:
        st.session_state.tarefa_em_edicao = None

    with tab_admin:
        tarefa_ed = st.session_state.tarefa_em_edicao
        
        if tarefa_ed:
            st.subheader("✏️ Editar Tarefa")
            texto_botao = "Salvar Alterações"
        else:
            st.subheader("Cadastrar Nova Tarefa")
            texto_botao = "Salvar Tarefa"

        with st.form("form_tarefa"):
            lista_disciplinas = obter_disciplinas_grade() + ["Outra"]
            
            # Linha 1: Disciplina e Atividade
            col1, col2 = st.columns(2)
            disc_idx = 0
            if tarefa_ed and tarefa_ed.get("disciplina") in lista_disciplinas:
                disc_idx = lista_disciplinas.index(tarefa_ed["disciplina"])
            disc = col1.selectbox("Disciplina", lista_disciplinas, index=disc_idx)
            
            default_ativ = tarefa_ed.get("atividade", "") if tarefa_ed else ""
            ativ = col2.text_input("Atividade (Ex: Lab 1, Projeto Final)", value=default_ativ)
            
            # Linha 2: Data Limite e Hora Limite
            col3, col4 = st.columns(2)
            from datetime import time as _time
            default_data = _date.today()
            default_hora = _time(23, 59)
            if tarefa_ed and tarefa_ed.get("data_limite"):
                data_limite_str = tarefa_ed["data_limite"]
                formatos = [
                    "%d-%m-%Y %H:%M:%S",
                    "%d-%m-%Y %H:%M",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d",
                    "%d/%m/%Y %H:%M",
                    "%d/%m/%Y",
                ]
                for fmt in formatos:
                    try:
                        dt = _datetime.strptime(data_limite_str.strip(), fmt)
                        default_data = dt.date()
                        default_hora = dt.time()
                        break
                    except ValueError:
                        continue
            data_lim = col3.date_input("Data Limite", value=default_data)
            hora_lim = col4.time_input("Hora Limite", value=default_hora)
            
            # Linha 3: Formato e Link
            col5, col6 = st.columns(2)
            formatos_lista = ["E-mail", "Link Externo"]
            formato_idx = 0
            if tarefa_ed and tarefa_ed.get("formato_envio") in formatos_lista:
                formato_idx = formatos_lista.index(tarefa_ed["formato_envio"])
            formato = col5.selectbox("Formato", formatos_lista, index=formato_idx)
            
            default_link = tarefa_ed.get("link_destino", "") if tarefa_ed else ""
            link = col6.text_input("Link de Destino (Opcional)", value=default_link)
            
            default_detalhes = tarefa_ed.get("detalhes", "") if tarefa_ed else ""
            detalhes = st.text_area("Detalhes adicionais", value=default_detalhes)
            
            if st.form_submit_button(texto_botao):
                tarefas = load_deadlines()
                if tarefa_ed:
                    # Modo de edição: Atualizar tarefa existente
                    for t in tarefas:
                        if t.get("id") == tarefa_ed["id"]:
                            t["disciplina"] = disc
                            t["atividade"] = ativ
                            t["data_limite"] = f'{data_lim.strftime("%d-%m-%Y")} {hora_lim.strftime("%H:%M:%S") if hasattr(hora_lim, "strftime") else hora_lim}'
                            t["formato_envio"] = formato
                            t["link_destino"] = link
                            t["detalhes"] = detalhes
                            break
                    save_deadlines(tarefas)
                    st.session_state.tarefa_em_edicao = None
                    st.success("Tarefa atualizada com sucesso!")
                else:
                    # Modo de criação: Cadastrar nova tarefa
                    nova_tarefa = {
                        "id": str(uuid.uuid4())[:8],
                        "disciplina": disc,
                        "atividade": ativ,
                        "data_limite": f'{data_lim.strftime("%d-%m-%Y")} {hora_lim}',
                        "formato_envio": formato,
                        "link_destino": link,
                        "detalhes": detalhes
                    }
                    tarefas.append(nova_tarefa)
                    save_deadlines(tarefas)
                    st.success("Tarefa cadastrada com sucesso!")
                time.sleep(0.5)
                st.rerun()

        # Botão de cancelar edição caso esteja editando
        if tarefa_ed:
            if st.button("❌ Cancelar Edição"):
                st.session_state.tarefa_em_edicao = None
                st.rerun()
        
        st.subheader("Tarefas Cadastradas")
        tarefas_atuais = load_deadlines()
        if tarefas_atuais:
            # Cabeçalhos da tabela customizada
            col_disc, col_ativ, col_prazo, col_acoes = st.columns([3, 3, 3, 2])
            with col_disc:
                st.markdown("**Disciplina**")
            with col_ativ:
                st.markdown("**Atividade**")
            with col_prazo:
                st.markdown("**Prazo**")
            with col_acoes:
                st.markdown("**Ações**")
            st.divider()

            for t in tarefas_atuais:
                col_d, col_at, col_pr, col_ac = st.columns([3, 3, 3, 2])
                with col_d:
                    st.write(t.get("disciplina", ""))
                with col_at:
                    st.write(t.get("atividade", ""))
                with col_pr:
                    st.write(t.get("data_limite", ""))
                with col_ac:
                    col_edit, col_del = st.columns(2)
                    with col_edit:
                        if st.button("✏️", key=f"edit_{t['id']}", help="Editar esta tarefa"):
                            st.session_state.tarefa_em_edicao = t
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_{t['id']}", help="Excluir esta tarefa"):
                            if delete_deadline(t["id"]):
                                if st.session_state.tarefa_em_edicao and st.session_state.tarefa_em_edicao["id"] == t["id"]:
                                    st.session_state.tarefa_em_edicao = None
                                st.success("Tarefa excluída com sucesso!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Erro ao excluir tarefa.")
                st.divider()
        else:
            st.info("Nenhuma tarefa cadastrada.")

    with tab_tarefas_admin:
        st.subheader("📋 Tarefas a Entregar")
        import pandas as pd
        tarefas_admin = load_deadlines()
        if tarefas_admin:
            # Adicionar coluna de urgência
            for t in tarefas_admin:
                t["⚡ Urgência"] = calcular_urgencia_tarefa(t.get("data_limite", ""))

            colunas_exibidas = ["⚡ Urgência", "disciplina", "atividade", "data_limite", "formato_envio", "link_destino", "detalhes"]
            colunas_presentes = [c for c in colunas_exibidas if c in tarefas_admin[0]]
            df_tarefas_admin = pd.DataFrame(tarefas_admin)[colunas_presentes]

            if st.checkbox("Mostrar apenas a vencer", key="filter_admin"):
                df_tarefas_admin = df_tarefas_admin[~df_tarefas_admin["data_limite"].apply(_tarefa_expirada)]

            # Renomear para melhor legibilidade
            df_tarefas_admin = df_tarefas_admin.rename(columns={
                "disciplina": "Disciplina",
                "atividade": "Atividade",
                "data_limite": "Data Limite",
                "formato_envio": "Formato",
                "link_destino": "Link",
                "detalhes": "Detalhes",
            })

            config_colunas = {
                "⚡ Urgência": st.column_config.TextColumn(
                    "⚡ Urgência",
                    alignment="center",
                    width="small",
                ),
                "Disciplina": st.column_config.TextColumn(
                    "Disciplina",
                    width="medium",
                ),
                "Atividade": st.column_config.TextColumn(
                    "Atividade",
                    width="medium",
                ),
                "Data Limite": st.column_config.TextColumn(
                    "Data Limite",
                    width="medium",
                ),
                "Formato": st.column_config.TextColumn(
                    "Formato",
                    width="small",
                ),
                "Link": st.column_config.LinkColumn(
                    "Link",
                    display_text="Acessar Link",
                    width="small",
                ),
                "Detalhes": st.column_config.TextColumn(
                    "Detalhes",
                    width="large",
                ),
            }

            st.dataframe(
                df_tarefas_admin,
                use_container_width=True,
                hide_index=True,
                column_config=config_colunas
            )

            st.divider()
            st.caption("🟢 Mais de 2 dias  •  🟡 Até 2 dias  •  🔴 1 dia ou menos  •  🟣 Vencida")
        else:
            st.info("Nenhuma tarefa cadastrada.")

    with tab_ragas:
        st.subheader("📈 Avaliação RAGAS (Proxy)")
        st.caption("Executa as 3 métricas proxy (Answer Relevancy, Faithfulness, Context Precision) contra o pipeline RAG atual.")

        perguntas_ragas = [
            "Quais os criterios de desligamento da residencia?",
            "Qual a distribuicao de peso das notas?",
            "O que diz o regulamento em caso de perda de prazo?",
            "Quantas faltas sao permitidas?",
            "Qual a carga horaria total da fase 2?",
        ]

        if "ragas_results" not in st.session_state:
            st.session_state.ragas_results = None

        if st.button("🚀 Executar Avaliação RAGAS", key="run_ragas"):
            with st.spinner("Avaliando pipeline... isso pode levar ~30s"):
                resultados_relevancia = []
                resultados_fidelidade = []
                resultados_precisao = []
                detalhes = []

                modelo_llm = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

                for q in perguntas_ragas:
                    # Retrieve
                    hits = pipeline.retrieve(q, k=5)
                    contexto = "\n\n".join(
                        f"[{h['source']}:p{h['page']}] {h['text']}" for h in hits
                    )

                    # Gerar resposta
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

                    # --- Answer Relevancy (proxy) ---
                    resp_lower = resposta.lower()
                    eh_valida = len(resp_lower) > 20 and "nao encontrado" not in resp_lower
                    score_relevancia = 1.0 if eh_valida else 0.0
                    resultados_relevancia.append(score_relevancia)

                    # --- Faithfulness (proxy) ---
                    tem_citacao = "[" in resposta and "]" in resposta and ":" in resposta
                    score_fidelidade = 1.0 if tem_citacao else 0.0
                    resultados_fidelidade.append(score_fidelidade)

                    # --- Context Precision (proxy) ---
                    fontes_unicas = len(set(h["source"] for h in hits))
                    score_precisao = min(fontes_unicas / max(len(hits), 1), 1.0)
                    resultados_precisao.append(score_precisao)

                    detalhes.append({
                        "Pergunta": q[:60] + "...",
                        "Relevância": f"{score_relevancia:.0%}",
                        "Fidelidade": f"{score_fidelidade:.0%}",
                        "Precisão": f"{score_precisao:.0%}",
                        "Chunks": len(hits),
                    })

                media_rel = sum(resultados_relevancia) / len(resultados_relevancia)
                media_fid = sum(resultados_fidelidade) / len(resultados_fidelidade)
                media_prec = sum(resultados_precisao) / len(resultados_precisao)

                st.session_state.ragas_results = {
                    "relevancy": media_rel,
                    "faithfulness": media_fid,
                    "precision": media_prec,
                    "details": detalhes,
                }

        if st.session_state.ragas_results:
            res = st.session_state.ragas_results

            col_r, col_f, col_p = st.columns(3)
            with col_r:
                delta_r = "✅ Passou" if res["relevancy"] >= 0.85 else "❌ Abaixo"
                st.metric("Answer Relevancy", f"{res['relevancy']:.0%}", delta=delta_r, help="Target ≥ 85%")
            with col_f:
                delta_f = "✅ Passou" if res["faithfulness"] >= 0.90 else "❌ Abaixo"
                st.metric("Faithfulness", f"{res['faithfulness']:.0%}", delta=delta_f, help="Target ≥ 90%")
            with col_p:
                delta_p = "✅ Passou" if res["precision"] >= 0.80 else "❌ Abaixo"
                st.metric("Context Precision", f"{res['precision']:.0%}", delta=delta_p, help="Target ≥ 80%")

            st.divider()
            st.subheader("Detalhes por Pergunta")
            import pandas as pd
            df_ragas = pd.DataFrame(res["details"])
            st.dataframe(df_ragas, use_container_width=True)
        else:
            st.info("Clique no botão acima para executar a avaliação.")

else:
    # ── Visão do ALUNO: abas Minhas Tarefas + Chatbot ──
    tab_tarefas_aluno, tab_chat_aluno = st.tabs(["📋 Minhas Tarefas", "💬 Chatbot"])

    with tab_tarefas_aluno:
        st.subheader("📋 Tarefas a Entregar")
        import pandas as pd
        tarefas_aluno = load_deadlines()
        if tarefas_aluno:
            # Adicionar coluna de urgência
            for t in tarefas_aluno:
                t["⚡ Urgência"] = calcular_urgencia_tarefa(t.get("data_limite", ""))

            colunas_exibidas = ["⚡ Urgência", "disciplina", "atividade", "data_limite", "formato_envio", "link_destino", "detalhes"]
            colunas_presentes = [c for c in colunas_exibidas if c in tarefas_aluno[0]]
            df_tarefas = pd.DataFrame(tarefas_aluno)[colunas_presentes]

            if st.checkbox("Mostrar apenas a vencer", key="filter_aluno"):
                df_tarefas = df_tarefas[~df_tarefas["data_limite"].apply(_tarefa_expirada)]

            # Renomear para melhor legibilidade
            df_tarefas = df_tarefas.rename(columns={
                "disciplina": "Disciplina",
                "atividade": "Atividade",
                "data_limite": "Data Limite",
                "formato_envio": "Formato",
                "link_destino": "Link",
                "detalhes": "Detalhes",
            })

            config_colunas = {
                "⚡ Urgência": st.column_config.TextColumn(
                    "⚡ Urgência",
                    alignment="center",
                    width="small",
                ),
                "Disciplina": st.column_config.TextColumn(
                    "Disciplina",
                    width="medium",
                ),
                "Atividade": st.column_config.TextColumn(
                    "Atividade",
                    width="medium",
                ),
                "Data Limite": st.column_config.TextColumn(
                    "Data Limite",
                    width="medium",
                ),
                "Formato": st.column_config.TextColumn(
                    "Formato",
                    width="small",
                ),
                "Link": st.column_config.LinkColumn(
                    "Link",
                    display_text="Acessar Link",
                    width="small",
                ),
                "Detalhes": st.column_config.TextColumn(
                    "Detalhes",
                    width="large",
                ),
            }

            st.dataframe(
                df_tarefas,
                use_container_width=True,
                hide_index=True,
                column_config=config_colunas
            )

            st.divider()
            st.caption("🟢 Mais de 2 dias  •  🟡 Até 2 dias  •  🔴 1 dia ou menos  •  🟣 Vencida")
        else:
            st.info("Nenhuma tarefa cadastrada ainda. O administrador ainda não lançou tarefas.")

with st.sidebar:
    st.divider()
    st.header("📊 Métricas")

    st.markdown(
        """
        <style>
        .custom-metric {
            background: #1e2025;
            border: 1px solid #283048;
            border-radius: 10px;
            padding: 6px 10px;
            margin-bottom: 6px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .custom-metric label {
            color: #a3a3a3;
            font-size: 1.0rem;
            font-weight: 500;
            display: block;
            line-height: 1.1;
        }
        .custom-metric value {
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: 700;
            display: block;
            line-height: 1.2;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _metrica(label: str, valor: object) -> None:
        st.markdown(
            f'<div class="custom-metric"><label>{label}</label><value>{valor}</value></div>',
            unsafe_allow_html=True,
        )

    _metrica("Chunks indexados", pipeline.collection.count())
    if hasattr(pipeline, "llm_model"):
        _metrica("Modelo LLM", pipeline.llm_model)

    _metrica("Busca", "Híbrida (BM25 + Semântica)")
    bm25_ok = pipeline.indice_bm25._indice is not None
    _metrica("Índice BM25", "✅ Ativo" if bm25_ok else "❌ Inativo")

    tracker = get_latency_tracker()
    p95 = tracker.p95()
    _metrica("Latência P95 (ms)", f"{p95:.0f}" if p95 else "—")
    _metrica("Requisições", tracker.count())

    exact_rate = exact_cache.hit_rate()
    sem_rate = semantic_cache.hit_rate()
    _metrica("Cache Hit (exact)", f"{exact_rate:.0%}" if exact_rate else "0%")
    _metrica("Cache Hit (semântico)", f"{sem_rate:.0%}" if sem_rate else "0%")

    if st.button("Limpar caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos")

if is_admin:
    chat_container = tab_chat
else:
    chat_container = tab_chat_aluno

with chat_container:
    col_chat_header, col_chat_clear = st.columns([6, 1])
    with col_chat_header:
        st.header("💬 Chat com o Edital")
    with col_chat_clear:
        if st.button("🗑️ Limpar chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "extra" in msg:
                with st.expander("🔍 Detalhes"):
                    for k, v in msg["extra"].items():
                        st.caption(f"{k}: {v}")

    if prompt := st.chat_input("Pergunte sobre o edital, prazos ou regras da residência..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with trace("chat_answer", query=prompt) as ctx:
                _call_start = time.perf_counter()
                trace_id = ctx["trace_id"]
                placeholder = st.empty()
                placeholder.markdown("_pensando..._")

                cached_exact = exact_cache.get(prompt)
                if cached_exact:
                    placeholder.markdown(cached_exact)
                    log_event("cache_hit", trace_id=trace_id, layer="exact")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": cached_exact,
                        "extra": {"Cache": "✅ Exact Hit"},
                    })
                    st.stop()

                try:
                    cached_sem = semantic_cache.get(prompt)
                except NotImplementedError:
                    cached_sem = None

                if cached_sem:
                    placeholder.markdown(cached_sem)
                    log_event("cache_hit", trace_id=trace_id, layer="semantic")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": cached_sem,
                        "extra": {"Cache": "✅ Semantic Hit"},
                    })
                    st.stop()

                try:
                    decision = classify_complexity(prompt)
                    model_used = decision.model
                    route_info = f"Rota: {decision.complexity} → `{model_used}` ({decision.reason})"
                    log_event("route_decision", trace_id=trace_id, **decision.__dict__)
                except NotImplementedError:
                    model_used = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
                    route_info = f"Modelo default: `{model_used}`"

                try:
                    hits = pipeline.retrieve(prompt, k=12)
                    context = "\n\n".join(
                        f"[{h['source']}:p{h['page']}] {h['text']}" for h in hits if h
                    )
                    context = context.strip()
                    if not hits:
                        log_event("retrieve_empty", trace_id=trace_id, msg="retrieve() returned 0 hits")
                except Exception as e:
                    log_event("retrieve_error", trace_id=trace_id, error=str(e))
                    hits = []
                    context = ""

                data_atual = datetime.now()
                dia_da_semana = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"][data_atual.weekday()]
                data_atual_formatada = data_atual.strftime("%d-%m-%Y %H:%M:%S")

                needs_tools = any(kw in prompt.lower() for kw in ["prazo", "cronograma", "entrega", "entregar", "deadlines", "tarefa", "semana"])

                tool_context = ""
                if needs_tools:
                    tool_context = sidi_tools.consultar_prazos_entregas()

                full_context = context
                if tool_context:
                    full_context += "\n\n[consultar_prazos_entregas]\n" + tool_context

                system_msg = (
                    "Voce e um assistente especializado no edital da residencia tecnologica SiDi.\n"
                    f"Data/hora atual do sistema: {data_atual_formatada} ({dia_da_semana}).\n\n"
                    "VOCE SO PODE RESPONDER COM BASE ESTRITAMENTE NO CONTEXTO ABAIXO.\n"
                    "Se a informacao solicitada nao estiver no contexto, diga exatamente 'Nao encontrado no corpus' — nunca use seu conhecimento proprio.\n"
                    "Sempre cite a fonte usando o formato [arquivo:pagina].\n\n"
                    "REGRAS:\n"
                    "- Liste as informacoes de forma concisa, em topicos.\n"
                    "- NUNCA repita a mesma informacao.\n"
                    "- Se o contexto mencionar valores, prazos ou condicoes, extraia exatamente como esta.\n"
                    "- Quando o usuario pedir informacao sobre disciplinas, meses ou professores, USE SOMENTE a GRADE (ignore resumos do edital sobre isso).\n"
                    "- Voce pode deduzir o mes a partir das datas no formato MM/DD/YY (ex: 12/4/26 e Dezembro de 2026).\n"
                    "- Se o usuario pedir para enviar lembrete por email, peca o email e o ID da tarefa — voce nao pode enviar diretamente.\n\n"
                    f"=== CONTEXTO ===\n{full_context}\n=== FIM DO CONTEXTO ==="
                )

                user_msg = prompt

                try:
                    response = pipeline.client.chat.completions.create(
                        model=model_used,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.1,
                        stream=False,
                    )
                    msg = response.choices[0].message
                    answer = msg.content
                    placeholder.markdown(answer)
                    extra = {"Routing": route_info}
                    if tool_context:
                        extra["Dados"] = "Edital + Prazos"
                    else:
                        extra["Busca"] = "Híbrida (BM25 + Semântica + RRF)"

                except NotImplementedError as e:
                    answer = f"⚠️ Pipeline não implementado: {e}"
                    extra = {"Erro": str(e)}

                elapsed_ms = (time.perf_counter() - _call_start) * 1000
                extra["Latência"] = f"{elapsed_ms:.0f}ms"

                placeholder.markdown(answer)

                exact_cache.put(prompt, answer)
                semantic_cache.put(prompt, answer)
                log_event("answer_generated", trace_id=trace_id, model=model_used)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "extra": extra,
                })

    if st.session_state.messages:
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption(
    "Lembrei, SiDi! — Projeto Portfólio | "
    "Dúvidas sobre o edital? Consulte o corpus ou pergunte ao assistente."
    )
