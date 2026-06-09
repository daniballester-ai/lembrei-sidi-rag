from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

DEADLINES_PATH = Path(__file__).resolve().parents[2] / "data" / "deadlines.json"


def load_deadlines() -> list[dict]:
    if not DEADLINES_PATH.exists():
        return []
    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_deadlines(data: list[dict]) -> None:
    with open(DEADLINES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def delete_deadline(id_tarefa: str) -> bool:
    atividades = load_deadlines()
    nova_lista = [a for a in atividades if a.get("id") != id_tarefa]
    if len(nova_lista) == len(atividades):
        return False
    save_deadlines(nova_lista)
    return True


def get_sidi_deadlines(mes: str | None = None) -> str:
    """Retorna o cronograma de entregas do SiDi filtrado por mes (opcional).

    Args:
        mes: Mes para filtrar no formato 'YYYY-MM' (ex: '2026-06').
             Se None, retorna todas as atividades cadastradas.
    """
    atividades = load_deadlines()
    if not atividades:
        return "Nenhuma atividade cadastrada no momento."

    if mes:
        ano, mes_num = mes.split("-")
        filtradas = [
            a for a in atividades
            if len(a.get("data_limite", "")) >= 10
            and a["data_limite"][6:10] == ano
            and a["data_limite"][3:5] == mes_num
        ]
        if not filtradas:
            return f"Nenhuma atividade encontrada para {mes}."
        atividades = filtradas

    linhas = ["**Cronograma de Atividades SiDi:**\n"]
    for a in atividades:
        linha = (
            f"- **{a.get('disciplina', 'N/A')}** — {a.get('atividade', 'N/A')}\n"
            f"  Data: {a.get('data_limite', 'N/A')}\n"
            f"  Formato: {a.get('formato_envio', 'N/A')}\n"
            f"  Detalhes: {a.get('detalhes', 'N/A')}"
        )
        if a.get("link_destino"):
            linha += f"\n  Link: {a['link_destino']}"
        linhas.append(linha)

    return "\n".join(linhas)


def consultar_prazos_entregas(filtro_mes: str | None = None) -> str:
    """Retorna o cronograma de tarefas e prazos de entrega do SiDi, opcionalmente filtrado por mês.

    Args:
        filtro_mes: Mês para filtrar as tarefas no formato 'YYYY-MM' (ex: '2026-06').
    """
    tarefas = load_deadlines()
    if not tarefas:
        return "Nenhuma tarefa cadastrada no momento."

    if filtro_mes:
        partes = filtro_mes.split("-")
        if len(partes) == 2:
            ano, mes_num = partes
            filtradas = []
            for t in tarefas:
                data = t.get("data_limite", "")
                if len(data) >= 10:
                    if data[6:10] == ano and data[3:5] == mes_num:
                        filtradas.append(t)
            tarefas = filtradas

    if not tarefas:
        return f"Nenhuma tarefa cadastrada para o mês {filtro_mes}."

    linhas = ["**Cronograma de Atividades SiDi:**\n"]
    for t in tarefas:
        linha = (
            f"- **{t.get('disciplina', 'N/A')}** — {t.get('atividade', 'N/A')}\n"
            f"  ID da Tarefa: {t.get('id', 'N/A')}\n"
            f"  Prazo Limite: {t.get('data_limite', 'N/A')}\n"
            f"  Formato de Envio: {t.get('formato_envio', 'N/A')}\n"
            f"  Detalhes: {t.get('detalhes', 'N/A')}"
        )
        if t.get("link_destino"):
            linha += f"\n  Link de Destino: {t['link_destino']}"
        linhas.append(linha)

    return "\n".join(linhas)


def schedule_deadline_email(email_aluno: str, id_tarefa: str) -> str:
    """Agenda um alerta por e-mail para o aluno sobre uma tarefa especifica.

    Args:
        email_aluno: Email institucional do aluno.
        id_tarefa: ID unico da tarefa cadastrada no sistema.
    """
    atividades = load_deadlines()
    tarefa = next((a for a in atividades if a.get("id") == id_tarefa), None)
    if not tarefa:
        return f"ERRO: Tarefa com ID '{id_tarefa}' nao encontrada."

    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        return "ERRO: RESEND_API_KEY nao configurada no .env"

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": "SiDi-Insight <onboarding@resend.dev>",
                "to": [email_aluno],
                "subject": f"Lembrete: {tarefa['atividade']} — {tarefa['disciplina']}",
                "text": (
                    f"Ola,\n\n"
                    f"Este e um alerta do SiDi-Insight Bot.\n\n"
                    f"Atividade: {tarefa['atividade']}\n"
                    f"Disciplina: {tarefa['disciplina']}\n"
                    f"Data limite: {tarefa['data_limite']}\n"
                    f"Formato de envio: {tarefa['formato_envio']}\n"
                    f"Link: {tarefa.get('link_destino', 'N/A')}\n"
                    f"Detalhes: {tarefa.get('detalhes', 'N/A')}\n\n"
                    f"Nao perca o prazo!\n"
                    f"— SiDi-Insight Bot"
                ),
            },
            timeout=10,
        )
        if resp.ok:
            return f"Email enviado com sucesso para {email_aluno} sobre a tarefa '{tarefa['atividade']}'."
        return f"ERRO ao enviar email: {resp.status_code} — {resp.text}"
    except Exception as e:
        return f"ERRO na requisicao: {e}"


def cadastrar_atividade(
    disciplina: str,
    atividade: str,
    data_limite: str,
    formato_envio: str = "Link Externo",
    link_destino: str = "",
    detalhes: str = "",
) -> str:
    """Cadastra uma nova atividade no banco de deadlines.

    Args:
        disciplina: Nome da disciplina.
        atividade: Nome/titulo da atividade.
        data_limite: Data limite no formato 'DD-MM-YYYY HH:MM'.
        formato_envio: 'E-mail' ou 'Link Externo'.
        link_destino: URL de destino (opcional).
        detalhes: Informacoes adicionais (opcional).
    """
    nova = {
        "id": str(uuid.uuid4())[:8],
        "disciplina": disciplina,
        "atividade": atividade,
        "data_limite": data_limite,
        "formato_envio": formato_envio,
        "link_destino": link_destino,
        "detalhes": detalhes,
    }
    dados = load_deadlines()
    dados.append(nova)
    save_deadlines(dados)
    return f"Atividade '{atividade}' cadastrada com ID {nova['id']}."


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "consultar_prazos_entregas",
            "description": "Consulta o cronograma de tarefas e prazos de entrega do SiDi cadastrados no sistema, retornando detalhes como data limite, formato de envio, link de destino e detalhes da atividade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filtro_mes": {
                        "type": "string",
                        "description": "Mês opcional para filtrar no formato YYYY-MM (ex: 2026-06). Se vazio, retorna todas as tarefas.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_deadline_email",
            "description": "Envia um alerta por email para o aluno sobre uma tarefa especifica",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_aluno": {
                        "type": "string",
                        "description": "Email institucional do aluno",
                    },
                    "id_tarefa": {
                        "type": "string",
                        "description": "ID unico da tarefa no sistema",
                    },
                },
                "required": ["email_aluno", "id_tarefa"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "get_sidi_deadlines": get_sidi_deadlines,
    "consultar_prazos_entregas": consultar_prazos_entregas,
    "schedule_deadline_email": schedule_deadline_email,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"

