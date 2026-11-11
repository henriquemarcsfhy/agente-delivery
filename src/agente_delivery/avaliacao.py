"""Avaliação do agente por cenários.

Cada cenário define as mensagens do cliente, o momento simulado, se o humano
confirma o pedido e um conjunto de checagens sobre o ESTADO FINAL e as
CHAMADAS DE FERRAMENTA, em vez de comparar o texto gerado. Isso deixa a
avaliação objetiva, reprodutível e barata (sem LLM-as-judge).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .agent import Agent
from .catalog import Catalogo, normalizar
from .llm import LLMClient
from .tools import FUSO, DeliveryTools, EstadoSessao

# Erros que não dizem nada sobre o modelo: limite de taxa, rede, timeout, falha do servidor.
ERROS_DE_INFRA = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}


@dataclass
class ResultadoCenario:
    id: str
    sucesso: bool
    checagens: dict[str, bool]
    passos: int
    chamadas_ferramenta: int
    erros_ferramenta: int
    rejeicoes_provedor: int
    tokens: int
    latencia_s: float
    resposta_final: str
    erro_execucao: str | None = None
    erro_infra: bool = False
    chamadas: list[dict[str, Any]] = field(default_factory=list)


def _args_contem(args: dict[str, Any] | None, esperado: dict[str, Any]) -> bool:
    if not args:
        return False
    for k, v in esperado.items():
        atual = args.get(k)
        if isinstance(v, str) and isinstance(atual, str):
            if normalizar(v) != normalizar(atual):
                return False
        elif atual != v:
            return False
    return True


def checar(
    cenario: dict[str, Any], tools: DeliveryTools, chamadas: list[dict[str, Any]], textos: list[str]
) -> dict[str, bool]:
    esperado = cenario["esperado"]
    estado = tools.estado
    pedido = estado.pedidos[-1] if estado.pedidos else None
    nomes = [c["nome"] for c in chamadas]
    r: dict[str, bool] = {}

    if "pedido_finalizado" in esperado:
        r["pedido_finalizado"] = (pedido is not None) == esperado["pedido_finalizado"]
    if "itens" in esperado:
        # Itens do pedido enviado, ou do carrinho se nada foi enviado.
        atuais = (
            {i["item_id"]: i["quantidade"] for i in pedido["itens"]}
            if pedido
            else {i: linha.quantidade for i, linha in estado.carrinho.items()}
        )
        r["itens"] = atuais == esperado["itens"]
    if "cupom" in esperado:
        cupom_final = pedido["cupom"] if pedido else estado.cupom
        r["cupom"] = cupom_final == esperado["cupom"]
    if "forma_pagamento" in esperado:
        r["forma_pagamento"] = bool(pedido) and pedido["forma_pagamento"] == esperado["forma_pagamento"]
    for nome in esperado.get("ferramentas_chamadas", []):
        r[f"chamou:{nome}"] = nome in nomes
    for nome in esperado.get("ferramentas_nao_chamadas", []):
        r[f"nao_chamou:{nome}"] = nome not in nomes
    for i, proibida in enumerate(esperado.get("chamadas_proibidas", [])):
        r[f"chamada_proibida_{i}"] = not any(
            c["nome"] == proibida["nome"] and _args_contem(c["argumentos"], proibida.get("args", {})) for c in chamadas
        )
    for item in esperado.get("itens_proibidos", []):
        no_carrinho = item in estado.carrinho
        no_pedido = bool(pedido) and any(i["item_id"] == item for i in pedido["itens"])
        r[f"sem_item:{item}"] = not (no_carrinho or no_pedido)
    # Procura em todas as respostas do agente no cenário, não só na última:
    # o agente pode explicar o problema num turno e só confirmar no seguinte.
    tudo = normalizar(" ".join(textos))
    for trecho in esperado.get("resposta_contem", []):
        r[f"resposta_contem:{trecho}"] = normalizar(trecho) in tudo
    return r


def rodar_cenario(cenario: dict[str, Any], llm: LLMClient, catalogo: Catalogo, max_passos: int = 8) -> ResultadoCenario:
    agora = datetime.fromisoformat(cenario.get("agora", "2026-11-11T20:00")).replace(tzinfo=FUSO)
    tools = DeliveryTools(catalogo, EstadoSessao(agora=agora))
    confirma = cenario.get("humano_confirma", True)
    agente = Agent(llm=llm, tools=tools, confirmar=lambda *_: confirma, max_passos=max_passos)

    chamadas: list[dict[str, Any]] = []
    textos: list[str] = []
    passos = 0
    texto_final = ""
    erro = None
    inicio = time.perf_counter()
    try:
        for msg in cenario["turnos"]:
            resp = agente.enviar(msg)
            chamadas.extend(resp.chamadas)
            passos += resp.passos
            texto_final = resp.texto
            textos.append(resp.texto)
    except Exception as e:  # falha de API, timeout etc. conta como falha do cenário
        erro = f"{type(e).__name__}: {e}"
        erro_infra = type(e).__name__ in ERROS_DE_INFRA
    else:
        erro_infra = False
    latencia = time.perf_counter() - inicio

    checagens = checar(cenario, tools, chamadas, textos) if erro is None else {}
    return ResultadoCenario(
        id=cenario["id"],
        sucesso=erro is None and all(checagens.values()),
        checagens=checagens,
        passos=passos,
        chamadas_ferramenta=len(chamadas),
        erros_ferramenta=sum(1 for c in chamadas if isinstance(c["resultado"], dict) and "erro" in c["resultado"]),
        rejeicoes_provedor=agente.rejeicoes_provedor,
        tokens=sum(agente.uso_tokens.values()),
        latencia_s=round(latencia, 2),
        resposta_final=texto_final,
        erro_execucao=erro,
        erro_infra=erro_infra,
        chamadas=chamadas,
    )
