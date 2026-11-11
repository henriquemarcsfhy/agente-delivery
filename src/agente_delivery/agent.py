"""Loop do agente: LLM -> tool calls -> observações -> LLM, até haver resposta em texto."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, ToolCall
from .prompts import montar_system_prompt
from .tools import FERRAMENTAS_SENSIVEIS, TOOL_SCHEMAS, DeliveryTools, ToolError

# Recebe (nome_da_ferramenta, argumentos, resumo_validado) e devolve True se o HUMANO confirmou.
ConfirmFn = Callable[[str, dict[str, Any], dict[str, Any]], bool]

MENSAGEM_LIMITE_PASSOS = "Desculpe, não consegui concluir essa etapa agora. Pode reformular o pedido ou tentar de novo?"


@dataclass
class EventoTrace:
    tipo: str  # "llm" | "tool" | "confirmacao"
    dados: dict[str, Any]
    duracao_ms: float = 0.0


@dataclass
class RespostaAgente:
    texto: str
    passos: int
    chamadas: list[dict[str, Any]] = field(default_factory=list)
    atingiu_limite: bool = False


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: DeliveryTools,
        confirmar: ConfirmFn,
        max_passos: int = 8,
        system_prompt: str | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.confirmar = confirmar
        self.max_passos = max_passos
        self.trace: list[EventoTrace] = []
        self.uso_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        prompt = system_prompt or montar_system_prompt(tools.estado.agora)
        self.mensagens: list[dict[str, Any]] = [{"role": "system", "content": prompt}]

    def enviar(self, texto_usuario: str) -> RespostaAgente:
        self.mensagens.append({"role": "user", "content": texto_usuario})
        chamadas_turno: list[dict[str, Any]] = []

        for passo in range(1, self.max_passos + 1):
            inicio = time.perf_counter()
            resposta = self.llm.complete(self.mensagens, TOOL_SCHEMAS)
            self._registrar("llm", {"conteudo": resposta.content, "tool_calls": len(resposta.tool_calls)}, inicio)
            for k, v in resposta.usage.items():
                self.uso_tokens[k] = self.uso_tokens.get(k, 0) + v
            self.mensagens.append(resposta.as_message())

            if not resposta.tool_calls:
                return RespostaAgente(texto=resposta.content or "", passos=passo, chamadas=chamadas_turno)

            for chamada in resposta.tool_calls:
                resultado = self._executar_chamada(chamada)
                chamadas_turno.append({"nome": chamada.name, "argumentos": chamada.arguments, "resultado": resultado})
                self.mensagens.append(
                    {
                        "role": "tool",
                        "tool_call_id": chamada.id,
                        "content": json.dumps(resultado, ensure_ascii=False),
                    }
                )

        # Fecha o turno de forma explícita para o histórico continuar válido.
        self.mensagens.append({"role": "assistant", "content": MENSAGEM_LIMITE_PASSOS})
        return RespostaAgente(
            texto=MENSAGEM_LIMITE_PASSOS, passos=self.max_passos, chamadas=chamadas_turno, atingiu_limite=True
        )

    def _executar_chamada(self, chamada: ToolCall) -> dict[str, Any]:
        inicio = time.perf_counter()
        if chamada.arguments is None:
            resultado: dict[str, Any] = {"erro": f"Não foi possível ler os argumentos: {chamada.parse_error}"}
        else:
            try:
                if chamada.name in FERRAMENTAS_SENSIVEIS:
                    resultado = self._executar_sensivel(chamada)
                else:
                    resultado = self.tools.executar(chamada.name, chamada.arguments)
            except ToolError as e:
                resultado = {"erro": str(e)}
        self._registrar("tool", {"nome": chamada.name, "argumentos": chamada.arguments, "resultado": resultado}, inicio)
        return resultado

    def _executar_sensivel(self, chamada: ToolCall) -> dict[str, Any]:
        # 1) valida em código, 2) pede confirmação ao humano FORA do LLM, 3) executa.
        # Assim nem uma injeção de prompt nem uma alucinação conseguem fechar um pedido sozinhas.
        resumo = self.tools.preparar_finalizacao(**chamada.arguments)
        confirmado = bool(self.confirmar(chamada.name, chamada.arguments, resumo))
        self._registrar("confirmacao", {"nome": chamada.name, "confirmado": confirmado}, time.perf_counter())
        if not confirmado:
            return {"erro": "O cliente NÃO confirmou o pedido. Nada foi enviado. Pergunte o que ele quer ajustar."}
        return self.tools.executar(chamada.name, chamada.arguments)

    def _registrar(self, tipo: str, dados: dict[str, Any], inicio: float) -> None:
        self.trace.append(EventoTrace(tipo, dados, (time.perf_counter() - inicio) * 1000))

    def salvar_trace(self, caminho: str) -> None:
        with open(caminho, "w", encoding="utf-8") as f:
            for ev in self.trace:
                f.write(
                    json.dumps({"tipo": ev.tipo, "duracao_ms": round(ev.duracao_ms, 1), **ev.dados}, ensure_ascii=False)
                    + "\n"
                )
