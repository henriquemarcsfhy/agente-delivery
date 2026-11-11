"""Clientes de LLM.

`OpenAICompatibleClient` fala o protocolo de Chat Completions com tools, então
funciona com OpenAI, Ollama, vLLM, Groq, OpenRouter, LM Studio e o endpoint
OpenAI-compatível do Gemini, bastando trocar `base_url` e `model`.

`ScriptedLLM` devolve respostas pré-programadas e é usado nos testes, para que
a lógica do agente seja testada de forma determinística e sem custo.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None = None  # None quando o JSON veio inválido
    parse_error: str | None = None

    @classmethod
    def from_raw(cls, id: str, name: str, raw_arguments: str | None) -> ToolCall:
        raw = raw_arguments or "{}"
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return cls(id, name, raw, None, "argumentos devem ser um objeto JSON")
            return cls(id, name, raw, parsed)
        except json.JSONDecodeError as e:
            return cls(id, name, raw, None, f"JSON inválido: {e.msg}")


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}}
                for c in self.tool_calls
            ]
        return msg


class ToolCallRejectedError(Exception):
    """O provedor recusou a tool call gerada pelo modelo antes de devolvê-la.

    Alguns provedores (ex.: Groq) validam nome e argumentos da tool call no servidor e,
    se o modelo gerar algo inválido, respondem HTTP 400 em vez de devolver a chamada.
    O agente trata isso como erro recuperável: avisa o modelo e tenta de novo.
    """


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 8,
    ):
        from openai import OpenAI  # import tardio: testes não precisam do SDK

        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base_url or os.environ.get("LLM_BASE_URL") or None,
            api_key=api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "nao-usado",
            timeout=timeout,
            # Planos gratuitos (Groq, Gemini) limitam requisições por minuto; o SDK
            # respeita o Retry-After e tenta de novo em vez de falhar o cenário.
            max_retries=max_retries,
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        from openai import BadRequestError

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
            )
        except BadRequestError as e:
            texto_erro = str(e).lower()
            if "tool call validation failed" in texto_erro or "tool_use_failed" in texto_erro:
                raise ToolCallRejectedError(str(e)) from e
            raise
        msg = resp.choices[0].message
        calls = [ToolCall.from_raw(c.id, c.function.name, c.function.arguments) for c in (msg.tool_calls or [])]
        usage = {}
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)


Passo = LLMResponse | Exception | Callable[[list[dict[str, Any]]], LLMResponse]


class ScriptedLLM:
    """LLM falso que segue um roteiro. Cada passo é uma resposta ou uma função das mensagens."""

    def __init__(self, roteiro: Sequence[Passo]):
        self._roteiro = list(roteiro)
        self.chamadas: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.chamadas.append([dict(m) for m in messages])
        if not self._roteiro:
            raise RuntimeError("ScriptedLLM: roteiro esgotado")
        passo = self._roteiro.pop(0)
        if isinstance(passo, Exception):
            raise passo
        return passo(messages) if callable(passo) else passo


def chamada(name: str, id: str | None = None, **arguments: Any) -> LLMResponse:
    """Atalho para testes: resposta do modelo com uma única tool call."""
    return LLMResponse(content=None, tool_calls=[ToolCall.from_raw(id or f"call_{name}", name, json.dumps(arguments))])


def texto(content: str) -> LLMResponse:
    return LLMResponse(content=content)
