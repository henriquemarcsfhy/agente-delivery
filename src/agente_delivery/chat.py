"""Chat no terminal.

python -m agente_delivery.chat --model gpt-4o-mini
python -m agente_delivery.chat --base-url http://localhost:11434/v1 --model qwen2.5:7b   # Ollama
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .agent import Agent
from .catalog import Catalogo
from .llm import OpenAICompatibleClient
from .tools import DeliveryTools


def confirmar_no_terminal(nome: str, args: dict[str, Any], resumo: dict[str, Any]) -> bool:
    print("\n=== CONFIRMAÇÃO NECESSÁRIA (fora do LLM) ===")
    for item in resumo["itens"]:
        obs = f" ({item['observacao']})" if item["observacao"] else ""
        print(f"  {item['quantidade']}x {item['nome']}{obs} .... {item['preco_total']}")
    print(f"  Subtotal {resumo['subtotal']} | Desconto {resumo['desconto']} | Frete {resumo['frete']}")
    print(f"  TOTAL {resumo['total']} | Pagamento: {args['forma_pagamento']}")
    return input("Confirmar pedido? [s/N] ").strip().lower() in {"s", "sim", "y"}


def main() -> None:
    p = argparse.ArgumentParser(description="Agente de pedidos de delivery")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--trace", help="Arquivo .jsonl para salvar o trace da sessão")
    p.add_argument("--debug", action="store_true", help="Mostra as chamadas de ferramenta")
    a = p.parse_args()

    agente = Agent(
        llm=OpenAICompatibleClient(model=a.model, base_url=a.base_url),
        tools=DeliveryTools(Catalogo.carregar()),
        confirmar=confirmar_no_terminal,
    )
    print("Agente de delivery. Digite 'sair' para encerrar.\n")
    try:
        while True:
            entrada = input("você> ").strip()
            if entrada.lower() in {"sair", "exit", "quit"}:
                break
            if not entrada:
                continue
            resposta = agente.enviar(entrada)
            if a.debug:
                for c in resposta.chamadas:
                    print(f"  [tool] {c['nome']}({json.dumps(c['argumentos'], ensure_ascii=False)})")
            print(f"agente> {resposta.texto}\n")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        if a.trace:
            agente.salvar_trace(a.trace)
        print(f"Tokens usados: {agente.uso_tokens}")


if __name__ == "__main__":
    main()
