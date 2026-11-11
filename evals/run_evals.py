"""Roda os cenários de avaliação contra um modelo real.

Exemplos:
    LLM_API_KEY=... python evals/run_evals.py --model gpt-4o-mini --repeticoes 3
    python evals/run_evals.py --base-url http://localhost:11434/v1 --model qwen2.5:7b

Gera evals/resultados/<modelo>_<data>.json e imprime uma tabela em Markdown
pronta para colar no README.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agente_delivery.avaliacao import rodar_cenario
from agente_delivery.catalog import Catalogo
from agente_delivery.llm import OpenAICompatibleClient

AQUI = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url")
    p.add_argument("--repeticoes", type=int, default=1, help="Rodadas por cenário (mede consistência)")
    p.add_argument("--temperatura", type=float, default=0.0)
    p.add_argument("--filtro", help="Roda só cenários cujo id contém este texto")
    a = p.parse_args()

    cenarios = json.loads((AQUI / "cenarios.json").read_text("utf-8"))
    if a.filtro:
        cenarios = [c for c in cenarios if a.filtro in c["id"]]
    llm = OpenAICompatibleClient(model=a.model, base_url=a.base_url, temperature=a.temperatura)
    catalogo = Catalogo.carregar()

    resultados = []
    for c in cenarios:
        for rodada in range(a.repeticoes):
            r = rodar_cenario(c, llm, catalogo)
            resultados.append({"categoria": c.get("categoria", "-"), "rodada": rodada, **asdict(r)})
            falhas = [k for k, v in r.checagens.items() if not v]
            status = "OK " if r.sucesso else "FALHOU"
            print(
                f"[{status}] {c['id']} #{rodada + 1}  {r.erro_execucao or ''}{' falhas=' + str(falhas) if falhas else ''}"
            )

    n = len(resultados)
    sucesso = sum(r["sucesso"] for r in resultados) / n
    print(f"\n## Resultados: `{a.model}` ({len(cenarios)} cenários x {a.repeticoes} rodadas)\n")
    print(f"- Taxa de sucesso: **{sucesso:.0%}**")
    print(
        f"- Chamadas de ferramenta por cenário (mediana): {statistics.median(r['chamadas_ferramenta'] for r in resultados)}"
    )
    print(f"- Tokens por cenário (mediana): {statistics.median(r['tokens'] for r in resultados)}")
    print(f"- Latência por cenário (mediana): {statistics.median(r['latencia_s'] for r in resultados):.1f}s\n")

    por_cat: dict[str, list[bool]] = defaultdict(list)
    for r in resultados:
        por_cat[r["categoria"]].append(r["sucesso"])
    print("| Categoria | Sucesso |\n|---|---|")
    for cat, vals in sorted(por_cat.items()):
        print(f"| {cat} | {sum(vals)}/{len(vals)} |")

    saida = AQUI / "resultados"
    saida.mkdir(exist_ok=True)
    nome = f"{a.model.replace('/', '_').replace(':', '_')}_{datetime.now():%Y%m%d_%H%M}.json"
    (saida / nome).write_text(
        json.dumps(
            {"modelo": a.model, "taxa_sucesso": sucesso, "resultados": resultados}, ensure_ascii=False, indent=2
        ),
        "utf-8",
    )
    print(f"\nDetalhes salvos em evals/resultados/{nome}")


if __name__ == "__main__":
    main()
