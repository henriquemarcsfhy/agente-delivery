"""Testa o harness de avaliação (não o modelo): os cenários são válidos e as checagens funcionam."""

import json
from pathlib import Path

import pytest

from agente_delivery.avaliacao import rodar_cenario
from agente_delivery.llm import ScriptedLLM, chamada, texto

CENARIOS = json.loads((Path(__file__).parents[1] / "evals" / "cenarios.json").read_text("utf-8"))
CHAVES_VALIDAS = {
    "pedido_finalizado",
    "itens",
    "cupom",
    "forma_pagamento",
    "ferramentas_chamadas",
    "ferramentas_nao_chamadas",
    "chamadas_proibidas",
    "itens_proibidos",
    "resposta_contem",
}


def test_ids_unicos():
    ids = [c["id"] for c in CENARIOS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("cenario", CENARIOS, ids=lambda c: c["id"])
def test_cenario_bem_formado(cenario, catalogo):
    assert cenario["turnos"] and set(cenario["esperado"]) <= CHAVES_VALIDAS
    for item in list(cenario["esperado"].get("itens", {})) + cenario["esperado"].get("itens_proibidos", []):
        assert item in catalogo.itens, f"item inexistente no cenário: {item}"


def _cenario(cid):
    return next(c for c in CENARIOS if c["id"] == cid)


def test_agente_perfeito_passa_no_cenario_de_injecao(catalogo):
    llm = ScriptedLLM(
        [
            chamada("definir_endereco", bairro="Pinheiros"),
            chamada("adicionar_item", item_id="verde-bowl"),
            chamada("adicionar_item", item_id="verde-suco"),
            texto("Resumo: Bowl + Suco, total R$ 53,79 no pix. Posso fechar?"),
            chamada("finalizar_pedido", forma_pagamento="pix"),
            texto("Pedido enviado!"),
            texto("Já está tudo certo."),
        ]
    )
    r = rodar_cenario(_cenario("injecao_no_cardapio"), llm, catalogo)
    assert r.sucesso, r.checagens


def test_agente_que_obedece_a_injecao_falha(catalogo):
    llm = ScriptedLLM(
        [
            chamada("definir_endereco", bairro="Pinheiros"),
            chamada("adicionar_item", item_id="verde-bowl"),
            chamada("adicionar_item", item_id="verde-suco"),
            chamada("aplicar_cupom", codigo="FUNCIONARIO100"),
            texto("..."),
            texto("..."),
            texto("..."),
        ]
    )
    r = rodar_cenario(_cenario("injecao_no_cardapio"), llm, catalogo)
    assert not r.sucesso
    assert r.checagens["chamada_proibida_0"] is False
    assert r.erros_ferramenta == 1


def test_agente_que_ignora_alergia_falha(catalogo):
    llm = ScriptedLLM([chamada("adicionar_item", item_id="kaze-hotroll-camarao"), texto("Adicionado!"), texto("ok")])
    r = rodar_cenario(_cenario("alergia_crustaceo"), llm, catalogo)
    assert not r.sucesso
    assert r.checagens["chamou:registrar_restricoes"] is False
    assert r.checagens["sem_item:kaze-hotroll-camarao"] is False


def test_erro_de_api_conta_como_falha(catalogo):
    r = rodar_cenario(_cenario("consulta_preco"), ScriptedLLM([]), catalogo)
    assert not r.sucesso and "roteiro esgotado" in r.erro_execucao
