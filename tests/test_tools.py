from datetime import datetime

import pytest

from agente_delivery.catalog import formatar_reais, normalizar
from agente_delivery.tools import FUSO, ToolError


def test_formatar_reais():
    assert formatar_reais(0) == "R$ 0,00"
    assert formatar_reais(5490) == "R$ 54,90"
    assert formatar_reais(123456) == "R$ 1.234,56"


def test_normalizar_remove_acentos_e_espacos():
    assert normalizar("  Vila   Madalena ") == "vila madalena"
    assert normalizar("Glúten") == "gluten"


def test_buscar_filtra_por_culinaria_e_bairro(tools):
    r = tools.buscar_restaurantes(busca="Japonesa", bairro="Moema")
    assert [x["restaurante_id"] for x in r["restaurantes"]] == ["sushi-kaze"]


@pytest.mark.parametrize(
    ("busca", "esperado"),
    [
        ("sushi", "sushi-kaze"),  # bug real encontrado na avaliação com gpt-oss-20b
        ("comida japonesa", "sushi-kaze"),
        ("pizza", "cantina-nonna"),
        ("Brasa Burger", "brasa-burger"),
        ("tiramisù", "cantina-nonna"),  # nome de prato
    ],
)
def test_buscar_por_sinonimo_nome_ou_prato(tools, busca, esperado):
    ids = [x["restaurante_id"] for x in tools.buscar_restaurantes(busca=busca)["restaurantes"]]
    assert esperado in ids


def test_busca_vazia_explica_o_que_existe(tools):
    tools.estado.agora = datetime(2026, 11, 11, 9, 0, tzinfo=FUSO)
    r = tools.buscar_restaurantes(busca="pizza")
    assert r["total"] == 0
    assert r["dica"]["fechados_agora_que_combinam"] == ["Cantina da Nonna (abre 18:00)"]
    assert "japonesa" in r["dica"]["culinarias_disponiveis"]


def test_buscar_respeita_horario(tools):
    tools.estado.agora = datetime(2026, 11, 11, 9, 0, tzinfo=FUSO)
    abertos = {x["restaurante_id"] for x in tools.buscar_restaurantes()["restaurantes"]}
    assert "cantina-nonna" not in abertos  # abre 18h
    todos = {x["restaurante_id"] for x in tools.buscar_restaurantes(apenas_abertos=False)["restaurantes"]}
    assert "cantina-nonna" in todos


def test_horario_que_atravessa_meia_noite(tools, catalogo):
    brasa = catalogo.restaurantes["brasa-burger"]  # 11:00-02:00
    assert brasa.aberto_em(datetime(2026, 11, 11, 1, 30, tzinfo=FUSO))
    assert not brasa.aberto_em(datetime(2026, 11, 11, 3, 0, tzinfo=FUSO))


def test_totais_com_frete(tools):
    tools.definir_endereco("Pinheiros")
    r = tools.adicionar_item("nonna-margherita", 2)
    assert r["subtotal"] == "R$ 109,80"
    assert r["frete"] == "R$ 5,99"
    assert r["total"] == "R$ 115,79"


def test_item_inexistente(tools):
    with pytest.raises(ToolError, match="não existe"):
        tools.adicionar_item("pizza-de-unicornio")


def test_carrinho_aceita_um_restaurante(tools):
    tools.adicionar_item("nonna-margherita")
    with pytest.raises(ToolError, match="já tem itens"):
        tools.adicionar_item("brasa-classico")
    tools.esvaziar_carrinho()
    tools.adicionar_item("brasa-classico")
    assert tools.estado.restaurante_id == "brasa-burger"


def test_alergia_bloqueia_item(tools):
    tools.registrar_restricoes(["crustaceos"])
    with pytest.raises(ToolError, match="restrição"):
        tools.adicionar_item("kaze-hotroll-camarao")
    assert tools.estado.carrinho == {}
    cardapio = tools.ver_cardapio("sushi-kaze")
    bloqueados = {i["item_id"] for i in cardapio["itens"] if i["bloqueado_por_restricao"]}
    assert bloqueados == {"kaze-hotroll-camarao"}


def test_restricao_registrada_depois_aponta_conflito(tools):
    tools.adicionar_item("nonna-tiramisu")
    r = tools.registrar_restricoes(["ovo"])
    assert r["itens_no_carrinho_com_conflito"] == ["nonna-tiramisu"]


def test_cupom_percentual(tools):
    tools.adicionar_item("nonna-calabresa")  # 52,90
    r = tools.aplicar_cupom("bemvindo10")
    assert r["cupom"] == "BEMVINDO10"
    assert r["desconto"] == "R$ 5,29"


def test_cupom_expirado(tools):
    tools.adicionar_item("kaze-combo20")
    with pytest.raises(ToolError, match="expirou"):
        tools.aplicar_cupom("VERAO2025")


def test_cupom_inexistente_nao_e_aplicado(tools):
    tools.adicionar_item("verde-bowl")
    with pytest.raises(ToolError, match="não existe"):
        tools.aplicar_cupom("FUNCIONARIO100")
    assert tools.estado.cupom is None


def test_cupom_restrito_a_restaurante(tools):
    tools.adicionar_item("kaze-combo20")
    with pytest.raises(ToolError, match="restaurante"):
        tools.aplicar_cupom("BRASA15")


def test_cupom_deixa_de_valer_quando_subtotal_cai(tools):
    tools.definir_endereco("pinheiros")
    tools.adicionar_item("brasa-duplo", 2)  # 77,80
    tools.aplicar_cupom("BRASA15")
    r = tools.remover_item("brasa-duplo", 1)  # 38,90 < 60,00
    assert r["desconto"] == "R$ 0,00"
    assert any("BRASA15" in a for a in r["avisos"])
    with pytest.raises(ToolError, match="deixou de valer"):
        tools.preparar_finalizacao("pix")


def test_frete_gratis(tools):
    tools.definir_endereco("moema")
    tools.adicionar_item("kaze-combo20")
    r = tools.aplicar_cupom("FRETEGRATIS")
    assert r["frete"] == "R$ 9,99"
    assert r["desconto"] == "R$ 9,99"
    assert r["total"] == "R$ 79,90"


@pytest.mark.parametrize(
    ("bairro", "item", "hora", "mensagem"),
    [
        (None, "nonna-margherita", 20, "Endereço"),
        ("moema", "verde-burger", 20, "não entrega"),
        ("pinheiros", "nonna-margherita", 9, "fechado"),
        ("pinheiros", "verde-suco", 20, "Pedido mínimo"),
    ],
)
def test_finalizacao_bloqueada_por_regras(tools, bairro, item, hora, mensagem):
    tools.estado.agora = datetime(2026, 11, 11, hora, 0, tzinfo=FUSO)
    if bairro:
        tools.definir_endereco(bairro)
    tools.adicionar_item(item)
    with pytest.raises(ToolError, match=mensagem):
        tools.finalizar_pedido("pix")
    assert tools.estado.pedidos == []


def test_finalizar_pedido_limpa_carrinho(tools):
    tools.definir_endereco("pinheiros")
    tools.adicionar_item("nonna-margherita", observacao="sem manjericão")
    pedido = tools.finalizar_pedido("pix")
    assert pedido["pedido_id"] == "PED-0001"
    assert pedido["itens"][0]["observacao"] == "sem manjericão"
    assert tools.estado.carrinho == {}


@pytest.mark.parametrize(
    ("args", "mensagem"),
    [
        ({}, "obrigatórios"),
        ({"item_id": "brasa-classico", "quantidade": 0}, ">= 1"),
        ({"item_id": "brasa-classico", "quantidade": True}, "integer"),
        ({"item_id": "brasa-classico", "preco": 1}, "não suportados"),
    ],
)
def test_validacao_de_argumentos(tools, args, mensagem):
    with pytest.raises(ToolError, match=mensagem):
        tools.executar("adicionar_item", args)


def test_forma_pagamento_invalida(tools):
    with pytest.raises(ToolError, match="deve ser um de"):
        tools.executar("finalizar_pedido", {"forma_pagamento": "bitcoin"})
