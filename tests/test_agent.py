import json

from agente_delivery.agent import MENSAGEM_LIMITE_PASSOS, Agent
from agente_delivery.llm import LLMResponse, ScriptedLLM, ToolCall, chamada, texto


def _agente(tools, roteiro, confirma=True, **kw):
    pedidos_de_confirmacao = []

    def confirmar(nome, args, resumo):
        pedidos_de_confirmacao.append(resumo)
        return confirma

    llm = ScriptedLLM(roteiro)
    return Agent(llm=llm, tools=tools, confirmar=confirmar, **kw), llm, pedidos_de_confirmacao


def test_resposta_direta_sem_ferramentas(tools):
    agente, _, _ = _agente(tools, [texto("Olá! Em que posso ajudar?")])
    r = agente.enviar("oi")
    assert r.texto == "Olá! Em que posso ajudar?"
    assert r.passos == 1 and r.chamadas == []


def test_resultado_da_ferramenta_volta_para_o_modelo(tools):
    agente, llm, _ = _agente(
        tools, [chamada("ver_cardapio", restaurante_id="brasa-burger"), texto("O Brasa Duplo custa R$ 38,90.")]
    )
    r = agente.enviar("quanto custa o brasa duplo?")
    assert r.passos == 2
    msg_tool = llm.chamadas[1][-1]
    assert msg_tool["role"] == "tool" and msg_tool["tool_call_id"] == "call_ver_cardapio"
    assert "R$ 38,90" in msg_tool["content"]


def test_erro_de_regra_vira_observacao_e_nao_excecao(tools):
    agente, llm, _ = _agente(
        tools, [chamada("adicionar_item", item_id="nao-existe"), texto("Não encontrei esse item.")]
    )
    r = agente.enviar("adiciona X")
    assert "erro" in r.chamadas[0]["resultado"]
    assert "erro" in json.loads(llm.chamadas[1][-1]["content"])


def test_json_invalido_nas_ferramentas(tools):
    quebrado = LLMResponse(content=None, tool_calls=[ToolCall.from_raw("c1", "adicionar_item", "{item_id: ")])
    agente, _, _ = _agente(tools, [quebrado, texto("ok")])
    r = agente.enviar("x")
    assert "JSON inválido" in r.chamadas[0]["resultado"]["erro"]


def test_varias_tool_calls_no_mesmo_passo(tools):
    paralelo = LLMResponse(
        content=None,
        tool_calls=[
            ToolCall.from_raw("a", "definir_endereco", json.dumps({"bairro": "Pinheiros"})),
            ToolCall.from_raw("b", "adicionar_item", json.dumps({"item_id": "brasa-classico"})),
        ],
    )
    agente, llm, _ = _agente(tools, [paralelo, texto("Adicionado!")])
    agente.enviar("um clássico pra pinheiros")
    ids = [m["tool_call_id"] for m in llm.chamadas[1] if m["role"] == "tool"]
    assert ids == ["a", "b"]
    assert tools.ver_carrinho()["frete"] == "R$ 4,99"


def _roteiro_ate_finalizar():
    return [
        chamada("definir_endereco", bairro="pinheiros"),
        chamada("adicionar_item", item_id="nonna-margherita"),
        chamada("finalizar_pedido", forma_pagamento="pix"),
        texto("Pronto!"),
    ]


def test_finalizacao_exige_confirmacao_humana(tools):
    agente, _, confirmacoes = _agente(tools, _roteiro_ate_finalizar(), confirma=True)
    agente.enviar("margherita pra pinheiros no pix, pode fechar")
    assert len(confirmacoes) == 1
    assert confirmacoes[0]["total"] == "R$ 60,89"
    assert len(tools.estado.pedidos) == 1


def test_humano_recusa_e_nada_e_enviado(tools):
    agente, _, _ = _agente(tools, _roteiro_ate_finalizar(), confirma=False)
    r = agente.enviar("...")
    assert tools.estado.pedidos == []
    assert "NÃO confirmou" in r.chamadas[-1]["resultado"]["erro"]
    assert "nonna-margherita" in tools.estado.carrinho


def test_nao_pede_confirmacao_de_pedido_invalido(tools):
    # Sem endereço: a validação falha antes de incomodar o humano.
    roteiro = [
        chamada("adicionar_item", item_id="nonna-margherita"),
        chamada("finalizar_pedido", forma_pagamento="pix"),
        texto("Qual o bairro?"),
    ]
    agente, _, confirmacoes = _agente(tools, roteiro)
    r = agente.enviar("...")
    assert confirmacoes == []
    assert "Endereço" in r.chamadas[-1]["resultado"]["erro"]


def test_limite_de_passos_evita_loop_infinito(tools):
    roteiro = [chamada("ver_carrinho", id=f"c{i}") for i in range(10)]
    agente, _, _ = _agente(tools, roteiro, max_passos=3)
    r = agente.enviar("...")
    assert r.atingiu_limite and r.passos == 3
    assert r.texto == MENSAGEM_LIMITE_PASSOS
    assert agente.mensagens[-1] == {"role": "assistant", "content": MENSAGEM_LIMITE_PASSOS}


def test_historico_persiste_entre_turnos(tools):
    agente, llm, _ = _agente(tools, [texto("Oi!"), texto("Claro.")])
    agente.enviar("primeira")
    agente.enviar("segunda")
    conteudos = [m["content"] for m in llm.chamadas[1] if m["role"] == "user"]
    assert conteudos == ["primeira", "segunda"]


def test_trace_registra_eventos(tools, tmp_path):
    agente, _, _ = _agente(tools, _roteiro_ate_finalizar())
    agente.enviar("...")
    tipos = [e.tipo for e in agente.trace]
    assert tipos.count("tool") == 3 and "confirmacao" in tipos
    arq = tmp_path / "trace.jsonl"
    agente.salvar_trace(str(arq))
    assert len(arq.read_text().splitlines()) == len(agente.trace)
