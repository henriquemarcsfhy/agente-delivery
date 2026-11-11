"""Catálogo fictício de restaurantes, itens e cupons.

Os dados ficam em `data/catalogo.json`. Valores monetários são sempre inteiros
em centavos, para evitar erros de arredondamento com float.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time
from importlib import resources


def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos: 'Pinheiros ' -> 'pinheiros', 'Glúten' -> 'gluten'."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.lower().split())


def formatar_reais(centavos: int) -> str:
    """1234 -> 'R$ 12,34'."""
    sinal = "-" if centavos < 0 else ""
    reais, cents = divmod(abs(centavos), 100)
    reais_fmt = f"{reais:,}".replace(",", ".")
    return f"{sinal}R$ {reais_fmt},{cents:02d}"


@dataclass(frozen=True)
class Item:
    id: str
    nome: str
    descricao: str
    preco_centavos: int
    alergenos: tuple[str, ...]
    restaurante_id: str


@dataclass(frozen=True)
class Restaurante:
    id: str
    nome: str
    culinaria: str
    abre: time
    fecha: time
    pedido_minimo_centavos: int
    frete_centavos_por_bairro: dict[str, int]
    itens: tuple[Item, ...] = field(default_factory=tuple)

    def aberto_em(self, momento: datetime) -> bool:
        agora = momento.time()
        if self.abre <= self.fecha:
            return self.abre <= agora < self.fecha
        # Horário que atravessa a meia-noite, ex.: 11:00 às 02:00.
        return agora >= self.abre or agora < self.fecha

    def entrega_em(self, bairro: str) -> bool:
        return normalizar(bairro) in self.frete_centavos_por_bairro


@dataclass(frozen=True)
class Cupom:
    codigo: str
    tipo: str  # "percentual" | "fixo" | "frete_gratis"
    valor: int  # percentual (0-100) ou centavos
    subtotal_minimo_centavos: int
    validade: date
    restaurantes: tuple[str, ...] | None


@dataclass(frozen=True)
class Catalogo:
    restaurantes: dict[str, Restaurante]
    itens: dict[str, Item]
    cupons: dict[str, Cupom]

    @classmethod
    def carregar(cls, caminho: str | None = None) -> Catalogo:
        if caminho is None:
            bruto = resources.files("agente_delivery").joinpath("data/catalogo.json").read_text("utf-8")
        else:
            with open(caminho, encoding="utf-8") as f:
                bruto = f.read()
        return cls.de_dict(json.loads(bruto))

    @classmethod
    def de_dict(cls, dados: dict) -> Catalogo:
        restaurantes: dict[str, Restaurante] = {}
        itens: dict[str, Item] = {}
        for r in dados["restaurantes"]:
            itens_r = tuple(
                Item(
                    id=i["id"],
                    nome=i["nome"],
                    descricao=i["descricao"],
                    preco_centavos=i["preco_centavos"],
                    alergenos=tuple(normalizar(a) for a in i["alergenos"]),
                    restaurante_id=r["id"],
                )
                for i in r["itens"]
            )
            restaurantes[r["id"]] = Restaurante(
                id=r["id"],
                nome=r["nome"],
                culinaria=normalizar(r["culinaria"]),
                abre=time.fromisoformat(r["abre"]),
                fecha=time.fromisoformat(r["fecha"]),
                pedido_minimo_centavos=r["pedido_minimo_centavos"],
                frete_centavos_por_bairro={normalizar(b): v for b, v in r["frete_centavos_por_bairro"].items()},
                itens=itens_r,
            )
            for item in itens_r:
                if item.id in itens:
                    raise ValueError(f"ID de item duplicado: {item.id}")
                itens[item.id] = item
        cupons = {
            c["codigo"].upper(): Cupom(
                codigo=c["codigo"].upper(),
                tipo=c["tipo"],
                valor=c["valor"],
                subtotal_minimo_centavos=c["subtotal_minimo_centavos"],
                validade=date.fromisoformat(c["validade"]),
                restaurantes=tuple(c["restaurantes"]) if c["restaurantes"] else None,
            )
            for c in dados["cupons"]
        }
        return cls(restaurantes=restaurantes, itens=itens, cupons=cupons)
