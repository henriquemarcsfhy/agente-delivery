from datetime import datetime

import pytest

from agente_delivery.catalog import Catalogo
from agente_delivery.tools import FUSO, DeliveryTools, EstadoSessao


@pytest.fixture(scope="session")
def catalogo() -> Catalogo:
    return Catalogo.carregar()


@pytest.fixture
def tools(catalogo) -> DeliveryTools:
    # Quarta-feira, 20h: todos os restaurantes abertos.
    return DeliveryTools(catalogo, EstadoSessao(agora=datetime(2026, 11, 11, 20, 0, tzinfo=FUSO)))
