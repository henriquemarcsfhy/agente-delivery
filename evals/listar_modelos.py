"""Lista os modelos disponíveis no endpoint configurado.

python evals/listar_modelos.py --base-url https://api.groq.com/openai/v1
"""

import argparse
import os

from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
a = p.parse_args()
cliente = OpenAI(base_url=a.base_url, api_key=os.environ.get("LLM_API_KEY", "nao-usado"))
for modelo in sorted(m.id for m in cliente.models.list()):
    print(modelo)
