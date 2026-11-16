"""
Cliente Gerador RAG UNICAMP (Sistema Sob Teste / SUT) — Fase 3
==============================================================

Objetivo:
Executar o Orientador Acadêmico sobre o Golden Dataset utilizando tanto o
`PROMPT_V1_INGENUO` (v1.0.0) quanto o `PROMPT_V2_ESTRITO` (v2.0.0).

Suporta dois modos de operação configurados via `.env` (`LLM_PROVIDER`):
  1. `antigravity` (Padrão): Carrega e valida via Pydantic as respostas geradas
     pelos Subagentes isolados do Antigravity em `data/results/respostas_geradas.json`
     (custo zero e sem necessidade de API Key externa).
  2. `gemini` / `openai`: Faz chamadas HTTP ao vivo para a API configurada usando
     Structured Outputs (`RespostaOrientador`) e salva os resultados em disco.

Como executar (dentro de 02-llm-eval-pipeline com .venv ativo):
  python src/rag_client/advisor.py
  python src/rag_client/advisor.py --caso GD-04
  python src/rag_client/advisor.py --caso GD-09
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from tabulate import tabulate

PROJECT_02_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_02_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_02_ROOT))

from src.dataset.build_golden_dataset import (  # noqa: E402
    GoldenTestCase,
    carregar_golden_dataset,
)
from src.prompts.prompt_templates import (  # noqa: E402
    PROMPT_V1_INGENUO,
    PROMPT_V2_ESTRITO,
    PromptVersion,
    RespostaOrientador,
)

load_dotenv(PROJECT_02_ROOT / ".env")

RESPOSTAS_CACHE_PATH = (
    PROJECT_02_ROOT / "data" / "results" / "respostas_geradas.json"
)


class RegistroGeracaoCaso(BaseModel):
    """Armazena a resposta gerada para um caso do Golden Dataset em uma versão de Prompt."""

    id: str
    versao_prompt: str
    usou_filtro_sql: bool
    latencia_ms: float
    resposta_orientador: RespostaOrientador


class PacoteRespostasGeradas(BaseModel):
    """Arquivo consolidado contendo as execuções do Prompt v1 e do Prompt v2."""

    provedor_llm: str
    modelo_gerador: str
    v1_ingenuo: dict[str, RegistroGeracaoCaso]
    v2_estrito: dict[str, RegistroGeracaoCaso]


def chamar_llm_api_externa(
    prompt_version: PromptVersion,
    caso: GoldenTestCase,
    provedor: str,
    modelo: str,
) -> RegistroGeracaoCaso:
    """
    Executa chamada real via SDK oficial (`google-genai` ou `openai`) caso o usuário
    configure `LLM_PROVIDER=gemini` ou `LLM_PROVIDER=openai` no arquivo `.env`.
    """
    chunks = (
        caso.retrieval_context
        if prompt_version.usa_filtro_relacional_sql
        else caso.retrieval_context_sem_filtro
    )
    sys_msg, user_msg = prompt_version.formatar_mensagens(caso.input, chunks)

    t0 = time.time()
    if provedor == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=modelo,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=sys_msg,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RespostaOrientador,
            ),
        )
        parsed = RespostaOrientador.model_validate_json(resp.text)

    elif provedor == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.beta.chat.completions.parse(
            model=modelo,
            temperature=0.0,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format=RespostaOrientador,
        )
        parsed = completion.choices[0].message.parsed
    else:
        raise ValueError(f"Provedor externo não reconhecido: {provedor}")

    latencia_ms = (time.time() - t0) * 1000
    return RegistroGeracaoCaso(
        id=caso.id,
        versao_prompt=prompt_version.versao,
        usou_filtro_sql=prompt_version.usa_filtro_relacional_sql,
        latencia_ms=round(latencia_ms, 1),
        resposta_orientador=parsed,
    )


def carregar_respostas_geradas(
    caminho: Path = RESPOSTAS_CACHE_PATH,
) -> PacoteRespostasGeradas:
    """Carrega e valida com Pydantic o pacote de respostas do Prompt v1 e Prompt v2."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de respostas não encontrado em {caminho}."
        )
    with open(caminho, "r", encoding="utf-8") as fp:
        dados = json.load(fp)
    return PacoteRespostasGeradas.model_validate(dados)


def exibir_comparativo_prompts(
    casos: list[GoldenTestCase],
    pacote: PacoteRespostasGeradas,
    filtro_caso_id: str | None = None,
) -> None:
    """Exibe no terminal a comparação lado a lado entre Prompt v1 e Prompt v2."""
    print("=" * 100)
    print(
        f" COMPARATIVO DE GERAÇÃO RAG: PROMPT V1 (INGÊNUO) vs PROMPT V2 (ESTRITO)\n"
        f" Provedor: {pacote.provedor_llm} | Modelo: {pacote.modelo_gerador}"
    )
    print("=" * 100)

    linhas_tabela = []
    for c in casos:
        r_v1 = pacote.v1_ingenuo[c.id].resposta_orientador
        r_v2 = pacote.v2_estrito[c.id].resposta_orientador

        linhas_tabela.append(
            [
                c.id,
                c.categoria,
                ", ".join(r_v1.disciplinas_citadas) or "(Nenhuma)",
                "SIM" if r_v1.recusa_ou_alerta_acionado else "NÃO",
                ", ".join(r_v2.disciplinas_citadas) or "(Nenhuma)",
                "SIM" if r_v2.recusa_ou_alerta_acionado else "NÃO",
                "SIM" if c.regras_deterministicas.deve_recusar_ou_alertar else "NÃO",
            ]
        )

    print(
        tabulate(
            linhas_tabela,
            headers=[
                "ID",
                "Categoria",
                "Siglas v1 (Sem Filtro)",
                "Recusa v1",
                "Siglas v2 (Híbrido)",
                "Recusa v2",
                "Gabarito Recusa",
            ],
            tablefmt="github",
        )
    )
    print("=" * 100)

    # Exibe inspeção detalhada de casos selecionados
    ids_para_detalhar = (
        [filtro_caso_id.upper()]
        if filtro_caso_id
        else ["GD-04", "GD-06", "GD-07", "GD-09"]
    )

    for cid in ids_para_detalhar:
        caso_match = next((x for x in casos if x.id == cid), None)
        if not caso_match:
            continue
        v1 = pacote.v1_ingenuo[cid].resposta_orientador
        v2 = pacote.v2_estrito[cid].resposta_orientador

        print(f"\n>>> INSPEÇÃO DETALHADA [{cid} - {caso_match.categoria}]: {caso_match.titulo_caso}")
        print(f"Pergunta do Aluno: \"{caso_match.input}\"")
        print("-" * 100)
        print(
            f"[PROMPT V1 - INGÊNUO + SEM FILTRO] "
            f"(Siglas={v1.disciplinas_citadas} | Instituto={v1.instituto_identificado} | Recusa={v1.recusa_ou_alerta_acionado}):\n"
            f"  {v1.resposta_texto}"
        )
        print("-" * 100)
        print(
            f"[PROMPT V2 - ESTRITO + HÍBRIDO SQL] "
            f"(Siglas={v2.disciplinas_citadas} | Instituto={v2.instituto_identificado} | Recusa={v2.recusa_ou_alerta_acionado}):\n"
            f"  {v2.resposta_texto}"
        )
        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cliente Gerador RAG UNICAMP — Comparativo Prompt v1 vs Prompt v2"
    )
    parser.add_argument(
        "--caso",
        type=str,
        default=None,
        help="ID de um caso específico para inspecionar em detalhe (ex: GD-01, GD-04, GD-09)",
    )
    args = parser.parse_args()

    casos_gd = carregar_golden_dataset()
    pacote_resp = carregar_respostas_geradas()
    exibir_comparativo_prompts(casos_gd, pacote_resp, filtro_caso_id=args.caso)
