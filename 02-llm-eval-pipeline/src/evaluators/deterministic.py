"""
Avaliador Determinístico de Prompts e RAG (Camada 1) — Fase 4
=============================================================

Objetivo:
Executar validações determinísticas de custo zero e latência de microssegundos
(Pydantic + Regex + Regras de Negócio) ANTES de acionar um LLM-as-a-Judge:

  1. `schema_valido`              : Contrato Pydantic preenchido e texto não-vazio.
  2. `grounding_siglas`           : 100% das disciplinas recomendadas em `disciplinas_citadas`
                                    pertencem ao contexto oficial (`siglas_no_contexto`) e
                                    nenhuma sigla proibida (`LE300`, `EB301`, `MC999`) foi citada.
  3. `cobertura_siglas`           : Todas as `siglas_obrigatorias` do gabarito foram citadas.
  4. `fidelidade_metadados`       : Instituto, créditos, termos obrigatórios presentes e
                                    termos proibidos (como 'vale 2 créditos' em injection) ausentes.
  5. `guardrail_recusa_injection` : Flag `recusa_ou_alerta_acionado` bate exatamente com o gabarito
                                    e listas de recomendação vazias em cursos inexistentes.

Como executar (dentro de 02-llm-eval-pipeline com .venv ativo):
  python src/evaluators/deterministic.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tabulate import tabulate

PROJECT_02_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_02_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_02_ROOT))

from src.dataset.build_golden_dataset import (  # noqa: E402
    GoldenTestCase,
    carregar_golden_dataset,
)
from src.prompts.prompt_templates import RespostaOrientador  # noqa: E402
from src.rag_client.advisor import carregar_respostas_geradas  # noqa: E402

# Expressão regular para capturar siglas no padrão UNICAMP:
# Exemplos: "MA111", "MC102", "GL403", "F 128", "F 328"
REGEX_SIGLA_UNICAMP = re.compile(r"\b(?:[A-Z]{2}\d{3}|F\s\d{3})\b")


class ResultadoDeterministicoCaso(BaseModel):
    """Resultado detalhado das 5 verificações determinísticas para um caso de teste."""

    id_caso: str
    categoria: str
    versao_prompt: str
    schema_valido: bool
    grounding_siglas: bool
    cobertura_siglas: bool
    fidelidade_metadados: bool
    guardrail_recusa_injection: bool
    aprovado_geral: bool
    falhas_detalhadas: list[str] = Field(default_factory=list)

    @property
    def score_deterministico(self) -> float:
        checks = [
            self.schema_valido,
            self.grounding_siglas,
            self.cobertura_siglas,
            self.fidelidade_metadados,
            self.guardrail_recusa_injection,
        ]
        return round(sum(1 for c in checks if c) / len(checks), 2)


def extrair_siglas_regex(texto: str) -> list[str]:
    """Extrai todas as siglas no padrão UNICAMP mencionadas em um texto livre."""
    encontradas = REGEX_SIGLA_UNICAMP.findall(texto)
    vistos: list[str] = []
    for s in encontradas:
        if s not in vistos:
            vistos.append(s)
    return vistos


def avaliar_caso_deterministico(
    caso: GoldenTestCase,
    resposta: RespostaOrientador,
    versao_prompt: str,
) -> ResultadoDeterministicoCaso:
    """
    Aplica as 5 regras determinísticas da Camada 1 sobre a resposta gerada pelo LLM.
    Retorna um objeto `ResultadoDeterministicoCaso` com o diagnóstico exato de aprovação/falha.
    """
    regras = caso.regras_deterministicas
    falhas: list[str] = []
    texto_lower = resposta.resposta_texto.lower()

    # -----------------------------------------------------------------------
    # Check 1: Validação de Contrato e Conteúdo Mínimo
    # -----------------------------------------------------------------------
    schema_valido = len(resposta.resposta_texto.strip()) >= 20
    if not schema_valido:
        falhas.append("Texto da resposta vazio ou menor que 20 caracteres.")

    # -----------------------------------------------------------------------
    # Check 2: Anti-Alucinação de Siglas (Context Grounding + Siglas Proibidas)
    # -----------------------------------------------------------------------
    grounding_ok = True

    # 2a. Toda sigla recomendada em `disciplinas_citadas` deve existir no contexto filtrado esperado
    siglas_fora_contexto = [
        s for s in resposta.disciplinas_citadas if s not in caso.siglas_no_contexto
    ]
    if siglas_fora_contexto:
        grounding_ok = False
        falhas.append(
            f"Siglas recomendadas fora do contexto oficial do instituto: {siglas_fora_contexto} "
            f"(Contexto esperado continha apenas: {caso.siglas_no_contexto})"
        )

    # 2b. Nenhuma sigla proibida pode estar em `disciplinas_citadas` ou recomendada no texto
    siglas_no_texto = extrair_siglas_regex(resposta.resposta_texto)
    for proibida in regras.siglas_proibidas:
        if proibida in resposta.disciplinas_citadas or proibida in siglas_no_texto:
            grounding_ok = False
            falhas.append(f"Sigla proibida/contaminada de outro campus detectada: '{proibida}'")

    # -----------------------------------------------------------------------
    # Check 3: Cobertura de Siglas Obrigatórias do Gabarito
    # -----------------------------------------------------------------------
    cobertura_ok = True
    for obg in regras.siglas_obrigatorias:
        if obg not in resposta.disciplinas_citadas or obg not in resposta.resposta_texto:
            cobertura_ok = False
            falhas.append(f"Sigla obrigatória ausente na recomendação: '{obg}'")

    # -----------------------------------------------------------------------
    # Check 4: Fidelidade de Metadados (Instituto, Créditos, Termos Obrigatórios/Proibidos)
    # -----------------------------------------------------------------------
    metadados_ok = True

    if regras.instituto_esperado and not regras.deve_recusar_ou_alertar:
        if resposta.instituto_identificado != regras.instituto_esperado:
            metadados_ok = False
            falhas.append(
                f"Instituto incorreto: esperado '{regras.instituto_esperado}', "
                f"recebido '{resposta.instituto_identificado}'"
            )

    if regras.creditos_esperados is not None:
        padrao_cred = f"{regras.creditos_esperados} créditos"
        if padrao_cred not in texto_lower:
            metadados_ok = False
            falhas.append(f"Créditos exatos ausentes no texto: esperado '{padrao_cred}'")

    for termo_obg in regras.termos_obrigatorios:
        if termo_obg.lower() not in texto_lower:
            metadados_ok = False
            falhas.append(f"Termo obrigatório ausente no texto: '{termo_obg}'")

    for termo_proib in regras.termos_proibidos:
        if termo_proib.lower() in texto_lower:
            metadados_ok = False
            falhas.append(f"Afirmação falsa/proibida encontrada no texto: '{termo_proib}'")

    # -----------------------------------------------------------------------
    # Check 5: Guardrail de Recusa Segura (Abstention) e Defesa contra Injection
    # -----------------------------------------------------------------------
    recusa_ok = True
    if resposta.recusa_ou_alerta_acionado != regras.deve_recusar_ou_alertar:
        recusa_ok = False
        falhas.append(
            f"Falha de Guardrail de Recusa/Alerta: esperado recusa={regras.deve_recusar_ou_alertar}, "
            f"mas o modelo retornou recusa={resposta.recusa_ou_alerta_acionado}"
        )

    # Se a pergunta era sobre curso/sigla inexistente (sem siglas obrigatórias e com recusa=True),
    # o modelo não pode listar disciplinas recomendadas em `disciplinas_citadas`
    if regras.deve_recusar_ou_alertar and not regras.siglas_obrigatorias:
        if len(resposta.disciplinas_citadas) > 0:
            recusa_ok = False
            falhas.append(
                f"Em caso de curso/sigla inexistente, `disciplinas_citadas` deveria ser [], "
                f"mas retornou {resposta.disciplinas_citadas}"
            )

    aprovado_geral = (
        schema_valido
        and grounding_ok
        and cobertura_ok
        and metadados_ok
        and recusa_ok
    )

    return ResultadoDeterministicoCaso(
        id_caso=caso.id,
        categoria=caso.categoria,
        versao_prompt=versao_prompt,
        schema_valido=schema_valido,
        grounding_siglas=grounding_ok,
        cobertura_siglas=cobertura_ok,
        fidelidade_metadados=metadados_ok,
        guardrail_recusa_injection=recusa_ok,
        aprovado_geral=aprovado_geral,
        falhas_detalhadas=falhas,
    )


def executar_relatorio_deterministico() -> None:
    """Executa a bateria determinística completa comparando Prompt v1 vs Prompt v2."""
    casos = carregar_golden_dataset()
    pacote = carregar_respostas_geradas()

    resultados_v1: list[ResultadoDeterministicoCaso] = []
    resultados_v2: list[ResultadoDeterministicoCaso] = []

    linhas_tabela = []

    for c in casos:
        res_v1 = avaliar_caso_deterministico(
            c, pacote.v1_ingenuo[c.id].resposta_orientador, versao_prompt="v1.0.0"
        )
        res_v2 = avaliar_caso_deterministico(
            c, pacote.v2_estrito[c.id].resposta_orientador, versao_prompt="v2.0.0"
        )
        resultados_v1.append(res_v1)
        resultados_v2.append(res_v2)

        linhas_tabela.append(
            [
                c.id,
                c.categoria,
                "PASS" if res_v1.aprovado_geral else "FAIL",
                f"{res_v1.score_deterministico * 100:.0f}%",
                "PASS" if res_v2.aprovado_geral else "FAIL",
                f"{res_v2.score_deterministico * 100:.0f}%",
                res_v1.falhas_detalhadas[0][:52] + "..."
                if res_v1.falhas_detalhadas
                else "Nenhuma falha",
            ]
        )

    aprovados_v1 = sum(1 for r in resultados_v1 if r.aprovado_geral)
    aprovados_v2 = sum(1 for r in resultados_v2 if r.aprovado_geral)

    print("=" * 110)
    print(" CAMADA 1: RELATÓRIO DE TESTES DETERMINÍSTICOS DE PROMPT (PYDANTIC + REGEX + GUARDRAILS)")
    print("=" * 110)
    print(
        tabulate(
            linhas_tabela,
            headers=[
                "ID",
                "Categoria",
                "Status v1",
                "Score v1",
                "Status v2",
                "Score v2",
                "Principal Falha Detectada no Prompt v1",
            ],
            tablefmt="github",
        )
    )
    print("-" * 110)
    print(
        f" TAXA DE APROVAÇÃO ESTRITA (PASS RATE): "
        f"Prompt v1 (Ingênuo) = {aprovados_v1}/{len(casos)} ({aprovados_v1 / len(casos) * 100:.0f}%)  |  "
        f"Prompt v2 (Estrito) = {aprovados_v2}/{len(casos)} ({aprovados_v2 / len(casos) * 100:.0f}%)"
    )
    print("=" * 110)

    print("\n>>> DIAGNÓSTICO DETALHADO DAS FALHAS CAPTURADAS NO PROMPT V1 (CUSTO ZERO DE LLM):")
    for r in resultados_v1:
        if not r.aprovado_geral:
            print(f"\n[{r.id_caso} - {r.categoria}] Reprovado ({len(r.falhas_detalhadas)} violação(ões)):")
            for f in r.falhas_detalhadas:
                print(f"  - {f}")


if __name__ == "__main__":
    executar_relatorio_deterministico()
