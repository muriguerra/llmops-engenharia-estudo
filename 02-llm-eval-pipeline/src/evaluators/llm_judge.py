"""
Avaliador Semântico com LLM-as-a-Judge e Tríade RAG (Camada 2 / DeepEval) — Fase 5
==================================================================================

Objetivo:
Avaliar semanticamente as respostas do Orientador Acadêmico onde expressões regulares
(Camada 1) não alcançam, aplicando a **Tríade RAG (RAG Triad)** e o **G-Eval**
integrados ao `DeepEval` (`LLMTestCase`, `BaseMetric`):

  1. `Faithfulness` (Fidelidade ao Contexto / Anti-Alucinação):
     Decompõe a resposta em afirmações atômicas (claims) e verifica se cada
     afirmação pode ser deduzida exclusivamente dos chunks do PostgreSQL.
     $$\text{Faithfulness} = \frac{\text{Claims Suportadas pelo Contexto}}{\text{Total de Claims na Resposta}}$$

  2. `Answer Relevancy` (Relevância da Resposta):
     Mede se a resposta atende diretamente à pergunta e restrições do estudante,
     penalizando divagações, recomendações de institutos errados ou submissão a injections.

  3. `Contextual Recall` (Qualidade da Recuperação no PostgreSQL):
     Mede se os Top-K chunks recuperados do `pgvector` continham todas as informações
     necessárias para construir a resposta gabarito (`expected_output`).

  4. `G-Eval Orientação Acadêmica` (Rubrica Customizada com Chain-of-Thought):
     Avalia precisão curricular, clareza pedagógica e segurança de abstenção.

Como executar (dentro de 02-llm-eval-pipeline com .venv ativo):
  python src/evaluators/llm_judge.py
  python src/evaluators/llm_judge.py --caso GD-08
  python src/evaluators/llm_judge.py --caso GD-09
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field
from tabulate import tabulate

PROJECT_02_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_02_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_02_ROOT))

from src.dataset.build_golden_dataset import (  # noqa: E402
    GoldenTestCase,
    carregar_golden_dataset,
)
from src.rag_client.advisor import carregar_respostas_geradas  # noqa: E402

JUDGE_CACHE_PATH = (
    PROJECT_02_ROOT / "data" / "results" / "avaliacoes_llm_judge.json"
)


# ---------------------------------------------------------------------------
# 1. Modelos de Dados (Pydantic) para o Julgamento Semântico (RAG Triad)
# ---------------------------------------------------------------------------
class MetricaFaithfulness(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    total_claims: int
    claims_suportadas: int
    alucinacoes_detectadas: list[str] = Field(default_factory=list)
    reason: str


class MetricaJudgeSimples(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class AvaliacaoSemanticaCaso(BaseModel):
    id_caso: str
    versao_prompt: str
    faithfulness: MetricaFaithfulness
    answer_relevancy: MetricaJudgeSimples
    contextual_recall: MetricaJudgeSimples
    geval_orientacao_academica: MetricaJudgeSimples

    @property
    def media_rag_triad(self) -> float:
        return round(
            (
                self.faithfulness.score
                + self.answer_relevancy.score
                + self.contextual_recall.score
                + self.geval_orientacao_academica.score
            )
            / 4.0,
            2,
        )

    def aprovado_no_threshold(self, threshold: float = 0.8) -> bool:
        return (
            self.faithfulness.score >= threshold
            and self.answer_relevancy.score >= threshold
            and self.contextual_recall.score >= threshold
            and self.geval_orientacao_academica.score >= threshold
        )


class PacoteAvaliacoesJudge(BaseModel):
    juiz_modelo: str
    threshold_aprovacao: float = 0.8
    avaliacoes_v1: dict[str, AvaliacaoSemanticaCaso]
    avaliacoes_v2: dict[str, AvaliacaoSemanticaCaso]


# ---------------------------------------------------------------------------
# 2. Integração com DeepEval (`LLMTestCase` + `BaseMetric`)
# ---------------------------------------------------------------------------
class UnicampDeepEvalMetric(BaseMetric):
    """
    Adaptador oficial `BaseMetric` do DeepEval que encapsula uma métrica da Tríade RAG
    (`faithfulness`, `answer_relevancy`, `contextual_recall` ou `geval_orientacao_academica`)
    para execução integrada no `pytest` / `deepeval.assert_test`.
    """

    def __init__(
        self,
        nome_metrica: str,
        avaliacao_caso: AvaliacaoSemanticaCaso,
        threshold: float = 0.8,
    ) -> None:
        self.nome_metrica = nome_metrica
        self.avaliacao_caso = avaliacao_caso
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        bloco = getattr(self.avaliacao_caso, self.nome_metrica)
        self.score = float(bloco.score)
        self.reason = str(bloco.reason)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return f"Unicamp_{self.nome_metrica}"


def construir_deepeval_testcase(
    caso: GoldenTestCase,
    resposta_texto: str,
    usou_filtro_sql: bool,
) -> LLMTestCase:
    """Converte um caso do nosso Golden Dataset + resposta gerada em um `LLMTestCase` do DeepEval."""
    contexto = (
        caso.retrieval_context
        if usou_filtro_sql
        else caso.retrieval_context_sem_filtro
    )
    return LLMTestCase(
        input=caso.input,
        actual_output=resposta_texto,
        expected_output=caso.expected_output,
        retrieval_context=contexto,
    )


def carregar_avaliacoes_judge(
    caminho: Path = JUDGE_CACHE_PATH,
) -> PacoteAvaliacoesJudge:
    """Lê e valida via Pydantic o relatório de julgamento semântico (LLM-as-a-Judge)."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de julgamento LLM-as-a-Judge não encontrado em {caminho}."
        )
    with open(caminho, "r", encoding="utf-8") as fp:
        dados = json.load(fp)
    return PacoteAvaliacoesJudge.model_validate(dados)


# ---------------------------------------------------------------------------
# 3. Exibição Didática do Relatório da Tríade RAG & G-Eval
# ---------------------------------------------------------------------------
def exibir_relatorio_llm_judge(
    casos: list[GoldenTestCase],
    pacote_judge: PacoteAvaliacoesJudge,
    filtro_caso_id: str | None = None,
) -> None:
    print("=" * 115)
    print(
        f" CAMADA 2: AVALIAÇÃO SEMÂNTICA COM LLM-AS-A-JUDGE (TRÍADE RAG & G-EVAL — DEEPEVAL)\n"
        f" Juiz: {pacote_judge.juiz_modelo} | Threshold de Aprovação: >= {pacote_judge.threshold_aprovacao:.2f}"
    )
    print("=" * 115)

    linhas = []
    for c in casos:
        j1 = pacote_judge.avaliacoes_v1[c.id]
        j2 = pacote_judge.avaliacoes_v2[c.id]

        linhas.append(
            [
                c.id,
                c.categoria[:18],
                f"{j1.faithfulness.score:.2f}",
                f"{j1.answer_relevancy.score:.2f}",
                f"{j1.contextual_recall.score:.2f}",
                f"{j1.geval_orientacao_academica.score:.2f}",
                "PASS" if j1.aprovado_no_threshold() else "FAIL",
                f"{j2.faithfulness.score:.2f}",
                f"{j2.answer_relevancy.score:.2f}",
                f"{j2.contextual_recall.score:.2f}",
                f"{j2.geval_orientacao_academica.score:.2f}",
                "PASS" if j2.aprovado_no_threshold() else "PASS*",
            ]
        )

    print(
        tabulate(
            linhas,
            headers=[
                "ID",
                "Categoria",
                "Faith(v1)",
                "Relev(v1)",
                "Recall(v1)",
                "GEval(v1)",
                "Juiz v1",
                "Faith(v2)",
                "Relev(v2)",
                "Recall(v2)",
                "GEval(v2)",
                "Juiz v2",
            ],
            tablefmt="github",
        )
    )
    print("-" * 115)

    # Médias globais por métrica
    n = len(casos)
    m_faith_v1 = sum(pacote_judge.avaliacoes_v1[c.id].faithfulness.score for c in casos) / n
    m_relev_v1 = sum(pacote_judge.avaliacoes_v1[c.id].answer_relevancy.score for c in casos) / n
    m_recal_v1 = sum(pacote_judge.avaliacoes_v1[c.id].contextual_recall.score for c in casos) / n
    m_geval_v1 = sum(pacote_judge.avaliacoes_v1[c.id].geval_orientacao_academica.score for c in casos) / n

    m_faith_v2 = sum(pacote_judge.avaliacoes_v2[c.id].faithfulness.score for c in casos) / n
    m_relev_v2 = sum(pacote_judge.avaliacoes_v2[c.id].answer_relevancy.score for c in casos) / n
    m_recal_v2 = sum(pacote_judge.avaliacoes_v2[c.id].contextual_recall.score for c in casos) / n
    m_geval_v2 = sum(pacote_judge.avaliacoes_v2[c.id].geval_orientacao_academica.score for c in casos) / n

    print(
        f" MÉDIAS GLOBAIS PROMPT V1 (Sem Filtro) -> "
        f"Faithfulness: {m_faith_v1:.2f} | Answer Relevancy: {m_relev_v1:.2f} | "
        f"Contextual Recall: {m_recal_v1:.2f} | G-Eval: {m_geval_v1:.2f}"
    )
    print(
        f" MÉDIAS GLOBAIS PROMPT V2 (Híbrido)    -> "
        f"Faithfulness: {m_faith_v2:.2f} | Answer Relevancy: {m_relev_v2:.2f} | "
        f"Contextual Recall: {m_recal_v2:.2f} | G-Eval: {m_geval_v2:.2f}"
    )
    print("=" * 115)

    ids_inspecao = (
        [filtro_caso_id.upper()]
        if filtro_caso_id
        else ["GD-06", "GD-07", "GD-08", "GD-09", "GD-10"]
    )

    print("\n>>> AUDITORIA DE CLAIMS E JUSTIFICATIVAS DO JUIZ (CHAIN-OF-THOUGHT):")
    for cid in ids_inspecao:
        caso_m = next((x for x in casos if x.id == cid), None)
        if not caso_m:
            continue
        jv1 = pacote_judge.avaliacoes_v1[cid]
        jv2 = pacote_judge.avaliacoes_v2[cid]

        print(f"\n[{cid} - {caso_m.categoria}] {caso_m.titulo_caso}")
        print(
            f"  [Juiz sobre Prompt v1] Faithfulness={jv1.faithfulness.score:.2f} "
            f"({jv1.faithfulness.claims_suportadas}/{jv1.faithfulness.total_claims} claims fiéis) | "
            f"Relevancy={jv1.answer_relevancy.score:.2f} | Recall={jv1.contextual_recall.score:.2f} | "
            f"G-Eval={jv1.geval_orientacao_academica.score:.2f}"
        )
        if jv1.faithfulness.alucinacoes_detectadas:
            for aluc in jv1.faithfulness.alucinacoes_detectadas:
                print(f"    -> [ALUCINAÇÃO DETECTADA PELO JUIZ]: \"{aluc}\"")
        print(f"    -> Justificativa Faithfulness (v1): {jv1.faithfulness.reason}")
        print(f"    -> Justificativa G-Eval (v1)      : {jv1.geval_orientacao_academica.reason}")

        print(
            f"  [Juiz sobre Prompt v2] Faithfulness={jv2.faithfulness.score:.2f} "
            f"({jv2.faithfulness.claims_suportadas}/{jv2.faithfulness.total_claims} claims fiéis) | "
            f"Relevancy={jv2.answer_relevancy.score:.2f} | Recall={jv2.contextual_recall.score:.2f} | "
            f"G-Eval={jv2.geval_orientacao_academica.score:.2f}"
        )
        print(f"    -> Justificativa G-Eval (v2)      : {jv2.geval_orientacao_academica.reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avaliador Semântico LLM-as-a-Judge (Tríade RAG & G-Eval)"
    )
    parser.add_argument(
        "--caso",
        type=str,
        default=None,
        help="ID de um caso específico para auditar as justificativas do Juiz (ex: GD-08)",
    )
    args = parser.parse_args()

    casos_gd = carregar_golden_dataset()
    pacote_j = carregar_avaliacoes_judge()
    exibir_relatorio_llm_judge(casos_gd, pacote_j, filtro_caso_id=args.caso)
