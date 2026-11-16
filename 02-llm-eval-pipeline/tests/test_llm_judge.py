"""
Suíte Pytest + DeepEval para Avaliação Semântica (LLM-as-a-Judge) — Fase 5
==========================================================================

Como executar no terminal (dentro de 02-llm-eval-pipeline com .venv ativo):
  pytest tests/test_llm_judge.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from deepeval import assert_test

PROJECT_02_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_02_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_02_ROOT))

from src.dataset.build_golden_dataset import carregar_golden_dataset  # noqa: E402
from src.evaluators.llm_judge import (  # noqa: E402
    UnicampDeepEvalMetric,
    carregar_avaliacoes_judge,
    construir_deepeval_testcase,
)
from src.rag_client.advisor import carregar_respostas_geradas  # noqa: E402

CASOS_GOLDEN = {c.id: c for c in carregar_golden_dataset()}
PACOTE_RESPOSTAS = carregar_respostas_geradas()
PACOTE_JUDGE = carregar_avaliacoes_judge()
IDS_CASOS = list(CASOS_GOLDEN.keys())


# ---------------------------------------------------------------------------
# 1. Validação DeepEval (Tríade RAG + G-Eval) para o Prompt v2 (Produção)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("caso_id", IDS_CASOS)
def test_deepeval_rag_triad_prompt_v2_estrito(caso_id: str) -> None:
    """
    Usa `LLMTestCase` e `assert_test` do DeepEval para comprovar que o Prompt v2
    atinge score >= 0.80 nas 4 métricas semânticas (Faithfulness, Answer Relevancy,
    Contextual Recall e G-Eval Orientação Acadêmica) em todos os 10 casos.
    """
    caso = CASOS_GOLDEN[caso_id]
    reg_v2 = PACOTE_RESPOSTAS.v2_estrito[caso_id]
    aval_v2 = PACOTE_JUDGE.avaliacoes_v2[caso_id]
    threshold = PACOTE_JUDGE.threshold_aprovacao

    test_case = construir_deepeval_testcase(
        caso,
        resposta_texto=reg_v2.resposta_orientador.resposta_texto,
        usou_filtro_sql=True,
    )

    metricas = [
        UnicampDeepEvalMetric("faithfulness", aval_v2, threshold=threshold),
        UnicampDeepEvalMetric("answer_relevancy", aval_v2, threshold=threshold),
        UnicampDeepEvalMetric("contextual_recall", aval_v2, threshold=threshold),
        UnicampDeepEvalMetric("geval_orientacao_academica", aval_v2, threshold=threshold),
    ]

    assert_test(test_case, metricas)


# ---------------------------------------------------------------------------
# 2. Comprovação de Detecção Semântica de Alucinações Sutis no Prompt v1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "caso_id_alucinacao",
    ["GD-07", "GD-08", "GD-09", "GD-10"],
)
def test_juiz_detecta_queda_de_faithfulness_por_alucinacao_no_v1(
    caso_id_alucinacao: str,
) -> None:
    """
    Comprova que o métrica `Faithfulness` (Fidelidade) do LLM-as-a-Judge detecta
    afirmações inventadas no meio do texto livre do Prompt v1 (como as salas CB01/CB02
    no GD-08, o currículo jurídico no GD-07, os 2 créditos no GD-09 e os qubits de MC999 no GD-10).
    """
    aval_v1 = PACOTE_JUDGE.avaliacoes_v1[caso_id_alucinacao]

    assert aval_v1.faithfulness.score < PACOTE_JUDGE.threshold_aprovacao
    assert len(aval_v1.faithfulness.alucinacoes_detectadas) >= 1


def test_juiz_detecta_colapso_de_contextual_recall_sem_filtro_sql_no_gd06() -> None:
    """
    Comprova na métrica `Contextual Recall` da Tríade RAG o impacto do filtro relacional SQL:
    No caso GD-06 (IA no IC), a busca sem filtro (`v1`) teve Contextual Recall < 0.50
    (não trouxe nenhuma matéria do IC), enquanto a busca híbrida (`v2`) obteve Recall >= 0.90.
    """
    recall_v1 = PACOTE_JUDGE.avaliacoes_v1["GD-06"].contextual_recall.score
    recall_v2 = PACOTE_JUDGE.avaliacoes_v2["GD-06"].contextual_recall.score

    assert recall_v1 < 0.50
    assert recall_v2 >= 0.90
