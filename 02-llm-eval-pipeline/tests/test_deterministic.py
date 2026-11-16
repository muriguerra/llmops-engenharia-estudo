"""
Suíte Pytest de Testes Determinísticos de Prompt (Camada 1) — Fase 4
====================================================================

Como executar no terminal (dentro de 02-llm-eval-pipeline com .venv ativo):
  pytest tests/test_deterministic.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_02_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_02_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_02_ROOT))

from src.dataset.build_golden_dataset import carregar_golden_dataset  # noqa: E402
from src.evaluators.deterministic import (  # noqa: E402
    avaliar_caso_deterministico,
    extrair_siglas_regex,
)
from src.rag_client.advisor import carregar_respostas_geradas  # noqa: E402

CASOS_GOLDEN = {c.id: c for c in carregar_golden_dataset()}
PACOTE_RESPOSTAS = carregar_respostas_geradas()
IDS_CASOS = list(CASOS_GOLDEN.keys())


# ---------------------------------------------------------------------------
# 1. Testes Unitários do Extrator Regex de Siglas UNICAMP
# ---------------------------------------------------------------------------
def test_extrator_regex_siglas_padrao_e_com_espaco() -> None:
    """Garante que a Regex captura tanto siglas padrão (MA111, MC102) quanto com espaço (F 128)."""
    texto = (
        "Recomendamos MA111 no IMECC, MC102 no IC e a disciplina de física F 128 no IFGW, "
        "além de evitar LE300 da FCA."
    )
    siglas = extrair_siglas_regex(texto)
    assert siglas == ["MA111", "MC102", "F 128", "LE300"]


# ---------------------------------------------------------------------------
# 2. Bateria Parametrizada: Prompt v2 (Produção) DEVE passar em 100% dos 10 casos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("caso_id", IDS_CASOS)
def test_prompt_v2_estrito_passa_regras_deterministicas(caso_id: str) -> None:
    """
    Valida que o Prompt v2 (Engenharia Estrita + Busca Híbrida SQL) cumpre
    todas as 5 regras determinísticas (Schema, Grounding de Siglas, Cobertura,
    Fidelidade de Metadados e Guardrail de Recusa/Injection) em cada caso do Golden Dataset.
    """
    caso = CASOS_GOLDEN[caso_id]
    resposta_v2 = PACOTE_RESPOSTAS.v2_estrito[caso_id].resposta_orientador

    resultado = avaliar_caso_deterministico(caso, resposta_v2, versao_prompt="v2.0.0")

    assert resultado.schema_valido, f"[{caso_id}] Falha de schema: {resultado.falhas_detalhadas}"
    assert resultado.grounding_siglas, f"[{caso_id}] Alucinação/contaminação de sigla: {resultado.falhas_detalhadas}"
    assert resultado.cobertura_siglas, f"[{caso_id}] Sigla obrigatória ausente: {resultado.falhas_detalhadas}"
    assert resultado.fidelidade_metadados, f"[{caso_id}] Erro de metadados/créditos: {resultado.falhas_detalhadas}"
    assert resultado.guardrail_recusa_injection, f"[{caso_id}] Falha no guardrail de recusa: {resultado.falhas_detalhadas}"
    assert resultado.aprovado_geral is True


# ---------------------------------------------------------------------------
# 3. Testes de Detecção de Regressão: Comprova que a Camada 1 barra o Prompt v1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "caso_id_armadilha",
    ["GD-03", "GD-04", "GD-06", "GD-07", "GD-08", "GD-09", "GD-10"],
)
def test_detector_barra_falhas_do_prompt_v1_ingenuo(caso_id_armadilha: str) -> None:
    """
    Comprova que a nossa barreira determinística (Camada 1) detecta e reprova
    automaticamente as falhas do Prompt v1 nos 7 cenários de desambiguação,
    busca sem filtro, fora de escopo e Prompt Injection.
    """
    caso = CASOS_GOLDEN[caso_id_armadilha]
    resposta_v1 = PACOTE_RESPOSTAS.v1_ingenuo[caso_id_armadilha].resposta_orientador

    resultado_v1 = avaliar_caso_deterministico(caso, resposta_v1, versao_prompt="v1.0.0")

    assert resultado_v1.aprovado_geral is False, (
        f"O validador determinístico deveria ter reprovado o Prompt v1 no caso {caso_id_armadilha}!"
    )
    assert len(resultado_v1.falhas_detalhadas) >= 1
