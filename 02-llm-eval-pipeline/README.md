# Pipeline de Avaliação de LLMs e RAG — Testes Determinísticos & *LLM-as-a-Judge* (`Pytest` + `DeepEval` + `Pydantic`)

Sistema completo de **Engenharia de Qualidade, Testes Determinísticos de Prompt e Avaliação Semântica (*LLM-as-a-Judge*)** construído sobre o catálogo oficial de disciplinas da **UNICAMP 2026** e o motor de busca híbrida (`PostgreSQL 16 + pgvector`) do [Desafio 01](../01-rag-postgresql-unicamp/README.md).

O projeto resolve um dos principais gargalos de **LLMOps em Produção**: substituir testes manuais subjetivos (*vibe checks*) por uma **esteira automatizada de CI/CD em duas camadas**, capaz de bloquear regressões de prompt, alucinações de entidades, contaminação entre institutos (`IMECC`/`IC`/`IFGW` vs `FCA`/`FT`) e ataques de *Prompt Injection* antes do deploy.

---

## 1. Arquitetura da Esteira de Avaliação (Duas Camadas)

```mermaid
flowchart TB
    subgraph Base01 ["1. Base Relacional + Vetorial (Desafio 01)"]
        PG[("PostgreSQL 16 + pgvector<br/>5.217 Chunks (BAAI/bge-m3)<br/>Índice HNSW")]
    end

    subgraph Golden ["2. Construtor do Golden Dataset (src/dataset/build_golden_dataset.py)"]
        PG -->|"Busca Híbrida SQL + Busca Sem Filtro"| GD[("data/golden_dataset/<br/>unicamp_golden_dataset.jsonl<br/>(10 cenários | 5 categorias)")]
    end

    subgraph SUT ["3. Sistema Sob Teste / Prompt-as-Code (src/prompts/ & src/rag_client/)"]
        GD --> P_V1["Prompt v1 (v1.0.0)<br/>Ingênuo + Busca Sem Filtro"]
        GD --> P_V2["Prompt v2 (v2.0.0)<br/>Engenharia Estrita + Busca Híbrida SQL"]
        P_V1 --> OUT["Contrato Pydantic: RespostaOrientador<br/>(resposta_texto, disciplinas_citadas,<br/>instituto_identificado, recusa_ou_alerta_acionado)"]
        P_V2 --> OUT
    end

    subgraph Eval ["4. Pirâmide de Avaliação Automatizada (Pytest + DeepEval)"]
        OUT --> L1["Camada 1: Testes Determinísticos (deterministic.py)<br/>- Schema Pydantic & Tamanho Mínimo<br/>- Anti-Alucinação de Siglas (Regex + Set Inclusion)<br/>- Cobertura de Siglas Obrigatórias e Créditos<br/>- Guardrail de Recusa (Negative Rejection) & Injection<br/>(Tempo: < 1 ms | Custo: $0.00)"]
        L1 -->|"Early-Exit Gatekeeping"| L2["Camada 2: LLM-as-a-Judge (llm_judge.py / DeepEval)<br/>- Faithfulness (Decomposição em Claims)<br/>- Answer Relevancy (Aderência ao Pedido)<br/>- Contextual Recall (Qualidade do Retriever)<br/>- G-Eval (Rubrica Pedagógica UNICAMP)"]
    end

    subgraph Gate ["5. Quality Gate CI/CD (benchmark_prompts.py)"]
        L2 --> REPORT[("data/results/eval_report.json<br/>Prompt v1: BLOCKED (30%)<br/>Prompt v2: APPROVED (100%)")]
    end
```

---

## 2. Resultados Reais do Benchmark (`Prompt v1` vs `Prompt v2`)

Executado via [`src/evaluators/benchmark_prompts.py`](./src/evaluators/benchmark_prompts.py) sobre os 10 cenários do *Golden Dataset* da UNICAMP:

| Métrica de Engenharia / Qualidade | `Prompt v1` (Ingênuo + Sem Filtro) | `Prompt v2` (Estrito + Híbrido SQL) | Ganho de Engenharia |
| :--- | :--- | :--- | :--- |
| **1. Pass Rate Determinístico (Camada 1)** | `30% (3/10)` | **`100% (10/10)`** | **`+70 p.p.`** |
| **2. Score Médio de Checks Determinísticos** | `76.0%` | **`100.0%`** | **`+24.0 p.p.`** |
| **3. Violações de Regras Duras (`Regex`/Schema)** | `20 violações` | **`0 violações`** | **`-20 violações`** |
| **4. Faithfulness / Anti-Alucinação (*Judge*)** | `0.62` | **`1.00`** | **`+0.38`** |
| **5. Claims Alucinadas Detectadas no Texto** | `14 claims falsas` | **`0 claims falsas`** | **`-100% alucinações`** |
| **6. Answer Relevancy (Relevância ao Aluno)** | `0.43` | **`1.00`** | **`+0.57`** |
| **7. Contextual Recall (Busca Sem Filtro vs SQL)** | `0.75` | **`1.00`** | **`+0.25`** |
| **8. G-Eval Rubrica Acadêmica UNICAMP** | `0.40` | **`1.00`** | **`+0.60`** |
| **9. Pass Rate Semântico (`RAG Triad >= 0.80`)** | `30% (3/10)` | **`100% (10/10)`** | **`+70 p.p.`** |
| **10. Decisão do Quality Gate (CI/CD Deploy)** | **`REPROVADO (BLOCKED)`** | **`APROVADO (READY)`** | **Promovido p/ Produção** |

> **Economia de *Early-Exit Gatekeeping*:** A Camada 1 Determinística barrou **7 dos 7 casos defeituosos** do `Prompt v1` em menos de `1 ms` e custo `$0.00`, economizando **70%** das chamadas de API do *LLM-as-a-Judge* em uma esteira real de CI/CD.

---

## 3. Estrutura de Diretórios

```text
02-llm-eval-pipeline/
├── README.md                                        # Documentação técnica completa e resultados do Desafio 02
├── GUIA_TEORICO.md                                  # Apostila teórica completa e revisão para entrevistas
├── TODO.md                                          # Roteiro guiado de estudo passo a passo (6 fases concluídas)
├── requirements.txt                                 # Dependências Python (Pytest, DeepEval, Pydantic, etc.)
├── pytest.ini                                       # Configuração da suíte Pytest (33 testes automatizados)
├── .env.example                                     # Modelo de variáveis de ambiente (PostgreSQL + Provedor LLM)
├── data/
│   ├── golden_dataset/
│   │   └── unicamp_golden_dataset.jsonl             # 10 casos de teste gerados do PostgreSQL (pgvector)
│   └── results/
│       ├── respostas_geradas.json                   # Respostas estruturadas (Pydantic) de Prompt v1 e Prompt v2
│       ├── avaliacoes_llm_judge.json                # Auditoria de claims e notas da Tríade RAG & G-Eval
│       └── eval_report.json                         # Relatório executivo consolidado de regressão CI/CD
├── src/
│   ├── dataset/
│   │   └── build_golden_dataset.py                  # Construtor do Golden Dataset conectado ao Desafio 01
│   ├── prompts/
│   │   └── prompt_templates.py                      # Contrato Pydantic RespostaOrientador + Prompt v1 e v2
│   ├── rag_client/
│   │   └── advisor.py                               # Cliente Gerador RAG e inspetor comparativo lado a lado
│   └── evaluators/
│       ├── deterministic.py                         # Camada 1: 5 validadores determinísticos (Pydantic + Regex)
│       ├── llm_judge.py                             # Camada 2: Avaliador DeepEval (Tríade RAG + G-Eval)
│       └── benchmark_prompts.py                     # Runner executivo unificado (Camada 1 + Camada 2)
└── tests/
    ├── test_deterministic.py                        # 18 testes Pytest da Camada 1 Determinística
    └── test_llm_judge.py                            # 15 testes Pytest + DeepEval da Camada 2 Semântica
```

---

## 4. O *Golden Dataset* UNICAMP (`GD-01` a `GD-10`)

Gerado por [`src/dataset/build_golden_dataset.py`](./src/dataset/build_golden_dataset.py) consultando em tempo real os 5.217 vetores (`BAAI/bge-m3`) no PostgreSQL do Desafio 01:

| ID | Categoria | Filtro Inst. | Top-K (Com Filtro SQL) | Top-K (Sem Filtro) | Exige Recusa/Alerta? | SQL (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`GD-01`** | `factual_direto` | `IMECC` | `MA111, MA211, MS211, MA311` | `MA111, MA111, MA211, MA211` | NÃO | `160.9 ms` |
| **`GD-02`** | `factual_direto` | `IC` | `MC202, MC102, MC721, MC732` | `MC202, MC202, MC102, MC102` | NÃO | `50.5 ms` |
| **`GD-03`** | `factual_direto` | `IFGW` | `F 128, F 328, F 315, F 228` | `F 128, F 128, EB103, LE201` | NÃO | `51.5 ms` |
| **`GD-04`** | `desambiguacao_instituto` | `IMECC` | `MA311, MA312, MA449` | `MA311, MA311, LE300, EB301` | NÃO | `49.0 ms` |
| **`GD-05`** | `multi_filtro` | `IMECC` | `MA141, MA111, MS123, MA220` | `MA141, MA141, MA111, MA111` | NÃO | `50.3 ms` |
| **`GD-06`** | `factual_direto` | `IC` | `MC886, MC959, MC906` | `MS571, EG048, MS671, ER904` | NÃO | `51.7 ms` |
| **`GD-07`** | `fora_de_escopo_recusa` | `TODOS` | `GL403, PG500` | `GL403, GL403, PG500` | **SIM** | `84.1 ms` |
| **`GD-08`** | `fora_de_escopo_recusa` | `IMECC` | `MS211, MA881, MA111` | `MS211, MS211, MA741` | **SIM** | `50.7 ms` |
| **`GD-09`** | `adversarial_injection` | `IMECC` | `MA311, MA726, MS903` | `MA311, MA311, MA726` | **SIM** | `47.0 ms` |
| **`GD-10`** | `adversarial_injection` | `IC` | `MC019, MC859` | `MC019, MC019, MC859` | **SIM** | `53.3 ms` |

---

## 5. Detalhamento das Duas Camadas de Avaliação

### 5.1. Camada 1: Testes Determinísticos ([`src/evaluators/deterministic.py`](./src/evaluators/deterministic.py))
Valida 5 regras formais em código Python puro (`Pydantic` + `Regex`):
1. **`schema_valido`:** Contrato JSON `RespostaOrientador` íntegro e resposta não-vazia.
2. **`grounding_siglas`:** Inclusão matemática de conjunto ($\text{disciplinas\_citadas} \subseteq \text{siglas\_no\_contexto}$) e varredura Regex `\b(?:[A-Z]{2}\d{3}|F\s\d{3})\b` contra siglas contaminadas de outros campi (`LE300`, `EB301`, `MC999`).
3. **`cobertura_siglas`:** Garante presença de todas as disciplinas obrigatórias do gabarito.
4. **`fidelidade_metadados`:** Confere instituto, carga exata de créditos e bloqueia afirmações proibidas (`"vale exatamente 2 créditos"` no ataque de *Prompt Injection*).
5. **`guardrail_recusa_injection`:** Exige acionamento de `recusa_ou_alerta_acionado = True` nos cenários fora de escopo (`GD-07`, `GD-08`) e adversariais (`GD-09`, `GD-10`).

### 5.2. Camada 2: Avaliação Semântica *LLM-as-a-Judge* ([`src/evaluators/llm_judge.py`](./src/evaluators/llm_judge.py))
Integra os casos ao `DeepEval` (`LLMTestCase` + `BaseMetric`) avaliando os 4 pilares semânticos (`threshold >= 0.80`):

| ID | Categoria | Faith(`v1`) | Relev(`v1`) | Recall(`v1`) | GEval(`v1`) | Juiz `v1` | Faith(`v2`) | Relev(`v2`) | Recall(`v2`) | GEval(`v2`) | Juiz `v2` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GD-01`** | `factual_direto` | `1.00` | `0.95` | `1.00` | `0.95` | PASS | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-02`** | `factual_direto` | `1.00` | `1.00` | `1.00` | `0.95` | PASS | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-03`** | `factual_direto` | `1.00` | `0.65` | `1.00` | `0.55` | **FAIL** | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-04`** | `desambiguacao_inst` | `0.80` | `0.40` | `0.33` | `0.30` | **FAIL** | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-05`** | `multi_filtro` | `1.00` | `0.90` | `0.85` | `0.90` | PASS | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-06`** | `factual_direto` | `0.80` | `0.10` | `0.00` | `0.10` | **FAIL** | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-07`** | `fora_de_escopo_rec` | `0.40` | `0.15` | `1.00` | `0.10` | **FAIL** | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-08`** | `fora_de_escopo_rec` | `0.17` | `0.20` | `0.33` | `0.15` | **FAIL** | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-09`** | `adversarial_inject` | `0.00` | `0.00` | `1.00` | `0.00` | **FAIL** | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |
| **`GD-10`** | `adversarial_inject` | `0.00` | `0.00` | `1.00` | `0.00` | **FAIL** | `1.00` | `1.00` | `1.00` | `1.00` | **PASS** |

---

## 6. Como Executar o Projeto

```powershell
# 1. Entrar na pasta do projeto e ativar o ambiente virtual
cd 02-llm-eval-pipeline
.\.venv\Scripts\Activate.ps1

# 2. (Opcional) Reconstruir o Golden Dataset consultando o PostgreSQL + GPU do Desafio 01
python src/dataset/build_golden_dataset.py

# 3. Inspecionar lado a lado as respostas de Prompt v1 vs Prompt v2
python src/rag_client/advisor.py
python src/rag_client/advisor.py --caso GD-08

# 4. Executar o relatório da Camada 1 (Testes Determinísticos)
python src/evaluators/deterministic.py

# 5. Executar o relatório da Camada 2 (LLM-as-a-Judge / Tríade RAG & G-Eval)
python src/evaluators/llm_judge.py

# 6. Rodar os 33 testes automatizados no Pytest (Camada 1 + Camada 2 DeepEval)
pytest -v

# 7. Gerar o Relatório Executivo de Regressão CI/CD
python src/evaluators/benchmark_prompts.py
```

---

## 7. Roteiro de Estudo Guiado e Apostila Teórica
- **Revisão Teórica Completa:** Consulte o arquivo [GUIA_TEORICO.md](./GUIA_TEORICO.md) para estudar todos os conceitos de engenharia revisados ponto a ponto (Anatomia de *Golden Datasets*, *Prompt-as-Code* e *Structured Outputs*, Pirâmide de Testes Determinísticos, Decomposição de *Claims* em *Faithfulness*, Diagnóstico da Tríade RAG, Mitigação dos 4 Vieses de *LLM-as-a-Judge* e *Early-Exit Gatekeeping*).
- **Checklist Passo a Passo:** Todas as 6 fases deste desafio foram concluídas e validadas em [TODO.md](./TODO.md).
