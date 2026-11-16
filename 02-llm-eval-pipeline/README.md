# Desafio 02: Pipeline de Avaliação de LLMs e RAG (Testes Determinísticos & *LLM-as-a-Judge*)

Pipeline de **Engenharia de Qualidade, Testes Determinísticos de Prompt e Avaliação Semântica (*LLM-as-a-Judge*)** construído sobre a base de dados e o motor de busca híbrida da **UNICAMP (Desafio 01)**.

O projeto resolve um dos maiores gargalos de **LLMOps em Produção**: como garantir que alterações em *System Prompts*, parâmetros de *Retrieval* (`Top-K`, filtros SQL) ou trocas de modelos de linguagem não introduzam **alucinações**, quebras de contrato JSON ou perda de precisão nas respostas acadêmicas antes do deploy.

---

## Arquitetura do Pipeline de Avaliação

```mermaid
flowchart LR
    subgraph Base01 ["Base do Desafio 01 (UNICAMP 2026)"]
        PG[("PostgreSQL + pgvector<br/>5.217 Chunks (bge-m3)")]
        JSONL[("chunks_2026.jsonl")]
    end

    subgraph Golden ["1. Golden Dataset (data/golden_dataset/)"]
        GD["Casos de Teste Curados<br/>(Pergunta + Filtros + Contexto Esperado<br/>+ Ground Truth + Regras)"]
    end

    subgraph SUT ["2. Sistema Sob Teste (src/prompts/ & src/rag_client/)"]
        P_V1["Prompt v1 (Ingênuo)"]
        P_V2["Prompt v2 (Engenharia Estrita + JSON/Citações)"]
        LLM_GEN["Gerador RAG UNICAMP<br/>(Contexto Top-K + LLM)"]
    end

    subgraph Eval ["3. Bateria de Avaliação Híbrida (tests/ & src/evaluators/)"]
        DET["Camada 1: Testes Determinísticos (Pytest + Pydantic)<br/>- Schema JSON válido<br/>- Regex de siglas reais (MA111, MC102)<br/>- Proibição de alucinar códigos fora do contexto<br/>- Guardrail de recusa fora de escopo"]
        JUDGE["Camada 2: LLM-as-a-Judge (DeepEval / G-Eval)<br/>- Faithfulness (Fidelidade à Ementa)<br/>- Answer Relevancy (Relevância)<br/>- Contextual Recall & Precision<br/>- Rubrica Acadêmica Customizada"]
    end

    subgraph Report ["4. Relatório de Regressão (data/results/)"]
        BENCH["Comparativo Prompt v1 vs Prompt v2<br/>Pass Rate (%), Latência e Scores"]
    end

    Base01 --> GD
    Base01 --> LLM_GEN
    GD --> LLM_GEN
    P_V1 --> LLM_GEN
    P_V2 --> LLM_GEN
    LLM_GEN --> DET
    LLM_GEN --> JUDGE
    DET --> BENCH
    JUDGE --> BENCH
```

---

## Estrutura de Diretórios do Projeto

```text
02-llm-eval-pipeline/
├── README.md                        # Visão geral e arquitetura do Desafio 02
├── TODO.md                          # Roteiro guiado de estudo passo a passo (6 fases)
├── requirements.txt                 # Dependências Python do projeto (Pytest, DeepEval, Pydantic, etc.)
├── .env.example                     # Modelo de variáveis de ambiente (Banco Desafio 01 + API Keys LLM)
├── data/
│   ├── golden_dataset/              # Dataset Padrão-Ouro (perguntas, ground truth e contextos UNICAMP)
│   └── results/                     # Relatórios JSON/Markdown de regressão (Prompt v1 vs Prompt v2)
├── src/
│   ├── dataset/                     # Construtor e validador do Golden Dataset a partir do Desafio 01
│   ├── rag_client/                  # Integração com a busca híbrida do Desafio 01 e chamada ao LLM
│   ├── prompts/                     # Versionamento de Prompts (v1 ingênuo vs v2 estruturado/restritivo)
│   └── evaluators/                  # Métricas determinísticas e juízes semânticos (LLM-as-a-Judge)
└── tests/                           # Suíte Pytest automatizada (Determinísticos + DeepEval)
```

---

## Como usar este repositório de estudo
Siga o roteiro detalhado em [TODO.md](./TODO.md), avançando uma fase por vez.
