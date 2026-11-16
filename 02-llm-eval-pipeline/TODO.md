# Roteiro de Estudo e To-Do: Desafio 02 (Pipeline de Avaliação de LLMs e RAG)

Este guia foi estruturado em **6 fases progressivas** para você estudar os conceitos de **Avaliação de LLMs (LLMOps)** e construir uma esteira de testes automatizados utilizando os dados reais da **UNICAMP (Desafio 01)** como base.

---

## Fase 1: Preparação do Ambiente e Escolha do Provedor LLM
**Objetivo:** Configurar o ambiente isolado do Desafio 02, conectar à base do Desafio 01 e definir o modelo gerador e o modelo juiz (*LLM-as-a-Judge*).

- [x] **1.1. Criar e validar o ambiente virtual Python (`02-llm-eval-pipeline/.venv`):**
  - Validar `python --version` (`Python 3.12.10` validado).
  - Criar ambiente virtual isolado: `python -m venv .venv` criado em `02-llm-eval-pipeline/.venv` com ponte `.pth` para compartilhar PyTorch CUDA e `sentence-transformers` do Desafio 01 sem duplicar disco.
- [x] **1.2. Definir o provedor de LLM para Geração e Juiz (*Judge*):**
  - Arquitetura híbrida definida: suporte nativo a execução via **Antigravity** (`LLM_PROVIDER=antigravity`, custo zero e sem necessidade de chave de API externa) e compatibilidade com `GEMINI_API_KEY` / `OPENAI_API_KEY` via `.env`.
  - Arquivo `.env` configurado (incluindo conexão ativa ao container `unicamp-pgvector` na porta `5432`).
- [x] **1.3. Instalar dependências (`requirements.txt`):**
  - `pytest`, `deepeval`, `pydantic` (testes determinísticos, contratos de saída e *LLM-as-a-Judge*).
  - `psycopg[binary]`, `pgvector`, `sentence-transformers` (integração com o Retriever do Desafio 01).
  - Clientes de LLM (`google-genai`, `openai`) e `python-dotenv`.

---

## Fase 2: Construção do *Golden Dataset* (Dataset Padrão-Ouro da UNICAMP)
**Objetivo:** Entender a anatomia de um *Golden Dataset* em LLMOps e construir casos de teste reais extraídos das 5.031 disciplinas da UNICAMP coletadas no Desafio 01.

- [ ] **2.1. Estudar a teoria de *Golden Datasets* para RAG:**
  - Diferença entre `input` (pergunta do aluno), `expected_output` (*ground truth* / gabarito), `retrieval_context` (chunks reais recuperados) e metadados esperados (`instituto_esperado`, `siglas_obrigatorias`, `cenario`).
  - Categorias essenciais de casos de teste:
    1. **Consultas Diretas de Ementa/Pré-requisito** (ex: `MA111`, `MA311`, `MC102`).
    2. **Consultas Multi-Filtro / Comparativas** (ex: matérias de Cálculo no `IMECC` vs outros institutos).
    3. **Casos de Borda (*Out-of-Scope* / *Negative Rejection*):** Perguntas sobre cursos/disciplinas que não existem no contexto recuperado (onde o LLM **deve recusar** responder em vez de alucinar).
    4. **Casos Adversariais (*Prompt Injection* leve):** Tentativa do usuário de fazer o orientador ignorar as regras da UNICAMP.
- [ ] **2.2. Implementar o gerador do Golden Dataset (`src/dataset/build_golden_dataset.py`):**
  - Consultar os dados reais do Desafio 01 (`01-rag-postgresql-unicamp/data/processed/chunks_2026.jsonl` e/ou PostgreSQL) para gerar `data/golden_dataset/unicamp_golden_dataset.jsonl` com casos 100% fundamentados no catálogo oficial de 2026.

---

## Fase 3: Versionamento de Prompts e Sistema Sob Teste (RAG Client)
**Objetivo:** Tratar Prompts como código versionado (*Prompt-as-Code*) e criar duas versões contrastantes do Orientador Acadêmico para medir evolução e regressão.

- [ ] **3.1. Estudar *Prompt Versioning* e Contratos de Saída Estruturada (`Pydantic`):**
  - Por que respostas em texto livre são difíceis de validar em produção?
  - Como exigir respostas estruturadas (ou seções padronizadas com citações explícitas de `[Sigla - Instituto]`, `créditos` e `pré-requisitos`).
- [ ] **3.2. Criar o catálogo de Prompts (`src/prompts/prompt_templates.py`):**
  - **`PROMPT_V1_INGENUO`:** Prompt curto e permissivo (*"Responda à dúvida do aluno sobre a UNICAMP usando o contexto"*), propenso a extrapolar conhecimento prévio do LLM ou omitir pré-requisitos/institutos.
  - **`PROMPT_V2_ESTRITO`:** Prompt engenheirado com regras explícitas de *grounding* estrito, citação obrigatória de sigla/instituto/créditos/pré-requisitos e regra de recusa segura (*abstention*) quando a informação não consta nos chunks.
- [ ] **3.3. Implementar o Pipeline Gerador (`src/rag_client/advisor.py`):**
  - Conectar a busca híbrida do Desafio 01 (ou o contexto do Golden Dataset) ao LLM escolhido, registrando tempo de latência, tokens consumidos e saída estruturada.

---

## Fase 4: Camada 1 — Testes Determinísticos de Prompt (`Pytest` + `Pydantic` + `Regex`)
**Objetivo:** Construir a primeira barreira de CI/CD (custo zero de API de juiz, execução em milissegundos) para capturar falhas estruturais e alucinações óbvias de siglas.

- [ ] **4.1. Estudar a Pirâmide de Testes em LLMOps:**
  - Por que rodar testes determinísticos **antes** de chamar um *LLM-as-a-Judge*? (Custo, determinismo, velocidade e validação de regras duras de negócio).
- [ ] **4.2. Implementar validadores determinísticos (`src/evaluators/deterministic.py`):**
  - **Validador de Contrato/Schema (`Pydantic`):** Verifica campos obrigatórios, tipos de dados e ausência de texto vazio.
  - **Validador Anti-Alucinação de Siglas (`Regex` + *Context Grounding*):** Extrai todas as siglas de disciplinas citadas pelo LLM (ex: `MA111`, `MC102`, `F 128`) via expressão regular e comprova matematicamente se **100% delas pertencem ao contexto recuperado**!
  - **Validador de Fidelidade Numérica:** Verifica se os créditos e pré-requisitos citados batem com os metadados do chunk.
  - **Validador de Recusa (*Abstention Check*):** Garante que perguntas sem contexto válido acionem a flag/mensagem de recusa sem inventar matérias.
- [ ] **4.3. Criar a suíte `tests/test_deterministic.py` e rodar com `pytest`:**
  - Validar todas as asserções determinísticas em lote.

---

## Fase 5: Camada 2 — Avaliação Semântica com *LLM-as-a-Judge* (`DeepEval` / *RAG Triad*)
**Objetivo:** Avaliar nuances semânticas onde expressões regulares não chegam, implementando a **Tríade RAG (*RAG Triad*)** e métricas customizadas (*G-Eval*).

- [ ] **5.1. Estudar a teoria do *LLM-as-a-Judge* e a Tríade RAG:**
  - **Faithfulness (Fidelidade / Anti-Alucinação):** Cada afirmação da resposta pode ser deduzida exclusivamente do `retrieval_context`?
  - **Answer Relevancy (Relevância da Resposta):** A resposta responde diretamente à pergunta do estudante sem divagar?
  - **Contextual Precision & Recall:** Os chunks recuperados no Top-K continham toda a informação necessária e estavam bem ranqueados?
  - **G-Eval (*Chain-of-Thought Judge*):** Como criar uma rubrica acadêmica customizada com pesos e passos de raciocínio para o modelo Juiz, mitigando vieses (*position bias*, *verbosity bias*).
- [ ] **5.2. Implementar os avaliadores semânticos (`src/evaluators/llm_judge.py` e `tests/test_llm_judge.py`):**
  - Configurar o modelo Juiz e executar as métricas de *Faithfulness*, *Answer Relevancy* e *G-Eval* sobre as respostas do Orientador Acadêmico.

---

## Fase 6: Benchmark Comparativo (`Prompt v1` vs `Prompt v2`) e Documentação Final
**Objetivo:** Rodar a bateria completa de regressão comparando `Prompt v1 (Ingênuo)` vs `Prompt v2 (Estrito)` + `Busca Sem Filtro` vs `Busca Híbrida`, gerando métricas reais e consolidando o `GUIA_TEORICO.md`.

- [ ] **6.1. Implementar o Runner de Benchmark (`src/evaluators/benchmark_prompts.py`):**
  - Executar o Golden Dataset inteiro nas configurações avaliadas.
  - Gerar tabela comparativa no terminal e salvar relatório em `data/results/eval_report.json`.
- [ ] **6.2. Atualizar `README.md` com as métricas reais obtidas e criar `GUIA_TEORICO.md`:**
  - Documentar todos os resultados reais e consolidar a apostila teórica completa do Desafio 02 para entrevistas de Engenharia de IA / LLMOps.
