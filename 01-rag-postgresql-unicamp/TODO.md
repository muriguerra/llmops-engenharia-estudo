# Roteiro de Estudo e To-Do: Desafio 01 (RAG com PostgreSQL e pgvector)

Este guia foi estruturado em **6 fases progressivas** para você estudar os conceitos teóricos e construir a infraestrutura passo a passo, sem pular fundamentos importantes.

---

## Fase 1: Preparação do Ambiente Base (Pré-requisitos)
**Objetivo:** Garantir que o ambiente de desenvolvimento no Windows esteja pronto para rodar containers e bibliotecas de IA em Python.

- [x] **1.1. Instalar e validar o Python (3.11 ou 3.12 recomendado):**
  - Verificar no terminal com `python --version` (`Python 3.12.10` validado).
  - Criar um ambiente virtual isolado na pasta do projeto (`python -m venv .venv` criado em `01-rag-postgresql-unicamp/.venv`).
- [x] **1.2. Instalar e validar o Docker Desktop:**
  - Garantir que o WSL2 esteja ativo e o Docker Desktop rodando (`Docker 29.9.3` e `Docker Compose v5.7.0` validados).
- [x] **1.3. Definir gerenciador de pacotes e dependências iniciais:**
  - [x] **1.3.a.** Criar `requirements.txt` com as bibliotecas do projeto:
    - `psycopg[binary]` e `pgvector` (driver PostgreSQL + suporte ao tipo `vector`)
    - `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`
    - `sentence-transformers` (para rodar o modelo `BAAI/bge-m3` localmente)
    - `pymupdf` (`fitz`) (para parsing rápido de PDFs acadêmicos)
    - `python-dotenv` (para variáveis de ambiente de conexão ao banco)
  - [x] **1.3.b.** Instalar as dependências dentro do `.venv` (`pip install -r requirements.txt`) e validar as importações.

---

## Fase 2: Infraestrutura de Banco de Dados (`PostgreSQL + pgvector`)
**Objetivo:** Subir o PostgreSQL com suporte vetorial persistente e entender a modelagem híbrida (relacional + vetorial).

- [x] **2.1. Estudar o conceito do `pgvector`:**
  - Entender o tipo de dado `vector(N)` (no caso do `bge-m3`, `N = 1024` dimensões).
  - Entender os 3 operadores de distância do `pgvector`:
    - `<->` Distância Euclidiana (L2)
    - `<#>` Produto Interno Negativo (Inner Product)
    - `<=>` **Distância de Cosseno** (o foco deste desafio; similaridade = `1 - distância`)
- [x] **2.2. Criar o `docker-compose.yml`:**
  - Utilizar a imagem oficial `pgvector/pgvector:pg16`.
  - Configurar volume persistente para não perder os embeddings ao reiniciar o container.
- [x] **2.3. Criar o script SQL de inicialização (`docker/initdb/01_schema.sql`):**
  - Habilitar a extensão: `CREATE EXTENSION IF NOT EXISTS vector;`
  - Modelar a tabela relacional combinada com vetor alinhada ao Caderno de Horários da DAC:
    - Colunas relacionais de filtro: `id`, `tipo_documento` (`disciplina` | `resolucao`), `ano`, `semestre` (`1S` | `2S`), `instituto` (`IMECC`, `IC`, `FEEC`, `CEL`), `codigo_disciplina` (`LA092`, `MA111`, `MC102`), `titulo`, `creditos`, `pre_requisitos`, `metadata_json` (`JSONB`).
    - Coluna de texto: `conteudo_chunk TEXT`.
    - Coluna vetorial: `embedding vector(1024)`.

---

## Fase 3: Coleta de Dados (Web Scraper DAC), Parsing e Estratégia de Chunking
**Objetivo:** Coletar as disciplinas reais de 2026 (`1S` e `2S`) direto do Caderno de Horários da DAC UNICAMP com *checkpoint* incremental e paralelismo, estruturando os blocos de texto com metadados ricos.

- [x] **3.1. Preparar dependências de Web Scraping e arquitetura de coleta (`data/raw/`):**
  - Adicionar `requests` e `beautifulsoup4` ao `requirements.txt` e instalar no `.venv`.
  - Definir o formato de *checkpoint* incremental em JSON Lines (`data/raw/disciplinas_2026.jsonl`).
- [x] **3.2. Construir o Web Scraper paralelo com Checkpoint (`src/ingestion/dac_scraper.py`):**
  - Varredura de `1S` e `2S` de `2026` no portal da DAC (`https://www.dac.unicamp.br/portal/caderno-de-horarios/2026/{semestre}/S/G`).
  - Extração de `ano`, `semestre`, `instituto`, `codigo_disciplina`, `titulo`, `ementa`, `creditos`, `pre_requisitos` e metadados extras (`continencia`, `equivalencia`, `url`).
  - Validar primeiro com o instituto `IMECC` e depois expandir para todos os institutos.
- [x] **3.3. Definir e testar a estratégia de Chunking e Enriquecimento (`src/ingestion/chunker.py`):**
  - Estudar o impacto de `chunk_size` e `chunk_overlap` com `RecursiveCharacterTextSplitter`.
  - Garantir que cada *chunk* receba cabeçalho contextual e herde os metadados relacionais da disciplina (fundamental para o `WHERE instituto = 'IMECC'`).

---

## Fase 4: Geração de Embeddings Locais com `BAAI/bge-m3`
**Objetivo:** Gerar vetores densos multilíngues de alta qualidade na sua máquina e persistir no PostgreSQL.

- [x] **4.1. Estudar as características do modelo `BAAI/bge-m3`:**
  - Por que o `bge-m3` é excelente para português? (Suporte multilíngue nativo, janela de até 8192 tokens e vetor denso de `1024` dimensões).
  - Entender a importância de normalizar os vetores (`normalize_embeddings=True`) ao trabalhar com similaridade de cosseno.
- [x] **4.2. Implementar o módulo de Embeddings (`src/embeddings/embedder.py`):**
  - Carregar o modelo via `SentenceTransformer("BAAI/bge-m3")` com suporte a GPU NVIDIA (`float16`).
  - Testar a geração de um vetor para uma frase simples e verificar se `len(vetor) == 1024`.
- [x] **4.3. Implementar o pipeline de Ingestão no Banco (`src/database/ingest.py`):**
  - Inserir em lote (*batch insert*) os 5.217 chunks + metadados relacionais + vetor `embedding` (1024 dimensões) na tabela `documentos_unicamp` do PostgreSQL.

---

## Fase 5: Índices Vetoriais (`HNSW` vs `IVFFlat`)
**Objetivo:** Dominar a diferença entre busca exata (*Sequential Scan*) e busca aproximada (*ANN - Approximate Nearest Neighbor*).

- [x] **5.1. Estudar a teoria dos índices no `pgvector`:**
  - **Sem índice (Exact KNN):** O banco calcula o cosseno contra 100% das linhas (precisão 100%, mas lento em grandes volumes).
  - **IVFFlat (*Inverted File with Flat Compression*):** Agrupa vetores em *clusters* (`lists`) via k-means. Requer que a tabela já tenha dados ao criar o índice; ajuste do parâmetro `ivfflat.probes` na hora da busca.
  - **HNSW (*Hierarchical Navigable Small World*):** Grafo multicamada de alta performance e excelente *recall*; pode ser criado com a tabela vazia; ajuste de `m`, `ef_construction` (criação) e `hnsw.ef_search` (busca).
- [x] **5.2. Criar os índices com a classe de operador correta (`vector_cosine_ops`):**
  - Testar a criação dos índices `IVFFlat` e `HNSW` (`src/database/benchmark_indexes.py`):
    `CREATE INDEX ON documentos_unicamp USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);`
  - Usar `EXPLAIN ANALYZE` no PostgreSQL para comprovar que o planejador está usando o índice e entender como filtros `WHERE` interagem com índices vetoriais (*pre-filtering* vs *post-filtering* e *iterative index scans* nas versões recentes do `pgvector`).

---

## Fase 6: Queries SQL Híbridas e Integração RAG
**Objetivo:** Cumprir o desafio central combinando filtros relacionais estritos com ordenação por similaridade de cosseno.

- [ ] **6.1. Escrever a Query SQL Híbrida (`src/retrieval/search.py`):**
  - Converter a pergunta do usuário em um vetor de 1024 dimensões usando o `bge-m3`.
  - Executar a query SQL parametrizada:
    ```sql
    SELECT
        codigo_disciplina,
        titulo,
        instituto,
        conteudo_chunk,
        1 - (embedding <=> %s::vector) AS similaridade_cosseno
    FROM documentos_unicamp
    WHERE instituto = %s
    ORDER BY embedding <=> %s::vector
    LIMIT 5;
    ```
- [ ] **6.2. Comparar resultados Com Filtro vs Sem Filtro:**
  - Fazer uma pergunta sobre "Cálculo" ou "Álgebra Linear" filtrando por `instituto = 'IMECC'` vs buscando na base inteira.
- [ ] **6.3. (Opcional/Fechamento) Conectar os Chunks recuperados a um LLM via LangChain:**
  - Montar o prompt final passando os Top-K chunks recuperados do PostgreSQL para responder perguntas acadêmicas citando a resolução/disciplina fonte.
