# Guia Teórico e Revisão de Engenharia — Desafio 01 (RAG Híbrido com PostgreSQL + pgvector)

Este documento consolida **toda a base teórica, decisões arquiteturais e aprendizados práticos de engenharia** construídos ao longo do **Desafio 01**. Ele foi desenhado como material de consulta rápida e revisão para entrevistas técnicas de **Engenharia de IA, Engenharia de Dados e LLMOps**.

---

## Sumário
1. [Módulo 1: O que é RAG e Por Que a Arquitetura Híbrida (SQL + Vetor)?](#módulo-1-o-que-é-rag-e-por-que-a-arquitetura-híbrida-sql--vetor)
2. [Módulo 2: Matemática dos Embeddings e Operadores do `pgvector`](#módulo-2-matemática-dos-embeddings-e-operadores-do-pgvector)
3. [Módulo 3: Infraestrutura Docker e Ciclo de Vida do PostgreSQL](#módulo-3-infraestrutura-docker-e-ciclo-de-vida-do-postgresql)
4. [Módulo 4: Engenharia de Coleta Web (Polite Scraping, Keep-Alive e Checkpoints JSONL)](#módulo-4-engenharia-de-coleta-web-polite-scraping-keep-alive-e-checkpoints-jsonl)
5. [Módulo 5: Estratégia de Chunking e *Contextual Chunk Header*](#módulo-5-estratégia-de-chunking-e-contextual-chunk-header)
6. [Módulo 6: Inferência Local em GPU (`BAAI/bge-m3`, CUDA e Precisão Mista `FP16`)](#módulo-6-inferência-local-em-gpu-baaibge-m3-cuda-e-precisão-mista-fp16)
7. [Módulo 7: Estruturas de Índices (`B-Tree` vs `IVFFlat` vs `HNSW`) e o Planejador SQL](#módulo-7-estruturas-de-índices-b-tree-vs-ivfflat-vs-hnsw-e-o-planejador-sql)
8. [Módulo 8: Do Retriever ao Agente (Como o `search.py` vira uma Tool de LLM)](#módulo-8-do-retriever-ao-agente-como-o-searchpy-vira-uma-tool-de-llm)

---

## Módulo 1: O que é RAG e Por Que a Arquitetura Híbrida (SQL + Vetor)?

### 1.1. O Problema das Alucinações e Conhecimento Estático
Modelos de Linguagem (LLMs) possuem duas limitações estruturais quando aplicados a dados corporativos ou acadêmicos específicos:
1. **Corte Temporal (*Knowledge Cutoff*):** O modelo não conhece dados publicados ou atualizados após seu treinamento (como o Caderno de Horários de 2026 da UNICAMP).
2. **Alucinação por Aproximação Probabilística:** Quando perguntado sobre uma sigla específica (ex: *"Quais os pré-requisitos de MA311?"*), um LLM sem acesso à fonte tenta "adivinhar" os tokens mais prováveis, frequentemente inventando códigos ou regras inexistentes.

O **RAG (*Retrieval-Augmented Generation*)** resolve isso transformando um problema de *memória* do LLM em um problema de *leitura interpretativa*: primeiro buscamos os documentos exatos em um banco de dados confiável (**Retrieval**) e depois injetamos esses documentos no prompt do LLM (**Augmented Generation**) exigindo resposta baseada exclusivamente nas fontes.

### 1.2. Banco Vetorial Puro vs. Banco Híbrido Relacional + Vetorial (`PostgreSQL + pgvector`)
Em uma busca semântica pura (apenas vetores), perguntas como *"Cálculo III, equações diferenciais e transformada de Laplace"* retornam documentos cujo **texto da ementa** é parecido, mas ignoram regras duras de negócio.
- **No nosso experimento prático:** A busca vetorial pura trouxe no Top 5 disciplinas de Campinas (`MA311` do `IMECC`) misturadas com disciplinas de Limeira (`LE300` da `FCA` e `EB301` da `FT`), pois suas ementas são quase idênticas!
- **A Vantagem do PostgreSQL + `pgvector`:** Permite unir na mesma tabela e na mesma transação ACID:
  - **Colunas Relacionais Estritas (`VARCHAR`, `INTEGER`):** Para filtros exatos (`WHERE instituto = 'IMECC' AND semestre = '1S' AND creditos <= 6 AND pre_requisitos = 'Nenhum'`).
  - **Coluna Semi-estruturada (`JSONB`):** Para metadados flexíveis (`continencia`, `equivalencia`, `url_origem`) sem precisar alterar o schema (`ALTER TABLE`) a cada campo extra.
  - **Coluna Vetorial (`vector(1024)`):** Para ordenação semântica densa.

---

## Módulo 2: Matemática dos Embeddings e Operadores do `pgvector`

### 2.1. O Tipo de Dado `vector(N)`
Um *embedding* é a tradução de um texto para uma coordenada geométrica em um espaço de $N$ dimensões (uma lista de $N$ números `float32`). Textos com significados semelhantes ficam posicionados próximos uns dos outros nesse espaço.
- No nosso projeto, $N = 1024$ porque essa é a dimensão fixa de saída do modelo **`BAAI/bge-m3`**.
- Declarar a coluna como `embedding vector(1024)` no PostgreSQL garante **integridade dimensional**: o banco rejeita automaticamente qualquer tentativa de inserir um vetor de tamanho diferente (ex: 768 ou 1536 dimensões).

### 2.2. Os 3 Operadores de Distância do `pgvector`

| Operador no SQL | Nome Matemático | O que calcula | Classe de Índice no `pgvector` | Quando utilizar |
| :--- | :--- | :--- | :--- | :--- |
| **`<->`** | Distância Euclidiana ($L_2$) | Linha reta entre dois pontos no espaço ($\sqrt{\sum (a_i - b_i)^2}$). Sensível à magnitude (comprimento) dos vetores. | `vector_l2_ops` | Dados onde a intensidade/magnitude absoluta do vetor importa. |
| **`<#>`** | Produto Interno Negativo (*Inner Product*) | $-\sum (a_i \cdot b_i)$. É o cálculo menos custoso para a CPU. | `vector_ip_ops` | Quando os vetores já estão obrigatoriamente normalizados e busca-se máxima velocidade bruta. |
| **`<=>`** | **Distância de Cosseno** | $1 - \cos(\theta)$. Mede apenas o **ângulo** entre os dois vetores, ignorando seus comprimentos. | **`vector_cosine_ops`** | **Padrão para RAG / Busca Textual.** Foco do nosso projeto. |

### 2.3. Distância de Cosseno (`ORDER BY`) vs. Similaridade de Cosseno (`SELECT`)
Uma dúvida muito comum ao escrever queries no `pgvector`:
- O operador `<=>` retorna a **Distância de Cosseno**, que varia de `0.0` (textos idênticos, ângulo de $0^\circ$) até `2.0` (vetores diametralmente opostos):
  $$\text{Distância de Cosseno}(A, B) = 1 - \text{Similaridade de Cosseno}(A, B)$$
- Por isso:
  1. No **`ORDER BY`**, ordenamos de forma crescente (`ASC`, que é o padrão do SQL) pela **menor distância**:
     ```sql
     ORDER BY embedding <=> %s::vector LIMIT 5
     ```
  2. No **`SELECT`**, para exibirmos uma nota intuitiva de `0` a `1` (onde `1.0000` significa 100% de similaridade), fazemos o complemento:
     ```sql
     1 - (embedding <=> %s::vector) AS similaridade_cosseno
     ```

---

## Módulo 3: Infraestrutura Docker e Ciclo de Vida do PostgreSQL

No arquivo [docker-compose.yml](./docker-compose.yml), aplicamos 4 conceitos essenciais de infraestrutura para LLMOps:

1. **Imagem Especializada (`pgvector/pgvector:pg16`):**
   - A imagem padrão `postgres:16` não traz os binários em C da extensão `pgvector`. Usar a imagem oficial do `pgvector` evita ter que compilar código C (`gcc`, `make`) dentro do container.
2. **Persistência com Volume Nomeado (`unicamp_pgvector_data:/var/lib/postgresql/data`):**
   - Containers são efêmeros por natureza. Como gerar milhares de embeddings consome tempo de GPU, o volume nomeado garante que os `33 MB` da tabela e os `41 MB` do índice `HNSW` sobrevivam mesmo após `docker compose down` ou reinicialização do Windows.
3. **Regra de Ouro do `/docker-entrypoint-initdb.d`:**
   - O PostgreSQL no Docker executa scripts `.sql` mapeados nessa pasta **exclusivamente na primeira vez em que o diretório de dados (`pgdata`) é inicializado do zero**.
   - Se você alterar o arquivo `01_schema.sql` depois que o volume já foi criado, um simples `docker compose restart` **não** reexecuta o SQL. Para recriar o banco do zero lendo o novo schema, usa-se `docker compose down -v` (onde `-v` remove o volume antigo) seguido de `docker compose up -d`.
4. **Resiliência (`restart: unless-stopped` + `healthcheck` com `pg_isready`):**
   - O container sobe sozinho assim que o Docker Desktop é iniciado no Windows, e o `pg_isready` testa se o processo do banco já aceita conexões SQL reais.

---

## Módulo 4: Engenharia de Coleta Web (Polite Scraping, Keep-Alive e Checkpoints JSONL)

Durante a coleta real das **5.031 disciplinas** dos 26 institutos da UNICAMP no portal da DAC ([src/ingestion/dac_scraper.py](./src/ingestion/dac_scraper.py)), resolvemos 4 problemas clássicos de Engenharia de Dados em produção:

### 4.1. Bloqueio de Firewall por *SYN Flood* vs. `HTTP Keep-Alive` (`threading.local()`)
- **O Problema:** Na primeira tentativa com 8 threads criando uma nova `requests.Session()` a cada página, o script tentou abrir centenas de novas conexões TCP/TLS na porta `443` em poucos segundos. O firewall da UNICAMP detectou o pico de pacotes `SYN` e aplicou um bloqueio temporário (*ConnectTimeoutError*) após a 9ª disciplina.
- **A Solução (`HTTP Keep-Alive` por Thread):** Usando `threading.local()`, cada *worker* cria **uma única sessão TCP persistente** (`Connection: keep-alive`) e reaproveita o mesmo túnel já aberto para todas as requisições seguintes, combinado com uma pausa educada (`--delay 0.6` + variação aleatória *jitter*). Resultado: baixamos todas as 5.031 disciplinas sem nenhum bloqueio adicional.

### 4.2. Por que `.jsonl` (JSON Lines) é superior ao `.json` para Checkpoints?
- Em um arquivo `.json` tradicional (`[ {...}, {...} ]`), se o programa parar na metade, o arquivo fica sintaticamente inválido (sem o `]` final).
- Em um arquivo **`.jsonl`**, cada linha é um JSON independente terminado em `\n`:
  - **Escrita O(1) Thread-Safe:** Cada disciplina concluída é gravada imediatamente no fim do arquivo (`open("a")` + `flush()`) protegida por `threading.Lock()`.
  - **Retomada Instantânea (*Resume*):** Ao iniciar, o script lê as linhas existentes, armazena o campo `"chave"` (`"2026_1S_IMECC_MA111"`) em um conjunto `set()` em memória (busca $O(1)$) e pula tudo o que já foi baixado.

### 4.3. Tratamento de URLs com Espaço Codificado (`%20`)
- No Instituto de Física (`IFGW`), as siglas de Física Geral possuem espaço interno para completar 5 caracteres (`F 128`, `F 328`), aparecendo no HTML como `.../IFGW/F%20128`.
- Ajustamos a expressão regular para capturar `%` (`[A-Z0-9%]+`), decodificamos com `urllib.parse.unquote()` para salvar `"F 128"` limpo no banco, e recodificamos com `urllib.parse.quote()` na hora da requisição HTTP.

---

## Módulo 5: Estratégia de Chunking e *Contextual Chunk Header*

No script [src/ingestion/chunker.py](./src/ingestion/chunker.py), transformamos `5.031` disciplinas brutas em `5.217` chunks prontos para vetorização:

### 5.1. `RecursiveCharacterTextSplitter` (`chunk_size=600`, `chunk_overlap=100`)
- Textos muito longos diluídos em um único vetor perdem especificidade semântica.
- O divisor recursivo tenta quebrar o texto respeitando a hierarquia natural da escrita: primeiro quebras de parágrafo (`\n\n`), depois linha (`\n`), depois ponto final (`. `) e ponto-e-vírgula (`; `).
- **O Papel do `chunk_overlap = 100`:** Quando uma ementa longa (como as `162` disciplinas que tinham mais de 600 caracteres, ex: `MA446 - Grupos e Representações`) é dividida em `Parte 1/2` e `Parte 2/2`, os últimos 100 caracteres da Parte 1 são repetidos no início da Parte 2 para que nenhuma frase ou conceito seja mutilado na fronteira do corte.

### 5.2. A Técnica do *Contextual Chunk Header*
Se vetorizarmos apenas o texto solto da ementa (`"Intervalos e desigualdades. Funções. Limites..."`), o vetor não conterá o código `MA111`, nem o nome `Cálculo I`, nem o instituto `IMECC`.
Por isso, injetamos um cabeçalho estruturado no topo de cada chunk antes de passá-lo para o modelo de embedding:

```text
[Instituto: IMECC | Disciplina: MA111 - Cálculo I | Período: 1S/2026 | Créditos: 6 | Pré-requisitos: Nenhum]
Ementa: Intervalos e desigualdades. Funções. Limites. Continuidade. Derivada e diferencial. Integral. Técnicas de integração.
```

Isso garante que:
1. O vetor de 1024 dimensões codifique simultaneamente a identidade da disciplina e seu conteúdo programático;
2. Todo chunk derivado de uma mesma disciplina herde todas as colunas relacionais (`instituto`, `semestre`, `creditos`, `pre_requisitos`) no PostgreSQL.

---

## Módulo 6: Inferência Local em GPU (`BAAI/bge-m3`, CUDA e Precisão Mista `FP16`)

### 6.1. Por que o modelo `BAAI/bge-m3`?
O nome **M3** resume suas três capacidades:
- **Multi-Lingual:** Excelente compreensão semântica nativa do Português acadêmico (superando modelos treinados apenas em inglês).
- **Multi-Granularity:** Suporta desde sentenças curtas até documentos de 8.192 tokens, gerando vetores densos de `1024` dimensões.
- **Multi-Functionality:** Capacidade de representação densa, esparsa (lexical) e multi-vetor.

### 6.2. Otimização de Hardware: PyTorch CUDA + `FP16 (float16)` em GPU de 4 GB VRAM
- **Armadilha do `pip` no Windows:** O comando padrão `pip install torch` instala o binário `+cpu`. Substituímos pelo binário compilado para CUDA (`--index-url https://download.pytorch.org/whl/cu124`).
- **Gerenciamento de VRAM com `torch.float16`:** Em precisão padrão de 32 bits (`float32`), o `BAAI/bge-m3` (~568M de parâmetros) consome ~2,3 GB de VRAM apenas para os pesos, correndo risco de `CUDA Out of Memory` em placas de 4 GB durante o processamento de lotes. Ao ativarmos `torch_dtype = torch.float16` e limitarmos `max_seq_length = 512`, o consumo caiu para **~1,1 GB de VRAM** e a placa atingiu **~55 chunks/segundo** (vetorizando e gravando 4.994 blocos em apenas `91,3s`!).
- **Normalização L2 (`normalize_embeddings=True`):** Garante que todo vetor saia com comprimento unitário ($\|v\|_2 = 1.0$), tornando o cálculo do cosseno mais rápido e estável.

---

## Módulo 7: Estruturas de Índices (`B-Tree` vs `IVFFlat` vs `HNSW`) e o Planejador SQL

### 7.1. Comparativo Conceitual e Analogias dos Índices

| Tipo de Índice | Alvo no Banco | Como Funciona (Analogia) | Prós e Contras |
| :--- | :--- | :--- | :--- |
| **`B-Tree`** | Colunas Escalares (`instituto`, `ano`, `creditos`) | **Dicionário / Lista Telefônica:** Árvore ordenada alfabeticamente/numericamente. Resolve `=`, `<`, `>`, `BETWEEN` em milissegundos. | Não funciona para vetores de 1024 dimensões (não calcula ângulos geométricos). |
| **`Seq Scan` (Sem Índice Vetorial)** | Coluna `embedding vector(1024)` | **Ler o livro inteiro página por página:** Calcula a distância de cosseno contra 100% das 5.217 linhas (KNN Exato). | Precisão (`Recall`) de 100%, zero memória extra, mas fica lento conforme a tabela cresce (`37,54 ms` para 5k linhas; ~3,7s para 500k linhas). |
| **`IVFFlat`** | Coluna `embedding vector(1024)` | **Dividir a cidade em Bairros (*Clusters*):** Agrupa os vetores em `lists = 100` regiões via *k-means*. Na busca, olha apenas os `ivfflat.probes = 5` bairros mais próximos da pergunta. | Rápido para criar (`0,76s`) e busca em `0,92 ms`, mas **não pode ser criado com a tabela vazia** e pode perder vizinhos que moram na fronteira de outro cluster. |
| **`HNSW`** | Coluna `embedding vector(1024)` | **Rede de Transporte em Camadas (Avião -> Metrô -> Rua):** Grafo multicamada navegável (`m = 16` conexões por nó, `ef_construction = 64`). Dá saltos longos no topo e refina na base. | **Padrão ouro de produção:** Pode ser criado com a tabela vazia, atualiza dinamicamente a cada `INSERT`, obteve **`0,582 ms` (`64,5x` mais rápido que o Seq Scan)** mantendo `100%` de `Recall@10`! |

### 7.2. Duas Lições Avançadas do Planejador do PostgreSQL
1. **O Custo de Página Aleatória (`random_page_cost`) em Tabelas Pequenas:**
   - Por padrão, o PostgreSQL assume `random_page_cost = 4.0` (herança da época dos HDs mecânicos). Em uma tabela pequena de `33 MB` (`5.217` linhas) onde o índice HNSW tem `41 MB`, o planejador pode achar que ler os `33 MB` sequencialmente é mais barato do que saltar pelo grafo HNSW. Em produção com SSDs, ajusta-se `random_page_cost = 1.1` (ou `SET enable_seqscan = off` em testes de benchmark) para que o planejador priorize o índice vetorial.
2. **Filtros `WHERE` + Índice HNSW (`SET hnsw.iterative_scan = relaxed_order`):**
   - Em versões antigas de bancos vetoriais, aplicar um filtro restritivo (`WHERE instituto = 'IMECC'`) junto com um índice HNSW que busca apenas `ef_search = 40` candidatos podia retornar menos resultados que o `LIMIT 5` (caso 38 dos 40 candidatos fossem de outros institutos).
   - No `pgvector 0.8+`, o comando `SET hnsw.iterative_scan = relaxed_order;` instrui o PostgreSQL a continuar caminhando iterativamente pelo grafo HNSW até encontrar exatamente os 5 documentos que satisfaçam todas as condições do `WHERE`!

---

## Módulo 8: Do Retriever ao Agente (Como o `search.py` vira uma Tool de LLM)

No fechamento do projeto ([src/retrieval/search.py](./src/retrieval/search.py)), demonstramos como o motor de busca híbrida atua como a **Ferramenta de Recuperação (*Retriever Tool*)** de um Agente ou LLM:

1. **Extração de Parâmetros Estruturados + Semânticos:**
   - Quando um estudante pergunta: *"Quero uma matéria no IMECC no 1º semestre sem pré-requisitos de até 6 créditos sobre derivadas"*, o papel de um Agente (ex: LangGraph) é quebrar essa frase em dois tipos de argumentos para chamar `buscar_disciplinas_hibrido()`:
     - **Filtros Rígidos (`WHERE`):** `instituto="IMECC"`, `semestre="1S"`, `max_creditos=6`, `sem_pre_requisitos=True`
     - **Busca Semântica (`<=>`):** `pergunta="Introdução ao cálculo diferencial e integral, limites, derivadas"`
2. **Injeção de Contexto no Prompt (`montar_prompt_rag`):**
   - Os Top-K chunks recuperados em `~50 ms` do PostgreSQL são formatados em um `ChatPromptTemplate` do LangChain contendo instrução de sistema (*System Prompt*) restritiva + as fontes numeradas com seus scores de similaridade e cabeçalhos contextuais completos, eliminando alucinações e garantindo citação exata da fonte oficial da UNICAMP.
