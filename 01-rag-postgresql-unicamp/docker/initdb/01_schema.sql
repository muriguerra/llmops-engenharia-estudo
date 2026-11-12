-- ============================================================================
-- Desafio 01: Motor de Busca Semântica (RAG com PostgreSQL + pgvector)
-- Script de Inicialização do Banco de Dados (01_schema.sql)
-- Alinhado com a estrutura real do Caderno de Horários / Catálogo da DAC UNICAMP
-- ============================================================================

-- 1. Habilita a extensão pgvector no banco de dados atual
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Criação da Tabela Híbrida (Relacional + Vetorial)
-- Combina os dados estruturados da DAC (para filtros WHERE exatos) com o vetor
-- denso de 1024 dimensões gerado pelo modelo BAAI/bge-m3 sobre a ementa/conteúdo.
CREATE TABLE IF NOT EXISTS documentos_unicamp (
    id SERIAL PRIMARY KEY,
    
    -- Metadados Relacionais da DAC / UNICAMP (usados em filtros WHERE)
    tipo_documento VARCHAR(50) NOT NULL DEFAULT 'disciplina', -- Ex: 'disciplina' ou 'resolucao'
    ano INTEGER NOT NULL,                     -- Ex: 2026 (Ano / Ano de Catálogo)
    semestre VARCHAR(10),                     -- Ex: '1S', '2S' (NULL para resoluções anuais)
    instituto VARCHAR(20) NOT NULL,           -- Ex: 'IMECC', 'IC', 'FEEC', 'CEL'
    codigo_disciplina VARCHAR(20),            -- Sigla da disciplina, ex: 'LA092', 'MA111', 'MC102'
    titulo TEXT NOT NULL,                     -- Nome da disciplina ou resolução, ex: 'Língua Inglesa II'
    creditos INTEGER,                         -- Número de créditos, ex: 4, 6 (permite WHERE creditos <= 4)
    pre_requisitos TEXT,                      -- Ex: 'LA091', 'MA111, MA141' ou 'AA200' / 'Nenhum'
    
    -- Conteúdo Textual (Ementa / Chunk) e Metadados Extras
    chunk_index INTEGER NOT NULL DEFAULT 0,   -- Posição do bloco (0 se a ementa inteira couber em 1 chunk)
    conteudo_chunk TEXT NOT NULL,             -- Texto da Ementa (ou trecho de resolução) que vai ao LLM
    metadata_json JSONB DEFAULT '{}'::jsonb,  -- Campos extras (ex: nivel 'Graduação/Tecnologia', arquivo_origem)
    
    -- Coluna Vetorial (BAAI/bge-m3 = 1024 dimensões)
    embedding vector(1024) NOT NULL,
    
    criado_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Índices Relacionais Tradicionais (B-Tree)
-- Aceleram a filtragem estruturada antes/durante a busca vetorial
CREATE INDEX IF NOT EXISTS idx_documentos_instituto ON documentos_unicamp (instituto);
CREATE INDEX IF NOT EXISTS idx_documentos_disciplina ON documentos_unicamp (codigo_disciplina);
CREATE INDEX IF NOT EXISTS idx_documentos_ano_semestre ON documentos_unicamp (ano, semestre);
CREATE INDEX IF NOT EXISTS idx_documentos_creditos ON documentos_unicamp (creditos);

-- NOTA DIDÁTICA:
-- Os índices vetoriais (HNSW e IVFFlat) NÃO foram criados aqui propositalmente!
-- Nós vamos criá-los na Fase 5 para podermos comparar na prática o plano de execução
-- (EXPLAIN ANALYZE) de uma busca exata (Seq Scan) vs busca aproximada (Index Scan).
