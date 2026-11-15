"""
Pipeline de Ingestão Vetorial em Lote no PostgreSQL + pgvector (Passo 4.3)
==========================================================================

Objetivo:
Ler os chunks processados (`data/processed/chunks_2026.jsonl`), gerar os vetores
de 1024 dimensões na GPU em lotes (batches) usando o modelo `BAAI/bge-m3` e
inserir os registros (metadados relacionais + texto + vetor) na tabela
`documentos_unicamp` do PostgreSQL de forma idempotente (com checkpoint no banco).

Como executar (dentro de 01-rag-postgresql-unicamp com .venv ativo):
  - Testar primeiro apenas com o IMECC (223 chunks, ~5 segundos na GPU):
      python src/database/ingest.py --institutos IMECC
  - Ingerir TODAS as 5.217 linhas dos 26 institutos:
      python src/database/ingest.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROCESSED_CHUNKS = ROOT_DIR / "data" / "processed" / "chunks_2026.jsonl"


def log_debug(msg: str) -> None:
    """Imprime mensagens de acompanhamento com timestamp e flush imediato no terminal."""
    agora = time.strftime("%H:%M:%S")
    print(f"[{agora}] {msg}", flush=True)


def obter_conexao_db() -> psycopg.Connection:
    """
    Carrega as variáveis do arquivo .env, abre conexão com o PostgreSQL no Docker
    (com timeout de 5 segundos para não travar caso o Docker esteja fechado)
    e registra o adaptador do tipo `vector` do pgvector.
    """
    load_dotenv(ROOT_DIR / ".env")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "unicamp_rag")
    user = os.getenv("POSTGRES_USER", "unicamp_user")
    password = os.getenv("POSTGRES_PASSWORD", "unicamp_pass")

    log_debug(f"[1/4] Conectando ao PostgreSQL em {host}:{port} (banco='{dbname}')...")
    try:
        conn = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=5,
        )
    except psycopg.OperationalError as err:
        print("\n" + "!" * 70, flush=True)
        print("[ERRO DE CONEXÃO COM O BANCO DE DADOS]", flush=True)
        print("Não foi possível conectar ao PostgreSQL na porta 5432.", flush=True)
        print("Verifique se o Docker Desktop está aberto e se o container está rodando:", flush=True)
        print("  -> Comando: docker compose up -d", flush=True)
        print(f"Detalhe técnico: {err}", flush=True)
        print("!" * 70 + "\n", flush=True)
        sys.exit(1)

    register_vector(conn)
    log_debug("[1/4] Conexão com PostgreSQL + extensão pgvector estabelecida com sucesso!")
    return conn


def buscar_chunks_ja_inseridos(conn: psycopg.Connection) -> set[str]:
    """
    Consulta no PostgreSQL quais combinações (chave_origem, chunk_index) já foram
    inseridas anteriormente, evitando duplicação caso o script seja rodado mais de uma vez.
    """
    ja_inseridos: set[str] = set()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT metadata_json->>'chave_origem' AS chave, chunk_index
            FROM documentos_unicamp
            WHERE metadata_json ? 'chave_origem';
            """
        )
        for chave, idx in cur.fetchall():
            if chave is not None:
                ja_inseridos.add(f"{chave}#{idx}")
    return ja_inseridos


def carregar_chunks_pendentes(
    caminho_jsonl: Path,
    ja_inseridos: set[str],
    institutos_filtro: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Lê o arquivo chunks_2026.jsonl e filtra apenas os que ainda não estão no banco."""
    if not caminho_jsonl.exists():
        raise FileNotFoundError(f"Arquivo de chunks não encontrado: {caminho_jsonl}")

    filtro_set = {i.upper() for i in institutos_filtro} if institutos_filtro else None
    pendentes: list[dict[str, Any]] = []

    with caminho_jsonl.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            doc = json.loads(linha)
            if filtro_set and doc.get("instituto", "").upper() not in filtro_set:
                continue

            chave_origem = (doc.get("metadata_json") or {}).get("chave_origem", "")
            chunk_idx = doc.get("chunk_index", 0)
            id_unico = f"{chave_origem}#{chunk_idx}"

            if id_unico not in ja_inseridos:
                pendentes.append(doc)

    return pendentes


def inserir_lote_no_postgres(
    conn: psycopg.Connection,
    lote_docs: list[dict[str, Any]],
    lote_vetores: list[list[float]],
) -> None:
    """Insere um lote (batch) de chunks + embeddings em uma única transação SQL."""
    sql_insert = """
        INSERT INTO documentos_unicamp (
            tipo_documento,
            ano,
            semestre,
            instituto,
            codigo_disciplina,
            titulo,
            creditos,
            pre_requisitos,
            chunk_index,
            conteudo_chunk,
            metadata_json,
            embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
    """

    parametros = [
        (
            doc.get("tipo_documento", "disciplina"),
            doc.get("ano", 2026),
            doc.get("semestre"),
            doc.get("instituto"),
            doc.get("codigo_disciplina"),
            doc.get("titulo"),
            doc.get("creditos"),
            doc.get("pre_requisitos"),
            doc.get("chunk_index", 0),
            doc.get("conteudo_chunk"),
            json.dumps(doc.get("metadata_json") or {}, ensure_ascii=False),
            vetor,
        )
        for doc, vetor in zip(lote_docs, lote_vetores)
    ]

    with conn.cursor() as cur:
        cur.executemany(sql_insert, parametros)
    conn.commit()


def executar_ingestao(
    institutos_filtro: list[str] | None = None,
    batch_size: int = 32,
) -> None:
    """Orquestra a leitura dos chunks, geração de embeddings em lote e INSERT no banco."""
    print("=" * 70, flush=True)
    print("PIPELINE DE INGESTÃO NO POSTGRESQL + PGVECTOR (PASSO 4.3)", flush=True)
    print("=" * 70, flush=True)

    # Passo 1: Conectar ao PostgreSQL ANTES de importar/carregar o modelo pesado!
    conn = obter_conexao_db()

    log_debug("[2/4] Consultando checkpoint de chunks já gravados no banco...")
    ja_inseridos = buscar_chunks_ja_inseridos(conn)
    log_debug(f"[2/4] {len(ja_inseridos)} chunks já existem na tabela `documentos_unicamp`.")

    pendentes = carregar_chunks_pendentes(PROCESSED_CHUNKS, ja_inseridos, institutos_filtro)
    total_pendentes = len(pendentes)

    if total_pendentes == 0:
        log_debug("[Concluído] Todos os chunks solicitados já estão gravados no PostgreSQL!")
        conn.close()
        return

    alvo_str = ", ".join(institutos_filtro) if institutos_filtro else "TODOS OS 26 INSTITUTOS"
    log_debug(f"[3/4] {total_pendentes} chunks na fila para ingestão (Alvo: {alvo_str}).")

    # Importa o PyTorch/SentenceTransformers apenas após validar o banco e a fila
    log_debug("[3/4] Importando PyTorch/SentenceTransformers e carregando BAAI/bge-m3...")
    from src.embeddings.embedder import UnicampEmbedder

    embedder = UnicampEmbedder()

    log_debug(f"[4/4] Iniciando geração de embeddings e INSERT no banco (batch_size={batch_size})...")
    inicio = time.time()
    inseridos_agora = 0
    total_lotes = (total_pendentes + batch_size - 1) // batch_size

    for num_lote, i in enumerate(range(0, total_pendentes, batch_size), start=1):
        t_lote = time.time()
        lote_docs = pendentes[i : i + batch_size]
        textos_lote = [d["conteudo_chunk"] for d in lote_docs]

        # 1. Gera os vetores de 1024 dimensões na GPU para o lote atual
        vetores_lote = embedder.gerar_embeddings(
            textos_lote,
            batch_size=batch_size,
            mostrar_progresso=False,
        )
        dt_emb = time.time() - t_lote

        # 2. Faz o INSERT em lote no PostgreSQL
        t_sql = time.time()
        inserir_lote_no_postgres(conn, lote_docs, vetores_lote)
        dt_sql = time.time() - t_sql

        inseridos_agora += len(lote_docs)

        # Exibe log detalhado a cada lote (quando <= 15 lotes) ou a cada 5 lotes
        if total_lotes <= 15 or num_lote % 5 == 0 or inseridos_agora == total_pendentes:
            decorrido = time.time() - inicio
            vel = inseridos_agora / decorrido if decorrido > 0 else 0
            log_debug(
                f"  -> Lote {num_lote}/{total_lotes} | "
                f"Acumulado: {inseridos_agora}/{total_pendentes} ({inseridos_agora * 100 // total_pendentes}%) | "
                f"GPU: {dt_emb:.2f}s | SQL: {dt_sql:.2f}s | Média: {vel:.1f} chunks/s"
            )

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documentos_unicamp;")
        total_no_banco = cur.fetchone()[0]

    conn.close()
    tempo_total = time.time() - inicio
    print("=" * 70, flush=True)
    print(f"INGESTÃO CONCLUÍDA EM {tempo_total:.1f}s!", flush=True)
    print(f"Novos chunks inseridos : {inseridos_agora}", flush=True)
    print(f"Total na tabela SQL    : {total_no_banco} linhas com vetor(1024)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão de chunks + embeddings no PostgreSQL")
    parser.add_argument(
        "--institutos",
        nargs="+",
        default=None,
        help="Opcional: filtrar institutos (ex: --institutos IMECC). Se omitido, ingere todos.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote de geração de embeddings e INSERT no banco (padrão: 32).",
    )
    args = parser.parse_args()

    executar_ingestao(institutos_filtro=args.institutos, batch_size=args.batch_size)
