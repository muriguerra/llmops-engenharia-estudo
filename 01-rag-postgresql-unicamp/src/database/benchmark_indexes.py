"""
Benchmark Prático de Índices Vetoriais no PostgreSQL + pgvector (Passo 5.2)
===========================================================================

Objetivo:
Comparar na prática, sobre os 5.217 vetores de 1024 dimensões da UNICAMP,
o Plano de Execução (`EXPLAIN ANALYZE`), a latência de busca (ms) e a
precisão (Recall@10) em 3 cenários:
  1. SEM Índice Vetorial (Exact KNN -> Sequential Scan)
  2. COM Índice IVFFlat  (Clusterização k-means -> `lists = 100`, `probes = 5`)
  3. COM Índice HNSW     (Grafo Multicamada -> `m = 16`, `ef_construction = 64`)

Ao final, o script mantém o índice HNSW ativo na tabela `documentos_unicamp`
para ser utilizado nas consultas RAG da Fase 6.

Como executar (dentro de 01-rag-postgresql-unicamp com .venv ativo):
    python src/database/benchmark_indexes.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.ingest import obter_conexao_db  # noqa: E402


def obter_vetor_referencia(conn) -> tuple[str, list[float]]:
    """
    Para que o benchmark meça puramente a velocidade do PostgreSQL (sem precisar
    esperar o carregamento do PyTorch), pegamos o vetor de 1024 dimensões de uma
    disciplina já gravada no banco (ex: MA111 - Cálculo I) como vetor de busca!
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT codigo_disciplina || ' - ' || titulo, embedding
            FROM documentos_unicamp
            WHERE codigo_disciplina = 'MA111' AND semestre = '1S'
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                "SELECT codigo_disciplina || ' - ' || titulo, embedding FROM documentos_unicamp LIMIT 1;"
            )
            row = cur.fetchone()
        return row[0], row[1]


def extrair_tempo_execucao(linhas_explain: list[str]) -> float:
    """Extrai o valor numérico de 'Execution Time: X.XXX ms' do EXPLAIN ANALYZE."""
    for linha in reversed(linhas_explain):
        m = re.search(r"Execution Time:\s*([\d\.]+)\s*ms", linha)
        if m:
            return float(m.group(1))
    return -1.0


def rodar_explain_e_topk(
    conn,
    vetor_query,
    titulo_cenario: str,
    comandos_sessao: list[str] | None = None,
    k: int = 10,
) -> tuple[float, list[int]]:
    """
    Executa EXPLAIN (ANALYZE, BUFFERS) para medir o tempo em ms e retorna também
    os IDs dos Top-K documentos encontrados para calcularmos o Recall (precisão).
    """
    sql_query = f"""
        SELECT id, instituto, codigo_disciplina, titulo, 1 - (embedding <=> %s::vector) AS sim
        FROM documentos_unicamp
        ORDER BY embedding <=> %s::vector
        LIMIT {k};
    """
    sql_explain = f"EXPLAIN (ANALYZE, BUFFERS) {sql_query}"

    with conn.cursor() as cur:
        if comandos_sessao:
            for cmd in comandos_sessao:
                cur.execute(cmd)

        # 1. Busca os IDs do Top-K para cálculo de Recall@K
        cur.execute(sql_query, (vetor_query, vetor_query))
        ids_topk = [row[0] for row in cur.fetchall()]

        # 2. Aquecimento de cache (warm-up) e medição oficial do EXPLAIN ANALYZE
        cur.execute(sql_explain, (vetor_query, vetor_query))
        _ = cur.fetchall()

        cur.execute(sql_explain, (vetor_query, vetor_query))
        plano = [row[0] for row in cur.fetchall()]

    tempo_ms = extrair_tempo_execucao(plano)
    print(f"\n--- {titulo_cenario} ---")
    for linha in plano:
        print(f"  {linha}")
    return tempo_ms, ids_topk


def calcular_recall(ids_exatos: list[int], ids_aproximados: list[int]) -> float:
    """Calcula a porcentagem dos Top-K exatos que o índice ANN conseguiu encontrar."""
    if not ids_exatos:
        return 100.0
    acertos = len(set(ids_exatos) & set(ids_aproximados))
    return (acertos / len(ids_exatos)) * 100.0


def executar_benchmark() -> None:
    print("=" * 82)
    print("BENCHMARK DE ÍNDICES VETORIAIS: SEQ SCAN vs IVFFLAT vs HNSW (PASSO 5.2)")
    print("=" * 82)

    conn = obter_conexao_db()
    nome_ref, vetor_ref = obter_vetor_referencia(conn)
    print(f"[Referência de Busca] Usando o vetor(1024) de: '{nome_ref}'")

    # Atualiza estatísticas da tabela no planejador do PostgreSQL
    with conn.cursor() as cur:
        cur.execute("ANALYZE documentos_unicamp;")
    conn.commit()

    # -------------------------------------------------------------------------
    # 1. CENÁRIO SEM ÍNDICE VETORIAL (Exact KNN / Sequential Scan)
    # -------------------------------------------------------------------------
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_documentos_embedding_hnsw;")
        cur.execute("DROP INDEX IF EXISTS idx_documentos_embedding_ivfflat;")
    conn.commit()

    tempo_seq, ids_seq = rodar_explain_e_topk(
        conn,
        vetor_ref,
        titulo_cenario="CENÁRIO 1: SEM ÍNDICE VETORIAL (Sequential Scan Exato em 5.217 vetores)",
        comandos_sessao=[
            "SET enable_seqscan = on;",
            "SET max_parallel_workers_per_gather = 0;",
        ],
    )

    # -------------------------------------------------------------------------
    # 2. CENÁRIO COM ÍNDICE IVFFLAT (lists = 100, probes = 5)
    # -------------------------------------------------------------------------
    print("\n[Construindo Índice IVFFlat] lists = 100 (agrupamento k-means)...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX idx_documentos_embedding_ivfflat
            ON documentos_unicamp
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
        cur.execute("SELECT pg_size_pretty(pg_relation_size('idx_documentos_embedding_ivfflat'));")
        tam_ivfflat = cur.fetchone()[0]
    conn.commit()
    tempo_build_ivf = time.time() - t0
    print(f"  -> Índice IVFFlat criado em {tempo_build_ivf:.2f}s (Tamanho em disco: {tam_ivfflat})")

    tempo_ivf, ids_ivf = rodar_explain_e_topk(
        conn,
        vetor_ref,
        titulo_cenario="CENÁRIO 2: COM ÍNDICE IVFFLAT (lists=100, ivfflat.probes=5)",
        comandos_sessao=[
            "SET enable_seqscan = off;",
            "SET ivfflat.probes = 5;",
        ],
    )
    recall_ivf = calcular_recall(ids_seq, ids_ivf)

    # Removemos o IVFFlat antes de criar o HNSW para não manter 2 índices vetoriais redundantes
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_documentos_embedding_ivfflat;")
    conn.commit()

    # -------------------------------------------------------------------------
    # 3. CENÁRIO COM ÍNDICE HNSW (m = 16, ef_construction = 64)
    # -------------------------------------------------------------------------
    print("\n[Construindo Índice HNSW] m = 16, ef_construction = 64 (Grafo Multicamada)...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX idx_documentos_embedding_hnsw
            ON documentos_unicamp
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
        )
        cur.execute("SELECT pg_size_pretty(pg_relation_size('idx_documentos_embedding_hnsw'));")
        tam_hnsw = cur.fetchone()[0]
    conn.commit()
    tempo_build_hnsw = time.time() - t0
    print(f"  -> Índice HNSW criado em {tempo_build_hnsw:.2f}s (Tamanho em disco: {tam_hnsw})")

    tempo_hnsw, ids_hnsw = rodar_explain_e_topk(
        conn,
        vetor_ref,
        titulo_cenario="CENÁRIO 3: COM ÍNDICE HNSW (m=16, ef_construction=64, hnsw.ef_search=40)",
        comandos_sessao=[
            "SET enable_seqscan = off;",
            "SET hnsw.ef_search = 40;",
        ],
    )
    recall_hnsw = calcular_recall(ids_seq, ids_hnsw)

    # -------------------------------------------------------------------------
    # 4. CENÁRIO BÔNUS: BUSCA HÍBRIDA (WHERE instituto = 'IMECC' + HNSW Iterative Scan)
    # -------------------------------------------------------------------------
    print("\n--- CENÁRIO 4: BUSCA HÍBRIDA (WHERE instituto = 'IMECC' + HNSW Iterative Scan) ---")
    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off;")
        cur.execute("SET enable_bitmapscan = off;")
        cur.execute("SET hnsw.iterative_scan = relaxed_order;")
        cur.execute(
            """
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT id, instituto, codigo_disciplina, titulo
            FROM documentos_unicamp
            WHERE instituto = 'IMECC'
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
            """,
            (vetor_ref,),
        )
        plano_hibrido = [r[0] for r in cur.fetchall()]
    for linha in plano_hibrido:
        print(f"  {linha}")

    conn.close()

    # -------------------------------------------------------------------------
    # RESUMO COMPARATIVO FINAL
    # -------------------------------------------------------------------------
    ganho_hnsw = (tempo_seq / tempo_hnsw) if tempo_hnsw > 0 else 0
    print("\n" + "=" * 82)
    print("RESUMO COMPARATIVO DE ENGENHARIA (5.217 VETORES DE 1024 DIMENSÕES)")
    print("=" * 82)
    print(
        f"{'Estratégia':<28} | {'Build Time':<10} | {'Tamanho':<8} | {'Busca (ms)':<12} | {'Recall@10':<10}"
    )
    print("-" * 82)
    print(
        f"{'1. Sem Índice (Seq Scan)':<28} | {'0.00s':<10} | {'0 MB':<8} | {tempo_seq:>8.3f} ms | {'100.0%':<10}"
    )
    print(
        f"{'2. Índice IVFFlat (lists=100)':<28} | {f'{tempo_build_ivf:.2f}s':<10} | {tam_ivfflat:<8} | {tempo_ivf:>8.3f} ms | {f'{recall_ivf:.1f}%':<10}"
    )
    print(
        f"{'3. Índice HNSW (m=16, ef=64)':<28} | {f'{tempo_build_hnsw:.2f}s':<10} | {tam_hnsw:<8} | {tempo_hnsw:>8.3f} ms | {f'{recall_hnsw:.1f}%':<10}"
    )
    print("-" * 82)
    print(f"Aceleração Real do HNSW sobre o Sequential Scan: ~{ganho_hnsw:.1f}x mais rápido na busca!")
    print("Nota: O índice `idx_documentos_embedding_hnsw` foi mantido ativo no PostgreSQL.")
    print("=" * 82)


if __name__ == "__main__":
    executar_benchmark()
