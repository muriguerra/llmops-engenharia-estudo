"""
Motor de Busca Semântica Híbrida (SQL Relacional + pgvector) — Fase 6
=====================================================================

Objetivo:
Combinar filtros relacionais estritos do PostgreSQL (`WHERE instituto = 'IMECC'`,
`semestre = '1S'`, `creditos <= 4`, `pre_requisitos = 'Nenhum'`) com a ordenação
vetorial por distância de cosseno (`ORDER BY embedding <=> %s::vector`) usando
o índice HNSW com `iterative_scan` e o modelo local `BAAI/bge-m3`.

Como executar (dentro de 01-rag-postgresql-unicamp com .venv ativo):
  1. Rodar a bateria demonstrativa completa (Sem Filtro vs Com Filtro IMECC + Prompt RAG):
       python src/retrieval/search.py

  2. Fazer uma pergunta personalizada com filtros customizados:
       python src/retrieval/search.py --pergunta "inteligência artificial e redes neurais" --instituto IC --top-k 5
       python src/retrieval/search.py --pergunta "cálculo e derivadas" --instituto IMECC --semestre 1S --sem-pre-req
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.ingest import obter_conexao_db  # noqa: E402
from src.embeddings.embedder import UnicampEmbedder  # noqa: E402


@dataclass
class ResultadoBusca:
    """Representa um chunk recuperado do PostgreSQL com seu score de similaridade."""

    instituto: str
    semestre: str
    codigo_disciplina: str
    titulo: str
    creditos: int | None
    pre_requisitos: str
    conteudo_chunk: str
    similaridade_cosseno: float
    url_origem: str | None = None


def buscar_disciplinas_hibrido(
    conn,
    vetor_pergunta: list[float],
    instituto: str | None = None,
    semestre: str | None = None,
    max_creditos: int | None = None,
    sem_pre_requisitos: bool = False,
    top_k: int = 5,
) -> tuple[list[ResultadoBusca], float]:
    """
    Executa a Query SQL Híbrida no PostgreSQL:
      - Aplica os filtros relacionais exatos na cláusula WHERE;
      - Ordena pela menor distância de cosseno (`embedding <=> %s::vector`) usando HNSW;
      - Calcula `1 - (embedding <=> %s::vector)` como similaridade (0.0 a 1.0).
    """
    condicoes_where: list[str] = []
    parametros_where: list[Any] = []

    if instituto:
        condicoes_where.append("instituto = %s")
        parametros_where.append(instituto.upper())

    if semestre:
        condicoes_where.append("semestre = %s")
        parametros_where.append(semestre.upper())

    if max_creditos is not None:
        condicoes_where.append("creditos <= %s")
        parametros_where.append(max_creditos)

    if sem_pre_requisitos:
        condicoes_where.append("pre_requisitos = 'Nenhum'")

    clausula_where = ""
    if condicoes_where:
        clausula_where = "WHERE " + " AND ".join(condicoes_where)

    sql_query = f"""
        SELECT
            instituto,
            semestre,
            codigo_disciplina,
            titulo,
            creditos,
            pre_requisitos,
            conteudo_chunk,
            ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS similaridade_cosseno,
            metadata_json->>'url_origem' AS url_origem
        FROM documentos_unicamp
        {clausula_where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    # Ordem dos parâmetros: [vetor_select] + [filtros_where...] + [vetor_order_by, top_k]
    parametros_finais = [vetor_pergunta, *parametros_where, vetor_pergunta, top_k]

    t0 = time.time()
    with conn.cursor() as cur:
        # Ativa o Iterative Scan do pgvector 0.8+ no índice HNSW:
        # Se o filtro WHERE descartar candidatos do grafo, o HNSW continua navegando
        # automaticamente até completar os `top_k` resultados que obedecem ao WHERE!
        cur.execute("SET hnsw.iterative_scan = relaxed_order;")
        cur.execute(sql_query, parametros_finais)
        linhas = cur.fetchall()
    tempo_sql_ms = (time.time() - t0) * 1000

    resultados = [
        ResultadoBusca(
            instituto=r[0],
            semestre=r[1],
            codigo_disciplina=r[2],
            titulo=r[3],
            creditos=r[4],
            pre_requisitos=r[5],
            conteudo_chunk=r[6],
            similaridade_cosseno=float(r[7]),
            url_origem=r[8],
        )
        for r in linhas
    ]
    return resultados, tempo_sql_ms


def exibir_resultados(
    titulo_secao: str,
    pergunta: str,
    filtros_desc: str,
    resultados: list[ResultadoBusca],
    tempo_sql_ms: float,
) -> None:
    """Imprime uma tabela formatada e didática com o Top-K retornado pelo PostgreSQL."""
    print("\n" + "=" * 95)
    print(f" {titulo_secao}")
    print("=" * 95)
    print(f"  Pergunta : '{pergunta}'")
    print(f"  Filtros  : {filtros_desc} | Tempo SQL (pgvector): {tempo_sql_ms:.2f} ms")
    print("-" * 95)
    print(
        f"  {'#':<2} | {'Similaridade':<12} | {'Inst.':<6} | {'Sem.':<4} | {'Sigla':<7} | "
        f"{'Créd.':<5} | {'Título da Disciplina'}"
    )
    print("-" * 95)

    if not resultados:
        print("  Nenhum resultado encontrado para os filtros informados.")
        return

    for idx, res in enumerate(resultados, start=1):
        cred_str = str(res.creditos) if res.creditos is not None else "N/D"
        print(
            f"  {idx:<2} | {res.similaridade_cosseno:<12.4f} | {res.instituto:<6} | "
            f"{res.semestre:<4} | {res.codigo_disciplina:<7} | {cred_str:<5} | {res.titulo}"
        )
        # Extrai apenas a linha da Ementa (sem o cabeçalho) para exibir um resumo limpo
        partes = res.conteudo_chunk.split("\nEmenta: ", maxsplit=1)
        resumo_ementa = partes[1] if len(partes) > 1 else res.conteudo_chunk
        print(f"     -> Pré-req: {res.pre_requisitos[:65]}")
        print(f"     -> Ementa : {resumo_ementa[:115]}...")
    print("=" * 95)


def montar_prompt_rag(pergunta: str, resultados: list[ResultadoBusca]) -> str:
    """
    Passo 6.3: Usa o ChatPromptTemplate do LangChain para estruturar o contexto
    recuperado do PostgreSQL em um prompt pronto para qualquer LLM responder
    citando a sigla e o instituto de cada disciplina.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um orientador acadêmico especialista no catálogo oficial da UNICAMP.\n"
                "Responda à dúvida do estudante utilizando EXCLUSIVAMENTE os dados das disciplinas "
                "recuperadas do banco PostgreSQL abaixo. Sempre cite a sigla (ex: MA111), o instituto "
                "(ex: IMECC), o número de créditos e os pré-requisitos.\n\n"
                "--- CONTEXTO RECUPERADO DO POSTGRESQL (TOP-{k} CHUNKS) ---\n"
                "{contexto}\n"
                "----------------------------------------------------------",
            ),
            ("human", "Pergunta do estudante: {pergunta}"),
        ]
    )

    blocos_contexto = []
    for idx, r in enumerate(resultados, start=1):
        blocos_contexto.append(
            f"[Fonte {idx} | Similaridade={r.similaridade_cosseno:.4f}]\n{r.conteudo_chunk}"
        )

    mensagens = template.format_messages(
        k=len(resultados),
        contexto="\n\n".join(blocos_contexto),
        pergunta=pergunta,
    )
    return "\n\n".join(f"[{m.type.upper()}]\n{m.content}" for m in mensagens)


def executar_demonstracao_comparativa(embedder: UnicampEmbedder, conn) -> None:
    """
    Passo 6.2: Demonstra na prática a diferença entre buscar na base inteira (Sem Filtro)
    vs filtrar por `WHERE instituto = 'IMECC'` vs aplicar múltiplos filtros relacionais!
    """
    pergunta = (
        "Equações diferenciais ordinárias, transformada de Laplace, séries de Fourier "
        "e aplicações em modelagem matemática"
    )
    print(f"\n[Gerando Vetor da Pergunta (1024 dims)] '{pergunta}'...")
    vetor_q = embedder.gerar_embedding_pergunta(pergunta)

    # 1. Busca Global SEM FILTRO (todos os 26 institutos concorrendo)
    res_global, t_global = buscar_disciplinas_hibrido(conn, vetor_q, top_k=5)
    exibir_resultados(
        titulo_secao="CENÁRIO A: BUSCA SEMÂNTICA GLOBAL (SEM FILTRO WHERE — 26 INSTITUTOS)",
        pergunta=pergunta,
        filtros_desc="Nenhum (Busca nos 5.217 chunks da UNICAMP)",
        resultados=res_global,
        tempo_sql_ms=t_global,
    )

    # 2. Busca Híbrida COM FILTRO (WHERE instituto = 'IMECC')
    res_imecc, t_imecc = buscar_disciplinas_hibrido(
        conn, vetor_q, instituto="IMECC", top_k=5
    )
    exibir_resultados(
        titulo_secao="CENÁRIO B: BUSCA HÍBRIDA COM FILTRO (WHERE instituto = 'IMECC')",
        pergunta=pergunta,
        filtros_desc="WHERE instituto = 'IMECC'",
        resultados=res_imecc,
        tempo_sql_ms=t_imecc,
    )

    # 3. Busca Híbrida MÚLTIPLOS FILTROS (IMECC + 1S + sem pré-requisito + até 6 créditos)
    pergunta_iniciante = "Introdução ao cálculo diferencial e integral, limites, derivadas e vetores"
    vetor_ini = embedder.gerar_embedding_pergunta(pergunta_iniciante)
    res_composto, t_composto = buscar_disciplinas_hibrido(
        conn,
        vetor_ini,
        instituto="IMECC",
        semestre="1S",
        max_creditos=6,
        sem_pre_requisitos=True,
        top_k=5,
    )
    exibir_resultados(
        titulo_secao="CENÁRIO C: BUSCA HÍBRIDA MULTI-FILTRO (CALOURO 1S SEM PRÉ-REQUISITOS)",
        pergunta=pergunta_iniciante,
        filtros_desc="WHERE instituto = 'IMECC' AND semestre = '1S' AND creditos <= 6 AND pre_requisitos = 'Nenhum'",
        resultados=res_composto,
        tempo_sql_ms=t_composto,
    )

    # 4. Demonstração do Prompt RAG final estruturado pelo LangChain (Passo 6.3)
    prompt_final = montar_prompt_rag(pergunta, res_imecc[:3])
    print("\n" + "=" * 95)
    print(" PASSO 6.3: EXEMPLO DE PROMPT RAG MONTADO PELO LANGCHAIN (TOP-3 DO IMECC)")
    print("=" * 95)
    print(prompt_final)
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Motor de Busca Semântica Híbrida (SQL + pgvector) — UNICAMP 2026"
    )
    parser.add_argument(
        "--pergunta",
        type=str,
        default=None,
        help="Pergunta em linguagem natural. Se omitida, roda a demonstração comparativa completa.",
    )
    parser.add_argument("--instituto", type=str, default=None, help="Filtrar por instituto (ex: IMECC, IC, FEEC)")
    parser.add_argument("--semestre", type=str, default=None, help="Filtrar por semestre (1S ou 2S)")
    parser.add_argument("--max-creditos", type=int, default=None, help="Filtrar máximo de créditos (ex: 4)")
    parser.add_argument(
        "--sem-pre-req",
        action="store_true",
        help="Se informado, retorna apenas disciplinas sem pré-requisitos ('Nenhum').",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Quantidade de resultados (padrão: 5)")
    args = parser.parse_args()

    conexao = obter_conexao_db()
    modelo_emb = UnicampEmbedder()

    if args.pergunta:
        vetor_usuario = modelo_emb.gerar_embedding_pergunta(args.pergunta)
        resultados_usr, tempo_ms = buscar_disciplinas_hibrido(
            conexao,
            vetor_usuario,
            instituto=args.instituto,
            semestre=args.semestre,
            max_creditos=args.max_creditos,
            sem_pre_requisitos=args.sem_pre_req,
            top_k=args.top_k,
        )
        filtros_txt = (
            f"instituto={args.instituto or 'TODOS'}, semestre={args.semestre or 'TODOS'}, "
            f"max_creditos={args.max_creditos or 'TODOS'}, sem_pre_req={args.sem_pre_req}"
        )
        exibir_resultados(
            titulo_secao="RESULTADO DA SUA BUSCA HÍBRIDA PERSONALIZADA",
            pergunta=args.pergunta,
            filtros_desc=filtros_txt,
            resultados=resultados_usr,
            tempo_sql_ms=tempo_ms,
        )
    else:
        executar_demonstracao_comparativa(modelo_emb, conexao)

    conexao.close()
