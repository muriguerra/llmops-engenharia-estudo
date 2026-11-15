"""
Pipeline de Chunking e Enriquecimento Contextual (Passo 3.3)
============================================================

Objetivo:
Ler as disciplinas brutas coletadas pelo scraper (data/raw/disciplinas_2026.jsonl),
aplicar o RecursiveCharacterTextSplitter do LangChain sobre as ementas (caso alguma
ementa ou resolução ultrapasse o limite de caracteres do bloco) e injetar um
Cabeçalho Contextual (Contextual Chunk Header) em cada fragmento antes de
salvar em `data/processed/chunks_2026.jsonl`.

Como executar (dentro de 01-rag-postgresql-unicamp com .venv ativo):
    python src/ingestion/chunker.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_INPUT = ROOT_DIR / "data" / "raw" / "disciplinas_2026.jsonl"
PROCESSED_OUTPUT = ROOT_DIR / "data" / "processed" / "chunks_2026.jsonl"


def criar_splitter(chunk_size: int = 600, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    """
    Configura o divisor recursivo do LangChain.
    Tenta quebrar primeiro por parágrafos (\\n\\n), depois por quebras de linha (\\n),
    depois por ponto final ('. '), preservando a semântica das frases.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )


def montar_cabecalho_contextual(registro: dict[str, Any], chunk_idx: int, total_chunks: int) -> str:
    """
    Cria o cabeçalho rico que será anexado ao texto do chunk antes de gerar o embedding.
    Isso permite que o vetor BAAI/bge-m3 codifique tanto os metadados da disciplina
    quanto o conteúdo programático da ementa.
    """
    instituto = registro.get("instituto", "N/D")
    codigo = registro.get("codigo_disciplina", "N/D")
    titulo = registro.get("titulo", "Sem título")
    semestre = registro.get("semestre", "")
    ano = registro.get("ano", 2026)
    creditos = registro.get("creditos")
    creditos_str = str(creditos) if creditos is not None else "N/D"
    pre_req = registro.get("pre_requisitos") or "Nenhum"

    parte_info = f" (Parte {chunk_idx + 1}/{total_chunks})" if total_chunks > 1 else ""

    return (
        f"[Instituto: {instituto} | Disciplina: {codigo} - {titulo}{parte_info} | "
        f"Período: {semestre}/{ano} | Créditos: {creditos_str} | Pré-requisitos: {pre_req}]"
    )


def processar_chunks(
    caminho_entrada: Path = RAW_INPUT,
    caminho_saida: Path = PROCESSED_OUTPUT,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> None:
    """Lê o JSONL bruto, divide textos longos em chunks e salva o JSONL processado."""
    if not caminho_entrada.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {caminho_entrada}")

    splitter = criar_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    total_disciplinas = 0
    disciplinas_sem_ementa = 0
    total_chunks_gerados = 0
    disciplinas_multi_chunk = 0
    contagem_por_instituto: Counter[str] = Counter()

    with (
        caminho_entrada.open("r", encoding="utf-8") as f_in,
        caminho_saida.open("w", encoding="utf-8") as f_out,
    ):
        for linha in f_in:
            linha = linha.strip()
            if not linha:
                continue

            registro = json.loads(linha)
            total_disciplinas += 1

            ementa_bruta = (registro.get("ementa") or "").strip()
            if not ementa_bruta:
                disciplinas_sem_ementa += 1
                ementa_bruta = "Ementa não cadastrada no sistema da DAC."

            # Divide a ementa caso ultrapasse chunk_size caracteres
            pedacos_texto = splitter.split_text(ementa_bruta)
            if not pedacos_texto:
                pedacos_texto = [ementa_bruta]

            if len(pedacos_texto) > 1:
                disciplinas_multi_chunk += 1

            for idx, pedaco in enumerate(pedacos_texto):
                cabecalho = montar_cabecalho_contextual(registro, idx, len(pedacos_texto))
                conteudo_chunk = f"{cabecalho}\nEmenta: {pedaco}"

                metadata_extra = dict(registro.get("metadata_json") or {})
                metadata_extra["total_chunks"] = len(pedacos_texto)
                metadata_extra["chave_origem"] = registro.get("chave")

                chunk_doc = {
                    "tipo_documento": registro.get("tipo_documento", "disciplina"),
                    "ano": registro.get("ano", 2026),
                    "semestre": registro.get("semestre"),
                    "instituto": registro.get("instituto"),
                    "codigo_disciplina": registro.get("codigo_disciplina"),
                    "titulo": registro.get("titulo"),
                    "creditos": registro.get("creditos"),
                    "pre_requisitos": registro.get("pre_requisitos"),
                    "chunk_index": idx,
                    "conteudo_chunk": conteudo_chunk,
                    "metadata_json": metadata_extra,
                }

                f_out.write(json.dumps(chunk_doc, ensure_ascii=False) + "\n")
                total_chunks_gerados += 1
                contagem_por_instituto[registro.get("instituto", "N/D")] += 1

    print("=" * 70)
    print("RESUMO DO PROCESSAMENTO DE CHUNKS (PASSO 3.3)")
    print("=" * 70)
    print(f"Disciplinas lidas (raw)          : {total_disciplinas}")
    print(f"Disciplinas com +1 chunk (>600c) : {disciplinas_multi_chunk}")
    print(f"Disciplinas sem ementa explícita : {disciplinas_sem_ementa}")
    print(f"Total de Chunks gerados          : {total_chunks_gerados}")
    print(f"Arquivo salvo em                 : {caminho_saida}")
    print("-" * 70)
    print("Top 10 Institutos com mais Chunks:")
    for inst, qtd in contagem_por_instituto.most_common(10):
        print(f"  - {inst:<8}: {qtd} chunks")
    print("=" * 70)


if __name__ == "__main__":
    processar_chunks()
