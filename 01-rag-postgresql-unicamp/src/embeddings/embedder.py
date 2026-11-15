"""
Módulo de Geração de Embeddings Locais com BAAI/bge-m3 (Passo 4.2)
==================================================================

Objetivo:
Carregar o modelo multilíngue `BAAI/bge-m3` (1024 dimensões) localmente via
SentenceTransformers, detectando automaticamente se há uma GPU NVIDIA (CUDA)
disponível e ativando precisão mista (FP16 / float16) para caber com folga em
GPUs de 4GB+ de VRAM e acelerar a inferência nos Tensor Cores.

Como testar este módulo isoladamente (dentro de 01-rag-postgresql-unicamp):
    python src/embeddings/embedder.py
"""

from __future__ import annotations

import time
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
EXPECTED_DIM = 1024


class UnicampEmbedder:
    """Encapsula o modelo BAAI/bge-m3 com suporte automático a GPU (FP16) e normalização."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model_kwargs = {}
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(
                f"[DEBUG Embedder] GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM) "
                "-> Ativando FP16 (float16)",
                flush=True,
            )
            model_kwargs["torch_dtype"] = torch.float16
        else:
            print(
                f"[DEBUG Embedder] Rodando em CPU (PyTorch={torch.__version__}). "
                "Para usar a GPU NVIDIA, instale o PyTorch com suporte a CUDA.",
                flush=True,
            )

        print(
            f"[DEBUG Embedder] Inicializando SentenceTransformer('{model_name}') em '{self.device}'...",
            flush=True,
        )
        inicio = time.time()
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs=model_kwargs,
        )
        self.model.max_seq_length = 512
        print(
            f"[DEBUG Embedder] Modelo pronto na memória em {time.time() - inicio:.2f}s "
            f"(max_seq_length={self.model.max_seq_length})!",
            flush=True,
        )

    def gerar_embeddings(
        self,
        textos: Sequence[str],
        batch_size: int = 16,
        mostrar_progresso: bool = True,
    ) -> list[list[float]]:
        """
        Gera vetores normalizados de 1024 dimensões para uma lista de textos.
        `normalize_embeddings=True` garante vetores unitários (norma = 1.0),
        ideais para o operador de distância de cosseno (<=>) do pgvector.
        """
        vetores_np = self.model.encode(
            list(textos),
            batch_size=batch_size,
            show_progress_bar=mostrar_progresso,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vetores_np.tolist()

    def gerar_embedding_pergunta(self, pergunta: str) -> list[float]:
        """Converte uma única pergunta do usuário em um vetor de 1024 dimensões."""
        vetor = self.gerar_embeddings([pergunta], batch_size=1, mostrar_progresso=False)[0]
        if len(vetor) != EXPECTED_DIM:
            raise ValueError(f"Dimensão inesperada: {len(vetor)} (esperado: {EXPECTED_DIM})")
        return vetor


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("TESTE DO MÓDULO DE EMBEDDINGS (BAAI/bge-m3)", flush=True)
    print("=" * 70, flush=True)
    embedder = UnicampEmbedder()

    frase_teste = (
        "[Instituto: IMECC | Disciplina: MA111 - Cálculo I] "
        "Ementa: Intervalos e desigualdades. Funções. Limites. Derivada e integral."
    )
    t0 = time.time()
    vetor = embedder.gerar_embedding_pergunta(frase_teste)
    dt = (time.time() - t0) * 1000

    norma = sum(x * x for x in vetor) ** 0.5

    print("-" * 70, flush=True)
    print(f"Frase de teste     : {frase_teste[:65]}...", flush=True)
    print(f"Dimensão do vetor  : {len(vetor)} (Esperado: {EXPECTED_DIM})", flush=True)
    print(f"Norma L2 do vetor  : {norma:.6f} (1.0 = perfeitamente normalizado)", flush=True)
    print(f"Primeiros 5 valores: {[round(v, 5) for v in vetor[:5]]}", flush=True)
    print(f"Tempo de inferência: {dt:.1f} ms", flush=True)
    print("=" * 70, flush=True)
