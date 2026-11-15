"""
Web Scraper Educado (Polite Scraper) com Checkpoint — Caderno de Horários DAC UNICAMP (2026)
===========================================================================================

Objetivo:
Coletar as disciplinas de Graduação (1S e 2S de 2026) diretamente do portal
da DAC UNICAMP reutilizando conexões TCP (HTTP Keep-Alive) e respeitando um
intervalo entre requisições (Rate Limiting) para evitar bloqueios temporários
no firewall da universidade.

Como executar (dentro da pasta 01-rag-postgresql-unicamp com .venv ativo):
  - Apenas IMECC (modo seguro padrão: 2 workers, 1.2s de pausa):
      python src/ingestion/dac_scraper.py --institutos IMECC
  - Modo sequencial ultra-seguro (1 única conexão TCP persistente):
      python src/ingestion/dac_scraper.py --institutos IMECC --workers 1 --delay 1.5
  - Vários institutos específicos:
      python src/ingestion/dac_scraper.py --institutos IMECC IC FEEC CEL
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import urllib.parse
import urllib3
from bs4 import BeautifulSoup

# O servidor www.dac.unicamp.br não envia o certificado intermediário completo
# na cadeia SSL. Desativamos o warning de InsecureRequestWarning para manter o log limpo.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.dac.unicamp.br/portal/caderno-de-horarios"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "disciplinas_2026.jsonl"

# Lock global para escrita segura no arquivo .jsonl
FILE_LOCK = threading.Lock()

# Armazenamento local por thread: garante que cada Worker reutilize 1 única
# conexão TCP (Keep-Alive) em vez de abrir centenas de conexões novas no firewall!
THREAD_LOCAL = threading.local()


def obter_sessao_da_thread() -> requests.Session:
    """
    Retorna a sessão HTTP persistente da thread atual.
    Reutilizar a mesma Session evita o bloqueio por 'SYN flood' no firewall da DAC.
    """
    if not hasattr(THREAD_LOCAL, "session"):
        session = requests.Session()
        session.verify = False
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
                "Connection": "keep-alive",
            }
        )
        THREAD_LOCAL.session = session
    return THREAD_LOCAL.session


def carregar_checkpoint(caminho_jsonl: Path) -> set[str]:
    """
    Lê o arquivo .jsonl existente e retorna um conjunto (set) com as chaves
    únicas ('2026_1S_IMECC_MA111') que já foram coletadas anteriormente.
    """
    chaves_coletadas: set[str] = set()
    if not caminho_jsonl.exists():
        return chaves_coletadas

    with caminho_jsonl.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                dado = json.loads(linha)
                if "chave" in dado:
                    chaves_coletadas.add(dado["chave"])
            except json.JSONDecodeError:
                continue
    return chaves_coletadas


def salvar_no_checkpoint(caminho_jsonl: Path, registro: dict[str, Any]) -> None:
    """Grava uma única disciplina no final do arquivo .jsonl de forma thread-safe."""
    caminho_jsonl.parent.mkdir(parents=True, exist_ok=True)
    linha_json = json.dumps(registro, ensure_ascii=False)
    with FILE_LOCK:
        with caminho_jsonl.open("a", encoding="utf-8") as f:
            f.write(linha_json + "\n")
            f.flush()


def listar_institutos(session: requests.Session, ano: int, semestre_num: int) -> list[str]:
    """Acessa a página do semestre e extrai a sigla de todos os institutos."""
    url = f"{BASE_URL}/{ano}/{semestre_num}/S/G"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()

    padrao = rf"caderno-de-horarios/{ano}/{semestre_num}/S/G/([A-Z0-9]+)"
    return sorted(set(re.findall(padrao, resp.text)))


def listar_disciplinas_do_instituto(
    session: requests.Session, ano: int, semestre_num: int, instituto: str
) -> list[str]:
    """
    Acessa a página de um instituto e retorna a lista de códigos de disciplinas.
    Suporta códigos com espaço codificado na URL (%20), como 'F%20128' -> 'F 128' no IFGW.
    """
    url = f"{BASE_URL}/{ano}/{semestre_num}/S/G/{instituto}"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()

    padrao = rf"caderno-de-horarios/{ano}/{semestre_num}/S/G/{instituto}/([A-Z0-9%]+)"
    brutos = set(re.findall(padrao, resp.text))
    codigos = sorted(
        {urllib.parse.unquote(c).strip() for c in brutos if urllib.parse.unquote(c).strip()}
    )
    return codigos


def extrair_dados_disciplina(
    html: str,
    ano: int,
    semestre_num: int,
    instituto: str,
    codigo_disciplina: str,
    url: str,
) -> dict[str, Any]:
    """Faz o parsing do HTML da página da disciplina usando BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    semestre_str = f"{semestre_num}S"
    chave = f"{ano}_{semestre_str}_{instituto}_{codigo_disciplina}"

    # 1. Extrair Título (Nome da Disciplina)
    titulo = codigo_disciplina
    tag_title = soup.find("title")
    if tag_title:
        texto_title = re.sub(r"\s+", " ", tag_title.get_text(strip=True))
        match_titulo = re.search(
            rf"{re.escape(codigo_disciplina)}\s*-\s*(.+?)\s*-\s*{semestre_num}S/{ano}",
            texto_title,
        )
        if match_titulo:
            titulo = match_titulo.group(1).strip()

    # 2. Localizar o bloco principal <div class="row disciplina">
    div_disciplina = soup.find("div", class_="disciplina")

    ementa = ""
    ano_catalogo = ano
    creditos: int | None = None
    pre_requisitos = "Nenhum"
    continencia = None
    equivalencia = None

    if div_disciplina:
        colunas = div_disciplina.find_all("div", class_="col-md-6", recursive=False)

        if len(colunas) >= 1:
            texto_col_esq = colunas[0].get_text(separator=" ", strip=True)
            ementa = re.sub(r"^Ementa:\s*", "", texto_col_esq, flags=re.IGNORECASE).strip()
            ementa = re.sub(r"\s+", " ", ementa)

        if len(colunas) >= 2:
            col_dir = colunas[1]

            for b_tag in col_dir.find_all("b", class_="prop"):
                label_texto = b_tag.get_text(strip=True).lower()
                parent_p = b_tag.find_parent("p")
                if not parent_p:
                    continue

                if "ano de catálogo" in label_texto:
                    span = parent_p.find("span")
                    if span and span.get_text(strip=True).isdigit():
                        ano_catalogo = int(span.get_text(strip=True))

                elif "créditos" in label_texto:
                    span = parent_p.find("span")
                    if span and span.get_text(strip=True).isdigit():
                        creditos = int(span.get_text(strip=True))

                elif "continência" in label_texto:
                    mark = parent_p.find("mark")
                    if mark:
                        continencia = re.sub(r"\s+", " ", mark.get_text(strip=True))

                elif "equivalência" in label_texto:
                    mark = parent_p.find("mark")
                    if mark:
                        equivalencia = re.sub(r"\s+", " ", mark.get_text(strip=True))

            ul_prereq = col_dir.find("ul", class_="prerequisitos")
            if ul_prereq:
                marks = [
                    re.sub(r"\s+", " ", m.get_text(strip=True))
                    for m in ul_prereq.find_all("mark")
                    if m.get_text(strip=True)
                ]
                if marks:
                    pre_requisitos = " | ".join(dict.fromkeys(marks))

    return {
        "chave": chave,
        "tipo_documento": "disciplina",
        "ano": ano,
        "semestre": semestre_str,
        "instituto": instituto,
        "codigo_disciplina": codigo_disciplina,
        "titulo": titulo,
        "creditos": creditos,
        "pre_requisitos": pre_requisitos,
        "ementa": ementa,
        "metadata_json": {
            "ano_catalogo": ano_catalogo,
            "nivel": "Graduação/Tecnologia",
            "continencia": continencia,
            "equivalencia": equivalencia,
            "url_origem": url,
        },
    }


def baixar_uma_disciplina(
    ano: int,
    semestre_num: int,
    instituto: str,
    codigo_disciplina: str,
    caminho_jsonl: Path,
    delay: float = 1.2,
) -> tuple[str, bool, str]:
    """
    Reutiliza a sessão HTTP da thread atual (Keep-Alive) e aplica um intervalo
    educado (delay + jitter) para não acionar o firewall da UNICAMP.
    """
    # Pausa educada com leve variação aleatória (jitter)
    if delay > 0:
        time.sleep(delay + random.uniform(0.1, 0.4))

    codigo_url = urllib.parse.quote(codigo_disciplina)
    url = f"{BASE_URL}/{ano}/{semestre_num}/S/G/{instituto}/{codigo_url}"
    session = obter_sessao_da_thread()
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        registro = extrair_dados_disciplina(
            html=resp.text,
            ano=ano,
            semestre_num=semestre_num,
            instituto=instituto,
            codigo_disciplina=codigo_disciplina,
            url=url,
        )
        salvar_no_checkpoint(caminho_jsonl, registro)
        return (registro["chave"], True, registro["titulo"])
    except Exception as exc:
        return (f"{ano}_{semestre_num}S_{instituto}_{codigo_disciplina}", False, str(exc))


def executar_scraper(
    ano: int = 2026,
    semestres: list[int] | None = None,
    institutos_filtro: list[str] | None = None,
    max_workers: int = 2,
    delay: float = 1.2,
    caminho_jsonl: Path = DEFAULT_OUTPUT,
) -> None:
    """Orquestra a descoberta de disciplinas, filtragem por checkpoint e download controlado."""
    if semestres is None:
        semestres = [1, 2]

    chaves_ja_salvas = carregar_checkpoint(caminho_jsonl)
    print(f"[Checkpoint] {len(chaves_ja_salvas)} disciplinas já encontradas em: {caminho_jsonl.name}")

    session_principal = obter_sessao_da_thread()
    tarefas_pendentes: list[tuple[int, int, str, str]] = []

    for sem in semestres:
        if institutos_filtro:
            institutos = [i.upper() for i in institutos_filtro]
        else:
            print(f"[Descoberta] Listando todos os institutos do {sem}S/{ano}...")
            institutos = listar_institutos(session_principal, ano, sem)
            print(f"[Descoberta] {len(institutos)} institutos encontrados no {sem}S/{ano}.")

        for inst in institutos:
            try:
                time.sleep(0.5)
                codigos = listar_disciplinas_do_instituto(session_principal, ano, sem, inst)
            except Exception as e:
                print(
                    f"[Aviso] O firewall da DAC ainda está em cooldown ou indisponível para {inst} ({sem}S/{ano}): {e}"
                )
                continue

            novas_no_inst = 0
            for cod in codigos:
                chave = f"{ano}_{sem}S_{inst}_{cod}"
                if chave not in chaves_ja_salvas:
                    tarefas_pendentes.append((ano, sem, inst, cod))
                    novas_no_inst += 1

            print(
                f"  -> {inst} ({sem}S/{ano}): {len(codigos)} disciplinas no total "
                f"| {novas_no_inst} pendentes para baixar."
            )

    total_pendentes = len(tarefas_pendentes)
    if total_pendentes == 0:
        print("\n[Concluído] Nenhuma disciplina pendente encontrada (ou servidor em cooldown).")
        return

    print(
        f"\n[Iniciando Coleta Segura] Baixando {total_pendentes} disciplinas "
        f"({max_workers} conexões persistentes Keep-Alive | pausa={delay}s)..."
    )
    inicio = time.time()
    sucessos = 0
    erros_consecutivos = 0
    erros = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(
                baixar_uma_disciplina, ano_t, sem_t, inst_t, cod_t, caminho_jsonl, delay
            ): (inst_t, cod_t)
            for (ano_t, sem_t, inst_t, cod_t) in tarefas_pendentes
        }

        for idx, futuro in enumerate(as_completed(futuros), start=1):
            chave, ok, detalhe = futuro.result()
            if ok:
                sucessos += 1
                erros_consecutivos = 0
            else:
                erros += 1
                erros_consecutivos += 1
                print(f"  [ERRO] {chave}: {detalhe}")
                if erros_consecutivos >= 5:
                    print(
                        "\n[Pausa de Segurança] 5 timeouts seguidos detectados (Rate Limit do firewall da DAC). "
                        "O progresso até aqui já está salvo no checkpoint!"
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

            if idx % 10 == 0 or idx == total_pendentes:
                decorrido = time.time() - inicio
                taxa = idx / decorrido if decorrido > 0 else 0
                print(
                    f"  [Progresso] {idx}/{total_pendentes} ({idx * 100 // total_pendentes}%) "
                    f"| Sucessos: {sucessos} | Erros: {erros} | Velocidade: {taxa:.2f} disc/s"
                )

    tempo_total = time.time() - inicio
    print(
        f"\n[Finalizado em {tempo_total:.1f}s] "
        f"Novas salvas nesta rodada: {sucessos} | Total acumulado no arquivo: {len(chaves_ja_salvas) + sucessos}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Web Scraper Educado do Caderno de Horários DAC UNICAMP (1S e 2S de 2026)"
    )
    parser.add_argument(
        "--institutos",
        nargs="+",
        default=["IMECC"],
        help="Lista de siglas de institutos (ex: --institutos IMECC IC). Padrão: IMECC",
    )
    parser.add_argument(
        "--todos",
        action="store_true",
        help="Se informado, ignora --institutos e coleta TODOS os 26 institutos da UNICAMP.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Número de conexões simultâneas persistentes (padrão seguro: 2).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.2,
        help="Segundos de espera entre requisições de cada worker (padrão: 1.2s).",
    )
    args = parser.parse_args()

    filtro = None if args.todos else args.institutos
    executar_scraper(
        ano=2026,
        semestres=[1, 2],
        institutos_filtro=filtro,
        max_workers=args.workers,
        delay=args.delay,
    )
