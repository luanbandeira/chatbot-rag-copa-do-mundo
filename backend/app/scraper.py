"""
Scraper de conteúdo da Wikipédia (PT) sobre a Copa do Mundo FIFA.

Para cada termo de busca definido em SEARCH_TERMS, resolve o título
canônico da página via API de busca da Wikipédia (evita erro de nome
digitado errado ou redirecionamento), baixa o texto puro (sem markup
wiki, sem HTML) e salva em backend/data/raw/<slug>.txt

Tem retry automático com espera progressiva em caso de erro 429 (limite
de requisições) e retoma de onde parou se for interrompido ou se algum
termo falhar — não baixa de novo o que já deu certo.

Uso:
    python -m app.scraper
"""
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests

WIKI_API = "https://pt.wikipedia.org/w/api.php"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROGRESS_FILE = OUTPUT_DIR / ".scraper_progress.json"

DELAY_BETWEEN_REQUESTS = 1.5  # segundos entre chamadas, para não bater rate limit
MAX_RETRIES = 4

# Termos de busca — não precisam ser o título exato, a API resolve o mais
# próximo. Cobre história geral, edições individuais e curiosidades.
SEARCH_TERMS = [
    "Copa do Mundo FIFA",
    "Lista de campeões da Copa do Mundo FIFA",
    "Recordes da Copa do Mundo FIFA",
    "Copa do Mundo FIFA de 2026",
    "Mascote da Copa do Mundo FIFA",
    "Bola oficial da Copa do Mundo FIFA",
    "Copa do Mundo FIFA de 1930",
    "Copa do Mundo FIFA de 1934",
    "Copa do Mundo FIFA de 1938",
    "Copa do Mundo FIFA de 1950",
    "Copa do Mundo FIFA de 1954",
    "Copa do Mundo FIFA de 1958",
    "Copa do Mundo FIFA de 1962",
    "Copa do Mundo FIFA de 1966",
    "Copa do Mundo FIFA de 1970",
    "Copa do Mundo FIFA de 1974",
    "Copa do Mundo FIFA de 1978",
    "Copa do Mundo FIFA de 1982",
    "Copa do Mundo FIFA de 1986",
    "Copa do Mundo FIFA de 1990",
    "Copa do Mundo FIFA de 1994",
    "Copa do Mundo FIFA de 1998",
    "Copa do Mundo FIFA de 2002",
    "Copa do Mundo FIFA de 2006",
    "Copa do Mundo FIFA de 2010",
    "Copa do Mundo FIFA de 2014",
    "Copa do Mundo FIFA de 2018",
    "Copa do Mundo FIFA de 2022",
]

HEADERS = {
    # A Wikimedia pede um User-Agent identificável para chamadas via API.
    "User-Agent": "ChatbotRAGCopaDoMundo/1.0 (projeto academico UNICAP; uso educacional)"
}


def request_with_retry(params: dict) -> Optional[dict]:
    """Faz a chamada à API com retry e espera progressiva em caso de 429/5xx."""
    wait = 2
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", wait))
            print(f"  [429] Limite de requisições, aguardando {retry_after}s (tentativa {attempt}/{MAX_RETRIES})...")
            time.sleep(retry_after)
            wait *= 2
            continue

        if resp.status_code >= 500:
            print(f"  [{resp.status_code}] Erro do servidor, aguardando {wait}s (tentativa {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
            wait *= 2
            continue

        resp.raise_for_status()
        return resp.json()

    print("  [FALHOU] Excedeu o número de tentativas.")
    return None


def resolve_title(query: str) -> Optional[str]:
    """Usa a API de busca da Wikipédia para achar o título canônico da página."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    data = request_with_retry(params)
    if not data:
        return None
    results = data.get("query", {}).get("search", [])
    if not results:
        return None
    return results[0]["title"]


def fetch_extract(title: str) -> Optional[str]:
    """Baixa o texto puro (sem markup) da página pelo título."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    data = request_with_retry(params)
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return None


def slugify(text: str) -> str:
    """Transforma um título em um nome de arquivo seguro (sem acento/espaço)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()
    ok_count = 0
    failed = []

    for term in SEARCH_TERMS:
        if progress.get(term) == "ok":
            print(f"[JÁ FEITO] {term}")
            ok_count += 1
            continue

        title = resolve_title(term)
        if not title:
            print(f"[FALHOU] Não encontrei página para: {term}")
            failed.append(term)
            progress[term] = "falhou"
            save_progress(progress)
            continue

        text = fetch_extract(title)
        if not text or len(text.strip()) < 200:
            print(f"[FALHOU] Extract vazio/curto para: {title}")
            failed.append(term)
            progress[term] = "falhou"
            save_progress(progress)
            continue

        filename = OUTPUT_DIR / f"{slugify(title)}.txt"
        filename.write_text(text, encoding="utf-8")
        print(f"[OK] {title} -> {filename.name} ({len(text)} caracteres)")
        ok_count += 1
        progress[term] = "ok"
        save_progress(progress)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nConcluído: {ok_count} páginas salvas, {len(failed)} falharam.")
    if failed:
        print("Termos que falharam (rode o script de novo para tentar retomar):", failed)


if __name__ == "__main__":
    main()