import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


# ─── Load Config ──────────────────────────────────────────────────────────────

def _get_cfg() -> dict:
    try:
        return json.loads(Path("config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


# ─── Fetcher Dasar ────────────────────────────────────────────────────────────

def _fetch(url: str, headers: dict = None, data: bytes = None, timeout: int = 5) -> Optional[str]:
    try:
        h = {"User-Agent": "Mozilla/5.0 (compatible; AstaBot/1.0)"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(url, headers=h, data=data)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read(1024 * 100).decode(charset, errors="replace")
    except Exception:
        return None


# ─── Source 1: Exchange Rate API ──────────────────────────────────────────────

_CURRENCY_PATTERN = re.compile(
    r"\b(kurs|nilai tukar|harga|rate|berapa).{0,30}"
    r"(dolar|dollar|usd|euro|eur|yen|jpy|sgd|pound|gbp|ringgit|myr)\b",
    re.IGNORECASE,
)
_CURRENCY_MAP = {
    "dolar": "USD", "dollar": "USD", "usd": "USD",
    "euro": "EUR", "eur": "EUR",
    "yen": "JPY", "jpy": "JPY",
    "sgd": "SGD",
    "pound": "GBP", "gbp": "GBP",
    "ringgit": "MYR", "myr": "MYR",
}

def _is_currency_query(query: str) -> bool:
    return bool(_CURRENCY_PATTERN.search(query))

def _get_exchange_rate(query: str, timeout: int = 5) -> str:
    query_lower = query.lower()
    target = next(
        (code for kw, code in _CURRENCY_MAP.items() if kw in query_lower),
        "USD"
    )
    raw = _fetch(f"https://open.er-api.com/v6/latest/{target}", timeout=timeout)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        if data.get("result") == "success":
            idr = data["rates"].get("IDR")
            if idr:
                return (
                    f"Kurs terkini (open.er-api.com):\n"
                    f"1 {target} = Rp {idr:,.2f}\n"
                    f"Update: {data.get('time_last_update_utc', 'N/A')}"
                )
    except Exception:
        pass
    return ""


# ─── Source 2: Tavily Search API ──────────────────────────────────────────────

def _tavily_search(query: str, timeout: int = 7) -> str:
    api_key = _get_cfg().get("tavily_api_key", "")
    if not api_key:
        return ""

    payload = json.dumps({
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
        "include_answer": True,
    }).encode("utf-8")

    raw = _fetch(
        "https://api.tavily.com/search",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        data=payload,
        timeout=timeout,
    )
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        parts = []
        if data.get("answer"):
            parts.append(f"Jawaban: {data['answer']}")
        for r in data.get("results", [])[:2]:
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            if content:
                parts.append(f"[{title}]\n{content}\nSumber: {url}")
        return "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


# ─── Source 3: Serper API ─────────────────────────────────────────────────────

def _serper_search(query: str, timeout: int = 5) -> str:
    api_key = _get_cfg().get("serper_api_key", "")
    if not api_key:
        return ""

    payload = json.dumps({
        "q": query,
        "num": 3,
        "hl": "id",
    }).encode("utf-8")

    raw = _fetch(
        "https://google.serper.dev/search",
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        },
        data=payload,
        timeout=timeout,
    )
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        parts = []
        if data.get("answerBox", {}).get("answer"):
            parts.append(f"Jawaban: {data['answerBox']['answer']}")
        elif data.get("answerBox", {}).get("snippet"):
            parts.append(f"Jawaban: {data['answerBox']['snippet']}")
        # Knowledge graph
        if data.get("knowledgeGraph", {}).get("description"):
            kg = data["knowledgeGraph"]
            parts.append(
                f"[{kg.get('title', '')}]\n{kg['description'][:300]}"
            )
        for r in data.get("organic", [])[:2]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")[:250]
            link = r.get("link", "")
            if snippet:
                parts.append(f"[{title}]\n{snippet}\nSumber: {link}")
        return "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


# ─── Source 4: DuckDuckGo Instant Answer ─────────────────────────────────────

def _ddg_instant(query: str, timeout: int = 5) -> str:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
    raw = _fetch(url, timeout=timeout)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        parts = []
        if data.get("Answer"):
            parts.append(f"Jawaban: {data['Answer']}")
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:400])
        if data.get("Infobox", {}).get("content"):
            for item in data["Infobox"]["content"][:3]:
                if item.get("label") and item.get("value"):
                    parts.append(f"{item['label']}: {item['value']}")
        return "\n".join(parts) if parts else ""
    except Exception:
        return ""


# ─── Source 5: Wikipedia API ─────────────────────────────────────────────────

def _wikipedia_search(query: str, timeout: int = 5) -> str:
    import datetime
    if re.search(r"\b(saat ini|sekarang|terkini)\b", query, re.IGNORECASE):
        query = f"{query} {datetime.datetime.now().year}"

    for lang in ["id", "en"]:
        encoded = urllib.parse.quote_plus(query)
        raw = _fetch(
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}&format=json&srlimit=1",
            timeout=timeout,
        )
        if not raw:
            continue
        try:
            title = json.loads(raw)["query"]["search"][0]["title"]
            title_enc = urllib.parse.quote(title)
            raw2 = _fetch(
                f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title_enc}",
                timeout=timeout,
            )
            if raw2:
                extract = json.loads(raw2).get("extract", "")
                if extract:
                    return f"[Wikipedia: {title}]\n{extract[:500]}"
        except Exception:
            continue
    return ""


# ─── Main Search Function ─────────────────────────────────────────────────────

def search_and_summarize(query: str, max_results: int = 3, timeout: int = 7) -> str:
    if _is_currency_query(query):
        result = _get_exchange_rate(query, timeout=timeout)
        if result:
            return result

    result = _tavily_search(query, timeout=timeout)
    if result:
        return result

    result = _serper_search(query, timeout=timeout)
    if result:
        return result

    result = _ddg_instant(query, timeout=timeout)
    if result:
        return result

    result = _wikipedia_search(query, timeout=timeout)
    if result:
        return result

    return ""
