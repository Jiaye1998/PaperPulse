from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import re
import socket
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .article_processing import normalize_title
from .config import config


USER_AGENT = "PaperPulse/0.1 (public scholarly abstract metadata retrieval)"
MAX_HTML_BYTES = 3_000_000
MAX_ABSTRACT_CHARS = 12_000
MAX_REDIRECTS = 4
GLOBAL_CONCURRENCY = 10
PER_HOST_CONCURRENCY = 2


@dataclass(frozen=True)
class AbstractCandidate:
    text: str
    provenance: str
    complete: bool
    source_url: str
    priority: int
    doi: str = ""


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    soup = BeautifulSoup(html.unescape(value), "html.parser")
    for element in soup(["script", "style", "img", "svg", "button"]):
        element.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    text = re.sub(r"^(?:abstract|summary)\s*[:.\-]?\s*", "", text, flags=re.IGNORECASE)
    return text[:MAX_ABSTRACT_CHARS].strip()


def _looks_truncated(text: str) -> bool:
    stripped = text.rstrip()
    return bool(
        len(stripped) < 160
        or re.search(r"(?:\.\.\.|…|\[\.\.\.\])(?:\s*\[[^]]+\])?\s*$", stripped)
    )


def _title_matches(requested_title: str, page_title: str) -> bool:
    requested = set(normalize_title(requested_title).split())
    page = set(normalize_title(page_title).split())
    if not requested or not page:
        return True
    overlap = len(requested & page) / max(1, min(len(requested), len(page)))
    return overlap >= 0.45


def _metadata_title_matches(requested_title: str, result_title: str) -> bool:
    requested_text = normalize_title(requested_title)
    result_text = normalize_title(result_title)
    if not requested_text or not result_text:
        return False
    if requested_text == result_text:
        return True
    requested = set(requested_text.split())
    result = set(result_text.split())
    return len(requested & result) / max(len(requested), len(result)) >= 0.82


def _json_ld_abstracts(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        abstract = value.get("abstract")
        if isinstance(abstract, str):
            found.append(abstract)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                found.extend(_json_ld_abstracts(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_json_ld_abstracts(nested))
    return found


def extract_abstract_from_html(
    html_text: str, requested_title: str, source_url: str
) -> AbstractCandidate | None:
    """Extract only explicit public abstract fields, never arbitrary body text."""
    soup = BeautifulSoup(html_text, "html.parser")
    meta: dict[str, list[str]] = defaultdict(list)
    for element in soup.find_all("meta"):
        key = str(
            element.get("name") or element.get("property") or element.get("itemprop") or ""
        ).strip().casefold()
        content = str(element.get("content") or "").strip()
        if key and content:
            meta[key].append(content)

    page_title = next(
        (
            _clean_text(value)
            for key in ("citation_title", "dc.title", "dcterms.title", "og:title")
            for value in meta.get(key, [])
            if _clean_text(value)
        ),
        _clean_text(soup.title.get_text(" ", strip=True) if soup.title else ""),
    )
    if page_title and not _title_matches(requested_title, page_title):
        return None

    candidates: list[AbstractCandidate] = []

    def add(value: Any, provenance: str, priority: int, strong: bool) -> None:
        text = _clean_text(value)
        if len(text) < 40 or normalize_title(text) == normalize_title(requested_title):
            return
        candidates.append(
            AbstractCandidate(
                text=text,
                provenance=provenance,
                complete=bool(strong and not _looks_truncated(text)),
                source_url=source_url,
                priority=priority,
            )
        )

    strong_meta = {
        "citation_abstract": "publisher_page_abstract",
        "dcterms.abstract": "publisher_page_abstract",
        "dc.abstract": "publisher_page_abstract",
        "dc.description.abstract": "publisher_page_abstract",
        "eprints.abstract": "publisher_page_abstract",
        "prism.abstract": "publisher_page_abstract",
        "article:abstract": "publisher_page_abstract",
        "og:article:abstract": "publisher_page_abstract",
    }
    for key, provenance in strong_meta.items():
        for value in meta.get(key, []):
            add(value, provenance, 100, True)

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or script.get_text() or "")
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        for value in _json_ld_abstracts(payload):
            add(value, "publisher_page_abstract", 90, True)

    selectors = (
        "section#abstract",
        "section.abstract",
        "[role='doc-abstract']",
        "div#abstract",
        "div.abstract",
        ".abstractSection",
        ".article-section__abstract",
        ".abstractInFull",
        ".hlFld-Abstract",
        "#Abs1-content",
        "[data-title='Abstract']",
    )
    for selector in selectors:
        for element in soup.select(selector)[:2]:
            copy = BeautifulSoup(str(element), "html.parser")
            for heading in copy.find_all(["h1", "h2", "h3", "h4"]):
                heading.decompose()
            add(copy.get_text(" ", strip=True), "publisher_page_abstract", 80, True)

    for key in ("dc.description", "dcterms.description", "description", "og:description"):
        for value in meta.get(key, []):
            add(value, "publisher_page_excerpt", 20, False)

    if not candidates:
        return None
    unique: dict[str, AbstractCandidate] = {}
    for candidate in candidates:
        key = normalize_title(candidate.text)
        existing = unique.get(key)
        if existing is None or candidate.priority > existing.priority:
            unique[key] = candidate
    return max(
        unique.values(),
        key=lambda item: (item.complete, item.priority, len(item.text)),
    )


def _address_is_public(hostname: str, port: int) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    if not addresses:
        return False
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address[4][0])
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


async def _url_is_public(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in {80, 443}:
        return False
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        return False
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return await asyncio.to_thread(_address_is_public, hostname, port)
    return ip.is_global


async def _fetch_public_html(
    client: httpx.AsyncClient, url: str
) -> tuple[str, str] | None:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not await _url_is_public(current):
            return None
        try:
            response = await client.get(current, follow_redirects=False)
        except httpx.HTTPError:
            return None
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if not location:
                return None
            current = urljoin(current, location)
            continue
        if response.is_error:
            return None
        content_type = response.headers.get("content-type", "").casefold()
        if "html" not in content_type and "xhtml" not in content_type:
            return None
        declared_size = response.headers.get("content-length")
        if declared_size and declared_size.isdigit() and int(declared_size) > MAX_HTML_BYTES:
            return None
        if len(response.content) > MAX_HTML_BYTES:
            return None
        return response.text, str(response.url)
    return None


def _crossref_candidate(payload: Any, title: str, source_url: str) -> AbstractCandidate | None:
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    titles = message.get("title")
    crossref_title = str(titles[0]) if isinstance(titles, list) and titles else ""
    if crossref_title and not _metadata_title_matches(title, crossref_title):
        return None
    text = _clean_text(message.get("abstract"))
    if len(text) < 40:
        return None
    return AbstractCandidate(
        text=text,
        provenance="crossref_abstract",
        complete=not _looks_truncated(text),
        source_url=source_url,
        priority=95,
        doi=str(message.get("DOI") or "").casefold(),
    )


def _crossref_search_candidate(
    payload: Any, title: str, source_url: str
) -> AbstractCandidate | None:
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    items = message.get("items") if isinstance(message, dict) else None
    if not isinstance(items, list):
        return None
    for work in items:
        if not isinstance(work, dict):
            continue
        candidate = _crossref_candidate({"message": work}, title, source_url)
        if candidate:
            return candidate
    return None


def _openalex_abstract(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in value.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions.append((index, str(word)))
    return " ".join(word for _, word in sorted(positions))


def _openalex_candidate(payload: Any, title: str, source_url: str) -> AbstractCandidate | None:
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    work = results[0]
    if work.get("title") and not _metadata_title_matches(title, str(work["title"])):
        return None
    text = _clean_text(_openalex_abstract(work.get("abstract_inverted_index")))
    if len(text) < 40:
        return None
    return AbstractCandidate(
        text=text,
        provenance="openalex_abstract",
        complete=not _looks_truncated(text),
        source_url=source_url,
        priority=90,
    )


def _arxiv_feed_candidate(article: dict[str, Any]) -> AbstractCandidate | None:
    raw = article.get("raw") if isinstance(article.get("raw"), dict) else {}
    if not raw.get("arxiv_id"):
        return None
    summary = str(article.get("summary", ""))
    marker = re.search(r"\bAbstract:\s*", summary, flags=re.IGNORECASE)
    text = _clean_text(summary[marker.end() :] if marker else summary)
    if len(text) < 160 or _looks_truncated(text):
        return None
    return AbstractCandidate(
        text=text,
        provenance="arxiv_feed_abstract",
        complete=True,
        source_url=str(article.get("url", "")),
        priority=100,
    )


def _apply_candidate(article: dict[str, Any], candidate: AbstractCandidate) -> None:
    raw = dict(article.get("raw") or {})
    article["summary"] = candidate.text
    article["summary_quality"] = 0.95 if candidate.complete else min(
        0.55, max(0.25, len(candidate.text) / 1_500)
    )
    raw.update(
        {
            "summary_source": candidate.provenance,
            "abstract_status": "complete" if candidate.complete else "excerpt",
            "abstract_source_url": candidate.source_url,
            "abstract_fetched_at": datetime.now(UTC).isoformat(),
        }
    )
    if candidate.doi:
        raw["doi"] = candidate.doi
    article["raw"] = raw


def _better_candidate(
    incoming: AbstractCandidate | None, current: AbstractCandidate | None
) -> bool:
    if incoming is None:
        return False
    if current is None:
        return True
    return (incoming.complete, incoming.priority, len(incoming.text)) > (
        current.complete,
        current.priority,
        len(current.text),
    )


def _apply_unverified_status(article: dict[str, Any]) -> None:
    raw = dict(article.get("raw") or {})
    summary = str(article.get("summary", "")).strip()
    source = str(raw.get("summary_source") or "none")
    raw["abstract_status"] = "excerpt" if summary else "unavailable"
    if not summary:
        raw["summary_source"] = "none"
    elif source == "feed_abstract_or_excerpt":
        article["summary_quality"] = min(float(article.get("summary_quality", 0.5)), 0.55)
    raw["abstract_fetched_at"] = datetime.now(UTC).isoformat()
    article["raw"] = raw


def _reuse_cached_complete(
    article: dict[str, Any], cached: dict[str, Any] | None
) -> bool:
    if not cached or normalize_title(str(cached.get("title", ""))) != normalize_title(
        str(article.get("title", ""))
    ):
        return False
    raw = cached.get("raw") if isinstance(cached.get("raw"), dict) else {}
    if raw.get("abstract_status") != "complete":
        return False
    provenance = str(raw.get("summary_source") or "")
    if provenance not in {
        "publisher_page_abstract",
        "publisher_browser_abstract",
        "crossref_abstract",
        "openalex_abstract",
        "arxiv_feed_abstract",
    }:
        return False
    _apply_candidate(
        article,
        AbstractCandidate(
            text=str(cached.get("summary", "")),
            provenance=provenance,
            complete=True,
            source_url=str(raw.get("abstract_source_url") or cached.get("url") or ""),
            priority=100,
            doi=str(raw.get("doi") or ""),
        ),
    )
    article["raw"]["abstract_fetched_at"] = raw.get("abstract_fetched_at")
    return True


async def enrich_articles_with_public_abstracts(
    articles: list[dict[str, Any]],
    cached_articles: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve public abstracts before ranking while keeping failures non-fatal."""
    cached_by_id = {str(item["id"]): item for item in (cached_articles or [])}
    stats = {
        "attempted": 0,
        "cache_hits": 0,
        "complete": 0,
        "excerpt": 0,
        "unavailable": 0,
        "browser_attempted": 0,
        "browser_complete": 0,
        "browser_available": False,
        "browser_error": "",
        "verification_required": [],
    }
    pending: list[dict[str, Any]] = []
    for article in articles:
        if _reuse_cached_complete(article, cached_by_id.get(str(article["id"]))):
            stats["cache_hits"] += 1
            stats["complete"] += 1
            continue
        arxiv = _arxiv_feed_candidate(article)
        if arxiv:
            _apply_candidate(article, arxiv)
            stats["complete"] += 1
            continue
        pending.append(article)

    from .browser_abstracts import resolve_with_persistent_browser

    browser_result = await resolve_with_persistent_browser(
        pending, extract_abstract_from_html
    )
    stats["browser_attempted"] = browser_result.attempted
    stats["browser_available"] = browser_result.available
    stats["browser_error"] = browser_result.error
    stats["verification_required"] = browser_result.challenges
    challenge_domains = {
        item["domain"] for item in browser_result.challenges if item.get("domain")
    }
    global_limit = asyncio.Semaphore(GLOBAL_CONCURRENCY)
    host_limits: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(PER_HOST_CONCURRENCY)
    )
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(12.0, connect=6.0),
        headers=headers,
    ) as client:

        async def enrich(article: dict[str, Any]) -> None:
            best = browser_result.candidates.get(str(article["id"]))
            if best and best.complete:
                _apply_candidate(article, best)
                stats["complete"] += 1
                stats["browser_complete"] += 1
                return

            stats["attempted"] += 1
            article_url = str(article.get("url", ""))
            host = urlparse(article_url).hostname or "unknown"
            if article_url and host.casefold() not in challenge_domains:
                async with global_limit, host_limits[host]:
                    fetched = await _fetch_public_html(client, article_url)
                if fetched:
                    html_text, final_url = fetched
                    candidate = extract_abstract_from_html(
                        html_text, str(article.get("title", "")), final_url
                    )
                    if _better_candidate(candidate, best):
                        best = candidate

            raw = article.get("raw") if isinstance(article.get("raw"), dict) else {}
            doi = str(raw.get("doi") or "").strip()
            if doi and not (best and best.complete):
                crossref_url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
                try:
                    async with global_limit, host_limits["api.crossref.org"]:
                        response = await client.get(
                            crossref_url, headers={"Accept": "application/json"}
                        )
                    if not response.is_error:
                        candidate = _crossref_candidate(
                            response.json(), str(article.get("title", "")), crossref_url
                        )
                        if _better_candidate(candidate, best):
                            best = candidate
                            doi = candidate.doi or doi
                except (httpx.HTTPError, ValueError):
                    pass

            if not (best and best.complete):
                crossref_search_url = "https://api.crossref.org/works"
                crossref_params: dict[str, str | int] = {
                    "query.title": str(article.get("title", "")),
                    "rows": 3,
                    "select": "DOI,title,abstract,type",
                }
                try:
                    async with global_limit, host_limits["api.crossref.org"]:
                        response = await client.get(
                            crossref_search_url,
                            params=crossref_params,
                            headers={"Accept": "application/json"},
                        )
                    if not response.is_error:
                        candidate = _crossref_search_candidate(
                            response.json(),
                            str(article.get("title", "")),
                            str(response.url),
                        )
                        if _better_candidate(candidate, best):
                            best = candidate
                            doi = candidate.doi or doi
                except (httpx.HTTPError, ValueError):
                    pass

            if not (best and best.complete):
                openalex_url = "https://api.openalex.org/works"
                params: dict[str, str | int] = {
                    "per-page": 1,
                    "select": "title,abstract_inverted_index",
                }
                if doi:
                    params["filter"] = f"doi:{doi}"
                else:
                    params["search"] = str(article.get("title", ""))
                if config.openalex_api_key:
                    params["api_key"] = config.openalex_api_key
                try:
                    async with global_limit, host_limits["api.openalex.org"]:
                        response = await client.get(
                            openalex_url,
                            params=params,
                            headers={"Accept": "application/json"},
                        )
                    if not response.is_error:
                        candidate = _openalex_candidate(
                            response.json(), str(article.get("title", "")), str(response.url)
                        )
                        if _better_candidate(candidate, best):
                            best = candidate
                except (httpx.HTTPError, ValueError):
                    pass

            if best:
                feed_text = str(article.get("summary", ""))
                if not best.complete and len(feed_text) > len(best.text):
                    _apply_unverified_status(article)
                else:
                    _apply_candidate(article, best)
            else:
                _apply_unverified_status(article)

            status = str(article.get("raw", {}).get("abstract_status") or "unavailable")
            stats[status if status in {"complete", "excerpt"} else "unavailable"] += 1

        await asyncio.gather(*(enrich(article) for article in pending))

    return articles, stats
