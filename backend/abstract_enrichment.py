from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import re
import socket
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Iterator
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .article_processing import extract_doi, extract_elsevier_pii, normalize_title
from .config import config


MAX_HTML_BYTES = 3_000_000
MAX_ABSTRACT_CHARS = 12_000
MAX_REDIRECTS = 4
GLOBAL_CONCURRENCY = 10
PER_HOST_CONCURRENCY = 2

# Every scholarly metadata API here answers DOI-set queries, so one request can
# resolve a whole batch. Serial batches also stay inside the published rate
# limits, which single-article fan-out does not.
CROSSREF_BATCH_SIZE = 40
OPENALEX_BATCH_SIZE = 40
EUROPEPMC_BATCH_SIZE = 20
# Scopus Search caps COMPLETE-view results at 25 per request.
SCOPUS_BATCH_SIZE = 25
CROSSREF_MIN_INTERVAL = 0.5
# OpenAlex answers only the residue Crossref and Europe PMC missed, so a slow
# cadence costs little and avoids the 429/backoff cycle that dominated runtime.
OPENALEX_MIN_INTERVAL = 1.0
EUROPEPMC_MIN_INTERVAL = 0.2
SCOPUS_MIN_INTERVAL = 0.3
API_RETRY_ATTEMPTS = 3
MAX_RETRY_WAIT_SECONDS = 8.0
# After this many bot-wall responses a publisher host is dropped for the rest of
# the run; further requests only burn time and sharpen the block.
HOST_BLOCK_THRESHOLD = 2
TITLE_SEARCH_LIMIT = 60


def _user_agent() -> str:
    contact = config.contact_email.strip()
    suffix = f"; mailto:{contact}" if contact else ""
    return f"PaperPulse/0.2 (public scholarly abstract metadata retrieval{suffix})"


USER_AGENT = _user_agent()


def _polite_params(params: dict[str, Any]) -> dict[str, Any]:
    contact = config.contact_email.strip()
    if contact:
        params = {**params, "mailto": contact}
    return params


# Physics feeds carry LaTeX in titles, and these characters are filter/query
# syntax to Crossref and OpenAlex rather than searchable text.
_SEARCH_UNSAFE = re.compile(r"[,|:;\\{}$^_&#%~<>\"()\[\]]+")


def _search_phrase(title: str) -> str:
    return " ".join(_SEARCH_UNSAFE.sub(" ", title).split())[:250]


def _chunked(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class _ServiceGate:
    """Pace one API, and stop calling it once it is clearly refusing us.

    Without the second half, an API that throttles the whole run turns every
    remaining lookup into a full retry ladder and the refresh never finishes.
    """

    def __init__(self, min_interval: float, failure_budget: int = 4) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0
        self._failure_budget = failure_budget
        self._failures = 0
        self.disabled = False

    async def __aenter__(self) -> "_ServiceGate":
        await self._lock.acquire()
        delay = self._min_interval - (time.monotonic() - self._last)
        if delay > 0:
            await asyncio.sleep(delay)
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self._last = time.monotonic()
        self._lock.release()

    def record_success(self) -> None:
        self._failures = 0

    def record_refusal(self) -> None:
        self._failures += 1
        if self._failures >= self._failure_budget:
            self.disabled = True


async def _get_json(
    client: httpx.AsyncClient,
    gate: _ServiceGate,
    url: str,
    params: dict[str, Any],
) -> Any:
    """GET JSON, backing off when the service asks us to slow down."""
    if gate.disabled:
        return None
    for attempt in range(API_RETRY_ATTEMPTS):
        async with gate:
            try:
                response = await client.get(
                    url, params=params, headers={"Accept": "application/json"}
                )
            except httpx.HTTPError:
                gate.record_refusal()
                return None
        if response.status_code in {429, 503}:
            if attempt == API_RETRY_ATTEMPTS - 1 or gate.disabled:
                gate.record_refusal()
                return None
            header = response.headers.get("retry-after", "")
            wait = float(header) if header.replace(".", "", 1).isdigit() else 0.0
            await asyncio.sleep(min(max(wait, 2.0 ** attempt), MAX_RETRY_WAIT_SECONDS))
            continue
        if response.is_error:
            # A plain "no such record" is a miss, not a reason to drop the service.
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        gate.record_success()
        return payload
    return None


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


_TRUNCATION_MARK = re.compile(r"(?:\.\.\.|…|\[\.\.\.\])(?:\s*\[[^]]+\])?\s*$")


def _looks_truncated(text: str) -> bool:
    stripped = text.rstrip()
    return bool(len(stripped) < 160 or _TRUNCATION_MARK.search(stripped))


def _metadata_looks_complete(text: str) -> bool:
    """Judge an abstract that came from a dedicated abstract field.

    Scraped pages need a length floor because a stray blurb can look like an
    abstract. A metadata API's abstract field cannot be anything else, so only an
    explicit truncation mark or a near-empty value disqualifies it.
    """
    stripped = text.rstrip()
    return bool(len(stripped) >= 100 and not _TRUNCATION_MARK.search(stripped))


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


def _crossref_candidate(
    payload: Any, title: str, source_url: str, reject_preprints: bool = False
) -> AbstractCandidate | None:
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    titles = message.get("title")
    crossref_title = str(titles[0]) if isinstance(titles, list) and titles else ""
    if crossref_title and not _metadata_title_matches(title, crossref_title):
        return None
    if reject_preprints and str(message.get("type") or "") == "posted-content":
        # Matching a journal article by title readily lands on the authors' own
        # preprint, whose text and DOI belong to a different version of the work.
        return None
    text = _clean_text(message.get("abstract"))
    if len(text) < 40:
        return None
    return AbstractCandidate(
        text=text,
        provenance="crossref_abstract",
        complete=_metadata_looks_complete(text),
        source_url=source_url,
        priority=95,
        doi=str(message.get("DOI") or "").casefold(),
    )


def _crossref_search_candidate(
    payload: Any, title: str, source_url: str, reject_preprints: bool = False
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
        candidate = _crossref_candidate(
            {"message": work}, title, source_url, reject_preprints
        )
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
        complete=_metadata_looks_complete(text),
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


def _doi_batch_candidate(
    article: dict[str, Any],
    text: Any,
    result_title: Any,
    provenance: str,
    priority: int,
    source_url: str,
    doi: str,
) -> AbstractCandidate | None:
    """Build a candidate for a work matched by DOI.

    The DOI is the identity here, so the title is only a guard against a wrong
    DOI (a bad URL derivation, or a feed quoting someone else's identifier).
    """
    cleaned = _clean_text(text)
    if len(cleaned) < 40:
        return None
    requested_title = str(article.get("title", ""))
    if result_title and not _title_matches(requested_title, _clean_text(result_title)):
        return None
    return AbstractCandidate(
        text=cleaned,
        provenance=provenance,
        complete=_metadata_looks_complete(cleaned),
        source_url=source_url,
        priority=priority,
        doi=doi,
    )


async def _crossref_doi_batch(
    client: httpx.AsyncClient,
    gate: _ServiceGate,
    targets: dict[str, dict[str, Any]],
) -> dict[str, AbstractCandidate]:
    """Resolve many DOIs per request; same-name Crossref filters are OR-ed."""
    found: dict[str, AbstractCandidate] = {}
    for chunk in _chunked(sorted(targets), CROSSREF_BATCH_SIZE):
        payload = await _get_json(
            client,
            gate,
            "https://api.crossref.org/works",
            _polite_params(
                {
                    "filter": ",".join(f"doi:{doi}" for doi in chunk),
                    "rows": len(chunk),
                    "select": "DOI,title,abstract",
                }
            ),
        )
        message = payload.get("message") if isinstance(payload, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
        for work in items or []:
            if not isinstance(work, dict):
                continue
            doi = str(work.get("DOI") or "").casefold()
            article = targets.get(doi)
            if article is None:
                continue
            titles = work.get("title")
            candidate = _doi_batch_candidate(
                article,
                work.get("abstract"),
                titles[0] if isinstance(titles, list) and titles else "",
                "crossref_abstract",
                95,
                f"https://doi.org/{doi}",
                doi,
            )
            if candidate:
                found[str(article["id"])] = candidate
    return found


async def _europepmc_doi_batch(
    client: httpx.AsyncClient,
    gate: _ServiceGate,
    targets: dict[str, dict[str, Any]],
) -> dict[str, AbstractCandidate]:
    """Europe PMC indexes preprint servers and biomedical journals Crossref misses."""
    found: dict[str, AbstractCandidate] = {}
    for chunk in _chunked(sorted(targets), EUROPEPMC_BATCH_SIZE):
        payload = await _get_json(
            client,
            gate,
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            {
                "query": " OR ".join(f'DOI:"{doi}"' for doi in chunk),
                "format": "json",
                "resultType": "core",
                "pageSize": min(100, len(chunk) * 2),
            },
        )
        result_list = payload.get("resultList") if isinstance(payload, dict) else None
        results = result_list.get("result") if isinstance(result_list, dict) else None
        for work in results or []:
            if not isinstance(work, dict):
                continue
            doi = str(work.get("doi") or "").casefold()
            article = targets.get(doi)
            if article is None:
                continue
            candidate = _doi_batch_candidate(
                article,
                work.get("abstractText"),
                work.get("title"),
                "europepmc_abstract",
                92,
                f"https://europepmc.org/article/{work.get('source', 'MED')}/{work.get('id', '')}",
                doi,
            )
            if candidate:
                found[str(article["id"])] = candidate
    return found


async def _openalex_doi_batch(
    client: httpx.AsyncClient,
    gate: _ServiceGate,
    targets: dict[str, dict[str, Any]],
) -> dict[str, AbstractCandidate]:
    found: dict[str, AbstractCandidate] = {}
    for chunk in _chunked(sorted(targets), OPENALEX_BATCH_SIZE):
        params = _polite_params(
            {
                "filter": "doi:" + "|".join(chunk),
                "per-page": len(chunk),
                "select": "doi,title,abstract_inverted_index",
            }
        )
        if config.openalex_api_key:
            params["api_key"] = config.openalex_api_key
        payload = await _get_json(client, gate, "https://api.openalex.org/works", params)
        results = payload.get("results") if isinstance(payload, dict) else None
        for work in results or []:
            if not isinstance(work, dict):
                continue
            doi = str(work.get("doi") or "").casefold()
            doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
            article = targets.get(doi)
            if article is None:
                continue
            candidate = _doi_batch_candidate(
                article,
                _openalex_abstract(work.get("abstract_inverted_index")),
                work.get("title"),
                "openalex_abstract",
                90,
                f"https://api.openalex.org/works/doi:{doi}",
                doi,
            )
            if candidate:
                found[str(article["id"])] = candidate
    return found


_COPYRIGHT_TAIL = re.compile(r"\s*©.{0,120}$")


async def _scopus_doi_batch(
    client: httpx.AsyncClient,
    gate: _ServiceGate,
    targets: dict[str, dict[str, Any]],
) -> dict[str, AbstractCandidate]:
    """Ask Scopus for the abstracts no open service carries.

    Scopus Search in COMPLETE view returns dc:description for up to 25 records per
    request, which is far cheaper against the weekly quota than one Abstract
    Retrieval call per DOI. Entitlement is checked per request, so a key without
    institutional access simply yields nothing rather than partial data.
    """
    found: dict[str, AbstractCandidate] = {}
    if not config.elsevier_api_key:
        return found
    headers = {"X-ELS-APIKey": config.elsevier_api_key, "Accept": "application/json"}
    if config.elsevier_insttoken:
        headers["X-ELS-Insttoken"] = config.elsevier_insttoken
    for chunk in _chunked(sorted(targets), SCOPUS_BATCH_SIZE):
        query = " OR ".join(f'DOI("{doi}")' for doi in chunk)
        if gate.disabled:
            break
        async with gate:
            try:
                response = await client.get(
                    "https://api.elsevier.com/content/search/scopus",
                    params={"query": query, "view": "COMPLETE", "count": len(chunk)},
                    headers=headers,
                )
            except httpx.HTTPError:
                gate.record_refusal()
                continue
        if response.is_error:
            # 401 is a bad key and 403 means the network is not entitled; either way
            # retrying the rest of the batches cannot succeed.
            gate.record_refusal()
            continue
        gate.record_success()
        try:
            payload = response.json()
        except ValueError:
            continue
        results = payload.get("search-results") if isinstance(payload, dict) else None
        entries = results.get("entry") if isinstance(results, dict) else None
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            doi = str(entry.get("prism:doi") or "").casefold()
            article = targets.get(doi)
            if article is None:
                continue
            text = _COPYRIGHT_TAIL.sub("", str(entry.get("dc:description") or ""))
            candidate = _doi_batch_candidate(
                article,
                text,
                entry.get("dc:title"),
                "scopus_abstract",
                94,
                f"https://doi.org/{doi}",
                doi,
            )
            if candidate:
                found[str(article["id"])] = candidate
    return found


# Ordered cheapest-first. Scopus is last because its weekly quota is finite and
# every earlier service is free, so it only ever sees what nothing else answered.
DOI_BATCH_RESOLVERS = (
    ("crossref", _crossref_doi_batch),
    ("europepmc", _europepmc_doi_batch),
    ("openalex", _openalex_doi_batch),
    ("scopus", _scopus_doi_batch),
)


async def _resolve_elsevier_dois(
    client: httpx.AsyncClient,
    gate: _ServiceGate,
    articles: Iterable[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    """Turn ScienceDirect PIIs into DOIs so the DOI-keyed services can see them.

    Elsevier deposits no abstracts itself, but a DOI still unlocks Europe PMC for
    the biomedical share of its catalogue — and it stops the title-search fallback
    from settling for a preprint of the same work.
    """
    targets: dict[str, dict[str, Any]] = {}
    for article in articles:
        if (article.get("raw") or {}).get("doi"):
            continue
        pii = extract_elsevier_pii(str(article.get("url", "")))
        if pii:
            targets.setdefault(pii, article)
    stats["elsevier_pii_seen"] = len(targets)
    if not targets:
        return
    found = 0
    for chunk in _chunked(sorted(targets), CROSSREF_BATCH_SIZE):
        payload = await _get_json(
            client,
            gate,
            "https://api.crossref.org/works",
            _polite_params(
                {
                    "filter": ",".join(f"alternative-id:{pii}" for pii in chunk),
                    "rows": len(chunk),
                    "select": "DOI,title,alternative-id",
                }
            ),
        )
        message = payload.get("message") if isinstance(payload, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
        for work in items or []:
            if not isinstance(work, dict):
                continue
            doi = str(work.get("DOI") or "").casefold()
            identifiers = work.get("alternative-id")
            titles = work.get("title")
            result_title = titles[0] if isinstance(titles, list) and titles else ""
            for identifier in identifiers if isinstance(identifiers, list) else []:
                article = targets.get(str(identifier).upper())
                if article is None or not doi:
                    continue
                if result_title and not _title_matches(
                    str(article.get("title", "")), _clean_text(result_title)
                ):
                    continue
                raw = dict(article.get("raw") or {})
                raw["doi"] = doi
                article["raw"] = raw
                found += 1
    stats["elsevier_doi_resolved"] = found


async def _resolve_by_doi(
    client: httpx.AsyncClient,
    gates: dict[str, _ServiceGate],
    articles: Iterable[dict[str, Any]],
    resolved: dict[str, AbstractCandidate],
    stats: dict[str, Any],
) -> None:
    """Walk the DOI-keyed services in order, carrying only the still-missing works."""
    targets: dict[str, dict[str, Any]] = {}
    for article in articles:
        doi = str((article.get("raw") or {}).get("doi") or "") or extract_doi(article)
        if doi:
            targets.setdefault(doi.casefold(), article)
    stats["doi_known"] = len(targets)
    for name, resolver in DOI_BATCH_RESOLVERS:
        pending = {
            doi: article
            for doi, article in targets.items()
            if str(article["id"]) not in resolved
        }
        if not pending:
            break
        found = await resolver(client, gates[name], pending)
        stats[f"{name}_batch_hits"] = len(found)
        resolved.update(found)


async def _resolve_by_title(
    client: httpx.AsyncClient,
    gates: dict[str, _ServiceGate],
    articles: list[dict[str, Any]],
    resolved: dict[str, AbstractCandidate],
    stats: dict[str, Any],
) -> None:
    """Last metadata resort for works whose DOI is unknown, one query each."""
    pending = [
        article
        for article in articles
        if str(article["id"]) not in resolved and str(article.get("title", "")).strip()
    ][:TITLE_SEARCH_LIMIT]
    stats["title_search_attempted"] = len(pending)
    hits = 0
    for article in pending:
        title = str(article.get("title", ""))
        phrase = _search_phrase(title)
        if not phrase:
            continue
        payload = await _get_json(
            client,
            gates["crossref"],
            "https://api.crossref.org/works",
            _polite_params(
                {
                    "query.bibliographic": phrase,
                    "rows": 3,
                    "select": "DOI,title,abstract,type",
                }
            ),
        )
        candidate = _crossref_search_candidate(
            payload, title, "https://api.crossref.org/works", reject_preprints=True
        )
        if candidate is None:
            payload = await _get_json(
                client,
                gates["openalex"],
                "https://api.openalex.org/works",
                _polite_params(
                    {
                        "filter": f"title.search:{phrase}",
                        "per-page": 1,
                        "select": "title,abstract_inverted_index",
                    }
                ),
            )
            candidate = _openalex_candidate(
                payload, title, "https://api.openalex.org/works"
            )
        if candidate and candidate.complete:
            resolved[str(article["id"])] = candidate
            hits += 1
    stats["title_search_hits"] = hits


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
        "europepmc_abstract",
        "scopus_abstract",
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


async def _resolve_publisher_pages(
    client: httpx.AsyncClient,
    articles: list[dict[str, Any]],
    resolved: dict[str, AbstractCandidate],
    blocked_hosts: set[str],
    stats: dict[str, Any],
) -> None:
    """Fetch the article page itself for works no metadata service could answer."""
    pending = [article for article in articles if str(article["id"]) not in resolved]
    stats["page_attempted"] = len(pending)
    global_limit = asyncio.Semaphore(GLOBAL_CONCURRENCY)
    host_limits: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(PER_HOST_CONCURRENCY)
    )
    refusals: dict[str, int] = defaultdict(int)

    async def fetch(article: dict[str, Any]) -> None:
        article_url = str(article.get("url", ""))
        host = (urlparse(article_url).hostname or "").casefold()
        if not article_url or host in blocked_hosts:
            return
        async with global_limit, host_limits[host]:
            if host in blocked_hosts:
                return
            fetched = await _fetch_public_html(client, article_url)
        if fetched is None:
            refusals[host] += 1
            if refusals[host] >= HOST_BLOCK_THRESHOLD:
                blocked_hosts.add(host)
            return
        html_text, final_url = fetched
        candidate = extract_abstract_from_html(
            html_text, str(article.get("title", "")), final_url
        )
        if candidate and candidate.complete:
            resolved[str(article["id"])] = candidate

    await asyncio.gather(*(fetch(article) for article in pending))
    stats["page_hits"] = sum(
        1
        for article in pending
        if str(article["id"]) in resolved
    )
    stats["blocked_hosts"] = sorted(blocked_hosts)


async def enrich_articles_with_public_abstracts(
    articles: list[dict[str, Any]],
    cached_articles: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve public abstracts before ranking while keeping failures non-fatal.

    Cheap batched metadata runs first, so the slow and bot-walled paths only ever
    see the works nothing else could answer.
    """
    cached_by_id = {str(item["id"]): item for item in (cached_articles or [])}
    stats: dict[str, Any] = {
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

    stats["attempted"] = len(pending)
    resolved: dict[str, AbstractCandidate] = {}
    blocked_hosts: set[str] = set()
    gates = {
        "crossref": _ServiceGate(CROSSREF_MIN_INTERVAL),
        "openalex": _ServiceGate(OPENALEX_MIN_INTERVAL),
        "europepmc": _ServiceGate(EUROPEPMC_MIN_INTERVAL),
        "scopus": _ServiceGate(SCOPUS_MIN_INTERVAL, failure_budget=2),
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }

    if pending:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=8.0),
            headers=headers,
        ) as client:
            await _resolve_elsevier_dois(client, gates["crossref"], pending, stats)
            await _resolve_by_doi(client, gates, pending, resolved, stats)
            await _resolve_by_title(client, gates, pending, resolved, stats)
            await _resolve_publisher_pages(
                client, pending, resolved, blocked_hosts, stats
            )

    unresolved = [article for article in pending if str(article["id"]) not in resolved]
    if unresolved:
        from .browser_abstracts import resolve_with_persistent_browser

        browser_result = await resolve_with_persistent_browser(
            unresolved, extract_abstract_from_html
        )
        stats["browser_attempted"] = browser_result.attempted
        stats["browser_available"] = browser_result.available
        stats["browser_error"] = browser_result.error
        stats["verification_required"] = browser_result.challenges
        stats["browser_refused_domains"] = browser_result.refused_domains
        for article_id, candidate in browser_result.candidates.items():
            if _better_candidate(candidate, resolved.get(article_id)):
                resolved[article_id] = candidate
                if candidate.complete:
                    stats["browser_complete"] += 1

    for article in pending:
        best = resolved.get(str(article["id"]))
        if best is None:
            _apply_unverified_status(article)
        else:
            feed_text = str(article.get("summary", ""))
            if not best.complete and len(feed_text) > len(best.text):
                _apply_unverified_status(article)
            else:
                _apply_candidate(article, best)
        status = str((article.get("raw") or {}).get("abstract_status") or "unavailable")
        stats[status if status in {"complete", "excerpt"} else "unavailable"] += 1

    return articles, stats
