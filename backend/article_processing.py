from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import OrderedDict
from typing import Any
from urllib.parse import unquote, urlparse


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


def _squash(value: str) -> str:
    return " ".join((value or "").split())


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return _squash(text)


def canonical_source_name(value: str) -> str:
    source = _squash(value).strip(" -:|")
    source = re.sub(r"^Wiley(?:-Blackwell)?\s*:\s*", "", source, flags=re.IGNORECASE)
    source = re.sub(
        r"\s*:\s*Table of Contents(?:\s*\([^)]*\))?\s*$",
        "",
        source,
        flags=re.IGNORECASE,
    )
    source = re.sub(
        r"\s+-\s+current\s+-\s+nature\.com\s+science\s+feeds\s*$",
        "",
        source,
        flags=re.IGNORECASE,
    )
    source = re.sub(
        r"\s+(?:-|:)\s+nature\.com\s+(?:science|subject)\s+feeds\s*$",
        "",
        source,
        flags=re.IGNORECASE,
    )
    source = re.sub(
        r"^ScienceDirect\s+Publication\s*:\s*", "", source, flags=re.IGNORECASE
    )
    # Keyword-monitor feeds append a user-defined topic in parentheses.
    source = re.sub(r"\s*\([^()]{2,80}\)\s*$", "", source).strip()
    if source.casefold() == "physics updates on arxiv.org":
        return "arXiv"
    return source or "Unknown source"


def canonical_url(value: str) -> str:
    parsed = urlparse(value or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    host = parsed.netloc.casefold().removeprefix("www.")
    path = unquote(parsed.path).rstrip("/").casefold()
    return f"{host}{path}"


def extract_doi(article: dict[str, Any]) -> str:
    haystack = " ".join(
        str(article.get(field, "")) for field in ("url", "title", "summary")
    )
    match = DOI_PATTERN.search(unquote(haystack))
    return match.group(0).rstrip(".,;)").casefold() if match else ""


def extract_arxiv_id(article: dict[str, Any]) -> str:
    haystack = " ".join(
        str(article.get(field, "")) for field in ("url", "title", "summary")
    )
    match = ARXIV_PATTERN.search(haystack)
    return match.group(1).casefold() if match else ""


def article_identity(article: dict[str, Any]) -> str:
    return article_identity_keys(article)[0]


def article_identity_keys(article: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    doi = extract_doi(article)
    if doi:
        keys.append(f"doi:{doi}")
    arxiv_id = extract_arxiv_id(article)
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")
    normalized_url = canonical_url(str(article.get("url", "")))
    if normalized_url:
        keys.append(f"url:{normalized_url}")
    title = normalize_title(str(article.get("title", "")))
    if len(title) >= 12:
        keys.append(f"title:{title}")
    if keys:
        return keys
    fallback = f"{title}|{article.get('source', '')}|{article.get('published_at', '')}"
    return ["fallback:" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()]


def classify_article(article: dict[str, Any]) -> dict[str, str | bool]:
    title = _squash(str(article.get("title", "")))
    summary = _squash(str(article.get("summary", "")))
    lower_title = title.casefold()
    lower_summary = summary.casefold()
    url = str(article.get("url", ""))
    arxiv_id = extract_arxiv_id(article)
    announce_match = re.search(r"Announce Type:\s*([\w-]+)", summary, re.IGNORECASE)
    announce_type = announce_match.group(1).casefold() if announce_match else ""
    is_preprint = bool(arxiv_id) or "researchsquare.com" in urlparse(url).netloc.casefold()

    if is_preprint:
        work_type = "Preprint"
        publication_status = "Preprint"
    elif "perspective" in lower_title or lower_summary.startswith("this perspective"):
        work_type = "Perspective"
        publication_status = "Published"
    elif (
        "review" in lower_title
        or lower_summary.startswith("this review")
        or "systematic review" in lower_summary[:500]
    ):
        work_type = "Review"
        publication_status = "Published"
    else:
        work_type = "Research article"
        publication_status = "Published"

    update_status = "New publication"
    is_update = False
    if is_preprint:
        if "replace" in announce_type:
            update_status = "Revised preprint"
            is_update = True
        elif "cross" in announce_type:
            update_status = "Cross-listed preprint"
        elif announce_type == "new":
            update_status = "New preprint"
        else:
            update_status = "Preprint"
    elif work_type == "Review":
        update_status = "New review"
    elif work_type == "Perspective":
        update_status = "New perspective"

    return {
        "doi": extract_doi(article),
        "arxiv_id": arxiv_id,
        "work_type": work_type,
        "publication_status": publication_status,
        "update_status": update_status,
        "is_update": is_update,
    }


def _article_folders(article: dict[str, Any]) -> list[str]:
    raw = article.get("raw") if isinstance(article.get("raw"), dict) else {}
    folders = raw.get("folders") if isinstance(raw, dict) else None
    values = folders if isinstance(folders, list) else [article.get("folder", "")]
    return sorted(
        {
            _squash(str(value))
            for value in values
            if value and str(value) != "Uncategorized"
        }
    )


def deduplicate_articles(
    articles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    works: OrderedDict[str, dict[str, Any]] = OrderedDict()
    identity_index: dict[str, str] = {}
    for raw_article in articles:
        article = dict(raw_article)
        original_source = _squash(str(article.get("source", "Unknown source")))
        canonical_source = canonical_source_name(original_source)
        article["source"] = canonical_source
        identity_keys = article_identity_keys(article)
        key = next(
            (identity_index[item_key] for item_key in identity_keys if item_key in identity_index),
            identity_keys[0],
        )
        folders = _article_folders(article)
        article_raw = dict(article.get("raw") or {})
        article_raw.update(classify_article(article))
        article_raw.update(
            {
                "canonical_key": key,
                "source_aliases": [original_source],
                "folders": folders,
                "duplicate_count": 1,
            }
        )
        article["raw"] = article_raw
        article["folders"] = folders
        article["folder"] = folders[0] if folders else "Uncategorized"

        existing = works.get(key)
        if existing is None:
            works[key] = article
            for item_key in identity_keys:
                identity_index[item_key] = key
            continue

        existing_raw = dict(existing.get("raw") or {})
        aliases = set(existing_raw.get("source_aliases") or [])
        aliases.add(original_source)
        merged_folders = sorted(set(existing.get("folders") or []) | set(folders))
        duplicate_count = int(existing_raw.get("duplicate_count", 1)) + 1

        existing_rank = (
            float(existing.get("summary_quality", 0.0)),
            len(str(existing.get("summary", ""))),
        )
        incoming_rank = (
            float(article.get("summary_quality", 0.0)),
            len(str(article.get("summary", ""))),
        )
        if incoming_rank > existing_rank:
            stable_id = existing["id"]
            imported_at = existing.get("imported_at")
            existing.update(article)
            existing["id"] = stable_id
            if imported_at:
                existing["imported_at"] = imported_at
            merged_raw = dict(article_raw)
        else:
            merged_raw = existing_raw
        merged_raw["source_aliases"] = sorted(aliases)
        merged_raw["folders"] = merged_folders
        merged_raw["duplicate_count"] = duplicate_count
        existing["raw"] = merged_raw
        existing["folders"] = merged_folders
        existing["folder"] = merged_folders[0] if merged_folders else "Uncategorized"
        for item_key in identity_keys:
            identity_index[item_key] = key

    unique = list(works.values())
    return unique, {
        "received_count": len(articles),
        "unique_count": len(unique),
        "duplicate_count": len(articles) - len(unique),
        "missing_summary_count": sum(not item.get("summary") for item in unique),
        "thin_summary_count": sum(
            0 < len(str(item.get("summary", ""))) < 200 for item in unique
        ),
    }


def evidence_is_grounded(evidence: str, summary: str) -> bool:
    needle = _squash(evidence).casefold()
    haystack = _squash(summary).casefold()
    return bool(needle and len(needle) >= 20 and needle in haystack)


_SUPPORT_STOP_WORDS = {
    "about", "after", "against", "also", "among", "because", "before", "being",
    "between", "could", "during", "effect", "from", "into", "more", "paper",
    "report", "reported", "response", "result", "results", "show", "shown",
    "study", "than", "that", "their", "there", "these", "this", "through",
    "using", "were", "which", "while", "with", "would",
}


def _support_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", value.casefold()):
        token = raw.strip("-")
        if token in _SUPPORT_STOP_WORDS or len(token) < 4:
            continue
        tokens.add(token)
        for suffix in ("ingly", "ation", "ions", "ing", "ied", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                tokens.add(token[: -len(suffix)])
                break
    return tokens


def claim_is_supported(claim: str, evidence: str, summary: str) -> bool:
    """Require a verbatim quote plus lexical entailment signals for a factual claim.

    This deliberately conservative check does not claim to solve scientific entailment;
    it prevents an unrelated real quote from being attached to an arbitrary claim.
    """
    if not evidence_is_grounded(evidence, summary):
        return False
    normalized_claim = _squash(claim).casefold()
    normalized_evidence = _squash(evidence).casefold()
    if not normalized_claim:
        return False
    if normalized_claim in normalized_evidence:
        return True
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", normalized_claim))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", normalized_evidence))
    if claim_numbers and not claim_numbers.issubset(evidence_numbers):
        return False
    claim_tokens = _support_tokens(normalized_claim)
    evidence_tokens = _support_tokens(normalized_evidence)
    if not claim_tokens:
        return False
    overlap = len(claim_tokens & evidence_tokens)
    required = 1 if len(claim_tokens) <= 4 else 2
    return overlap >= required or overlap / len(claim_tokens) >= 0.3


def source_abstract(article: dict[str, Any]) -> tuple[str, str]:
    """Return only publisher/feed abstract text suitable for factual analysis.

    Inoreader-generated summaries and unverified feed excerpts remain useful for
    ranking but are never promoted to source evidence. Only abstracts confirmed by a
    publisher page, a scholarly metadata service, arXiv, or demo fixtures qualify.
    """
    summary = str(article.get("summary", "")).strip()
    raw = article.get("raw") if isinstance(article.get("raw"), dict) else {}
    provenance = str(raw.get("summary_source") or "")
    if provenance in {
        "publisher_page_abstract",
        "publisher_browser_abstract",
        "crossref_abstract",
        "openalex_abstract",
        "arxiv_feed_abstract",
        "demo_abstract",
    }:
        return summary, provenance
    return "", provenance or "unverified_legacy_excerpt"


def first_evidence(summary: str, limit: int = 360) -> str:
    text = _squash(summary)
    if not text:
        return "No abstract or feed summary was available."
    candidate = text[:limit]
    if len(text) > limit and " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0]
    return candidate
