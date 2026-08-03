from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote, urlencode, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import config
from .db import get_oauth_token, get_setting, save_oauth_token, set_setting


AUTH_ENDPOINT = "https://www.inoreader.com/oauth2/auth"
TOKEN_ENDPOINT = "https://www.inoreader.com/oauth2/token"
API_ROOT = "https://www.inoreader.com/reader/api/0"
READING_LIST = "user/-/state/com.google/reading-list"
READ_STATE = "user/-/state/com.google/read"


class InoreaderConfigurationError(RuntimeError):
    pass


def _oauth_error(response: httpx.Response, action: str) -> InoreaderConfigurationError:
    detail = "unknown response"
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(
                payload.get("error_description")
                or payload.get("error")
                or payload.get("message")
                or detail
            )
    except ValueError:
        if response.text:
            detail = response.text[:240]
    return InoreaderConfigurationError(
        f"Inoreader {action} failed (HTTP {response.status_code}): {detail}"
    )


def oauth_configured() -> bool:
    return bool(config.inoreader_client_id and config.inoreader_client_secret)


def connected() -> bool:
    return get_oauth_token("inoreader") is not None


def authorization_url() -> str:
    if not oauth_configured():
        raise InoreaderConfigurationError(
            "Add INOREADER_CLIENT_ID and INOREADER_CLIENT_SECRET to .env first."
        )
    state = get_setting("oauth_state")
    try:
        state_age = time.time() - float(get_setting("oauth_state_created_at", "0"))
    except ValueError:
        state_age = 601
    if not state or state_age > 600:
        state = secrets.token_urlsafe(32)
        set_setting("oauth_state", state)
        set_setting("oauth_state_created_at", time.time())
    set_setting("inoreader_last_error", "")
    params = {
        "client_id": config.inoreader_client_id,
        "redirect_uri": config.inoreader_redirect_uri,
        "response_type": "code",
        "scope": "read",
        "state": state,
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


async def exchange_code(code: str, state: str) -> None:
    if not code:
        raise ValueError("Inoreader did not return an authorization code.")
    expected = get_setting("oauth_state")
    if not expected or not secrets.compare_digest(expected, state):
        raise ValueError("The Inoreader authorization state did not match.")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_ENDPOINT,
            data={
                "code": code,
                "redirect_uri": config.inoreader_redirect_uri,
                "client_id": config.inoreader_client_id,
                "client_secret": config.inoreader_client_secret,
                "scope": "",
                "grant_type": "authorization_code",
            },
            headers={"User-Agent": "PaperPulse/0.1"},
        )
        if response.is_error:
            raise _oauth_error(response, "token exchange")
        token = response.json()
    save_oauth_token(
        "inoreader",
        token["access_token"],
        token.get("refresh_token"),
        time.time() + float(token.get("expires_in", 3600)),
        token.get("scope", "read"),
    )
    set_setting("oauth_state", "")
    set_setting("oauth_state_created_at", "0")


async def _valid_token() -> str:
    token = get_oauth_token("inoreader")
    if not token:
        raise InoreaderConfigurationError("Inoreader is not connected.")
    if token.get("expires_at") and float(token["expires_at"]) > time.time() + 60:
        return str(token["access_token"])
    if not token.get("refresh_token"):
        raise InoreaderConfigurationError("Reconnect Inoreader to refresh access.")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_ENDPOINT,
            data={
                "client_id": config.inoreader_client_id,
                "client_secret": config.inoreader_client_secret,
                "grant_type": "refresh_token",
                "refresh_token": token["refresh_token"],
            },
            headers={"User-Agent": "PaperPulse/0.1"},
        )
        if response.is_error:
            raise _oauth_error(response, "token refresh")
        refreshed = response.json()
    save_oauth_token(
        "inoreader",
        refreshed["access_token"],
        refreshed.get("refresh_token") or token.get("refresh_token"),
        time.time() + float(refreshed.get("expires_in", 3600)),
        refreshed.get("scope", token.get("scope", "read")),
    )
    return str(refreshed["access_token"])


def _plain_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for element in soup(["script", "style", "img", "svg"]):
        element.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def _safe_http_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    parsed = urlparse(candidate)
    return candidate if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _article_from_item(item: dict[str, Any]) -> dict[str, Any]:
    intelligence = item.get("summaries") or []
    intelligence_text = (
        _plain_text(intelligence[0].get("summary", ""))
        if intelligence and isinstance(intelligence[0], dict)
        else ""
    )
    feed_text = _plain_text((item.get("summary") or {}).get("content", ""))
    summary = intelligence_text or feed_text
    quality = 1.0 if intelligence_text else min(0.95, max(0.2, len(summary) / 900))
    links: list[Any] = []
    for key in ("canonical", "alternate"):
        value = item.get(key)
        if isinstance(value, list):
            links.extend(value)
    article_url = next(
        (
            safe
            for link in links
            if isinstance(link, dict)
            for safe in [_safe_http_url(link.get("href"))]
            if safe
        ),
        "",
    )
    origin_value = item.get("origin")
    origin = origin_value if isinstance(origin_value, dict) else {}
    categories = item.get("categories") if isinstance(item.get("categories"), list) else []
    folder_names = [
        category.split("/label/", 1)[1]
        for category in categories
        if isinstance(category, str) and "/label/" in category
    ]
    try:
        timestamp = int(item.get("published") or item.get("updated") or time.time())
        published = datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
    except (OverflowError, TypeError, ValueError):
        published = datetime.now(UTC).isoformat()
    return {
        "id": str(item["id"]),
        "title": _plain_text(item.get("title", "Untitled article")),
        "summary": summary,
        "source": origin.get("title", "Unknown source"),
        "source_url": _safe_http_url(origin.get("htmlUrl", "")),
        "url": article_url,
        "author": item.get("author", ""),
        "published_at": published,
        "folder": folder_names[0] if folder_names else "Uncategorized",
        "summary_quality": quality,
        "raw": {"timestampUsec": item.get("timestampUsec"), "categories": categories},
    }


async def fetch_unread(unread_window_days: int = 7, max_items: int = 1000) -> tuple[list[dict[str, Any]], dict[str, str]]:
    access_token = await _valid_token()
    start = datetime.now(UTC) - timedelta(days=unread_window_days)
    stream_id = quote(READING_LIST, safe="")
    endpoint = f"{API_ROOT}/stream/contents/{stream_id}"
    params: dict[str, str | int] = {
        "n": 100,
        "ot": int(start.timestamp()),
        "xt": READ_STATE,
        "output": "json",
        "summaries": "1",
        "includeAllDirectStreamIds": "false",
    }
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": "PaperPulse/0.1"}
    articles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rate: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=45) as client:
        while len(articles) < max_items:
            response = await client.get(endpoint, params=params, headers=headers)
            if response.is_error:
                raise _oauth_error(response, "unread feed request")
            payload = response.json()
            for item in payload.get("items", []):
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                article_id = str(item["id"])
                if article_id in seen_ids:
                    continue
                seen_ids.add(article_id)
                articles.append(_article_from_item(item))
            rate = {
                "limit": response.headers.get("X-Reader-Zone1-Limit", ""),
                "usage": response.headers.get("X-Reader-Zone1-Usage", ""),
                "reset_after": response.headers.get("X-Reader-Limits-Reset-After", ""),
            }
            continuation = payload.get("continuation")
            if not continuation or len(articles) >= max_items:
                break
            params["c"] = continuation
    return articles[:max_items], rate
