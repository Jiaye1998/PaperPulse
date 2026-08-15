from __future__ import annotations

import asyncio
import ipaddress
import re
import shutil
import socket
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .config import config


BROWSER_DOMAIN_CONCURRENCY = 4
ABSTRACT_CONTROL = re.compile(r"^\s*(?:abstract|summary)\s*$", re.IGNORECASE)
CHALLENGE_TEXT = re.compile(
    r"(?:just a moment|verify you are human|verification required|captcha|"
    r"radware captcha|checking your browser|security check|unusual traffic|"
    r"access denied|bot manager|enable javascript and cookies to continue)",
    re.IGNORECASE,
)
_PUBLIC_HOST_CACHE: dict[tuple[str, int], bool] = {}


@dataclass
class BrowserBatchResult:
    candidates: dict[str, Any] = field(default_factory=dict)
    attempted: int = 0
    available: bool = False
    challenges: list[dict[str, str]] = field(default_factory=list)
    error: str = ""


def _friendly_browser_error(error: Exception) -> str:
    message = str(error).casefold()
    if "executable doesn't exist" in message or "chrome" in message and "not found" in message:
        return "Google Chrome is unavailable; public metadata fallbacks were used."
    if "processsingleton" in message or "user data directory is already in use" in message:
        return (
            "Close the PaperPulse verification Chrome window, then refresh again; "
            "public metadata fallbacks were used this time."
        )
    return "The publisher browser could not start; public metadata fallbacks were used."


def _host_is_public(hostname: str, port: int) -> bool:
    key = (hostname, port)
    if key in _PUBLIC_HOST_CACHE:
        return _PUBLIC_HOST_CACHE[key]
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        allowed = bool(addresses) and all(
            ipaddress.ip_address(address[4][0]).is_global for address in addresses
        )
    except (socket.gaierror, ValueError):
        allowed = False
    _PUBLIC_HOST_CACHE[key] = allowed
    return allowed


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
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return await asyncio.to_thread(_host_is_public, hostname, port)
    return address.is_global


async def _challenge_detected(page: Any, html_text: str) -> bool:
    title = ""
    try:
        title = await page.title()
    except Exception:
        pass
    visible_prefix = ""
    try:
        visible_prefix = (await page.locator("body").inner_text(timeout=1_500))[:2_000]
    except Exception:
        pass
    if CHALLENGE_TEXT.search(f"{title}\n{visible_prefix}\n{html_text[:4_000]}"):
        return True
    for selector in (
        "iframe[src*='captcha' i]",
        "[id*='captcha' i]",
        "[class*='captcha' i]",
        "input[name='cf-turnstile-response']",
        "iframe[src*='challenge' i]",
    ):
        try:
            if await page.locator(selector).count():
                return True
        except Exception:
            continue
    return False


async def _try_abstract_control(page: Any) -> None:
    for role in ("button", "link", "tab"):
        try:
            locator = page.get_by_role(role, name=ABSTRACT_CONTROL)
            if await locator.count():
                await locator.first.click(timeout=2_500)
                await page.wait_for_timeout(600)
                return
        except Exception:
            continue


async def resolve_with_persistent_browser(
    articles: list[dict[str, Any]],
    extractor: Callable[[str, str, str], Any],
) -> BrowserBatchResult:
    """Render publisher pages in Chrome and extract only explicit public abstracts.

    A verification page pauses the rest of that publisher domain. PaperPulse never
    attempts to solve a CAPTCHA; the saved Chrome profile can be opened visibly by a
    user through the verification endpoint.
    """
    result = BrowserBatchResult()
    if not config.browser_abstracts or not articles:
        return result
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ImportError:
        result.error = "Playwright is not installed."
        return result

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        article_url = str(article.get("url", ""))
        parsed = urlparse(article_url)
        if parsed.hostname and await _url_is_public(article_url):
            groups[parsed.hostname.casefold()].append(article)

    try:
        async with async_playwright() as playwright:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(config.browser_profile_dir),
                channel="chrome",
                headless=config.browser_headless,
                accept_downloads=False,
                locale="en-US",
            )
            result.available = True
            domain_limit = asyncio.Semaphore(BROWSER_DOMAIN_CONCURRENCY)

            async def guard_navigation(route: Any, request: Any) -> None:
                if request.is_navigation_request() and not await _url_is_public(
                    request.url
                ):
                    await route.abort()
                    return
                await route.continue_()

            await context.route("**/*", guard_navigation)

            async def process_domain(
                domain: str, domain_articles: list[dict[str, Any]]
            ) -> None:
                async with domain_limit:
                    page = await context.new_page()
                    page.set_default_timeout(3_000)
                    try:
                        for index, article in enumerate(domain_articles):
                            result.attempted += 1
                            article_url = str(article.get("url", ""))
                            try:
                                await page.goto(
                                    article_url,
                                    wait_until="domcontentloaded",
                                    timeout=config.browser_timeout_ms,
                                )
                            except PlaywrightTimeoutError:
                                # A page can expose usable metadata before all trackers finish.
                                pass
                            except Exception:
                                continue
                            await page.wait_for_timeout(500)
                            try:
                                html_text = await page.content()
                            except Exception:
                                continue
                            if await _challenge_detected(page, html_text):
                                result.challenges.append(
                                    {
                                        "domain": domain,
                                        "url": article_url,
                                        "reason": "Browser verification required",
                                        "affected_articles": str(
                                            len(domain_articles) - index
                                        ),
                                    }
                                )
                                break

                            final_url = page.url
                            candidate = extractor(
                                html_text, str(article.get("title", "")), final_url
                            )
                            if candidate is None or not bool(
                                getattr(candidate, "complete", False)
                            ):
                                await _try_abstract_control(page)
                                try:
                                    expanded_html = await page.content()
                                except Exception:
                                    expanded_html = html_text
                                expanded = extractor(
                                    expanded_html,
                                    str(article.get("title", "")),
                                    page.url,
                                )
                                if expanded is not None and (
                                    candidate is None
                                    or (
                                        bool(getattr(expanded, "complete", False)),
                                        int(getattr(expanded, "priority", 0)),
                                        len(str(getattr(expanded, "text", ""))),
                                    )
                                    > (
                                        bool(getattr(candidate, "complete", False)),
                                        int(getattr(candidate, "priority", 0)),
                                        len(str(getattr(candidate, "text", ""))),
                                    )
                                ):
                                    candidate = expanded
                            if candidate is not None:
                                candidate = candidate.__class__(
                                    text=candidate.text,
                                    provenance=(
                                        "publisher_browser_abstract"
                                        if candidate.complete
                                        else "publisher_browser_excerpt"
                                    ),
                                    complete=candidate.complete,
                                    source_url=page.url,
                                    priority=max(110, int(candidate.priority)),
                                    doi=getattr(candidate, "doi", ""),
                                )
                                result.candidates[str(article["id"])] = candidate
                    finally:
                        await page.close()

            try:
                await asyncio.gather(
                    *(process_domain(domain, items) for domain, items in groups.items())
                )
            finally:
                await context.close()
    except Exception as error:
        result.error = _friendly_browser_error(error)
    return result


def _chrome_executable() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home()
        / "AppData"
        / "Local"
        / "Google"
        / "Chrome"
        / "Application"
        / "chrome.exe",
    ]
    command = shutil.which("chrome") or shutil.which("chrome.exe")
    if command:
        candidates.insert(0, Path(command))
    return next((candidate for candidate in candidates if candidate.exists()), None)


def open_verification_browser(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("The stored publisher verification URL is invalid.")
    executable = _chrome_executable()
    if not executable:
        raise RuntimeError("Google Chrome was not found on this computer.")
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    creation_flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen(
        [
            str(executable),
            f"--user-data-dir={config.browser_profile_dir}",
            "--profile-directory=Default",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-mode",
            "--new-window",
            url,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creation_flags,
    )
