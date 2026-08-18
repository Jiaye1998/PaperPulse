from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .abstract_enrichment import enrich_articles_with_public_abstracts
from .article_processing import deduplicate_articles, source_abstract
from .browser_abstracts import open_verification_browser
from .config import config
from .crypto import encrypt_bytes
from .db import (
    add_refresh_cost,
    archive_recommendations,
    complete_refresh_run,
    create_refresh_run,
    feedback_counts,
    feedback_recommendations,
    get_articles_by_ids,
    get_idea_lab,
    get_profile,
    get_setting,
    get_settings,
    init_db,
    idea_lab_context,
    latest_dashboard,
    list_articles,
    purge_demo_data,
    refresh_history,
    save_profile,
    save_idea_lab,
    save_recommendations,
    set_feedback,
    set_setting,
    source_catalog,
    update_profile_data,
    upsert_articles,
)
from .demo_data import DEMO_PROFILE, ensure_demo_data
from .inoreader import (
    InoreaderConfigurationError,
    authorization_url,
    connected,
    exchange_code,
    fetch_unread,
    oauth_configured,
)
from .models import (
    BrowserVerificationRequest,
    FeedbackRequest,
    ProfileUpdate,
    ResearchProfile,
    SettingsUpdate,
)
from .idea_lab import generate_idea_lab
from .profile_service import build_research_profile, extract_cv_text
from .ranking import rank_articles


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if config.demo_mode and not connected():
        ensure_demo_data()
    yield


app = FastAPI(
    title="PaperPulse API",
    description="Local research-intelligence backend for PaperPulse.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.frontend_url, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)
refresh_lock = asyncio.Lock()
idea_lab_jobs: set[tuple[int, str]] = set()


def _browser_verification_required() -> list[dict[str, str]]:
    try:
        payload = json.loads(get_setting("browser_verification_required", "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [
        {
            "domain": str(item.get("domain", "")),
            "url": str(item.get("url", "")),
            "reason": str(item.get("reason", "Browser verification required")),
            "affected_articles": str(item.get("affected_articles", "")),
        }
        for item in payload
        if isinstance(item, dict) and item.get("domain") and item.get("url")
    ]


async def _build_idea_lab(
    refresh_id: int,
    article: dict[str, object],
    recommendation: dict[str, object],
    profile: ResearchProfile,
) -> tuple[dict[str, object], float]:
    article_id = str(recommendation["article_id"])
    try:
        lab, cost = await asyncio.to_thread(
            generate_idea_lab, article, recommendation, profile
        )
        status = "complete" if lab.get("critic_status") == "reviewed" else "partial"
        stored = save_idea_lab(
            refresh_id,
            article_id,
            status,
            lab=lab,
            estimated_cost=cost,
        )
        return stored, cost
    except Exception as error:
        stored = save_idea_lab(
            refresh_id,
            article_id,
            "failed",
            error=str(error)[:600],
        )
        return stored, 0.0


async def _build_auto_idea_labs(
    refresh_id: int,
    articles: list[dict[str, object]],
    recommendations: list[dict[str, object]],
    profile: ResearchProfile,
    limit: int = 5,
) -> tuple[int, float]:
    by_id = {str(article["id"]): article for article in articles}
    semaphore = asyncio.Semaphore(2)
    eligible_recommendations = [
        recommendation
        for recommendation in recommendations[:limit]
        if source_abstract(by_id[str(recommendation["article_id"])])[0]
    ]

    async def one(recommendation: dict[str, object]) -> tuple[dict[str, object], float]:
        async with semaphore:
            article = by_id[str(recommendation["article_id"])]
            return await _build_idea_lab(
                refresh_id, article, recommendation, profile
            )

    if not eligible_recommendations:
        return 0, 0.0
    results = await asyncio.gather(*(one(item) for item in eligible_recommendations))
    successful = sum(result[0].get("status") in {"complete", "partial"} for result in results)
    return successful, sum(result[1] for result in results)


def _status() -> dict[str, object]:
    profile = get_profile()
    return {
        "openai_configured": bool(config.openai_api_key),
        "inoreader_oauth_configured": oauth_configured(),
        "inoreader_connected": connected(),
        "inoreader_last_error": get_setting("inoreader_last_error"),
        "profile_configured": bool(profile and profile.get("filename") != "demo-profile"),
        "demo_mode": config.demo_mode,
        "analysis_model": config.analysis_model,
        "embedding_model": config.embedding_model,
        "data_location": str(config.data_dir),
        "local_encryption": True,
        "browser_abstracts": config.browser_abstracts,
        "browser_available": not bool(get_setting("browser_last_error")),
        "browser_last_error": get_setting("browser_last_error"),
        "browser_verification_required": _browser_verification_required(),
    }


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "PaperPulse", "status": _status()}


@app.get("/api/dashboard")
def dashboard() -> dict[str, object]:
    data = latest_dashboard()
    return {
        **data,
        "saved": feedback_recommendations({"save_for_later"}),
        "feedback_history": feedback_recommendations(),
        "feedback_counts": feedback_counts(),
        "archive": archive_recommendations(),
        "history_runs": refresh_history(),
        "source_catalog": source_catalog(),
        "profile": get_profile(),
        "settings": get_settings(),
        "status": _status(),
    }


@app.get("/api/archive")
def archive(query: str = "", run_id: int | None = None) -> dict[str, object]:
    return {
        "recommendations": archive_recommendations(query=query, run_id=run_id),
        "history_runs": refresh_history(),
    }


@app.patch("/api/settings")
def update_settings(update: SettingsUpdate) -> dict[str, object]:
    for key, value in update.model_dump(exclude_none=True).items():
        if key in {"source_preferences", "folder_preferences"}:
            value = {name: preference for name, preference in value.items() if preference != "normal"}
        set_setting(key, value)
    return get_settings()


@app.post("/api/browser-verification")
def browser_verification(request: BrowserVerificationRequest) -> dict[str, object]:
    match = next(
        (
            item
            for item in _browser_verification_required()
            if item["domain"].casefold() == request.domain.casefold()
        ),
        None,
    )
    if not match:
        raise HTTPException(404, "No pending verification was found for this publisher.")
    try:
        open_verification_browser(match["url"])
    except (RuntimeError, ValueError, OSError) as error:
        raise HTTPException(502, str(error)) from error
    return {
        "ok": True,
        "domain": match["domain"],
        "message": (
            "Complete the publisher verification in Chrome, close that PaperPulse "
            "browser window, then refresh again."
        ),
    }


@app.post("/api/profile/cv")
async def upload_cv(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    filename = file.filename or "cv"
    if not filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(400, "Upload a PDF or DOCX CV.")
    payload = await file.read()
    if len(payload) > 10 * 1024 * 1024:
        raise HTTPException(413, "CV files are limited to 10 MB.")
    try:
        text = await asyncio.to_thread(extract_cv_text, filename, payload)
        profile = await asyncio.to_thread(build_research_profile, text)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(502, f"CV analysis failed: {error}") from error
    safe_name = Path(filename).name.replace(" ", "_")
    stored = config.uploads_dir / f"{safe_name}.enc"
    await asyncio.to_thread(stored.write_bytes, encrypt_bytes(payload))
    save_profile(filename, text, profile.model_dump())
    return {"filename": filename, "profile": profile.model_dump()}


@app.put("/api/profile")
def update_profile(update: ProfileUpdate) -> dict[str, object]:
    update_profile_data(update.profile.model_dump())
    return {"profile": update.profile.model_dump()}


@app.get("/api/inoreader/auth/start")
def inoreader_auth_start() -> dict[str, str]:
    try:
        return {"authorization_url": authorization_url()}
    except InoreaderConfigurationError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/inoreader/callback")
async def inoreader_callback(code: str = "", state: str = "", error: str = "") -> RedirectResponse:
    if error:
        set_setting("inoreader_last_error", error)
        query = urlencode({"inoreader": "error", "message": error})
        return RedirectResponse(f"{config.frontend_url}?{query}")
    try:
        await exchange_code(code, state)
        purge_demo_data()
        set_setting("inoreader_last_error", "")
        query = urlencode({"inoreader": "connected"})
    except Exception as exc:
        safe_error = str(exc)
        set_setting("inoreader_last_error", safe_error)
        query = urlencode({"inoreader": "error", "message": safe_error})
    return RedirectResponse(f"{config.frontend_url}?{query}")


@app.post("/api/refresh")
async def refresh() -> dict[str, object]:
    # Checking and then acquiring lets two simultaneous callers both pass the
    # check and queue up, running two full refreshes instead of rejecting one.
    if refresh_lock.locked():
        raise HTTPException(409, "A refresh is already in progress.")
    try:
        await asyncio.wait_for(refresh_lock.acquire(), timeout=0.01)
    except (asyncio.TimeoutError, TimeoutError):
        raise HTTPException(409, "A refresh is already in progress.") from None
    try:
        settings = get_settings()
        refresh_id = create_refresh_run()
        # The demo path never enriches abstracts, so the funnel counters need a
        # value before either branch runs.
        abstract_stats: dict[str, object] = {}
        try:
            profile_payload = get_profile()
            live_connection = connected()
            if live_connection:
                if not profile_payload or profile_payload.get("filename") == "demo-profile":
                    raise InoreaderConfigurationError(
                        "Upload and review a CV before running a live refresh."
                    )
                purge_demo_data()
                raw_incoming, rate = await fetch_unread(settings["first_sync_days"])
                set_setting("inoreader_last_error", "")
                incoming, ingest_stats = deduplicate_articles(raw_incoming)
                cached_articles = get_articles_by_ids(
                    [str(article["id"]) for article in incoming]
                )
                incoming, abstract_stats = await enrich_articles_with_public_abstracts(
                    incoming, cached_articles
                )
                set_setting(
                    "browser_verification_required",
                    abstract_stats.get("verification_required", []),
                )
                set_setting(
                    "browser_last_error",
                    str(abstract_stats.get("browser_error") or ""),
                )
                ingest_stats["missing_summary_count"] = sum(
                    not article.get("summary") for article in incoming
                )
                ingest_stats["thin_summary_count"] = sum(
                    0 < len(str(article.get("summary", ""))) < 200
                    for article in incoming
                )
                upsert_articles(incoming)
                articles = get_articles_by_ids([article["id"] for article in incoming])
                source_note = (
                    f"Inoreader zone 1 usage: {rate.get('usage') or '—'} / "
                    f"{rate.get('limit') or '—'}"
                )
                if abstract_stats.get("title_search_skipped_over_limit"):
                    source_note += (
                        f"; {abstract_stats['title_search_skipped_over_limit']} works "
                        f"passed the title-search limit unqueried"
                    )
                if rate.get("truncated"):
                    source_note += (
                        f"; WARNING: more unread items exist than the "
                        f"{rate.get('scan_limit')}-item scan limit, so the oldest "
                        f"were not considered — mark items read in Inoreader, or "
                        f"narrow the scan window"
                    )
                resolution = ", ".join(
                    f"{label} {abstract_stats.get(key, 0)}"
                    for label, key in (
                        ("cache", "cache_hits"),
                        ("arxiv feed", "arxiv_feed_hits"),
                        ("crossref", "crossref_batch_hits"),
                        ("europepmc", "europepmc_batch_hits"),
                        ("openalex", "openalex_batch_hits"),
                        ("title search", "title_search_hits"),
                        ("publisher page", "page_hits"),
                        ("browser", "browser_complete"),
                    )
                    if abstract_stats.get(key)
                )
                if ingest_stats.get("non_research_count"):
                    breakdown = ", ".join(
                        f"{kind} {count}"
                        for kind, count in sorted(
                            (ingest_stats.get("non_research_breakdown") or {}).items()
                        )
                    )
                    source_note += (
                        f"; excluded {ingest_stats['non_research_count']} non-research "
                        f"entries ({breakdown})"
                    )
                source_note += (
                    f"; public abstracts: {abstract_stats['complete']} complete, "
                    f"{abstract_stats['excerpt']} excerpt-only, "
                    f"{abstract_stats['unavailable']} unavailable"
                    + (f" (resolved by {resolution})" if resolution else "")
                    + f"; verification domains: "
                    f"{len(abstract_stats.get('verification_required', []))}."
                )
            elif config.demo_mode:
                raw_articles = list_articles(limit=1000)
                articles, ingest_stats = deduplicate_articles(raw_articles)
                upsert_articles(articles)
                source_note = "Demo articles used — connect Inoreader for live unread items."
            else:
                raise InoreaderConfigurationError("Connect Inoreader before refreshing.")

            profile = ResearchProfile.model_validate(
                profile_payload["profile"] if profile_payload else DEMO_PROFILE.model_dump()
            )
            recommendations, estimated_cost, rank_note, candidate_count = await asyncio.to_thread(
                rank_articles,
                articles,
                profile,
                settings["top_n"],
                settings["candidate_multiplier"],
                settings["ranking_mode"],
                settings["source_preferences"],
                settings["folder_preferences"],
            )
            save_recommendations(refresh_id, recommendations)
            idea_lab_count, idea_lab_cost = await _build_auto_idea_labs(
                refresh_id,
                articles,
                recommendations,
                profile,
            )
            article_by_id = {str(article["id"]): article for article in articles}
            idea_lab_eligible = sum(
                bool(source_abstract(article_by_id[str(item["article_id"])])[0])
                for item in recommendations[:5]
            )
            estimated_cost += idea_lab_cost
            status = "completed" if live_connection else "demo"
            note = (
                f"{source_note} {rank_note} Deep Idea Labs generated: "
                f"{idea_lab_count}/{idea_lab_eligible} verified-abstract candidates "
                f"within the first five ranks."
            )
            complete_refresh_run(
                refresh_id,
                status,
                ingest_stats["received_count"],
                len(recommendations),
                estimated_cost,
                note,
                unique_count=ingest_stats["unique_count"],
                duplicate_count=ingest_stats["duplicate_count"],
                candidate_count=candidate_count,
                missing_summary_count=ingest_stats["missing_summary_count"],
                thin_summary_count=ingest_stats["thin_summary_count"],
                excluded_count=int(ingest_stats.get("non_research_count", 0)),
                complete_abstract_count=int(abstract_stats.get("complete", 0)),
                excerpt_abstract_count=int(abstract_stats.get("excerpt", 0)),
                idea_lab_count=idea_lab_count,
            )
            return dashboard()
        except Exception as error:
            complete_refresh_run(refresh_id, "failed", 0, 0, note=str(error))
            status_code = 400 if isinstance(error, InoreaderConfigurationError) else 502
            raise HTTPException(status_code, str(error)) from error
    finally:
        refresh_lock.release()


@app.post("/api/articles/{article_id:path}/feedback")
def article_feedback(article_id: str, feedback: FeedbackRequest) -> dict[str, object]:
    if not set_feedback(article_id, feedback.value):
        raise HTTPException(404, "Article not found.")
    return dashboard()


@app.post("/api/refreshes/{refresh_id}/articles/{article_id:path}/idea-lab")
async def article_idea_lab(refresh_id: int, article_id: str) -> dict[str, object]:
    job_key = (refresh_id, article_id)
    existing = get_idea_lab(refresh_id, article_id)
    if (
        existing
        and existing.get("status") in {"complete", "partial"}
        and int(existing.get("version") or 0) >= 3
    ):
        return {"idea_lab": existing}
    if job_key in idea_lab_jobs:
        raise HTTPException(409, "This Idea Lab is already being generated.")
    context = idea_lab_context(refresh_id, article_id)
    if not context:
        raise HTTPException(404, "Recommendation not found in this brief.")
    profile_payload = get_profile()
    if not profile_payload:
        raise HTTPException(400, "Configure a research profile before generating ideas.")
    profile = ResearchProfile.model_validate(profile_payload["profile"])
    idea_lab_jobs.add(job_key)
    save_idea_lab(refresh_id, article_id, "running")
    try:
        lab, cost = await _build_idea_lab(
            refresh_id,
            context,
            context,
            profile,
        )
        add_refresh_cost(refresh_id, cost)
        if lab.get("status") == "failed":
            raise HTTPException(502, str(lab.get("error") or "Idea generation failed."))
        return {"idea_lab": lab}
    finally:
        idea_lab_jobs.discard(job_key)
