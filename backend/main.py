from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .config import config
from .crypto import encrypt_bytes
from .db import (
    archive_recommendations,
    complete_refresh_run,
    create_refresh_run,
    feedback_counts,
    feedback_recommendations,
    get_articles_by_ids,
    get_profile,
    get_setting,
    get_settings,
    init_db,
    latest_dashboard,
    list_articles,
    purge_demo_data,
    refresh_history,
    save_profile,
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
from .models import FeedbackRequest, ProfileUpdate, ResearchProfile, SettingsUpdate
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
    if refresh_lock.locked():
        raise HTTPException(409, "A refresh is already in progress.")

    async with refresh_lock:
        settings = get_settings()
        refresh_id = create_refresh_run()
        try:
            profile_payload = get_profile()
            live_connection = connected()
            if live_connection:
                if not profile_payload or profile_payload.get("filename") == "demo-profile":
                    raise InoreaderConfigurationError(
                        "Upload and review a CV before running a live refresh."
                    )
                purge_demo_data()
                incoming, rate = await fetch_unread(settings["first_sync_days"])
                upsert_articles(incoming)
                articles = get_articles_by_ids([article["id"] for article in incoming])
                source_note = (
                    f"Inoreader zone 1 usage: {rate.get('usage') or '—'} / "
                    f"{rate.get('limit') or '—'}"
                )
            elif config.demo_mode:
                articles = list_articles(limit=1000)
                source_note = "Demo articles used — connect Inoreader for live unread items."
            else:
                raise InoreaderConfigurationError("Connect Inoreader before refreshing.")

            profile = ResearchProfile.model_validate(
                profile_payload["profile"] if profile_payload else DEMO_PROFILE.model_dump()
            )
            recommendations, estimated_cost, rank_note = await asyncio.to_thread(
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
            status = "completed" if live_connection else "demo"
            note = f"{source_note} {rank_note}"
            complete_refresh_run(
                refresh_id,
                status,
                len(articles),
                len(recommendations),
                estimated_cost,
                note,
            )
            return dashboard()
        except Exception as error:
            complete_refresh_run(refresh_id, "failed", 0, 0, note=str(error))
            status_code = 400 if isinstance(error, InoreaderConfigurationError) else 502
            raise HTTPException(status_code, str(error)) from error


@app.post("/api/articles/{article_id:path}/feedback")
def article_feedback(article_id: str, feedback: FeedbackRequest) -> dict[str, object]:
    if not set_feedback(article_id, feedback.value):
        raise HTTPException(404, "Article not found.")
    return dashboard()
