from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .config import config
from .crypto import decrypt_text, encrypt_bytes, encrypt_text


ARTICLE_ENCRYPTED_FIELDS = {
    "title", "summary", "source", "source_url", "url", "author", "folder",
    "raw_json", "embedding_json",
}
RECOMMENDATION_ENCRYPTED_FIELDS = {
    "reason", "core_finding", "innovation", "connection", "idea", "labels_json",
}
PROFILE_ENCRYPTED_FIELDS = {"filename", "original_text", "profile_json"}
TOKEN_ENCRYPTED_FIELDS = {"access_token", "refresh_token", "scope"}
FEEDBACK_ENCRYPTED_FIELDS = {"value"}
SECURE_SETTINGS = {"source_preferences", "folder_preferences", "oauth_state"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(config.database_path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    try:
        yield db
        db.commit()
    finally:
        db.close()


def init_db() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS research_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            filename TEXT,
            original_text TEXT NOT NULL DEFAULT '',
            profile_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS oauth_tokens (
            provider TEXT PRIMARY KEY,
            access_token TEXT NOT NULL,
            refresh_token TEXT,
            expires_at REAL,
            scope TEXT,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            source_url TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            author TEXT NOT NULL DEFAULT '',
            published_at TEXT NOT NULL,
            folder TEXT NOT NULL DEFAULT '',
            summary_quality REAL NOT NULL DEFAULT 0.5,
            raw_json TEXT NOT NULL DEFAULT '{}',
            embedding_json TEXT,
            embedding_model TEXT,
            imported_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS refresh_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            scanned_count INTEGER NOT NULL DEFAULT 0,
            selected_count INTEGER NOT NULL DEFAULT 0,
            estimated_cost REAL NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            refresh_id INTEGER NOT NULL,
            article_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            relevance_score REAL NOT NULL,
            novelty_score REAL NOT NULL,
            inspiration_score REAL NOT NULL,
            confidence REAL NOT NULL,
            reason TEXT NOT NULL,
            core_finding TEXT NOT NULL,
            innovation TEXT NOT NULL,
            connection TEXT NOT NULL,
            idea TEXT NOT NULL,
            idea_is_speculative INTEGER NOT NULL DEFAULT 1,
            labels_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            PRIMARY KEY (refresh_id, article_id),
            FOREIGN KEY (refresh_id) REFERENCES refresh_runs(id),
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS feedback (
            article_id TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS articles_published_idx ON articles(published_at DESC)",
        "CREATE INDEX IF NOT EXISTS recs_refresh_rank_idx ON recommendations(refresh_id, rank)",
    ]
    with connection() as db:
        for statement in statements:
            db.execute(statement)
        article_columns = {
            str(row["name"]) for row in db.execute("PRAGMA table_info(articles)").fetchall()
        }
        if "embedding_model" not in article_columns:
            db.execute("ALTER TABLE articles ADD COLUMN embedding_model TEXT")
        defaults = {
            "top_n": "20",
            "first_sync_days": "7",
            "candidate_multiplier": "2",
            "last_successful_sync": "",
            "oauth_state": "",
            "oauth_state_created_at": "0",
            "ranking_mode": "balanced",
            "source_preferences": "{}",
            "folder_preferences": "{}",
        }
        for key, value in defaults.items():
            db.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                (key, value),
            )
        _migrate_encrypted_columns(db)
    _migrate_uploaded_files()


def _migrate_encrypted_columns(db: sqlite3.Connection) -> None:
    for table, fields in (
        ("research_profile", PROFILE_ENCRYPTED_FIELDS),
        ("oauth_tokens", TOKEN_ENCRYPTED_FIELDS),
        ("articles", ARTICLE_ENCRYPTED_FIELDS),
        ("recommendations", RECOMMENDATION_ENCRYPTED_FIELDS),
        ("feedback", FEEDBACK_ENCRYPTED_FIELDS),
    ):
        columns = ", ".join(fields)
        for row in db.execute(f"SELECT rowid AS _rowid, {columns} FROM {table}").fetchall():
            updates = {
                field: encrypt_text(row[field])
                for field in fields
                if row[field] is not None and not str(row[field]).startswith("enc:v1:")
            }
            if updates:
                assignments = ", ".join(f"{field} = ?" for field in updates)
                db.execute(
                    f"UPDATE {table} SET {assignments} WHERE rowid = ?",
                    [*updates.values(), row["_rowid"]],
                )
    for key in SECURE_SETTINGS:
        row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row and not str(row["value"]).startswith("enc:v1:"):
            db.execute(
                "UPDATE settings SET value = ? WHERE key = ?",
                (encrypt_text(str(row["value"])), key),
            )


def _migrate_uploaded_files() -> None:
    uploads_dir = getattr(config, "uploads_dir", None)
    if not uploads_dir:
        return
    for path in Path(uploads_dir).iterdir():
        if not path.is_file() or path.suffix == ".enc":
            continue
        encrypted_path = path.with_name(path.name + ".enc")
        payload = path.read_bytes()
        encrypted = encrypt_bytes(payload)
        # Verify a complete encrypted token before removing the plaintext original.
        from .crypto import decrypt_bytes

        if decrypt_bytes(encrypted) != payload:
            raise RuntimeError(f"Could not verify encrypted CV copy for {path.name}.")
        encrypted_path.write_bytes(encrypted)
        path.unlink()


def get_setting(key: str, default: str = "") -> str:
    with connection() as db:
        row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    value = str(row["value"])
    return str(decrypt_text(value)) if key in SECURE_SETTINGS else value


def set_setting(key: str, value: Any) -> None:
    stored = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    if key in SECURE_SETTINGS:
        stored = str(encrypt_text(stored))
    with connection() as db:
        db.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, stored),
        )


def get_settings() -> dict[str, Any]:
    return {
        "top_n": int(get_setting("top_n", "20")),
        "first_sync_days": int(get_setting("first_sync_days", "7")),
        "candidate_multiplier": int(get_setting("candidate_multiplier", "2")),
        "ranking_mode": get_setting("ranking_mode", "balanced"),
        "source_preferences": json.loads(get_setting("source_preferences", "{}")),
        "folder_preferences": json.loads(get_setting("folder_preferences", "{}")),
    }


def save_profile(filename: str, original_text: str, profile: dict[str, Any]) -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO research_profile(id, filename, original_text, profile_json, updated_at)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename = excluded.filename,
                original_text = excluded.original_text,
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (
                encrypt_text(filename),
                encrypt_text(original_text),
                encrypt_text(json.dumps(profile)),
                utc_now(),
            ),
        )


def update_profile_data(profile: dict[str, Any]) -> None:
    """Update the editable profile without discarding the extracted CV text."""
    with connection() as db:
        existing = db.execute(
            "SELECT filename FROM research_profile WHERE id = 1"
        ).fetchone()
        if existing:
            db.execute(
                "UPDATE research_profile SET profile_json = ?, updated_at = ? WHERE id = 1",
                (encrypt_text(json.dumps(profile)), utc_now()),
            )
        else:
            db.execute(
                """
                INSERT INTO research_profile(id, filename, original_text, profile_json, updated_at)
                VALUES (1, ?, '', ?, ?)
                """,
                (
                    encrypt_text("manual-profile"),
                    encrypt_text(json.dumps(profile)),
                    utc_now(),
                ),
            )


def get_profile() -> dict[str, Any] | None:
    with connection() as db:
        row = db.execute("SELECT * FROM research_profile WHERE id = 1").fetchone()
    if not row:
        return None
    return {
        "filename": decrypt_text(row["filename"]),
        "profile": json.loads(str(decrypt_text(row["profile_json"]))),
        "updated_at": row["updated_at"],
    }


def save_oauth_token(
    provider: str,
    access_token: str,
    refresh_token: str | None,
    expires_at: float | None,
    scope: str | None,
) -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO oauth_tokens(provider, access_token, refresh_token, expires_at, scope, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                access_token = excluded.access_token,
                refresh_token = COALESCE(excluded.refresh_token, oauth_tokens.refresh_token),
                expires_at = excluded.expires_at,
                scope = excluded.scope,
                updated_at = excluded.updated_at
            """,
            (
                provider,
                encrypt_text(access_token),
                encrypt_text(refresh_token),
                expires_at,
                encrypt_text(scope),
                utc_now(),
            ),
        )


def get_oauth_token(provider: str) -> dict[str, Any] | None:
    with connection() as db:
        row = db.execute(
            "SELECT * FROM oauth_tokens WHERE provider = ?", (provider,)
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    for field in TOKEN_ENCRYPTED_FIELDS:
        item[field] = decrypt_text(item.get(field))
    return item


def upsert_articles(articles: list[dict[str, Any]]) -> int:
    if not articles:
        return 0
    with connection() as db:
        for article in articles:
            db.execute(
                """
                INSERT INTO articles(
                    id, title, summary, source, source_url, url, author,
                    published_at, folder, summary_quality, raw_json, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    summary = excluded.summary,
                    source = excluded.source,
                    source_url = excluded.source_url,
                    url = excluded.url,
                    author = excluded.author,
                    published_at = excluded.published_at,
                    folder = excluded.folder,
                    summary_quality = excluded.summary_quality,
                    raw_json = excluded.raw_json
                """,
                (
                    article["id"],
                    encrypt_text(article["title"]),
                    encrypt_text(article.get("summary", "")),
                    encrypt_text(article.get("source", "")),
                    encrypt_text(article.get("source_url", "")),
                    encrypt_text(article.get("url", "")),
                    encrypt_text(article.get("author", "")),
                    article["published_at"],
                    encrypt_text(article.get("folder", "")),
                    article.get("summary_quality", 0.5),
                    encrypt_text(json.dumps(article.get("raw", {}))),
                    utc_now(),
                ),
            )
    return len(articles)


def list_articles(limit: int = 1000) -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            "SELECT * FROM articles ORDER BY published_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_article_row(row) for row in rows]


def get_articles_by_ids(article_ids: list[str]) -> list[dict[str, Any]]:
    if not article_ids:
        return []
    unique_ids = list(dict.fromkeys(article_ids))
    placeholders = ",".join("?" for _ in unique_ids)
    with connection() as db:
        rows = db.execute(
            f"SELECT * FROM articles WHERE id IN ({placeholders})",
            unique_ids,
        ).fetchall()
    by_id = {str(row["id"]): _article_row(row) for row in rows}
    return [by_id[article_id] for article_id in unique_ids if article_id in by_id]


def _article_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for field in ARTICLE_ENCRYPTED_FIELDS:
        item[field] = decrypt_text(item.get(field))
    return item


def purge_demo_data() -> dict[str, int]:
    """Remove seeded demo articles and every refresh run contaminated by them."""
    with connection() as db:
        contaminated_rows = db.execute(
            "SELECT DISTINCT refresh_id FROM recommendations WHERE article_id LIKE ?",
            ("demo-%",),
        ).fetchall()
        contaminated_ids = [int(row["refresh_id"]) for row in contaminated_rows]

        deleted_recommendations = 0
        deleted_runs = 0
        if contaminated_ids:
            placeholders = ",".join("?" for _ in contaminated_ids)
            cursor = db.execute(
                f"DELETE FROM recommendations WHERE refresh_id IN ({placeholders})",
                contaminated_ids,
            )
            deleted_recommendations = cursor.rowcount
            cursor = db.execute(
                f"DELETE FROM refresh_runs WHERE id IN ({placeholders})",
                contaminated_ids,
            )
            deleted_runs = cursor.rowcount

        db.execute("DELETE FROM feedback WHERE article_id LIKE ?", ("demo-%",))
        cursor = db.execute("DELETE FROM articles WHERE id LIKE ?", ("demo-%",))
        return {
            "articles": cursor.rowcount,
            "recommendations": deleted_recommendations,
            "runs": deleted_runs,
        }


def set_article_embedding(article_id: str, embedding: list[float]) -> None:
    with connection() as db:
        db.execute(
            "UPDATE articles SET embedding_json = ?, embedding_model = ? WHERE id = ?",
            (encrypt_text(json.dumps(embedding)), config.embedding_model, article_id),
        )


def create_refresh_run(status: str = "running") -> int:
    with connection() as db:
        cursor = db.execute(
            "INSERT INTO refresh_runs(started_at, status) VALUES (?, ?)",
            (utc_now(), status),
        )
        return int(cursor.lastrowid)


def complete_refresh_run(
    refresh_id: int,
    status: str,
    scanned_count: int,
    selected_count: int,
    estimated_cost: float = 0,
    note: str = "",
) -> None:
    with connection() as db:
        db.execute(
            """
            UPDATE refresh_runs SET completed_at = ?, status = ?, scanned_count = ?,
                selected_count = ?, estimated_cost = ?, note = ? WHERE id = ?
            """,
            (
                utc_now(),
                status,
                scanned_count,
                selected_count,
                estimated_cost,
                note,
                refresh_id,
            ),
        )


def save_recommendations(
    refresh_id: int, recommendations: list[dict[str, Any]]
) -> None:
    with connection() as db:
        for rank, rec in enumerate(recommendations, start=1):
            db.execute(
                """
                INSERT OR REPLACE INTO recommendations(
                    refresh_id, article_id, rank, relevance_score, novelty_score,
                    inspiration_score, confidence, reason, core_finding, innovation,
                    connection, idea, idea_is_speculative, labels_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    refresh_id,
                    rec["article_id"],
                    rank,
                    rec["relevance_score"],
                    rec["novelty_score"],
                    rec["inspiration_score"],
                    rec["confidence"],
                    encrypt_text(rec["reason"]),
                    encrypt_text(rec["core_finding"]),
                    encrypt_text(rec["innovation"]),
                    encrypt_text(rec["connection"]),
                    encrypt_text(rec["idea"]),
                    int(rec.get("idea_is_speculative", True)),
                    encrypt_text(json.dumps(rec.get("labels", []))),
                    utc_now(),
                ),
            )


def latest_dashboard() -> dict[str, Any]:
    with connection() as db:
        run = db.execute(
            "SELECT * FROM refresh_runs WHERE status IN ('completed', 'demo') "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return {"run": None, "recommendations": []}
        rows = db.execute(
            """
            SELECT r.*, a.title, a.summary, a.source, a.source_url, a.url,
                   a.author, a.published_at, a.folder, a.summary_quality,
                   f.value AS feedback
            FROM recommendations r
            JOIN articles a ON a.id = r.article_id
            LEFT JOIN feedback f ON f.article_id = a.id
            WHERE r.refresh_id = ?
            ORDER BY r.rank ASC
            """,
            (run["id"],),
        ).fetchall()
    recommendations = [_recommendation_row(row) for row in rows]
    return {"run": dict(run), "recommendations": recommendations}


def _recommendation_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for field in ARTICLE_ENCRYPTED_FIELDS | RECOMMENDATION_ENCRYPTED_FIELDS:
        if field in item:
            item[field] = decrypt_text(item.get(field))
    item["labels"] = json.loads(str(item.pop("labels_json")))
    if "feedback" in item:
        item["feedback"] = decrypt_text(item.get("feedback"))
    item["idea_is_speculative"] = bool(item["idea_is_speculative"])
    return item


def feedback_recommendations(values: set[str] | None = None, limit: int = 200) -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            """
            WITH latest AS (
                SELECT article_id, MAX(refresh_id) AS refresh_id
                FROM recommendations
                GROUP BY article_id
            )
            SELECT r.*, a.title, a.summary, a.source, a.source_url, a.url,
                   a.author, a.published_at, a.folder, a.summary_quality,
                   f.value AS feedback
            FROM feedback f
            JOIN articles a ON a.id = f.article_id
            JOIN latest l ON l.article_id = a.id
            JOIN recommendations r
              ON r.article_id = l.article_id AND r.refresh_id = l.refresh_id
            ORDER BY f.updated_at DESC
            LIMIT 2000
            """
        ).fetchall()
    items = [_recommendation_row(row) for row in rows]
    if values:
        items = [item for item in items if item.get("feedback") in values]
    return items[:limit]


def archive_recommendations(
    query: str = "", run_id: int | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    params: list[Any] = []
    run_clause = ""
    if run_id is not None:
        run_clause = "WHERE r.refresh_id = ?"
        params.append(run_id)
    with connection() as db:
        rows = db.execute(
            f"""
            SELECT r.*, a.title, a.summary, a.source, a.source_url, a.url,
                   a.author, a.published_at, a.folder, a.summary_quality,
                   f.value AS feedback, rr.completed_at AS run_completed_at,
                   rr.status AS run_status
            FROM recommendations r
            JOIN articles a ON a.id = r.article_id
            JOIN refresh_runs rr ON rr.id = r.refresh_id
            LEFT JOIN feedback f ON f.article_id = a.id
            {run_clause}
            ORDER BY rr.completed_at DESC, r.rank ASC
            LIMIT 2500
            """,
            params,
        ).fetchall()
    items = [_recommendation_row(row) for row in rows]
    normalized = query.strip().casefold()
    if normalized:
        fields = (
            "title", "source", "folder", "summary", "reason", "core_finding",
            "innovation", "connection", "idea",
        )
        items = [
            item
            for item in items
            if normalized in " ".join(str(item.get(field, "")) for field in fields).casefold()
        ]
    return items[:limit]


def refresh_history(limit: int = 100) -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            """
            SELECT * FROM refresh_runs
            WHERE status IN ('completed', 'demo')
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def source_catalog() -> dict[str, list[str]]:
    with connection() as db:
        rows = db.execute("SELECT source, folder FROM articles").fetchall()
    sources = {
        str(decrypt_text(row["source"]))
        for row in rows
        if decrypt_text(row["source"])
    }
    folders = {
        str(decrypt_text(row["folder"]))
        for row in rows
        if decrypt_text(row["folder"])
    }
    return {"sources": sorted(sources), "folders": sorted(folders)}


def feedback_counts() -> dict[str, int]:
    with connection() as db:
        rows = db.execute("SELECT value FROM feedback").fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        value = str(decrypt_text(row["value"]))
        counts[value] = counts.get(value, 0) + 1
    return counts


def set_feedback(article_id: str, value: str) -> bool:
    now = utc_now()
    with connection() as db:
        exists = db.execute(
            "SELECT 1 FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
        if not exists:
            return False
        db.execute(
            """
            INSERT INTO feedback(article_id, value, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(article_id) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (article_id, encrypt_text(value), now, now),
        )
    return True


def feedback_examples() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "positive": [],
        "negative": [],
        "known": [],
    }
    with connection() as db:
        rows = db.execute(
            """
            SELECT f.value, a.id, a.title, a.summary, a.embedding_json
            FROM feedback f JOIN articles a ON a.id = f.article_id
            WHERE a.embedding_model = ?
            ORDER BY f.updated_at DESC LIMIT 80
            """,
            (config.embedding_model,),
        ).fetchall()
    for row in rows:
        item = dict(row)
        item["value"] = decrypt_text(item.get("value"))
        for field in ("title", "summary", "embedding_json"):
            item[field] = decrypt_text(item.get(field))
        if item.get("embedding_json"):
            item["embedding"] = json.loads(str(item.pop("embedding_json")))
        if item["value"] == "read":
            continue
        if item["value"] == "already_known":
            bucket = "known"
        elif item["value"] in {"relevant", "inspiring", "save_for_later"}:
            bucket = "positive"
        else:
            bucket = "negative"
        groups[bucket].append(item)
    return groups
