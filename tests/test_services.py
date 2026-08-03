from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

from backend import db
from backend.crypto import decrypt_text
from backend.demo_data import DEMO_ARTICLES, DEMO_PROFILE
from backend.inoreader import _article_from_item
from backend.models import FeedbackRequest
from backend.ranking import _fallback_recommendations, _select_valid_recommendations


@contextmanager
def isolated_database() -> Iterator[Path]:
    database_path = Path.cwd() / ".paperpulse-test.sqlite3"
    related = [database_path, Path(f"{database_path}-wal"), Path(f"{database_path}-shm")]
    for path in related:
        path.unlink(missing_ok=True)
    try:
        yield database_path
    finally:
        for path in related:
            path.unlink(missing_ok=True)


class ServiceTests(unittest.TestCase):
    def test_inoreader_item_prefers_intelligence_summary(self) -> None:
        article = _article_from_item(
            {
                "id": "item-1",
                "title": "Test article",
                "published": 1_700_000_000,
                "canonical": [{"href": "https://example.com/article"}],
                "summary": {"content": "<p>Feed summary</p>"},
                "summaries": [{"summary": "<p>Intelligence summary</p>"}],
                "origin": {"title": "Example Journal", "htmlUrl": "https://example.com"},
                "categories": ["user/1/label/Photonics"],
            }
        )
        self.assertEqual(article["summary"], "Intelligence summary")
        self.assertEqual(article["folder"], "Photonics")
        self.assertEqual(article["summary_quality"], 1.0)

    def test_title_only_article_remains_eligible(self) -> None:
        article = _article_from_item(
            {
                "id": "item-2",
                "title": "A title-only signal",
                "published": 1_700_000_000,
                "origin": {"title": "Sparse Feed"},
            }
        )
        self.assertEqual(article["summary"], "")
        self.assertGreater(article["summary_quality"], 0)
        self.assertLess(article["summary_quality"], 0.5)

    def test_inoreader_rejects_unsafe_links_and_uses_valid_alternate(self) -> None:
        article = _article_from_item(
            {
                "id": "item-unsafe",
                "title": "Safe link selection",
                "published": 1_700_000_000,
                "canonical": [{"href": "javascript:alert(1)"}],
                "alternate": [{"href": "https://example.com/real-article"}],
                "origin": {"title": "Example", "htmlUrl": "file:///private/feed"},
            }
        )
        self.assertEqual(article["url"], "https://example.com/real-article")
        self.assertEqual(article["source_url"], "")

    def test_fallback_ranking_returns_requested_count(self) -> None:
        articles = [
            {
                "id": f"demo-{index}",
                "title": item["title"],
                "summary": item["summary"],
                "source": item["source"],
                "summary_quality": 0.9,
            }
            for index, item in enumerate(DEMO_ARTICLES)
        ]
        recommendations = _fallback_recommendations(articles, DEMO_PROFILE, 5)
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(rec["article_id"] for rec in recommendations))
        self.assertTrue(all(rec["idea_is_speculative"] for rec in recommendations))

    def test_fallback_ranking_allows_fewer_articles_than_top_n(self) -> None:
        articles = [
            {
                "id": f"real-{index}",
                "title": item["title"],
                "summary": item["summary"],
                "source": item["source"],
                "summary_quality": 0.9,
            }
            for index, item in enumerate(DEMO_ARTICLES[:4])
        ]
        recommendations = _fallback_recommendations(articles, DEMO_PROFILE, 20)
        self.assertEqual(len(recommendations), 4)

    def test_local_read_feedback_is_valid(self) -> None:
        self.assertEqual(FeedbackRequest(value="read").value, "read")

    def test_ai_results_are_deduplicated_and_low_value_items_are_removed(self) -> None:
        base = {
            "novelty_score": 0.8,
            "confidence": 0.8,
            "reason": "Reason",
            "core_finding": "Finding",
            "innovation": "Innovation",
            "connection": "Connection",
            "idea": "Idea",
            "idea_is_speculative": True,
            "labels": [],
        }
        selected = _select_valid_recommendations(
            [
                {
                    **base,
                    "article_id": "irrelevant",
                    "relevance_score": 0.01,
                    "inspiration_score": 0.01,
                },
                {
                    **base,
                    "article_id": "valuable",
                    "relevance_score": 0.9,
                    "inspiration_score": 0.2,
                },
                {
                    **base,
                    "article_id": "valuable",
                    "relevance_score": 0.8,
                    "inspiration_score": 0.3,
                },
            ],
            {"irrelevant", "valuable"},
            10,
        )
        self.assertEqual([item["article_id"] for item in selected], ["valuable"])

    def test_demo_cleanup_feedback_history_and_profile_preservation(self) -> None:
        with isolated_database() as database_path:
            fake_config = SimpleNamespace(
                database_path=database_path,
                embedding_model="text-embedding-test",
            )
            with patch.object(db, "config", fake_config):
                db.init_db()
                now = "2026-08-02T00:00:00+00:00"
                articles = [
                    {
                        "id": "demo-1",
                        "title": "Demo",
                        "published_at": now,
                        "raw": {"demo": True},
                    },
                    {
                        "id": "real-1",
                        "title": "Real",
                        "published_at": now,
                        "url": "https://example.com/real",
                    },
                ]
                db.upsert_articles(articles)
                contaminated_run = db.create_refresh_run()
                base = {
                    "relevance_score": 0.8,
                    "novelty_score": 0.8,
                    "inspiration_score": 0.8,
                    "confidence": 0.8,
                    "reason": "Reason",
                    "core_finding": "Finding",
                    "innovation": "Innovation",
                    "connection": "Connection",
                    "idea": "Idea",
                    "idea_is_speculative": True,
                    "labels": ["Field match"],
                }
                db.save_recommendations(
                    contaminated_run,
                    [{**base, "article_id": "demo-1"}, {**base, "article_id": "real-1"}],
                )
                db.complete_refresh_run(contaminated_run, "completed", 2, 2)
                removed = db.purge_demo_data()
                self.assertEqual(removed["articles"], 1)
                self.assertEqual([item["id"] for item in db.list_articles()], ["real-1"])
                self.assertIsNone(db.latest_dashboard()["run"])

                clean_run = db.create_refresh_run()
                db.save_recommendations(clean_run, [{**base, "article_id": "real-1"}])
                db.complete_refresh_run(clean_run, "completed", 1, 1)
                self.assertTrue(db.set_feedback("real-1", "save_for_later"))
                self.assertFalse(db.set_feedback("missing", "relevant"))
                self.assertEqual(db.feedback_counts()["save_for_later"], 1)
                with db.connection() as connection:
                    raw_feedback = connection.execute(
                        "SELECT value FROM feedback WHERE article_id = ?", ("real-1",)
                    ).fetchone()["value"]
                self.assertTrue(raw_feedback.startswith("enc:v1:"))
                saved = db.feedback_recommendations({"save_for_later"})[0]
                self.assertEqual(saved["article_id"], "real-1")
                self.assertEqual(saved["refresh_id"], clean_run)

                db.save_profile("cv.docx", "extracted CV text", DEMO_PROFILE.model_dump())
                updated = {**DEMO_PROFILE.model_dump(), "headline": "Edited headline"}
                db.update_profile_data(updated)
                with db.connection() as connection:
                    stored = connection.execute(
                        "SELECT original_text, profile_json FROM research_profile WHERE id = 1"
                    ).fetchone()
                self.assertTrue(stored["original_text"].startswith("enc:v1:"))
                self.assertEqual(decrypt_text(stored["original_text"]), "extracted CV text")
                self.assertIn("Edited headline", str(decrypt_text(stored["profile_json"])))

                catalog = db.source_catalog()
                self.assertEqual(catalog["sources"], [])
                archive = db.archive_recommendations(query="real")
                self.assertEqual([item["article_id"] for item in archive], ["real-1"])

    def test_strictness_threshold_changes_selection(self) -> None:
        base = {
            "article_id": "borderline",
            "relevance_score": 0.2,
            "novelty_score": 0.8,
            "inspiration_score": 0.2,
            "confidence": 0.8,
            "reason": "Reason",
            "core_finding": "Finding",
            "innovation": "Innovation",
            "connection": "Connection",
            "idea": "Idea",
            "idea_is_speculative": True,
            "labels": [],
        }
        self.assertEqual(len(_select_valid_recommendations([base], {"borderline"}, 10, 0.15)), 1)
        self.assertEqual(len(_select_valid_recommendations([base], {"borderline"}, 10, 0.4)), 0)


if __name__ == "__main__":
    unittest.main()
