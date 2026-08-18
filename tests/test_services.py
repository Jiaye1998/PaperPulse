from __future__ import annotations

import asyncio
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

from backend import db
from backend.abstract_enrichment import (
    API_RETRY_ATTEMPTS,
    AbstractCandidate,
    _ServiceGate,
    _crossref_candidate,
    _crossref_search_candidate,
    _doi_batch_candidate,
    _get_json,
    _metadata_looks_complete,
    enrich_articles_with_public_abstracts,
    extract_abstract_from_html,
)
from backend.article_processing import (
    canonical_source_name,
    claim_is_supported,
    classify_article,
    deduplicate_articles,
    derive_doi_from_url,
    evidence_is_grounded,
    extract_doi,
    extract_elsevier_pii,
    is_preprint_source,
    non_research_kind,
    source_abstract,
)
from backend.browser_abstracts import CHALLENGE_TEXT, open_verification_browser
from backend.crypto import decrypt_text
from backend.demo_data import DEMO_ARTICLES, DEMO_PROFILE
from backend.inoreader import _article_from_item, fetch_unread
from backend.idea_lab import (
    _abstract_diagnostic,
    _generate_ideas,
    _idea_diversity_score,
    _merge_works,
    _sanitize_ideas,
    generate_idea_lab,
)
from backend.models import FeedbackRequest
from backend.ranking import (
    _exact_selection,
    _fallback_recommendations,
    _sanitize_research_structure,
    _select_valid_recommendations,
    rank_articles,
)


async def _no_sleep(_seconds: float) -> None:
    """Collapse retry backoff so throttling tests stay fast."""


async def _fake_token() -> str:
    return "test-token"


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
    def test_browser_challenge_detection_is_specific(self) -> None:
        self.assertIsNotNone(CHALLENGE_TEXT.search("Just a moment... verify you are human"))
        self.assertIsNotNone(CHALLENGE_TEXT.search("Radware Captcha"))
        self.assertIsNone(
            CHALLENGE_TEXT.search(
                "Abstract We report a public experimental study of optical resonances."
            )
        )

    def test_browser_verified_abstract_is_accepted_as_source_evidence(self) -> None:
        article = {
            "summary": "A complete abstract captured from the publisher page.",
            "raw": {"summary_source": "publisher_browser_abstract"},
        }
        self.assertEqual(
            source_abstract(article),
            (
                "A complete abstract captured from the publisher page.",
                "publisher_browser_abstract",
            ),
        )

    @patch("backend.browser_abstracts.subprocess.Popen")
    @patch("backend.browser_abstracts._chrome_executable")
    def test_manual_verification_uses_dedicated_persistent_profile(
        self, chrome_executable: object, popen: object
    ) -> None:
        chrome_executable.return_value = Path("C:/Program Files/Google/Chrome/chrome.exe")
        open_verification_browser("https://publisher.example/article/1")
        command = popen.call_args.args[0]
        self.assertIn("--profile-directory=Default", command)
        self.assertTrue(
            any(str(value).startswith("--user-data-dir=") for value in command)
        )
        self.assertEqual(command[-1], "https://publisher.example/article/1")

    def test_inoreader_item_prefers_feed_text_over_intelligence_summary(self) -> None:
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
        self.assertEqual(article["summary"], "Feed summary")
        self.assertNotIn("Intelligence summary", article["summary"])
        self.assertEqual(article["folder"], "Photonics")
        self.assertEqual(article["raw"]["summary_source"], "feed_abstract_or_excerpt")
        self.assertLessEqual(article["summary_quality"], 0.95)
        abstract, provenance = source_abstract(article)
        self.assertEqual(abstract, "")
        self.assertEqual(provenance, "feed_abstract_or_excerpt")

    def test_inoreader_summary_fallback_is_not_source_evidence(self) -> None:
        article = _article_from_item(
            {
                "id": "item-summary-only",
                "title": "Summary-only item",
                "published": 1_700_000_000,
                "summaries": [{"summary": "Inoreader-generated fallback text only."}],
                "origin": {"title": "Example Journal"},
            }
        )
        self.assertEqual(article["raw"]["summary_source"], "inoreader_summary_fallback")
        self.assertLessEqual(article["summary_quality"], 0.35)
        abstract, provenance = source_abstract(article)
        self.assertEqual(abstract, "")
        self.assertEqual(provenance, "inoreader_summary_fallback")
        legacy_abstract, legacy_provenance = source_abstract(
            {
                "summary": (
                    "INOREADER SUMMARY: Generated text.\n\n"
                    "FEED ABSTRACT OR EXCERPT: Publisher abstract text."
                ),
                "raw": {"summary_source": "intelligence+feed"},
            }
        )
        self.assertEqual(legacy_abstract, "")
        self.assertEqual(legacy_provenance, "intelligence+feed")

    def test_public_page_citation_abstract_is_confirmed_and_beats_description(self) -> None:
        candidate = extract_abstract_from_html(
            """
            <html><head>
              <meta name="citation_title" content="A programmable photonic resonator">
              <meta name="description" content="A short journal teaser...">
              <meta name="citation_abstract" content="We demonstrate a programmable photonic resonator with independently controlled coupling and loss. Measurements across twelve devices show stable tuning over repeated thermal cycles, and the reported model explains the observed resonance shift without introducing an additional fitted mechanism.">
            </head></html>
            """,
            "A programmable photonic resonator",
            "https://publisher.example/paper",
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertTrue(candidate.complete)
        self.assertEqual(candidate.provenance, "publisher_page_abstract")
        self.assertIn("twelve devices", candidate.text)

    def test_page_description_remains_excerpt_and_wrong_title_is_rejected(self) -> None:
        excerpt = extract_abstract_from_html(
            """
            <html><head>
              <meta property="og:title" content="Matched scientific title">
              <meta property="og:description" content="This public page gives only a shortened description of the reported experiment and therefore cannot establish the complete abstract.">
            </head></html>
            """,
            "Matched scientific title",
            "https://publisher.example/paper",
        )
        self.assertIsNotNone(excerpt)
        assert excerpt is not None
        self.assertFalse(excerpt.complete)
        self.assertEqual(excerpt.provenance, "publisher_page_excerpt")
        self.assertIsNone(
            extract_abstract_from_html(
                "<meta name='citation_title' content='An unrelated biology paper'><meta name='citation_abstract' content='A sufficiently long abstract about cells and proteins that belongs to a different article entirely and must not be attached to the requested photonics record.'>",
                "Matched scientific title",
                "https://publisher.example/wrong",
            )
        )

    def test_crossref_abstract_requires_matching_title(self) -> None:
        payload = {
            "message": {
                "title": ["Matched catalyst study"],
                "abstract": "<jats:p>We report a catalyst study with controlled synthesis, spectroscopy, and repeated activity measurements. The complete metadata abstract describes both the comparison group and the main quantitative observation across the tested conditions.</jats:p>",
            }
        }
        candidate = _crossref_candidate(
            payload, "Matched catalyst study", "https://api.crossref.org/works/10.1/test"
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertTrue(candidate.complete)
        self.assertIsNone(
            _crossref_candidate(
                payload,
                "Completely different photonics work",
                "https://api.crossref.org/works/10.1/test",
            )
        )
        search_candidate = _crossref_search_candidate(
            {"message": {"items": [{"title": ["Unrelated work"]}, payload["message"]]}},
            "Matched catalyst study",
            "https://api.crossref.org/works?query.title=matched",
        )
        self.assertIsNotNone(search_candidate)
        assert search_candidate is not None
        self.assertEqual(search_candidate.doi, "")

    def test_corrections_retractions_and_front_matter_leave_the_pool(self) -> None:
        excluded = [
            "Author Correction: Structural mechanism of cGAS inhibition",
            "Publisher Correction: A charged cage for excitons",
            "Correction to “Designing the Future of Hardmetals”",
            "Corrigendum to a study of perovskite interfaces",
            "Erratum: Thermal transport in layered oxides",
            "Retraction Note to: Exosome-mediated tendon repair",
            "Retracted: Graphene growth on liquid copper",
            "Withdrawn: Preliminary results on spin transport",
            "Expression of Concern: Image duplication in Figure 3",
            "Addendum: Extended data for the catalytic cycle",
            "In This Issue",
            "Issue Information",
            "Inside Front Cover: Molecular sieving membranes",
            "Editorial Board",
        ]
        for title in excluded:
            self.assertTrue(
                non_research_kind({"title": title, "url": "https://example.org/a"}),
                f"should be excluded: {title}",
            )

        kept = [
            # Substantive post-publication debate is real science and stays.
            "Comment on “Anomalous thermal conductivity in twisted bilayers”",
            "Reply to Zhang et al.: The role of surface defects",
            "Matters Arising: Reassessing the reported quantum yield",
            # Research whose title merely contains a trigger word.
            "Correction algorithms for phase retrieval in ptychography",
            "Editorial control of gene expression by CRISPR base editors",
            "Retraction of consent in longitudinal clinical cohorts",
            "Addenda-free synthesis of high-entropy alloys",
        ]
        for title in kept:
            self.assertEqual(
                non_research_kind({"title": title, "url": "https://example.org/a"}),
                "",
                f"should be kept: {title}",
            )

        # Video and general-news subscriptions can never yield a research abstract.
        for url in (
            "https://www.youtube.com/watch?v=abc123",
            "https://phys.org/news/2026-08-quantum.html",
            "https://www.wired.com/story/some-story",
        ):
            self.assertEqual(
                non_research_kind({"title": "A plausible science headline", "url": url}),
                "non_scholarly_source",
                f"should be excluded: {url}",
            )
        # A publisher host that merely contains a listed name must survive.
        self.assertEqual(
            non_research_kind(
                {"title": "Real study", "url": "https://pubs.acs.org/doi/10.1021/x"}
            ),
            "",
        )

        # Nature reserves the d##### DOI family for magazine content.
        self.assertEqual(
            non_research_kind(
                {"title": "Can invisible watermarks curb AI slop?",
                 "url": "https://www.nature.com/articles/d41586-026-02503-7"}
            ),
            "news",
        )
        self.assertEqual(
            non_research_kind(
                {"title": "Nanoporous monolayer metal enables confinement",
                 "url": "https://www.nature.com/articles/s41565-026-02240-y"}
            ),
            "",
        )

        unique, stats = deduplicate_articles(
            [
                {"id": "1", "title": "Erratum: something", "url": "https://a.org/1",
                 "source": "J", "summary": "x"},
                {"id": "2", "title": "A real study of layered oxide cathodes",
                 "url": "https://a.org/2", "source": "J", "summary": "x"},
                {"id": "3", "title": "A real study of layered oxide cathodes",
                 "url": "https://a.org/2", "source": "J", "summary": "x"},
            ]
        )
        self.assertEqual(len(unique), 1)
        self.assertEqual(stats["non_research_count"], 1)
        self.assertEqual(stats["non_research_breakdown"], {"correction": 1})
        # The excluded entry must not be miscounted as a duplicate of what remains.
        self.assertEqual(stats["duplicate_count"], 1)

    def test_sciencedirect_pii_is_extracted_and_preprints_lose_title_matches(self) -> None:
        self.assertEqual(
            extract_elsevier_pii(
                "https://www.sciencedirect.com/science/article/pii/S0925838826041976?dgcid=rss_sd_all"
            ),
            "S0925838826041976",
        )
        self.assertEqual(
            extract_elsevier_pii(
                "https://www.sciencedirect.com/science/article/abs/pii/S221128552600594X"
            ),
            "S221128552600594X",
        )
        # Elsevier's society sites punctuate the same identifier.
        self.assertEqual(
            extract_elsevier_pii("https://www.cell.com/cell/fulltext/S0092-8674(26)00828-7"),
            "S0092867426008287",
        )
        self.assertEqual(
            extract_elsevier_pii(
                "https://www.cell.com/action/showPdf?pii=S0092-8674%2826%2900828-7"
            ),
            "S0092867426008287",
        )
        self.assertEqual(extract_elsevier_pii("https://www.nature.com/articles/s41467-1"), "")

        body = (
            "AlN/GaN heterojunctions exhibit significant potential for high-efficiency "
            "millimeter-wave devices due to their thin barrier layers and high-density "
            "two-dimensional electron gas measured across the grown wafers."
        )
        title = "Interfacial rearrangement of metal atoms controlled by adsorption layers"
        preprint = {
            "message": {
                "items": [
                    {
                        "DOI": "10.2139/ssrn.6604099",
                        "type": "posted-content",
                        "title": [title],
                        "abstract": body,
                    }
                ]
            }
        }
        # The journal article is what the feed pointed at; its SSRN preprint carries a
        # different DOI and a possibly older text, so title matching must not take it.
        self.assertIsNone(
            _crossref_search_candidate(
                preprint, title, "https://api.crossref.org/works", reject_preprints=True
            )
        )
        accepted = _crossref_search_candidate(
            preprint, title, "https://api.crossref.org/works"
        )
        self.assertIsNotNone(accepted)

    def test_preprint_servers_keep_their_own_posted_content_records(self) -> None:
        # ChemRxiv, bioRxiv and friends publish nothing but posted-content, and
        # their article URLs carry no DOI, so title search is the only route to
        # their abstracts. Rejecting posted-content outright would silence them.
        for url in (
            "https://chemrxiv.org/engage/chemrxiv/article-details/68a1f2c3b4d5",
            "https://www.biorxiv.org/content/10.1101/2024.01.02.573210v1",
            "https://www.researchsquare.com/article/rs-10515961/latest",
            "https://arxiv.org/abs/2501.00001",
        ):
            self.assertTrue(
                is_preprint_source({"url": url, "title": "t", "summary": ""}),
                f"should count as a preprint source: {url}",
            )
        for url in (
            "https://pubs.acs.org/doi/10.1021/acsami.5c00001",
            "https://www.sciencedirect.com/science/article/pii/S0925838826041976",
            "https://pubs.rsc.org/en/content/articlelanding/2026/sc/d5sc00001a",
        ):
            self.assertFalse(
                is_preprint_source({"url": url, "title": "t", "summary": ""}),
                f"should not count as a preprint source: {url}",
            )

        body = (
            "We report a dynamic kinetic resolution guided by numerical simulation, "
            "with conversion and enantiomeric excess tracked across twelve substrates "
            "under otherwise identical catalytic conditions."
        )
        title = "Numerical simulation-guided development of dynamic kinetic resolution"
        payload = {
            "message": {
                "items": [
                    {
                        "DOI": "10.26434/chemrxiv.15001461/v2",
                        "type": "posted-content",
                        "title": [title],
                        "abstract": body,
                    }
                ]
            }
        }
        kept = _crossref_search_candidate(
            payload,
            title,
            "https://api.crossref.org/works",
            reject_preprints=not is_preprint_source(
                {"url": "https://chemrxiv.org/engage/chemrxiv/article-details/x"}
            ),
        )
        self.assertIsNotNone(kept)
        assert kept is not None
        self.assertTrue(kept.complete)

        # The same record must still be refused for a journal article.
        self.assertIsNone(
            _crossref_search_candidate(
                payload,
                title,
                "https://api.crossref.org/works",
                reject_preprints=not is_preprint_source(
                    {"url": "https://pubs.acs.org/doi/10.1021/jacs.5c00001"}
                ),
            )
        )

    def test_publisher_url_shapes_yield_doi_for_metadata_lookup(self) -> None:
        self.assertEqual(
            derive_doi_from_url("https://www.researchsquare.com/article/rs-10515961/latest"),
            "10.21203/rs.3.rs-10515961/v1",
        )
        self.assertEqual(
            derive_doi_from_url("https://www.researchsquare.com/article/rs-884422/v3"),
            "10.21203/rs.3.rs-884422/v3",
        )
        self.assertEqual(
            derive_doi_from_url("https://www.nature.com/articles/s41467-025-62831-6"),
            "10.1038/s41467-025-62831-6",
        )
        self.assertEqual(
            derive_doi_from_url("https://www.biorxiv.org/content/10.1101/2024.01.02.573210v1"),
            "10.1101/2024.01.02.573210v1",
        )
        # An RSC article URL ends in the DOI suffix itself.
        self.assertEqual(
            derive_doi_from_url(
                "https://pubs.rsc.org/en/content/articlelanding/2026/cc/d6cc03277j"
            ),
            "10.1039/d6cc03277j",
        )
        self.assertEqual(
            derive_doi_from_url(
                "https://pubs.rsc.org/en/content/articlehtml/2026/ta/d5ta04521e/unauth"
            ),
            "10.1039/d5ta04521e",
        )
        self.assertEqual(
            derive_doi_from_url("https://pubs.rsc.org/en/journals/journalissues/cc"), ""
        )
        # News pieces and unknown hosts must not produce an invented DOI.
        self.assertEqual(derive_doi_from_url("https://www.nature.com/articles/d41586-025-1"), "")
        self.assertEqual(derive_doi_from_url("https://www.youtube.com/watch?v=abc"), "")
        self.assertEqual(
            extract_doi({"url": "https://www.researchsquare.com/article/rs-42/latest"}),
            "10.21203/rs.3.rs-42/v1",
        )

    def test_doi_batch_rejects_abstract_belonging_to_another_work(self) -> None:
        article = {"id": "a1", "title": "Perovskite solar cell interface passivation"}
        body = (
            "We passivate the perovskite interface and track device stability over "
            "one thousand hours of continuous operation under standard illumination."
        )
        accepted = _doi_batch_candidate(
            article, body, "Perovskite solar cell interface passivation",
            "crossref_abstract", 95, "https://doi.org/10.1/x", "10.1/x",
        )
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertTrue(accepted.complete)
        self.assertEqual(accepted.doi, "10.1/x")
        self.assertIsNone(
            _doi_batch_candidate(
                article, body, "A genome-wide survey of marine archaea",
                "crossref_abstract", 95, "https://doi.org/10.1/x", "10.1/x",
            )
        )
        # A metadata abstract field stays trustworthy below the scraped-page floor,
        # but an explicit truncation mark still demotes it.
        short = "We report a compact source of narrowband heralded single photons in the telecom band."
        candidate = _doi_batch_candidate(
            article, short, "", "crossref_abstract", 95, "https://doi.org/10.1/x", "10.1/x"
        )
        assert candidate is not None
        self.assertFalse(candidate.complete)
        self.assertTrue(_metadata_looks_complete(body))
        self.assertFalse(_metadata_looks_complete(body[:200] + " ..."))

    def test_throttled_service_is_dropped_instead_of_retried_all_run(self) -> None:
        calls = {"count": 0}

        class _Response:
            status_code = 429
            headers = {"retry-after": "0"}
            is_error = True

            def json(self) -> dict[str, object]:
                return {}

        class _Client:
            async def get(self, url: str, **kwargs: object) -> _Response:
                calls["count"] += 1
                return _Response()

        async def exercise() -> _ServiceGate:
            gate = _ServiceGate(0.0, failure_budget=2)
            client = _Client()
            with patch("backend.abstract_enrichment.asyncio.sleep", new=_no_sleep):
                for _ in range(25):
                    self.assertIsNone(
                        await _get_json(client, gate, "https://api.example.org/works", {})
                    )
            return gate

        gate = asyncio.run(exercise())
        self.assertTrue(gate.disabled)
        # Two exhausted ladders spend the budget; the remaining 23 calls are free.
        self.assertEqual(calls["count"], 2 * API_RETRY_ATTEMPTS)

    def test_enrichment_stages_run_cheapest_first_and_browser_sees_only_residue(
        self,
    ) -> None:
        body = (
            "We report a layered oxide cathode synthesised by a molten-salt route and "
            "characterised by operando diffraction across two hundred charge cycles, "
            "with capacity retention compared against an unmodified reference cell."
        )
        articles = [
            {
                "id": "cached",
                "title": "Cached work",
                "url": "https://example.org/cached",
                "summary": body,
                "raw": {},
            },
            {
                "id": "arxiv",
                "title": "Preprint work",
                "url": "https://arxiv.org/abs/2501.00001",
                "summary": f"Announce Type: new Abstract: {body}",
                "raw": {"arxiv_id": "2501.00001"},
            },
            {
                "id": "doi",
                "title": "Crossref work",
                "url": "https://onlinelibrary.wiley.com/doi/10.1002/test.1",
                "summary": "Short feed blurb ...",
                "raw": {"doi": "10.1002/test.1"},
            },
            {
                "id": "residue",
                "title": "Bot-walled work",
                "url": "https://paywall.example.com/article/9",
                "summary": "Short feed blurb ...",
                "raw": {},
            },
        ]
        cached = [
            {
                "id": "cached",
                "title": "Cached work",
                "url": "https://example.org/cached",
                "summary": body,
                "raw": {
                    "abstract_status": "complete",
                    "summary_source": "crossref_abstract",
                    "abstract_source_url": "https://doi.org/10.1/cached",
                    "abstract_fetched_at": "2026-01-01T00:00:00+00:00",
                },
            }
        ]
        crossref_payload = {
            "message": {
                "items": [
                    {"DOI": "10.1002/test.1", "title": ["Crossref work"], "abstract": body}
                ]
            }
        }
        seen_by_browser: list[str] = []

        async def fake_get_json(_client, gate, url, _params):
            if "crossref" in url:
                return crossref_payload
            return None

        async def fake_pages(_client, articles_arg, resolved, _blocked, stats):
            stats["page_attempted"] = len(
                [a for a in articles_arg if str(a["id"]) not in resolved]
            )
            stats["page_hits"] = 0

        async def fake_browser(pending, _extractor):
            seen_by_browser.extend(str(item["id"]) for item in pending)
            return SimpleNamespace(
                candidates={
                    "residue": AbstractCandidate(
                        text=body,
                        provenance="publisher_browser_abstract",
                        complete=True,
                        source_url="https://paywall.example.com/article/9",
                        priority=110,
                    )
                },
                attempted=len(pending),
                available=True,
                challenges=[],
                refused_domains=[],
                error="",
            )

        with (
            patch("backend.abstract_enrichment._get_json", new=fake_get_json),
            patch("backend.abstract_enrichment._resolve_publisher_pages", new=fake_pages),
            patch(
                "backend.browser_abstracts.resolve_with_persistent_browser",
                new=fake_browser,
            ),
        ):
            enriched, stats = asyncio.run(
                enrich_articles_with_public_abstracts(articles, cached)
            )

        by_id = {item["id"]: item for item in enriched}
        self.assertEqual(by_id["cached"]["raw"]["summary_source"], "crossref_abstract")
        self.assertEqual(by_id["arxiv"]["raw"]["summary_source"], "arxiv_feed_abstract")
        self.assertEqual(by_id["doi"]["raw"]["summary_source"], "crossref_abstract")
        self.assertEqual(
            by_id["residue"]["raw"]["summary_source"], "publisher_browser_abstract"
        )
        for item in enriched:
            self.assertEqual(item["raw"]["abstract_status"], "complete")
        # Cached and arXiv works never reach the network; the browser is handed only
        # what every cheaper stage failed to resolve.
        self.assertEqual(seen_by_browser, ["residue"])
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["attempted"], 2)
        self.assertEqual(stats["crossref_batch_hits"], 1)
        self.assertEqual(stats["complete"], 4)
        self.assertEqual(stats["browser_complete"], 1)

    def test_unread_scan_reports_truncation_and_cannot_spin_on_a_cursor(self) -> None:
        calls = {"n": 0}

        def item(index: int) -> dict[str, object]:
            return {
                "id": f"item-{index}",
                "title": f"Article {index}",
                "published": 1_700_000_000,
                "canonical": [{"href": f"https://example.org/{index}"}],
                "origin": {"title": "Example Journal"},
            }

        class _Response:
            is_error = False
            headers = {"X-Reader-Zone1-Limit": "100", "X-Reader-Zone1-Usage": "9"}

            def __init__(self, payload): self._payload = payload
            def json(self): return self._payload

        class _Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

            async def get(self, url, **kwargs):
                if url.endswith("/tag/list"):
                    return _Response({"tags": []})
                calls["n"] += 1
                start = (calls["n"] - 1) * 2
                # Always offer another page, and always the same cursor.
                return _Response(
                    {"items": [item(start), item(start + 1)], "continuation": "same-cursor"}
                )

        with (
            patch("backend.inoreader._valid_token", new=_fake_token),
            patch("backend.inoreader.httpx.AsyncClient", new=lambda **kw: _Client()),
        ):
            articles, rate = asyncio.run(fetch_unread(7, max_items=4))

        self.assertEqual(len(articles), 4)
        self.assertEqual(rate["truncated"], "true")
        self.assertEqual(rate["scan_limit"], "4")

        # A repeated cursor must end the scan rather than loop forever.
        calls["n"] = 0
        with (
            patch("backend.inoreader._valid_token", new=_fake_token),
            patch("backend.inoreader.httpx.AsyncClient", new=lambda **kw: _Client()),
        ):
            articles, rate = asyncio.run(fetch_unread(7, max_items=1000))
        self.assertEqual(rate["truncated"], "true")
        self.assertLessEqual(calls["n"], 3)

    def test_inoreader_uses_only_confirmed_folder_labels(self) -> None:
        article = _article_from_item(
            {
                "id": "item-folders",
                "title": "Folder filtering",
                "published": 1_700_000_000,
                "categories": [
                    "user/1/label/Photonics",
                    "user/1/label/Relevant",
                ],
                "origin": {"title": "Example"},
            },
            {"Photonics"},
        )
        self.assertEqual(article["raw"]["folders"], ["Photonics"])

    def test_source_normalization_and_work_classification(self) -> None:
        self.assertEqual(
            canonical_source_name("Wiley: Advanced Materials: Table of Contents"),
            "Advanced Materials",
        )
        self.assertEqual(
            canonical_source_name("Materials science : nature.com subject feeds"),
            "Materials science",
        )
        self.assertEqual(
            canonical_source_name("ScienceDirect Publication: Journal of Photochemistry"),
            "Journal of Photochemistry",
        )
        metadata = classify_article(
            {
                "title": "Updated preprint",
                "url": "https://arxiv.org/abs/2608.12345v2",
                "summary": "Announce Type: replace-cross",
            }
        )
        self.assertEqual(metadata["work_type"], "Preprint")
        self.assertEqual(metadata["update_status"], "Revised preprint")
        self.assertTrue(metadata["is_update"])

    def test_duplicate_works_merge_folders_and_keep_best_summary(self) -> None:
        base = {
            "title": "One scientific result",
            "published_at": "2026-08-01T00:00:00+00:00",
        }
        unique, stats = deduplicate_articles(
            [
                {**base, "id": "feed-a", "url": "https://publisher.example/article", "summary": "short", "summary_quality": 0.2, "source": "Wiley: Journal A", "folder": "Optics", "raw": {"summary_source": "inoreader_summary_fallback"}},
                {**base, "id": "feed-b", "url": "https://aggregator.example/record", "summary": "a much more complete abstract", "summary_quality": 0.8, "source": "Journal A", "folder": "Methods", "raw": {"summary_source": "feed_abstract_or_excerpt"}},
            ]
        )
        self.assertEqual(stats["received_count"], 2)
        self.assertEqual(stats["unique_count"], 1)
        self.assertEqual(stats["duplicate_count"], 1)
        self.assertEqual(unique[0]["id"], "feed-a")
        self.assertEqual(unique[0]["summary"], "a much more complete abstract")
        self.assertEqual(unique[0]["folders"], ["Methods", "Optics"])
        self.assertEqual(unique[0]["raw"]["duplicate_count"], 2)
        self.assertEqual(unique[0]["raw"]["summary_source"], "feed_abstract_or_excerpt")

    def test_evidence_must_be_verbatim_and_substantial(self) -> None:
        summary = "The device achieved a measured quality factor of 1200 at room temperature."
        self.assertTrue(evidence_is_grounded("achieved a measured quality factor of 1200", summary))
        self.assertFalse(evidence_is_grounded("quality factor exceeded 5000", summary))
        self.assertTrue(
            claim_is_supported(
                "The measured quality factor was 1200.",
                "achieved a measured quality factor of 1200",
                summary,
            )
        )
        self.assertFalse(
            claim_is_supported(
                "A phase transition caused the optical response.",
                "achieved a measured quality factor of 1200",
                summary,
            )
        )

    def test_research_structure_drops_claims_without_matching_evidence(self) -> None:
        summary = "The film reached 90 percent transmission after thermal annealing."
        structure = _sanitize_research_structure(
            {
                "observation": {
                    "text": "High transmission was measured.",
                    "evidence": "reached 90 percent transmission after thermal annealing",
                },
                "mechanism": {
                    "text": "A phase transition caused the response.",
                    "evidence": "phase transition caused the response",
                },
                "method": {"text": "Not available", "evidence": ""},
                "controllable_variables": [
                    {"text": "Annealing", "evidence": "after thermal annealing"}
                ],
                "limitation_or_gap": {"text": "Not available", "evidence": ""},
                "central_claim": {
                    "text": "Annealing changes transmission.",
                    "evidence": "reached 90 percent transmission after thermal annealing",
                },
                "causal_links": [
                    {
                        "cause": "Thermal annealing",
                        "effect": "Higher transmission",
                        "evidence": "reached 90 percent transmission after thermal annealing",
                    },
                    {
                        "cause": "A hidden catalyst",
                        "effect": "Higher transmission",
                        "evidence": "hidden catalyst increased transmission",
                    },
                ],
                "boundary_conditions": [],
                "inferred_assumptions": [
                    {
                        "text": "Annealing rather than an unreported covariate drives the change.",
                        "abstract_basis": "after thermal annealing",
                    }
                ],
                "alternative_explanations": [
                    {
                        "text": "A hidden catalyst may drive the change.",
                        "abstract_basis": "hidden catalyst",
                    }
                ],
                "unknowns": ["The abstract does not report cycling stability."],
            },
            summary,
        )
        self.assertTrue(structure["observation"]["evidence"])
        self.assertEqual(structure["mechanism"]["text"], "Not available in the abstract.")
        self.assertEqual(len(structure["controllable_variables"]), 1)
        self.assertEqual(len(structure["causal_links"]), 1)
        self.assertEqual(len(structure["inferred_assumptions"]), 1)
        self.assertEqual(structure["alternative_explanations"], [])
        diagnostic = _abstract_diagnostic(structure)
        self.assertGreater(diagnostic["coverage_score"], 0)
        self.assertIn("mechanism", diagnostic["missing_elements"])

    def test_abstract_only_ideas_keep_typed_reasoning_and_distinct_operators(self) -> None:
        structure = {
            "central_claim": {"text": "Annealing changes transmission.", "evidence": "annealing increased transmission"},
            "observation": {"text": "Transmission increased.", "evidence": "increased transmission"},
            "mechanism": {"text": "Not available in the abstract.", "evidence": ""},
            "method": {"text": "Not available in the abstract.", "evidence": ""},
            "controllable_variables": [],
            "limitation_or_gap": {"text": "Not available in the abstract.", "evidence": ""},
            "causal_links": [],
            "boundary_conditions": [],
            "inferred_assumptions": [],
            "alternative_explanations": [],
            "unknowns": [],
        }
        raw_idea = {
            "title": "Test",
            "hypothesis": "Annealing produces a measurable transmission change.",
            "why_it_might_work": "The abstract reports the association.",
            "derivation_operator": "discriminate_cause",
            "abstract_gap_targeted": "The abstract does not distinguish causation from correlation.",
            "assumption_tested": "Annealing is the active cause.",
            "competing_explanation": "An unreported covariate produces the response.",
            "discriminating_outcome": "Only controlled annealing changes transmission.",
            "reasoning_chain": [
                {"kind": "abstract_evidence", "statement": "Transmission increased.", "anchor": "observation"},
                {"kind": "abstract_evidence", "statement": "A mechanism was reported.", "anchor": "mechanism"},
                {"kind": "assumption", "statement": "Annealing is causal.", "anchor": "none"},
            ],
            "minimum_test": "Compare annealed and matched non-annealed controls.",
            "independent_variables": ["Annealing"],
            "dependent_variables": ["Transmission"],
            "controls": ["Non-annealed control"],
            "expected_result": "Annealing changes transmission.",
            "falsification_criterion": "Matched controls show the same change.",
            "main_risk": "The abstract omits covariates.",
            "evidence_anchors": ["observation", "mechanism"],
            "novelty_search_query": "annealing optical transmission causal control",
        }
        payload = {"ideas": {idea_id: dict(raw_idea) for idea_id in (
            "direct_validation", "method_transfer", "high_risk_hypothesis"
        )}}
        ideas = _sanitize_ideas(payload, structure)
        self.assertEqual([idea["derivation_operator"] for idea in ideas], [
            "discriminate_cause", "transfer_mechanism", "invert_assumption"
        ])
        self.assertNotIn("mechanism", ideas[0]["evidence_anchors"])
        self.assertEqual(ideas[0]["reasoning_chain"][1]["kind"], "explicit_inference")
        self.assertEqual(ideas[0]["reasoning_chain"][1]["anchor"], "none")
        self.assertLess(_idea_diversity_score(ideas), 0.45)

    def test_failed_quality_repair_keeps_first_usable_idea_draft(self) -> None:
        raw_idea = {
            "title": "A usable first draft",
            "hypothesis": "A controlled input produces a measurable optical change.",
            "why_it_might_work": "The abstract reports a related observation.",
            "derivation_operator": "discriminate_cause",
            "abstract_gap_targeted": "The reported association does not establish a cause.",
            "assumption_tested": "The controlled input is the active cause of the change.",
            "competing_explanation": "An uncontrolled covariate produces the same change.",
            "discriminating_outcome": "Only the controlled input changes the matched response.",
            "reasoning_chain": [
                {"kind": "explicit_inference", "statement": "The observation motivates a causal test.", "anchor": "none"},
                {"kind": "assumption", "statement": "The input is causally active.", "anchor": "none"},
                {"kind": "explicit_inference", "statement": "Matched controls distinguish the explanations.", "anchor": "none"},
            ],
            "minimum_test": "Compare the controlled input with a matched negative control.",
            "independent_variables": ["Controlled input"],
            "dependent_variables": ["Optical response"],
            "controls": ["Matched negative control"],
            "expected_result": "Only the controlled input changes the response.",
            "falsification_criterion": "The matched control produces the same response.",
            "main_risk": "The abstract omits possible covariates.",
            "evidence_anchors": ["observation"],
            "novelty_search_query": "controlled optical input matched response",
        }
        payload = {
            "ideas": {
                idea_id: dict(raw_idea)
                for idea_id in (
                    "direct_validation",
                    "method_transfer",
                    "high_risk_hypothesis",
                )
            }
        }
        first_response = SimpleNamespace(output_text=json.dumps(payload), usage=None)
        structure = {
            "central_claim": {"text": "Claim", "evidence": "A detailed source abstract observation."},
            "observation": {"text": "Observation", "evidence": "A detailed source abstract observation."},
            "mechanism": {"text": "Not available in the abstract.", "evidence": ""},
            "method": {"text": "Not available in the abstract.", "evidence": ""},
            "controllable_variables": [],
            "limitation_or_gap": {"text": "Not available in the abstract.", "evidence": ""},
            "causal_links": [],
            "boundary_conditions": [],
            "inferred_assumptions": [],
            "alternative_explanations": [],
            "unknowns": [],
        }
        with (
            patch("backend.idea_lab.config", SimpleNamespace(openai_api_key="configured")),
            patch("backend.idea_lab.OpenAI", return_value=object()),
            patch(
                "backend.idea_lab._call_idea_generator",
                side_effect=[first_response, RuntimeError("repair unavailable")],
            ),
        ):
            ideas, cost = _generate_ideas(
                {
                    "title": "Source",
                    "summary": "A detailed source abstract observation.",
                    "raw": {"summary_source": "publisher_page_abstract", "abstract_status": "complete"},
                },
                {"research_structure": structure},
                DEMO_PROFILE,
            )
        self.assertEqual(len(ideas), 3)
        self.assertEqual(ideas[0]["title"], "A usable first draft")
        self.assertEqual(cost, 0)

    def test_related_work_merge_deduplicates_databases_and_excludes_source(self) -> None:
        article = {
            "title": "Original paper",
            "raw": {"doi": "10.1000/original"},
        }
        merged = _merge_works(
            [
                [
                    {"title": "Original paper", "doi": "10.1000/original", "database": "OpenAlex", "relevance": 99},
                    {"title": "Related method", "doi": "10.1000/related", "database": "OpenAlex", "relevance": 12},
                ],
                [
                    {"title": "Related method", "doi": "10.1000/related", "database": "Crossref", "relevance": 8},
                ],
            ],
            article,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["databases"], ["Crossref", "OpenAlex"])

    def test_idea_lab_pipeline_keeps_three_directions_and_caps_novelty(self) -> None:
        ideas = [
            {
                "id": idea_id,
                "direction": label,
                "title": label,
                "hypothesis": "A testable hypothesis.",
                "why_it_might_work": "Because of the grounded observation.",
                "reasoning_steps": ["Observation", "Prediction"],
                "minimum_test": "Measure a response against a control.",
                "independent_variables": ["Input"],
                "dependent_variables": ["Response"],
                "controls": ["Baseline"],
                "expected_result": "A directional change.",
                "falsification_criterion": "No change relative to control.",
                "main_risk": "The abstract omits mechanism detail.",
                "evidence_anchors": ["observation"],
                "novelty_search_query": f"{label} optical response",
            }
            for idea_id, label in (
                ("direct_validation", "Direct validation"),
                ("method_transfer", "Method transfer"),
                ("high_risk_hypothesis", "High-risk hypothesis"),
            )
        ]
        evaluations = {
            idea["id"]: {
                "testability": 0.8,
                "feasibility": 0.7,
                "potential_impact": 0.9,
                "evidence_strength": 0.6,
                "novelty_confidence": 0.99,
                "logical_support": "supported",
                "primary_concern": "Needs validation.",
                "verdict": "Worth testing.",
            }
            for idea in ideas
        }
        related = [{"title": "Related", "doi": "10.1/related", "database": "OpenAlex", "relevance": 2}]
        with (
            patch("backend.idea_lab.config", SimpleNamespace(openai_api_key="configured")),
            patch("backend.idea_lab._generate_ideas", return_value=(ideas, 0.01)),
            patch("backend.idea_lab._search_idea", return_value=(related, {"OpenAlex": "ok"}, 0.001)),
            patch("backend.idea_lab._arxiv_search", return_value=([], "ok", 0.0)),
            patch(
                "backend.idea_lab._critic_review",
                return_value=(
                    {
                        "evaluations": evaluations,
                        "best_idea_id": "method_transfer",
                        "overall_caveat": "Limited metadata search.",
                    },
                    0.02,
                ),
            ),
        ):
            lab, cost = generate_idea_lab(
                {"title": "Source", "summary": "A detailed source abstract " * 5, "raw": {"summary_source": "publisher_page_abstract", "abstract_status": "complete"}},
                {"research_structure": {
                    "central_claim": {"text": "Observed", "evidence": "source abstract"},
                    "observation": {"text": "Observed", "evidence": "source abstract"},
                    "mechanism": {"text": "Not available in the abstract.", "evidence": ""},
                    "method": {"text": "Not available in the abstract.", "evidence": ""},
                    "controllable_variables": [],
                    "limitation_or_gap": {"text": "Not available in the abstract.", "evidence": ""},
                    "causal_links": [],
                    "boundary_conditions": [],
                    "inferred_assumptions": [],
                    "alternative_explanations": [],
                    "unknowns": [],
                }},
                DEMO_PROFILE,
            )
        self.assertEqual([idea["id"] for idea in lab["ideas"]], [
            "direct_validation", "method_transfer", "high_risk_hypothesis"
        ])
        self.assertEqual(lab["best_idea_id"], "method_transfer")
        self.assertTrue(all(idea["evaluation"]["novelty_confidence"] <= 0.8 for idea in lab["ideas"]))
        self.assertAlmostEqual(cost, 0.03)

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

    def test_exact_selection_fills_model_shortfall_to_requested_count(self) -> None:
        candidates = [
            {
                "id": f"article-{index}",
                "candidate_key": f"A{index:03d}",
                "source": f"Journal {index % 5}",
                "summary_quality": 0.9,
                "profile_similarity": 0.8 - index / 100,
                "raw": {},
            }
            for index in range(1, 31)
        ]
        model_output = [
            {
                "candidate_key": "A001",
                "relevance_score": 0.9,
                "novelty_score": 0.7,
                "inspiration_score": 0.8,
                "confidence": 0.9,
                "labels": ["Field match"],
            }
        ]
        selected = _exact_selection(model_output, candidates, 20, "balanced")
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({item[0]["id"] for item in selected}), 20)

    def test_public_ranking_contract_returns_exact_n_without_api(self) -> None:
        articles = [
            {
                "id": f"contract-{index}",
                "title": f"Photonics research result {index}",
                "summary": "A sufficiently detailed source excerpt about optical materials and measurements.",
                "source": f"Journal {index % 4}",
                "folder": "Optics",
                "folders": ["Optics"],
                "summary_quality": 0.8,
                "raw": {},
            }
            for index in range(30)
        ]
        with patch("backend.ranking.config", SimpleNamespace(openai_api_key="")):
            recommendations, cost, _, candidate_count = rank_articles(
                articles, DEMO_PROFILE, 20
            )
        self.assertEqual(len(recommendations), 20)
        self.assertEqual(candidate_count, 30)
        self.assertEqual(cost, 0)

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
                    "evidence": "Evidence copied from the source excerpt.",
                    "research_structure": {
                        "observation": {
                            "text": "A result was observed.",
                            "evidence": "Evidence copied from the source excerpt.",
                        }
                    },
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
                db.complete_refresh_run(
                    clean_run,
                    "completed",
                    3,
                    1,
                    unique_count=2,
                    duplicate_count=1,
                    candidate_count=2,
                    missing_summary_count=0,
                    thin_summary_count=1,
                )
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
                self.assertEqual(saved["evidence"], "Evidence copied from the source excerpt.")
                self.assertEqual(saved["research_structure"]["observation"]["text"], "A result was observed.")
                run = db.latest_dashboard()["run"]
                self.assertEqual(run["unique_count"], 2)
                self.assertEqual(run["duplicate_count"], 1)
                self.assertEqual(run["candidate_count"], 2)

                lab = db.save_idea_lab(
                    clean_run,
                    "real-1",
                    "complete",
                    lab={"ideas": [{"id": "direct_validation"}], "best_idea_id": "direct_validation"},
                    estimated_cost=0.02,
                )
                self.assertEqual(lab["status"], "complete")
                with db.connection() as connection:
                    raw_lab = connection.execute(
                        "SELECT lab_json FROM idea_labs WHERE refresh_id = ? AND article_id = ?",
                        (clean_run, "real-1"),
                    ).fetchone()["lab_json"]
                self.assertTrue(raw_lab.startswith("enc:v1:"))
                dashboard_item = db.latest_dashboard()["recommendations"][0]
                self.assertEqual(dashboard_item["idea_lab"]["best_idea_id"], "direct_validation")
                self.assertEqual(db.idea_lab_context(clean_run, "real-1")["article_id"], "real-1")

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

    def test_changed_abstract_invalidates_cached_embedding(self) -> None:
        with isolated_database() as database_path:
            fake_config = SimpleNamespace(
                database_path=database_path,
                embedding_model="text-embedding-test",
            )
            with patch.object(db, "config", fake_config):
                db.init_db()
                article = {
                    "id": "abstract-change",
                    "title": "Stable title",
                    "summary": "Short feed excerpt.",
                    "published_at": "2026-08-02T00:00:00+00:00",
                    "raw": {"abstract_status": "excerpt"},
                }
                db.upsert_articles([article])
                db.set_article_embedding("abstract-change", [0.1, 0.2])
                db.upsert_articles([article])
                self.assertIsNotNone(db.list_articles()[0]["embedding_json"])
                db.upsert_articles(
                    [
                        {
                            **article,
                            "summary": "A newly retrieved complete public abstract.",
                            "raw": {"abstract_status": "complete"},
                        }
                    ]
                )
                self.assertIsNone(db.list_articles()[0]["embedding_json"])

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
