"""Check whether this machine's Scopus entitlement returns abstract text.

Elsevier grants the abstract only to an entitled institution, and entitlement is
decided per request by network or institution token. Run this from wherever
PaperPulse will run before relying on Scopus for ScienceDirect coverage:

    python scripts/check_scopus_access.py
    python scripts/check_scopus_access.py 10.1016/j.jallcom.2026.190128
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import config  # noqa: E402


SAMPLE_DOIS = (
    "10.1016/j.jallcom.2026.190128",
    "10.1016/j.optlastec.2026.116148",
    "10.1016/j.nanoen.2026.112290",
)


def main() -> int:
    dois = sys.argv[1:] or list(SAMPLE_DOIS)
    if not config.elsevier_api_key:
        print("ELSEVIER_API_KEY is not set in .env — nothing to check.")
        return 1

    headers = {"X-ELS-APIKey": config.elsevier_api_key, "Accept": "application/json"}
    if config.elsevier_insttoken:
        headers["X-ELS-Insttoken"] = config.elsevier_insttoken
        print("Using an institution token.")
    else:
        print("No institution token; entitlement will be decided by your network.")

    query = " OR ".join(f'DOI("{doi}")' for doi in dois)
    try:
        response = httpx.get(
            "https://api.elsevier.com/content/search/scopus",
            params={"query": query, "view": "COMPLETE", "count": len(dois)},
            headers=headers,
            timeout=45,
        )
    except httpx.HTTPError as error:
        print(f"Request failed: {error}")
        return 1

    print(f"HTTP {response.status_code}")
    if response.status_code == 401:
        print("The key was rejected. Check ELSEVIER_API_KEY.")
        return 1
    if response.status_code == 403:
        print(
            "Authenticated but not entitled to the COMPLETE view.\n"
            "Run from the institutional network, or request an institution token."
        )
        return 1
    if response.is_error:
        print(response.text[:400])
        return 1

    entries = (response.json().get("search-results") or {}).get("entry") or []
    print(f"{len(entries)} record(s) returned.\n")
    with_abstract = 0
    for entry in entries:
        doi = entry.get("prism:doi", "?")
        abstract = entry.get("dc:description") or ""
        title = (entry.get("dc:title") or "")[:70]
        if abstract:
            with_abstract += 1
        print(f"{doi}\n  title    : {title}")
        print(f"  abstract : {len(abstract)} chars")
        if abstract:
            print(f"  preview  : {abstract[:200]}...")
        print()

    print(f"Abstracts available: {with_abstract}/{len(entries)}")
    if with_abstract:
        print("Scopus will supply ScienceDirect abstracts from this machine.")
        return 0
    print(
        "Records resolve but carry no abstract text, which means this entitlement\n"
        "does not include it. PaperPulse will keep falling back to an excerpt."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
