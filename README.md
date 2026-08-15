# PaperPulse

**Find what matters in your daily research feed.**

Created by [Jiaye1998](https://github.com/Jiaye1998).

**[Visit the PaperPulse website →](https://jiaye1998.github.io/PaperPulse/)**

PaperPulse is a local-first research intelligence dashboard. It reads currently unread Inoreader items, learns an editable research profile from a PDF or DOCX CV, and creates an English brief with the requested number of unique articles whenever enough eligible works are available.

![PaperPulse social card](public/og.png)

## Highlights

- Read-only Inoreader OAuth; no read/star/save writeback
- CV-based personalization with an editable research lens
- Three-stage OpenAI pipeline: embeddings, candidate selection, then isolated per-article analysis with validated source evidence
- Strict, balanced, and exploratory discovery modes
- Boost, lower, or exclude individual sources and Inoreader folders
- An exact brief target from 1 to 100 when enough unique eligible works exist
- DOI/arXiv/URL/title deduplication, canonical source names, and preprint/update metadata
- Browser-first public abstract retrieval with a reusable, dedicated Chrome profile
- A visible coverage funnel from feed items to unique works, ranked candidates, and delivered articles
- A deep Idea Lab for the top five articles, with three hypothesis directions and an independent critic
- Prior-art leads from OpenAlex, Crossref, and arXiv with explicit novelty uncertainty
- Searchable brief archive and per-refresh history
- Relevant, Inspiring, Not useful, Save, Known, and local Read feedback
- Encrypted local CV files, OAuth tokens, article cache, recommendations, and embeddings
- SQLite storage, Docker support, and no credentials or reading data in Git

## Quick start on Windows

1. Run `setup.ps1` in PowerShell.
2. Open `.env` and add your OpenAI and Inoreader credentials.
3. Run `run.ps1`.
4. Open [http://localhost:3000](http://localhost:3000).

PowerShell may require `Set-ExecutionPolicy -Scope Process Bypass` for the current window.

## Quick start with Docker

1. Copy `.env.example` to `.env` and add your credentials.
2. Run:

   ```bash
   docker compose up --build
   ```

3. Open [http://localhost:3000](http://localhost:3000).

The `./data` directory is mounted into the API container, so your library survives restarts.

## Manual setup

Requirements: Python 3.11+, Node.js 22.13+, Inoreader Pro, and an OpenAI API key.

```bash
npm install
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python start.py
```

On macOS or Linux:

```bash
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python start.py
```

## Inoreader OAuth

Create a Web OAuth application in Inoreader and configure this redirect URI:

```text
http://localhost:8000/api/inoreader/callback
```

Put the client ID and client secret in `.env`. PaperPulse requests only the `read` scope. Then open Settings and select **Connect**.

## How selection works

Every refresh fetches articles that are still unread within the configured scan window, up to 1,000 feed entries. Entries are merged into unique works using DOI, arXiv ID, canonical URL, or normalized title. Source aliases are normalized and all confirmed Inoreader folder memberships are preserved. Before final ranking, PaperPulse opens each public article link in a real Chrome session and reads only explicit abstract metadata or an abstract section. The dedicated browser profile is reused across refreshes. If a publisher displays a human-verification page, PaperPulse stops automated requests to that domain and shows a button for opening the page visibly; complete the verification manually, close that Chrome window, and refresh again. PaperPulse never attempts to solve or bypass a CAPTCHA.

When the publisher page does not expose a complete abstract, PaperPulse falls back to direct public page metadata, then Crossref and OpenAlex DOI metadata; arXiv feed abstracts are accepted directly. Complete results are cached locally. Paywalls, sign-ins, private-network addresses, arbitrary page body text, and truncated descriptions are never treated as full abstracts. Browser-first refreshes may take longer the first time, while cached abstracts make later runs faster.

Abstract status is explicit: `complete`, `excerpt`, or `unavailable`. Feed excerpts and Inoreader-generated summaries may still help title-level ranking, but they are never used as factual evidence or Idea Lab input. Confirmed complete abstracts receive a ranking preference when relevance is otherwise comparable. The requested result count remains exact when enough unique articles exist; an excerpt-only item can fill a remaining slot, but its detailed scientific analysis is intentionally withheld. Excluded sources and folders are removed before ranking; boost/lower rules affect ordering without changing article facts.

PaperPulse returns exactly N articles when at least N unique items remain after exclusions. If fewer than N eligible works exist, it returns all of them and reports the shortfall in the coverage funnel. Detailed analysis is generated separately for each selected article, and every accepted analysis must include a verbatim evidence excerpt from that same article.

## Deep Idea Lab

For each new brief, articles within the first five ranks that have a confirmed complete public abstract automatically receive a deeper abstract-only analysis. Other verified-abstract articles can be expanded on demand. The system atomizes the abstract into its central claim, supported observations, mechanism, method, causal links, boundary conditions, variables, gaps, unknowns, inferred assumptions, and plausible alternative explanations. Every factual element and every inference basis must point to a verbatim abstract quote. A conservative claim-to-quote overlap check also rejects unrelated real quotes attached to unsupported claims; unsupported fields remain explicitly unavailable.

The three directions use different reasoning operators rather than three paraphrases: direct validation discriminates a claimed cause from its strongest alternative, method transfer tests whether a supported mechanism survives a new context, and the high-risk hypothesis inverts one inferred assumption. Every idea exposes its typed evidence/inference/assumption chain, the abstract gap it targets, a competing explanation, and the observation that would distinguish them. A deterministic quality check requests one rewrite when the ideas are underspecified or too similar. If that optional rewrite fails, the first usable draft is preserved with its remaining warnings instead of failing the whole Idea Lab.

An independent critic scores testability, generic feasibility, potential impact, evidence strength, and novelty confidence. OpenAlex and Crossref are queried for each idea, while arXiv supplies additional article-level context. These searches are deliberately treated as incomplete prior-art leads: a low number of results never proves novelty, and the interface never claims global priority.

## Privacy and encryption

The following stay in the local `data/` directory and are excluded from Git:

- encrypted uploaded CV files and extracted profile
- encrypted OAuth access and refresh tokens
- encrypted titles, summaries, URLs, embeddings, recommendations, and idea notes
- local feedback and brief history
- a dedicated Chrome profile used only for publisher-page abstract retrieval

The automatically generated key is stored as `data/.paperpulse.key`. This prevents casual inspection of the database, but anyone who obtains both the data directory and that key can decrypt it. The dedicated Chrome profile uses Chrome's normal local profile storage and is not encrypted by PaperPulse. Visiting publisher pages also exposes ordinary browser request information, such as your IP address, to those publishers. For stronger separation on a new installation, set `PAPERPULSE_ENCRYPTION_KEY` before the first start and keep that value outside the project and data backups. Do not change keys after data has been created: key rotation is not yet automated. Back up the active key because encrypted data cannot be recovered without it.

CV text is sent to OpenAI when the profile is built. During refresh, candidate titles and summaries are sent for embeddings and candidate selection. Each selected article is then sent separately for detailed analysis and evidence validation. The top five abstracts also receive Idea Lab generation and an independent critic pass. Technical novelty queries and article metadata—not the CV—are sent to OpenAlex, Crossref, and arXiv. Requests use `store: false` where supported.

## Development and checks

```bash
npm test
python -m unittest discover -s tests -p "test_*.py"
```

The dashboard uses React with vinext. The local API uses FastAPI and SQLite. GitHub Actions runs both test suites on pushes and pull requests.

## License

MIT
