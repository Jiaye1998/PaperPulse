# PaperPulse

**Find what matters in your daily research feed.**

Created by [Jiaye1998](https://github.com/Jiaye1998).

PaperPulse is a local-first research intelligence dashboard. It reads currently unread Inoreader items, learns an editable research profile from a PDF or DOCX CV, and creates an English shortlist of the strongest articles—up to your chosen maximum, never padded to a quota.

![PaperPulse social card](public/og.png)

## Highlights

- Read-only Inoreader OAuth; no read/star/save writeback
- CV-based personalization with an editable research lens
- Two-stage OpenAI ranking: embeddings, then detailed analysis of a shortlist
- Strict, balanced, and exploratory discovery modes
- Boost, lower, or exclude individual sources and Inoreader folders
- Any shortlist maximum from 1 to 100
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

Every refresh fetches articles that are still unread within the configured scan window, up to 1,000 entries. Cached articles that are no longer unread are not reconsidered. Thin or missing summaries remain eligible with lower confidence. Excluded sources and folders are removed before ranking; boost/lower rules affect ordering without changing article facts.

The result may contain fewer than N articles—or none—when the available items do not meet the active discovery mode.

## Privacy and encryption

The following stay in the local `data/` directory and are excluded from Git:

- encrypted uploaded CV files and extracted profile
- encrypted OAuth access and refresh tokens
- encrypted titles, summaries, URLs, embeddings, recommendations, and idea notes
- local feedback and brief history

The automatically generated key is stored as `data/.paperpulse.key`. This prevents casual inspection of the database, but anyone who obtains both the data directory and that key can decrypt it. For stronger separation on a new installation, set `PAPERPULSE_ENCRYPTION_KEY` before the first start and keep that value outside the project and data backups. Do not change keys after data has been created: key rotation is not yet automated. Back up the active key because encrypted data cannot be recovered without it.

CV text is sent to OpenAI when the profile is built. During refresh, candidate titles and summaries are sent for embeddings, and shortlisted candidates are sent for detailed analysis. Requests use `store: false` where supported.

## Development and checks

```bash
npm test
python -m unittest discover -s tests -p "test_*.py"
```

The dashboard uses React with vinext. The local API uses FastAPI and SQLite. GitHub Actions runs both test suites on pushes and pull requests.

## License

MIT
