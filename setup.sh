#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env. Add your OpenAI and Inoreader credentials before connecting."
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt
npm install
echo "PaperPulse is installed. Run .venv/bin/python start.py to start it."
