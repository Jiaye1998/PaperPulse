from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent


def main() -> None:
    env = os.environ.copy()
    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=ROOT,
        env=env,
    )
    npm = "npm.cmd" if os.name == "nt" else "npm"
    frontend = subprocess.Popen([npm, "run", "dev"], cwd=ROOT, env=env)

    def stop(*_: object) -> None:
        for process in (frontend, backend):
            if process.poll() is None:
                process.terminate()
        for process in (frontend, backend):
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if backend.poll() is not None or frontend.poll() is not None:
            stop()
        try:
            with urlopen("http://localhost:3000", timeout=1):
                break
        except (OSError, URLError):
            time.sleep(0.5)
    else:
        print("PaperPulse did not become ready within 30 seconds.", file=sys.stderr)
        stop()
    webbrowser.open("http://localhost:3000")
    print("PaperPulse is running at http://localhost:3000")
    while backend.poll() is None and frontend.poll() is None:
        time.sleep(0.5)
    stop()


if __name__ == "__main__":
    main()
