from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .config import config


PREFIX = "enc:v1:"


def _key_file() -> Path:
    data_dir = getattr(config, "data_dir", None)
    if data_dir is not None:
        return Path(data_dir) / ".paperpulse.key"
    return Path(config.database_path).parent / ".paperpulse.key"


@lru_cache(maxsize=8)
def _fernet(key_path: str, configured_key: str) -> Fernet:
    path = Path(key_path)
    if configured_key:
        key = configured_key.encode("ascii")
    elif path.exists():
        key = path.read_bytes().strip()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        path.write_bytes(key)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    try:
        return Fernet(key)
    except (ValueError, TypeError) as error:
        raise RuntimeError(
            "PAPERPULSE_ENCRYPTION_KEY must be a valid Fernet key."
        ) from error


def cipher() -> Fernet:
    return _fernet(
        str(_key_file().resolve()),
        os.getenv("PAPERPULSE_ENCRYPTION_KEY", "").strip(),
    )


def encrypt_text(value: str | None) -> str | None:
    if value is None or value.startswith(PREFIX):
        return value
    token = cipher().encrypt(value.encode("utf-8")).decode("ascii")
    return PREFIX + token


def decrypt_text(value: str | None) -> str | None:
    if value is None or not value.startswith(PREFIX):
        return value
    try:
        return cipher().decrypt(value[len(PREFIX) :].encode("ascii")).decode("utf-8")
    except InvalidToken as error:
        raise RuntimeError(
            "PaperPulse could not decrypt local data. Restore the matching encryption key."
        ) from error


def encrypt_bytes(value: bytes) -> bytes:
    return cipher().encrypt(value)


def decrypt_bytes(value: bytes) -> bytes:
    try:
        return cipher().decrypt(value)
    except InvalidToken as error:
        raise RuntimeError(
            "PaperPulse could not decrypt the local file. Restore the matching encryption key."
        ) from error
