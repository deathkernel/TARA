"""Load optional local .env configuration for TARA without exposing secrets."""
from __future__ import annotations

from pathlib import Path
import os


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs if the local .env file exists.

    Existing environment variables always win. Values are intentionally kept
    local; callers should never commit the .env file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
