from __future__ import annotations

from os import environ
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOADED = False


def load_project_env() -> None:
    global _LOADED
    if _LOADED:
        return

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        _LOADED = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        environ.setdefault(key, value)

    _LOADED = True
