from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Mapping

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:  # pragma: no cover - dependency is declared for normal use.
    dotenv_values = None
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ENV_FILE_VAR = "LLM_SHARED_ENV_FILE"
_LOAD_LOCK = Lock()
_DOTENV_LOADED = False


def load_missing_env_file(path: Path | str) -> None:
    """Fill absent or blank environment values from another dotenv file."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return
    if dotenv_values is not None:
        for key, value in dotenv_values(resolved).items():
            if value and not os.environ.get(str(key)):
                os.environ[str(key)] = str(value)
    elif load_dotenv is not None:  # pragma: no cover - compatibility fallback.
        load_dotenv(resolved, override=False)


def load_project_env() -> None:
    """Load this repository's root .env exactly once, independent of cwd."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    with _LOAD_LOCK:
        if _DOTENV_LOADED:
            return
        if load_dotenv is not None:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
            shared_path = os.environ.get(SHARED_ENV_FILE_VAR, "").strip()
            if shared_path:
                load_missing_env_file(shared_path)
        _DOTENV_LOADED = True


def env_value(
    name: str,
    env_override: Mapping[str, str] | None = None,
    default: str = "",
) -> str:
    load_project_env()
    if env_override is not None and name in env_override:
        return str(env_override[name])
    return os.environ.get(name, default)


def merged_environment(env_override: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child-process environment without mutating the parent process."""
    load_project_env()
    environment = dict(os.environ)
    if env_override:
        environment.update({str(key): str(value) for key, value in env_override.items()})
    return environment
