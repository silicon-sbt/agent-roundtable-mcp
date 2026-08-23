from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from llm_gateway.local_claude import list_local_claude_models
from llm_gateway.local_codex import list_local_codex_models
from llm_gateway.proxy import list_proxy_model_ids

from .environment import PROJECT_ROOT, SHARED_ENV_FILE_VAR

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - dependency is declared for normal use.
    dotenv_values = None


CATALOG_CONFIG_PATH = PROJECT_ROOT / "config" / "model_catalog.json"

CLIPROXY_PROVIDERS = {"codex_proxy", "antigravity", "grok", "copilot"}

# Built-in defaults, used when config/model_catalog.json is missing or invalid.
_DEFAULT_PROVIDER_MODEL_ENV = {
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
    "openrouter": "OPENROUTER_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
}

_DEFAULT_PROVIDER_KEY_PREFIXES = {
    "gemini": ["GEMINI_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
}

_DEFAULT_DEEPSEEK_OFFICIAL_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
    "deepseek-reasoner",
]


def _load_catalog_config(path: Path = CATALOG_CONFIG_PATH):
    """Load provider catalog data, falling back to built-in defaults."""
    model_env = dict(_DEFAULT_PROVIDER_MODEL_ENV)
    key_prefixes = {k: list(v) for k, v in _DEFAULT_PROVIDER_KEY_PREFIXES.items()}
    deepseek_models = list(_DEFAULT_DEEPSEEK_OFFICIAL_MODELS)

    try:
        providers = json.loads(path.read_text(encoding="utf-8")).get("providers", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return model_env, key_prefixes, deepseek_models

    if isinstance(providers, dict):
        for name, cfg in providers.items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("model_env"):
                model_env[name] = str(cfg["model_env"])
            if isinstance(cfg.get("key_prefixes"), list):
                key_prefixes[name] = [str(item) for item in cfg["key_prefixes"]]
            if name == "deepseek" and isinstance(cfg.get("official_models"), list):
                deepseek_models = [str(item) for item in cfg["official_models"]]
    return model_env, key_prefixes, deepseek_models


PROVIDER_MODEL_ENV, PROVIDER_KEY_PREFIXES, DEEPSEEK_OFFICIAL_MODELS = _load_catalog_config()


@dataclass(frozen=True)
class ModelCatalogResult:
    models: list[str]
    source: str
    error: str | None = None


def _env_values(
    root_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if environ is not None:
        return {key: value for key, value in environ.items() if value}

    values: dict[str, str] = {}
    env_path = Path(root_dir or PROJECT_ROOT) / ".env"
    if dotenv_values is not None:
        if env_path.exists():
            values.update(
                {
                    str(key): str(value)
                    for key, value in dotenv_values(env_path).items()
                    if value
                }
            )
        shared_path = str(
            os.environ.get(SHARED_ENV_FILE_VAR)
            or values.get(SHARED_ENV_FILE_VAR)
            or ""
        ).strip()
        if shared_path:
            shared_values = dotenv_values(Path(shared_path).expanduser())
            for key, value in shared_values.items():
                if value:
                    values.setdefault(str(key), str(value))
    values.update({key: value for key, value in os.environ.items() if value})
    return values


def _sort_key_env_names(prefix: str, values: Mapping[str, str]) -> list[str]:
    numbered: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for name, value in values.items():
        match = pattern.match(name)
        if match and value:
            numbered.append((int(match.group(1)), name))

    names = [name for _, name in sorted(numbered)]
    if values.get(prefix):
        names.append(prefix)
    return list(dict.fromkeys(names))


def list_api_key_env_names(
    provider: str,
    *,
    root_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    values = _env_values(root_dir, environ)
    names: list[str] = []
    for prefix in PROVIDER_KEY_PREFIXES.get(provider, []):
        names.extend(_sort_key_env_names(prefix, values))
    return list(dict.fromkeys(names))


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def fallback_model_options(
    provider: str,
    *,
    current_model: str | None = None,
    root_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    values = _env_values(root_dir, environ)
    if provider == "deepseek":
        options = list(DEEPSEEK_OFFICIAL_MODELS)
    else:
        options = [current_model.strip()] if current_model and current_model.strip() else []
    model_env = PROVIDER_MODEL_ENV.get(provider)
    if model_env:
        options.extend(_split_csv(values.get(model_env)))
    return list(dict.fromkeys(options))


def parse_openrouter_models(payload: Mapping[str, Any]) -> list[str]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    ids = [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]
    return sorted({model_id for model_id in ids if model_id})


def parse_gemini_models(payload: Mapping[str, Any]) -> list[str]:
    data = payload.get("models", [])
    if not isinstance(data, list):
        return []

    models: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods", [])
        if isinstance(methods, list) and "generateContent" not in methods:
            continue
        name = str(item.get("name", "")).strip()
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name:
            models.add(name)
    return sorted(models)


def _parse_openai_compatible_models(payload: Mapping[str, Any]) -> list[str]:
    return parse_openrouter_models(payload)


def _read_json_url(url: str, *, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_model_options(
    provider: str,
    *,
    api_key_env: str | None = None,
    current_model: str | None = None,
    root_dir: Path | str | None = None,
    timeout: int = 20,
) -> ModelCatalogResult:
    values = _env_values(root_dir)
    fallback = fallback_model_options(provider, current_model=current_model, root_dir=root_dir)
    api_key = values.get(api_key_env or "") if api_key_env else None

    try:
        if provider == "claude":
            models = list_local_claude_models()
            return ModelCatalogResult(
                list(dict.fromkeys(models + fallback)),
                "claude_aliases",
                None,
            )

        if provider == "codex":
            models = list_local_codex_models(env_override=values)
            error = None if models else "Codex model cache has no visible models."
            return ModelCatalogResult(models or fallback, "codex_cache", error)

        if provider in CLIPROXY_PROVIDERS:
            proxy_environment = dict(values)
            proxy_environment["CLI_PROXY_TIMEOUT"] = str(timeout)
            models = list_proxy_model_ids(
                provider,
                project_root=root_dir or PROJECT_ROOT,
                env_override=proxy_environment,
            )
            error = None if models else f"CLIProxyAPI exposed no text models owned by {provider}."
            return ModelCatalogResult(models or fallback, "cliproxy", error)

        if provider == "deepseek":
            return ModelCatalogResult(fallback, "official", None)

        if provider == "openrouter":
            headers = {"Accept": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = _read_json_url(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=timeout,
            )
            models = parse_openrouter_models(payload)
            return ModelCatalogResult(models or fallback, "openrouter", None)

        if provider == "gemini":
            if not api_key:
                return ModelCatalogResult(fallback, "env", "No Gemini API key selected.")
            payload = _read_json_url(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
            models = parse_gemini_models(payload)
            return ModelCatalogResult(models or fallback, "gemini", None)

        if provider == "openai":
            if not api_key:
                return ModelCatalogResult(fallback, "env", "No OpenAI API key selected.")
            payload = _read_json_url(
                "https://api.openai.com/v1/models",
                headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            models = _parse_openai_compatible_models(payload)
            return ModelCatalogResult(models or fallback, "openai", None)

        return ModelCatalogResult(fallback, "env", None)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, RuntimeError) as exc:
        source = {
            "codex": "codex_cache",
            "codex_proxy": "cliproxy",
            "antigravity": "cliproxy",
            "grok": "cliproxy",
            "copilot": "cliproxy",
            "deepseek": "official",
            "openrouter": "openrouter",
            "gemini": "gemini",
            "openai": "openai",
        }.get(provider, "env")
        return ModelCatalogResult(fallback, source, str(exc))
