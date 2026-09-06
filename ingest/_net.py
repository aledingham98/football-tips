"""Shared HTTP + secrets helpers for the ingestion layer.

* :func:`make_session` - a ``requests.Session`` with a browser-ish User-Agent and,
  where the ``truststore`` package is available, the OS trust store injected
  (needed behind TLS-intercepting proxies; a no-op on clean CI).
* :func:`get_secret` - read a credential from the environment, falling back to
  ``.streamlit/secrets.toml`` so the same code works locally and on Streamlit
  Cloud / GitHub Actions.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

import requests

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_SECRETS_PATH = Path(".streamlit/secrets.toml")


def make_session(extra_headers: dict[str, str] | None = None) -> requests.Session:
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Accept-Language": "en-GB,en;q=0.9"})
    if extra_headers:
        s.headers.update(extra_headers)
    return s


@lru_cache(maxsize=1)
def _secrets_file() -> dict:
    if _SECRETS_PATH.exists():
        return tomllib.loads(_SECRETS_PATH.read_text())
    return {}


def get_secret(*path: str, default: str | None = None, env: str | None = None) -> str | None:
    """``get_secret("football_data_org", "api_key", env="FOOTBALL_DATA_ORG_API_KEY")``.

    Environment variable wins; then the nested key in ``.streamlit/secrets.toml``;
    then ``default``.
    """
    if env and (val := os.environ.get(env)):
        return val
    node: object = _secrets_file()
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if isinstance(node, str) else default
