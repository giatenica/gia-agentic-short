""" 
Subprocess Environment Utilities
===============================
Helpers for building minimal environment dictionaries for subprocess execution.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional


def build_minimal_subprocess_env(
    *,
    sanitize_env: bool = True,
    allowlist: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """Return an environment dict suitable for subprocess execution.

    When sanitize_env is True, only a small allowlist is inherited from the parent
    environment to reduce accidental secret leakage.

    Args:
        sanitize_env: If False, inherits the full parent env.
        allowlist: Optional extra allowlist keys to include.

    Returns:
        Dict[str, str] to pass as subprocess env.
    """

    base_allowlist = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    }

    if allowlist is not None:
        for key in allowlist:
            if isinstance(key, str) and key:
                base_allowlist.add(key)

    env: Dict[str, str] = {}
    parent = os.environ

    for key in base_allowlist:
        value = parent.get(key)
        if value is not None:
            env[key] = value

    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONNOUSERSITE", "1")

    if not sanitize_env:
        inherited = dict(parent)
        inherited.update(env)
        return inherited

    return env
