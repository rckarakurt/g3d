"""SSL / download helpers for Google Colab (CERTIFICATE_VERIFY_FAILED fix)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SSL_CONFIGURED = False


def configure_colab_ssl(*, unverified_fallback: bool = True) -> None:
    """Install certifi and patch SSL for gdown / urllib / requests."""
    global _SSL_CONFIGURED

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "certifi", "requests"],
        check=False,
    )
    try:
        import certifi
        import os

        bundle = certifi.where()
        os.environ["SSL_CERT_FILE"] = bundle
        os.environ["REQUESTS_CA_BUNDLE"] = bundle
    except ImportError:
        pass

    if unverified_fallback:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: S501

    _SSL_CONFIGURED = True


def download_url(url: str, dest: str | Path, *, timeout: int = 300) -> Path:
    """Download file — Colab SSL-safe (requests verify=False + urllib fallback)."""
    configure_colab_ssl()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    import requests

    try:
        with requests.get(
            url,
            timeout=timeout,
            verify=False,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as resp:
            resp.raise_for_status()
            with dest.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        if dest.stat().st_size > 0:
            return dest
    except Exception as exc:
        print("requests indirme basarisiz, urllib deneniyor:", exc)

    import ssl
    import urllib.request

    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        dest.write_bytes(resp.read())
    return dest


def http_get_text(url: str, *, timeout: int = 60) -> str:
    configure_colab_ssl()
    import requests

    resp = requests.get(
        url,
        timeout=timeout,
        verify=False,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    return resp.text


def gdown_download(url_or_id: str, dest: str | Path, **kwargs):
    configure_colab_ssl()
    import gdown

    return gdown.download(url_or_id, str(dest), **kwargs)


def gdown_download_folder(**kwargs):
    configure_colab_ssl()
    import gdown

    return gdown.download_folder(**kwargs)
