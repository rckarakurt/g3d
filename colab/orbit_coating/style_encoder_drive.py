"""Kvasir fine-tune style encoder — Google Drive indirme."""

from __future__ import annotations

from pathlib import Path

from colab_net import configure_colab_ssl, gdown_download

# https://drive.google.com/file/d/1YW4dMwODu7jPNGTnQx4VQqFZVpbNLwqY/view
STYLE_ENCODER_GDRIVE_ID = "1YW4dMwODu7jPNGTnQx4VQqFZVpbNLwqY"
STYLE_ENCODER_GDRIVE_URL = (
    "https://drive.google.com/file/d/1YW4dMwODu7jPNGTnQx4VQqFZVpbNLwqY/view"
)
DEFAULT_STYLE_ENCODER_PATH = Path("/content/vrcaps_checkpoints/style_encoder_final.bin")


def ensure_style_encoder(dest: Path | None = None) -> Path:
    """Drive'da yoksa Google Drive linkinden indirir (~3 GB)."""
    configure_colab_ssl()
    dest = Path(dest or DEFAULT_STYLE_ENCODER_PATH)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print("Style encoder hazir:", dest, f"({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Style encoder indiriliyor (Google Drive, ~3 GB)...")
    print("Link:", STYLE_ENCODER_GDRIVE_URL)

    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)

    gdown_download(
        f"https://drive.google.com/uc?id={STYLE_ENCODER_GDRIVE_ID}",
        str(dest),
        quiet=False,
    )
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Style encoder indirilemedi: {dest}\n"
            f"Manuel: {STYLE_ENCODER_GDRIVE_URL}"
        )
    print("Indirme OK:", dest, f"({dest.stat().st_size / 1e6:.0f} MB)")
    return dest
