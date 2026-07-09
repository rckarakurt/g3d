"""Google Drive style-reference paths for Colab (mount + vrcaps lookup)."""

from __future__ import annotations

import os
from pathlib import Path

DRIVE_VRCAPS_DEFAULT = Path("/content/drive/MyDrive/vrcaps")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def drive_mounted() -> bool:
    return Path("/content/drive").is_dir()


def ensure_drive_mounted(*, force_remount: bool = False) -> Path:
    """Colab'da Google Drive'i bagla; MyDrive yolunu dondur."""
    my_drive = Path("/content/drive/MyDrive")
    if drive_mounted() and my_drive.exists():
        return my_drive
    try:
        from google.colab import drive as colab_drive
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive mount gerekli. Colab disinda calistiriyorsaniz "
            "VRCAPS_DRIVE_VRCAPS ortam degiskeni ile yerel klasor verin."
        ) from exc
    print("Google Drive baglaniyor...")
    colab_drive.mount("/content/drive", force_remount=force_remount)
    if not my_drive.exists():
        raise RuntimeError(
            "Drive mount tamamlandi ama MyDrive bulunamadi: /content/drive/MyDrive"
        )
    return my_drive


def resolve_drive_vrcaps_root(*, mount: bool = True) -> Path:
    """MyDrive/vrcaps klasorunu bul (env override + yaygin yedek yollar)."""
    env = os.environ.get("VRCAPS_DRIVE_VRCAPS", "").strip()
    if env:
        root = Path(env)
        if root.is_dir():
            return root
        raise FileNotFoundError(f"VRCAPS_DRIVE_VRCAPS klasoru yok: {root}")

    if mount:
        ensure_drive_mounted()
    elif not drive_mounted():
        raise RuntimeError(
            "Drive mount yok. Bolum 3b oncesi drive.mount yapin veya Bolum 5b calistirin."
        )

    candidates: list[Path] = [
        DRIVE_VRCAPS_DEFAULT,
        Path("/content/drive/My Drive/vrcaps"),
    ]
    my_drive = Path("/content/drive/MyDrive")
    if my_drive.is_dir():
        direct = my_drive / "vrcaps"
        if direct not in candidates:
            candidates.insert(0, direct)
        try:
            for hit in my_drive.glob("**/vrcaps"):
                if hit.is_dir() and hit not in candidates:
                    candidates.append(hit)
        except OSError:
            pass

    checked: list[str] = []
    for root in candidates:
        checked.append(str(root))
        if root.is_dir():
            return root

    listing = ""
    if my_drive.is_dir():
        try:
            names = sorted(p.name for p in my_drive.iterdir())
            listing = "\nMyDrive icerik (ilk 30): " + ", ".join(names[:30])
        except OSError:
            pass
    raise FileNotFoundError(
        "Drive'da vrcaps klasoru bulunamadi.\n"
        f"Aranan: {checked}\n"
        "Beklenen: MyDrive/vrcaps/kvasir_style_ref.jpg ve polyp_texture.jpg"
        f"{listing}"
    )


def resolve_drive_style_ref(stem: str, drive_vrcaps: Path | None = None) -> Path:
    """Drive/vrcaps altinda stem.jpg / stem.png (buyuk-kucuk harf duyarsiz)."""
    root = Path(drive_vrcaps) if drive_vrcaps is not None else resolve_drive_vrcaps_root()
    if not root.is_dir():
        raise FileNotFoundError(f"vrcaps klasoru yok: {root}")

    for name in (f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png", f"{stem}.webp"):
        candidate = root / name
        if candidate.is_file() and candidate.stat().st_size > 1_000:
            return candidate

    stem_lower = stem.lower()
    matches: list[Path] = []
    try:
        for entry in root.iterdir():
            if not entry.is_file():
                continue
            if entry.stem.lower() != stem_lower:
                continue
            if entry.suffix.lower() not in IMAGE_EXTS:
                continue
            if entry.stat().st_size > 1_000:
                matches.append(entry)
    except OSError as exc:
        raise FileNotFoundError(f"vrcaps klasoru okunamadi: {root}") from exc

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches, key=lambda p: p.name.lower())[0]

    try:
        files = sorted(p.name for p in root.iterdir() if p.is_file())
    except OSError:
        files = []
    raise FileNotFoundError(
        f"Drive stil referansi yok: {root / stem}.[jpg|jpeg|png]\n"
        f"vrcaps dosyalari: {files[:40]}"
    )
