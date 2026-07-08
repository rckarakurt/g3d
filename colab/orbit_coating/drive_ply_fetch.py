"""Download polyp PLY from a shared Google Drive folder (gdown)."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from colab_net import configure_colab_ssl, gdown_download, gdown_download_folder, http_get_text

# Dogrudan .ply klasoru
# https://drive.google.com/drive/folders/1P9a6WMqMLzmyvg53fOqZV6FRxA-w3veo
POLYP_DRIVE_FOLDER_ID = "1P9a6WMqMLzmyvg53fOqZV6FRxA-w3veo"
POLYP_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1P9a6WMqMLzmyvg53fOqZV6FRxA-w3veo"
)

DEFAULT_OUT_DIR = Path("/content/drive_ply_inbox")
DEFAULT_POLYP_NAME = "polyp_0004.ply"
SELECTED_PLY_MARKER = Path("/content/vrcaps_checkpoints/selected_ply.json")


def _ensure_gdown() -> None:
    configure_colab_ssl()
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        check=True,
    )


def _find_ply_in_tree(root: Path, polyp_name: str) -> Path | None:
    root = Path(root)
    exact = [p for p in root.rglob("*.ply") if p.name == polyp_name]
    if exact:
        return exact[0].resolve()

    all_ply = sorted(root.rglob("*.ply"))
    if len(all_ply) == 1:
        return all_ply[0].resolve()
    if all_ply:
        print(f"Uyari: {polyp_name} yok, ilk .ply kullaniliyor: {all_ply[0].name}")
        return all_ply[0].resolve()
    return None


def _list_ply_files_embedded(folder_id: str) -> list[tuple[str, str]]:
    """Parse Drive embedded folder view → [(filename, file_id), ...] for .ply only."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    html = http_get_text(url, timeout=60)

    pairs: list[tuple[str, str]] = []
    for block in re.split(r'class="flip-entry"', html):
        if ".ply" not in block.lower():
            continue
        name_m = re.search(r'class="flip-entry-title">([^<]+)', block)
        id_m = re.search(r'/file/d/([a-zA-Z0-9_-]+)/', block)
        if name_m and id_m:
            name = name_m.group(1).strip()
            if name.lower().endswith(".ply"):
                pairs.append((name, id_m.group(1)))

    if pairs:
        return pairs

    ids = re.findall(r"/file/d/([a-zA-Z0-9_-]{20,})/", html)
    names = re.findall(r'flip-entry-title">([^<]+\.ply)<', html, flags=re.I)
    if len(ids) == len(names):
        return list(zip(names, ids))
    return []


def _download_ply_files_selective(
    folder_id: str,
    out_dir: Path,
    polyp_name: str,
) -> Path:
    """Buyuk klasorler icin: yalnizca .ply dosyalarini tek tek indir."""
    _ensure_gdown()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _list_ply_files_embedded(folder_id)
    if not pairs:
        raise FileNotFoundError(
            f"Klasorde .ply listelenemedi (embedded view). folder_id={folder_id}"
        )

    print(f"Secici indirme: {len(pairs)} .ply bulundu")
    downloaded: list[Path] = []
    for name, file_id in pairs:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 1000:
            downloaded.append(dest)
            continue
        print(f"  indiriliyor: {name}")
        gdown_download(id=file_id, output=str(dest), quiet=False)
        if dest.exists():
            downloaded.append(dest)

    ply_path = _find_ply_in_tree(out_dir, polyp_name)
    if ply_path is None:
        raise FileNotFoundError(
            f"{polyp_name} indirilemedi. Bulunan: {[p.name for p in downloaded]}"
        )
    return ply_path


def download_drive_folder(
    folder_id: str = POLYP_DRIVE_FOLDER_ID,
    out_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download all files from a public/shared Drive folder."""
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    marker = out_dir / ".download_ok"

    if marker.exists() and not force and any(out_dir.rglob("*.ply")):
        print("Drive klasoru zaten indirilmis:", out_dir)
        return out_dir

    _ensure_gdown()

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Drive klasoru indiriliyor...")
    print(POLYP_DRIVE_FOLDER_URL)
    try:
        gdown_download_folder(
            id=folder_id,
            output=str(out_dir),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        err = type(exc).__name__ + " " + str(exc)
        if "MaximumLimit" in err or "MaximumLimit" in type(exc).__name__:
            print("Klasor cok buyuk (>50 dosya) — yalnizca .ply secici indiriliyor...")
            _download_ply_files_selective(folder_id, out_dir, DEFAULT_POLYP_NAME)
        else:
            raise

    marker.write_text(folder_id, encoding="utf-8")
    return out_dir


def fetch_ply_from_drive_folder(
    polyp_name: str = DEFAULT_POLYP_NAME,
    *,
    folder_id: str = POLYP_DRIVE_FOLDER_ID,
    out_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download shared folder and return path to ``polyp_name``."""
    configure_colab_ssl()
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)

    if force_download and out_dir.exists():
        shutil.rmtree(out_dir)

    cached = _find_ply_in_tree(out_dir, polyp_name) if out_dir.exists() else None
    if cached and not force_download:
        print("Onbellekte PLY var:", cached)
    else:
        download_drive_folder(folder_id, out_dir, force=force_download)
        cached = _find_ply_in_tree(out_dir, polyp_name)
        if cached is None:
            cached = _download_ply_files_selective(folder_id, out_dir, polyp_name)

    ply_path = cached
    SELECTED_PLY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_PLY_MARKER.write_text(
        json.dumps(
            {
                "ply_path": str(ply_path),
                "polyp_name": polyp_name,
                "folder_id": folder_id,
                "folder_url": POLYP_DRIVE_FOLDER_URL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Secilen PLY:", ply_path)
    return ply_path


def load_selected_ply() -> Path | None:
    if not SELECTED_PLY_MARKER.exists():
        return None
    data = json.loads(SELECTED_PLY_MARKER.read_text(encoding="utf-8"))
    path = Path(data["ply_path"])
    return path if path.exists() else None
