"""Standard Colab /content paths — Drive optional backup only."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

CONTENT = Path("/content")
REPO_ENV = "VRCAPS_REPO"
DEFAULT_REPO = CONTENT / "g3d"
LEGACY_ORBIT = CONTENT / "vrcaps_scripts"
LEGACY_GAZE = LEGACY_ORBIT / "gaze"


def repo_root() -> Path:
    env = os.environ.get(REPO_ENV, "").strip()
    if env:
        return Path(env)
    if DEFAULT_REPO.exists():
        return DEFAULT_REPO
    return DEFAULT_REPO


def orbit_scripts_dir() -> Path:
    cloned = repo_root() / "colab" / "orbit_coating"
    if (cloned / "coating_utils.py").exists():
        return cloned
    if (LEGACY_ORBIT / "coating_utils.py").exists():
        return LEGACY_ORBIT
    return cloned


def gaze_scripts_dir() -> Path:
    cloned = repo_root() / "captures"
    if (cloned / "gaze_composite.py").exists():
        return cloned
    if (LEGACY_GAZE / "gaze_composite.py").exists():
        return LEGACY_GAZE
    return cloned


def install_sys_path() -> tuple[Path, Path]:
    """Idempotent sys.path setup for Colab (GitHub clone or legacy embed)."""
    orbit = orbit_scripts_dir()
    gaze = gaze_scripts_dir()
    for path in (str(orbit), str(gaze)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return orbit, gaze


def print_repo_info() -> None:
    orbit, gaze = orbit_scripts_dir(), gaze_scripts_dir()
    print("VRCaps repo:", repo_root())
    print("  orbit:", orbit, ("OK" if orbit.exists() else "YOK"))
    print("  gaze:", gaze, ("OK" if gaze.exists() else "YOK"))
PLY_OUT = CONTENT / "ply_styleshot_out"
VIEW_BANK = PLY_OUT / "view_bank"
UNITY_DATASET = CONTENT / "medical_gan_dataset"
GAZE_COMPOSITE_OUT = CONTENT / "gaze_composite"

DRIVE_ROOT = Path("/content/drive/MyDrive/vrcaps")
# Kullanici Drive yolu: MyDrive/vrcaps/medical_gan_dataset
DATASET_DRIVE = DRIVE_ROOT / "medical_gan_dataset"
PLY_OUT_DRIVE = DRIVE_ROOT / "ply_styleshot_out"
GAZE_COMPOSITE_DRIVE = DRIVE_ROOT / "gaze_composite"
GAZE_OUT_DRIVE = DRIVE_ROOT / "gaze_out"


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
        raise RuntimeError("Drive mount tamamlandi ama MyDrive bulunamadi: /content/drive/MyDrive")
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
        DRIVE_ROOT,
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


def resolve_unity_dataset() -> Path:
    """Prefer /content/medical_gan_dataset; fallback to Drive if mounted."""
    if UNITY_DATASET.exists() and (UNITY_DATASET / "rgb").is_dir():
        return UNITY_DATASET
    if drive_mounted() and DATASET_DRIVE.exists() and (DATASET_DRIVE / "rgb").is_dir():
        return DATASET_DRIVE
    return UNITY_DATASET


def resolve_view_bank() -> Path:
    """Bolum 3 ciktisi — her zaman once /content."""
    if VIEW_BANK.exists() and (VIEW_BANK / "view_manifest.json").exists():
        return VIEW_BANK
    drive_bank = PLY_OUT_DRIVE / "view_bank"
    if drive_mounted() and drive_bank.exists():
        return drive_bank
    return VIEW_BANK


def sync_dataset_from_drive(dest: Path | None = None) -> Path:
    """MyDrive/vrcaps/medical_gan_dataset -> /content/medical_gan_dataset kopyala."""
    dest = Path(dest or UNITY_DATASET)
    if not drive_mounted():
        raise RuntimeError("Drive mount yok. Bolum 5b once drive.mount yapar.")
    if not (DATASET_DRIVE / "rgb").is_dir():
        raise FileNotFoundError(
            f"Drive dataset yok: {DATASET_DRIVE}\n"
            "Beklenen: MyDrive/vrcaps/medical_gan_dataset/ (rgb/, depth/, poses/)"
        )
    skip = {"splataam", "depth_preview", "gaze_rgb", "gaze_rgb_local", "figures", "gaze_composite"}

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in skip or n.endswith(".rar")}

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DATASET_DRIVE, dest, ignore=_ignore)
    return dest


def copy_tree_if_requested(src: Path, dst: Path, *, enabled: bool) -> None:
    if not enabled or not drive_mounted() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print("Drive yedek:", dst)
