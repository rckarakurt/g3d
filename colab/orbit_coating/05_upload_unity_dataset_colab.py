# 05 — Unity dataset: MyDrive/vrcaps/medical_gan_dataset → /content
from __future__ import annotations

import sys
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()
from colab_content_paths import DATASET_DRIVE, UNITY_DATASET, sync_dataset_from_drive

# ============ AYARLAR ============
# Drive yolu (Colab tam yol):
#   /content/drive/MyDrive/vrcaps/medical_gan_dataset
#
# Sizin Drive yolu:
#   MyDrive/vrcaps/medical_gan_dataset
#
SOURCE = "drive_folder"  # "drive_folder" | "drive_rar" | "archive"

# RAR yedek (klasor yoksa dene) — MyDrive/vrcaps/ altinda
DRIVE_RAR_NAME = "unity-vr-caps-dataset.rar"  # veya medical_gan_dataset.rar

ARCHIVE_PATH = None  # SOURCE="archive" ise /content/...zip
EXTRACT_TO = UNITY_DATASET
# ==================================


def _load_rar_from_drive() -> None:
    import shutil
    import subprocess
    import zipfile
    from colab_content_paths import DRIVE_ROOT

    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    for name in (DRIVE_RAR_NAME, "medical_gan_dataset.rar", "unity-vr-caps-dataset.rar"):
        rar_path = DRIVE_ROOT / name
        if not rar_path.exists():
            continue
        print("Drive RAR:", rar_path)
        local = Path("/content") / name
        if not local.exists():
            shutil.copy2(rar_path, local)
        staging = Path("/content/_dataset_unpack")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        subprocess.run(["apt-get", "install", "-y", "-qq", "unrar"], check=False)
        subprocess.run(["unrar", "x", "-o+", str(local), str(staging) + "/"], check=True)
        # finalize inline
        dest = EXTRACT_TO
        if dest.exists():
            shutil.rmtree(dest)
        if (staging / "medical_gan_dataset" / "rgb").exists():
            shutil.move(str(staging / "medical_gan_dataset"), str(dest))
        elif len(list(staging.iterdir())) == 1 and (next(staging.iterdir()) / "rgb").exists():
            shutil.move(str(next(staging.iterdir())), str(dest))
        elif (staging / "rgb").exists():
            shutil.move(str(staging), str(dest))
        shutil.rmtree(staging, ignore_errors=True)
        return
    raise FileNotFoundError(
        f"Drive'da ne klasor ne RAR bulundu.\n"
        f"  Klasor: {DATASET_DRIVE}\n"
        f"  RAR:    {DRIVE_ROOT / DRIVE_RAR_NAME}"
    )


print("Drive kaynak:", DATASET_DRIVE)
print("Hedef:       ", EXTRACT_TO)

if SOURCE == "drive_folder":
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    if (DATASET_DRIVE / "rgb").is_dir():
        sync_dataset_from_drive(EXTRACT_TO)
    else:
        print("Klasor yok, RAR deneniyor...")
        _load_rar_from_drive()
elif SOURCE == "drive_rar":
    _load_rar_from_drive()
else:
    raise ValueError("archive modu icin eski script veya ARCHIVE_PATH kullanin")

n_rgb = len(list((EXTRACT_TO / "rgb").glob("*.png")))
if n_rgb == 0:
    raise RuntimeError(f"rgb/ bos — kontrol edin: {EXTRACT_TO}")
print(f"\nHazir: {EXTRACT_TO}  ({n_rgb} kare)")
print("Sonraki: Bolum 3 → 6 → 7")
