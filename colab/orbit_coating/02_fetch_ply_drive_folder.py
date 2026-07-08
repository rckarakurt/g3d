# 2 — Drive paylasim klasorunden polyp PLY indir
import sys
from pathlib import Path

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()

from colab_net import configure_colab_ssl

configure_colab_ssl()

from drive_ply_fetch import (
    POLYP_DRIVE_FOLDER_URL,
    fetch_ply_from_drive_folder,
    load_selected_ply,
)

POLYP_NAME = "polyp_0004.ply"
FORCE_REDOWNLOAD = False  # eski indirme varsa bir kez True yapin

existing = load_selected_ply()
if existing and not FORCE_REDOWNLOAD:
    PLY_PATH = existing
    print("Onceki indirme kullaniliyor:", PLY_PATH)
else:
    PLY_PATH = fetch_ply_from_drive_folder(
        POLYP_NAME,
        force_download=FORCE_REDOWNLOAD,
    )

print("Kaynak klasor:", POLYP_DRIVE_FOLDER_URL)
print("PLY hazir:", PLY_PATH)
