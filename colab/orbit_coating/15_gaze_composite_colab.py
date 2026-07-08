# 15 — Aci eslestirmeli composite: Unity RGB + mesh (view_plane_deg)
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

get_ipython().run_line_magic("pip", "install -q opencv-python-headless scipy")

# ============ AYARLAR ============
# Her Unity karesi icin:
#   view_plane_deg (gaze_views.csv) -> view_bank'teki en yakin mesh acisi -> yapistir

SCALE_BOOST = 8.0
STRICT_ANGLE_MATCH = True   # True: kare bazli aci (yumusatma yok)
BANK_BLEND = True           # 5° arasi iki mesh gorunumu karistir
GLOBAL_SCALE = True

SMOOTH_WINDOW = 5           # STRICT=False ise
MAX_FRAMES = None           # test: 20
WRITE_DEBUG = True          # unity | mesh | composite (ilk N kare)
DEBUG_MAX_FRAMES = 12
MP4_FPS = 15.0
COPY_TO_DRIVE = False
# ================================

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()

from colab_content_paths import (
    DATASET_DRIVE,
    GAZE_COMPOSITE_DRIVE,
    GAZE_COMPOSITE_OUT,
    UNITY_DATASET,
    copy_tree_if_requested,
    drive_mounted,
    resolve_view_bank,
    sync_dataset_from_drive,
)
from export_gaze_views import export_gaze_views
from gaze_composite import build_strip, composite_dataset


def ensure_dataset_on_content() -> Path:
    if (UNITY_DATASET / "rgb").is_dir():
        return UNITY_DATASET
    if drive_mounted() and (DATASET_DRIVE / "rgb").is_dir():
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        print("Drive dataset -> /content kopyalaniyor...")
        return sync_dataset_from_drive(UNITY_DATASET)
    raise FileNotFoundError(
        f"Dataset yok.\n  Bolum 5b: MyDrive/vrcaps/medical_gan_dataset\n  veya {UNITY_DATASET}"
    )


def require_view_bank(view_bank: Path) -> None:
    manifest = view_bank / "view_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"View bank yok: {view_bank}\nOnce Bolum 3 calistirin."
        )
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    views = meta.get("views", [])
    azs = [float(v["azimuth_deg"]) for v in views]
    print(f"View bank: {view_bank}")
    print(f"  {len(views)} aci  ({min(azs):.0f}..{max(azs):.0f})")


def show_previews(out_dir: Path) -> None:
    from IPython.display import Image, display

    debug = sorted((out_dir / "debug").glob("match_*.png"))[:4]
    for p in debug:
        print(p.name)
        display(Image(filename=str(p), width=900))
    strip = out_dir / "composite_strip.png"
    if strip.exists():
        print("composite_strip.png")
        display(Image(filename=str(strip), width=900))


dataset = ensure_dataset_on_content()
view_bank = resolve_view_bank()
require_view_bank(view_bank)

gaze_csv = dataset / "poses" / "gaze_views.csv"
if not gaze_csv.exists():
    print("gaze_views.csv uretiliyor...")
    export_gaze_views(dataset, write_plot=False)

if GAZE_COMPOSITE_OUT.exists():
    shutil.rmtree(GAZE_COMPOSITE_OUT)
GAZE_COMPOSITE_OUT.mkdir(parents=True, exist_ok=True)

smooth_angles = not STRICT_ANGLE_MATCH
smooth_distance = not STRICT_ANGLE_MATCH

print("\n=== Aci eslestirmeli composite ===")
print("  dataset:  ", dataset)
print("  view_bank:", view_bank)
print("  strict:   ", STRICT_ANGLE_MATCH)

summary = composite_dataset(
    dataset,
    view_bank,
    GAZE_COMPOSITE_OUT,
    scale_boost=SCALE_BOOST,
    global_scale=GLOBAL_SCALE,
    smooth_window=SMOOTH_WINDOW,
    smooth_angles=smooth_angles,
    smooth_distance=smooth_distance,
    bank_blend=BANK_BLEND,
    max_frames=MAX_FRAMES,
    write_mp4=True,
    write_debug=WRITE_DEBUG,
    debug_max_frames=DEBUG_MAX_FRAMES,
    mp4_fps=MP4_FPS,
)
strip = build_strip(GAZE_COMPOSITE_OUT)
print("Kare:", summary["composite_count"])
print("Cikti:", GAZE_COMPOSITE_OUT / "rgb")
print("Debug:", GAZE_COMPOSITE_OUT / "debug")
print("MP4:", GAZE_COMPOSITE_OUT / "trajectory_composite.mp4")

copy_tree_if_requested(GAZE_COMPOSITE_OUT, GAZE_COMPOSITE_DRIVE, enabled=COPY_TO_DRIVE)
show_previews(GAZE_COMPOSITE_OUT)
