# 14 — Gaze anchor + viewing angles + trajectory map
from __future__ import annotations

import json
import sys
from pathlib import Path

get_ipython().run_line_magic("pip", "install -q pandas scipy matplotlib")

# ============ AYARLAR ============
COPY_TO_DRIVE = False  # True = istege bagli Drive yedegi

MARKED_RGB = True
MARKED_MAX_FRAMES = None
# ================================

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()

from colab_content_paths import (
    GAZE_OUT_DRIVE,
    copy_tree_if_requested,
    resolve_unity_dataset,
)
from export_gaze_trajectory_map import export_gaze_trajectory_map
from unity_dataset_angles import ensure_geometric_gaze_views

if MARKED_RGB:
    from export_gaze_rgb_marked import export_marked_rgb


def _require_dataset(root: Path) -> None:
    missing = []
    for rel in ("camera.json", "rgb", "depth"):
        if not (root / rel).exists():
            missing.append(str(root / rel))
    pose_candidates = [
        root / "poses" / "poses_with_directions.csv",
        root / "poses" / "poses_per_frame.csv",
        root / "poses" / "position_rotation.csv",
    ]
    if not any(p.exists() for p in pose_candidates):
        missing.append(str(root / "poses/*.csv"))
    if missing:
        raise FileNotFoundError(
            "Dataset eksik:\n  "
            + "\n  ".join(missing)
            + "\n\nOnce Bolum 5b calistirin (Drive -> /content/medical_gan_dataset)."
        )
    n_rgb = len(list((root / "rgb").glob("*.png")))
    n_depth = len(list((root / "depth").glob("*.png")))
    print(f"Dataset OK: {root}")
    print(f"  rgb={n_rgb}  depth={n_depth}")
    cam = json.loads((root / "camera.json").read_text(encoding="utf-8"))
    print(f"  image {cam['image_width']}x{cam['image_height']}  depth_scale={cam['depth_scale']}")


def _show_colab_previews(dataset: Path) -> None:
    from IPython.display import Image, display

    traj = dataset / "trajectory"
    for name in ("gaze_trajectory_map.png", "gaze_analysis.png", "trajectory_with_directions.png"):
        path = traj / name
        if path.exists():
            print(name)
            display(Image(filename=str(path)))
    csv_path = dataset / "poses" / "gaze_views.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        cols = ["frame", "view_bank_az_deg", "view_elevation_deg", "is_gazing", "distance_m"]
        if "view_bank_az_raw_deg" in df.columns:
            cols.insert(2, "view_bank_az_raw_deg")
        print("gaze_views.csv (ilk 5 satir):")
        display(df[cols].head())
        bank = df["view_bank_az_deg"].astype(float)
        print(
            f"view_bank_az_deg: {bank.min():.1f} .. {bank.max():.1f}  "
            f"(n={len(df)}, gazing={int(df['is_gazing'].sum())})"
        )


dataset = resolve_unity_dataset()
_require_dataset(dataset)

print("\n1/3 Gaze anchor + geometric bank acilari (view_bank_az_deg)...")
meta = ensure_geometric_gaze_views(dataset, write_plot=True)
print("  anchor:", meta["anchor_pos"])
print("  bank az span:", meta.get("bank_az_span_deg"))
print("  angle method:", meta.get("angle_method", "geometric"))
print("  gazing:", meta.get("gazing_frame_count"), "/", meta.get("frame_count"))

print("\n2/3 Trajectory map (view_bank_az_deg)...")
map_path = export_gaze_trajectory_map(dataset, angle_field="view_bank_az_deg")
print("  ->", map_path)

if MARKED_RGB:
    print("\n3/3 RGB overlay (crosshair + view_bank_az_deg)...")
    out_rgb = dataset / "gaze_rgb" / "rgb"
    export_marked_rgb(dataset, out_rgb, gazing_only=False, clean=True)
    print("  ->", out_rgb)

copy_tree_if_requested(dataset, GAZE_OUT_DRIVE, enabled=COPY_TO_DRIVE)

print("\n=== Tamamlandi ===")
print("Cikti:", dataset)
_show_colab_previews(dataset)
