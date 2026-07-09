# 16 — Makale: 7 sabit aci composite (Unity mucosa + StyleShot polyp view bank)
from __future__ import annotations

import sys
from pathlib import Path

get_ipython().run_line_magic("pip", "install -q opencv-python-headless scipy")

# ============ AYARLAR ============
# Sabit eslestirme (export_angle_strip_15deg ile ayni):
#   image_0067 -45°  | image_0078 -30° | image_0087 -15° | image_0098 0°
#   image_0121 +15°  | image_0132 +30° | image_0143 +45°

SCALE_BOOST = 8.0
LAB_MATCH = True
COPY_TO_DRIVE = False
OUT_DIR = Path("/content/paper_angle_composites")
# ================================

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()

for _mod in list(sys.modules):
    if _mod in ("gaze_composite", "paper_angle_composites"):
        del sys.modules[_mod]

from colab_content_paths import (
    PLY_OUT_DRIVE,
    UNITY_DATASET,
    copy_tree_if_requested,
    drive_mounted,
    resolve_unity_dataset,
    resolve_view_bank,
)
from paper_angle_composites import PAPER_ANGLE_PAIRS, export_paper_angle_composites
from unity_dataset_angles import ensure_geometric_gaze_views, print_strip_reference_mapping


def show_outputs(out_dir: Path) -> None:
    from IPython.display import Image, display

    strip = out_dir / "paper_pairs_strip_7.png"
    if strip.exists():
        print("Pairs strip (Unity | Polyp | Composite ust uste, 7 kolon):")
        display(Image(filename=str(strip), width=1200))
    comp_strip = out_dir / "paper_composites_strip_7.png"
    if comp_strip.exists():
        print("Composites strip (yan yana):")
        display(Image(filename=str(comp_strip), width=1200))
    singles = sorted((out_dir / "singles").glob("composite_*.png"))
    for path in singles[:3]:
        print(path.name)
        display(Image(filename=str(path), width=480))
    if len(singles) > 3:
        print(f"... +{len(singles) - 3} daha singles/ klasorunde")


dataset = resolve_unity_dataset()
view_bank = resolve_view_bank()
if not (view_bank / "view_manifest.json").exists():
    raise FileNotFoundError(
        f"View bank yok: {view_bank}\nOnce Bolum 3 (StyleShot + view bank) calistirin."
    )
if not (dataset / "rgb").is_dir():
    raise FileNotFoundError(f"Unity dataset yok: {dataset}\nOnce Bolum 5b calistirin.")

print("Dataset:", dataset)
print("View bank:", view_bank)
print("\nGaze acilari (dogrulama):")
ensure_geometric_gaze_views(dataset, write_plot=False, force=True)
print_strip_reference_mapping(dataset)

print("\n=== 7 aci makale composite ===")
for frame, az in PAPER_ANGLE_PAIRS:
    print(f"  image_{frame:04d}.png  <->  polyp {az:+.0f}°")

if OUT_DIR.exists():
    import shutil

    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = export_paper_angle_composites(
    dataset,
    view_bank,
    OUT_DIR,
    scale_boost=SCALE_BOOST,
    lab_match=LAB_MATCH,
)
print("\nTamamlandi.")
print("  singles:", OUT_DIR / "singles")
print("  pairs strip:", OUT_DIR / summary["pairs_strip"])
print("  composites strip:", OUT_DIR / summary["composites_strip"])
print("  composites stack:", OUT_DIR / summary["composites_stack"])
print("  pair stacks:", OUT_DIR / "pairs_stacked")
print("  manifest:", OUT_DIR / "paper_composites_manifest.json")

drive_out = PLY_OUT_DRIVE.parent / "paper_angle_composites"
copy_tree_if_requested(OUT_DIR, drive_out, enabled=COPY_TO_DRIVE)
show_outputs(OUT_DIR)
