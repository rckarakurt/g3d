# 16 — Makale: 7 sentetik composite (polyp mucosa ortasina bindirilmis)
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

get_ipython().run_line_magic("pip", "install -q opencv-python-headless scipy")

REPO_DIR = Path("/content/g3d")

# ============ AYARLAR ============
# Her cift: Unity mucosa + ayni acili textured polyp -> TEK bindirilmis goruntu
#   image_0067 -45° | 0078 -30° | 0087 -15° | 0098 0°
#   0121 +15° | 0132 +30° | 0143 +45°

SCALE_BOOST = 16.0   # polyp capraz cizgisi (px); 12=onceki, 16=biraz daha buyuk
LAB_MATCH = True
COPY_TO_DRIVE = False
OUT_DIR = Path("/content/paper_angle_composites")
# ================================

sys.path.insert(0, str(REPO_DIR))
from vrcaps_colab_bootstrap import bootstrap

bootstrap()

_paper_py = REPO_DIR / "captures" / "paper_angle_composites.py"
if not _paper_py.exists():
    raise FileNotFoundError(f"Eksik: {_paper_py}\nOnce Bolum 0 calistirin.")
_paper_src = _paper_py.read_text(encoding="utf-8")
if "pairs_stacked" in _paper_src or "paper_pairs_strip" in _paper_src:
    raise RuntimeError(
        "Eski paper_angle_composites.py — Bolum 0 git pull veya "
        "!rm -rf /content/g3d sonra Bolum 0."
    )

_rev = subprocess.run(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--oneline"],
    capture_output=True,
    text=True,
    check=False,
)
if _rev.stdout.strip():
    print("g3d:", _rev.stdout.strip())

from colab_content_paths import install_sys_path

install_sys_path()

for _mod in list(sys.modules):
    if _mod in ("gaze_composite", "paper_angle_composites"):
        del sys.modules[_mod]

from colab_content_paths import (
    PLY_OUT_DRIVE,
    copy_tree_if_requested,
    resolve_unity_dataset,
    resolve_view_bank,
)
from paper_angle_composites import PAPER_ANGLE_PAIRS, export_paper_angle_composites
from unity_dataset_angles import ensure_geometric_gaze_views, print_strip_reference_mapping


def show_outputs(out_dir: Path) -> None:
    from IPython.display import Image, display

    legacy = [
        out_dir / "paper_pairs_strip_7.png",
        out_dir / "pairs_stacked",
    ]
    for path in legacy:
        if path.exists():
            print("UYARI: Eski cikti silinmeli — Bolum 0 + Bolum 8 tekrar calistirin:", path)

    strip = out_dir / "paper_composites_strip_7.png"
    if not strip.exists():
        raise FileNotFoundError(f"Composite strip yok: {strip}")
    print("Makale figuru (yalnizca bindirilmis 7 composite, yan yana):")
    display(Image(filename=str(strip), width=1400))

    singles = sorted((out_dir / "singles").glob("composite_*.png"))
    if len(singles) != 7:
        raise RuntimeError(f"Beklenen 7 composite, bulunan: {len(singles)}")
    print(f"\nTekil sentetik goruntuler ({len(singles)} adet — polyp mucosa uzerinde):")
    for path in singles:
        print(f"  {path.name}")
        display(Image(filename=str(path), width=520))


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

print("\n=== 7 sentetik composite (polyp -> mucosa ortasi) ===")
for frame, az in PAPER_ANGLE_PAIRS:
    print(f"  image_{frame:04d}.png  +  polyp {az:+.0f}°  ->  tek bindirilmis PNG")

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
print("  singles (7 PNG):", OUT_DIR / "singles")
strip = summary.get("composites_strip_path") or OUT_DIR / summary["composites_strip"]
print("  ECCV strip PNG:", strip)
if summary.get("composites_strip_pdf"):
    print("  ECCV strip PDF:", summary["composites_strip_pdf"])
if summary.get("angle_pairs_latex"):
    print("  LaTeX table:  ", summary["angle_pairs_latex"])
print("  caption hint: ", summary.get("latex_caption_hint"))
print("  manifest:     ", OUT_DIR / "paper_composites_manifest.json")

drive_out = PLY_OUT_DRIVE.parent / "paper_angle_composites"
copy_tree_if_requested(OUT_DIR, drive_out, enabled=COPY_TO_DRIVE)
show_outputs(OUT_DIR)
