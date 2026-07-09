# 17 — Makale: stil referansi + StyleShot UV texture tablosu + validation
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_DIR = Path("/content/g3d")
OUT_ROOT = Path("/content/ply_styleshot_out")
TEX_DIR = OUT_ROOT / "uv_texture"
VIEW_BANK = OUT_ROOT / "view_bank"
FIG_OUT = Path("/content/paper_texture_figure")

PANEL_SIZE = 640
# ================================

sys.path.insert(0, str(REPO_DIR))
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path, resolve_view_bank

install_sys_path()

from texture_paper_figure import export_texture_paper_figure

view_bank = resolve_view_bank()
if not (TEX_DIR / "polyp_uv_texture.png").is_file():
    raise FileNotFoundError(f"Once Bolum 3: {TEX_DIR / 'polyp_uv_texture.png'}")

print("Texture:", TEX_DIR / "polyp_uv_texture.png")
print("View bank:", view_bank)

summary = export_texture_paper_figure(
    TEX_DIR,
    FIG_OUT,
    panel_size=PANEL_SIZE,
    view_bank_dir=view_bank if view_bank.exists() else None,
)

print("\nMakale tablosu:", summary["table_figure"])
print("Validation:", FIG_OUT / "texture_validation_report.json")
val = summary["validation"]
print(
    f"  luminance_hist_corr={val['luminance_hist_corr']:.3f}  "
    f"deltaE~={val['lab_delta']['deltaE_approx']:.1f}  "
    f"green_frac={val['texture_green_fraction']:.4f}  "
    f"entropy={val['texture_entropy_bits']:.2f}"
)
if "ssim_texture_vs_view0" in val:
    print(f"  ssim_texture_vs_view0={val['ssim_texture_vs_view0']:.3f}")

try:
    from IPython.display import Image as IPImage, display

    display(IPImage(filename=summary["table_figure"]))
except Exception:
    pass

print("\nTamamlandi:", FIG_OUT)
