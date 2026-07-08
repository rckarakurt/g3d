# 13 — Full 360° azimuth preview (0..360, arka gorunumler dahil)
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ============ AYARLAR ============
OUT_ROOT = Path("/content/ply_styleshot_out")
PREVIEW_DIR = OUT_ROOT / "view_preview_full360"

PREVIEW_STEP_DEG = 15
PREVIEW_HALF_SPAN_DEG = 90   # -90 .. +90
ORBIT_MODE = "lumen"

PREVIEW_RENDER_SIZE = 512
PREVIEW_THUMB_HEIGHT = 180
PREVIEW_GRID_COLS = 8
# ================================

import sys
from pathlib import Path

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()
from turntable_render import (
    preview_azimuth_grid,
    render_turntable_views,
    show_view_strip_colab,
)

uv_npz = OUT_ROOT / "mesh_uv" / "polyp_uv.npz"
tex_path = OUT_ROOT / "uv_texture" / "polyp_uv_texture.png"
meta_path = OUT_ROOT / "pipeline_meta.json"

if not uv_npz.exists() or not tex_path.exists():
    raise FileNotFoundError(
        f"UV/texture yok.\n  {uv_npz}\n  {tex_path}\nOnce Bolum 3 (pipeline) calistirin."
    )

obliquity_deg = 18.0
wall_mounted = True
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    obliquity_deg = float(meta.get("obliquity_deg", obliquity_deg))
    wall_mounted = bool(meta.get("wall_mounted", wall_mounted))

pack = np.load(uv_npz)
verts = pack["vertices"]
faces = pack["faces"]
uvs = pack["uvs"]
texture = cv2.cvtColor(cv2.imread(str(tex_path)), cv2.COLOR_BGR2RGB)

azimuths = preview_azimuth_grid(PREVIEW_STEP_DEG, PREVIEW_HALF_SPAN_DEG)
print(f"Onizleme: {ORBIT_MODE} orbit, {len(azimuths)} aci ({azimuths[0]:.0f}..{azimuths[-1]:.0f})")
print("  0=duz yuz (+Z), -90=sol yan, +90=sag yan  (Y ekseni, duvara yapisik)")

manifest = render_turntable_views(
    verts,
    faces,
    uvs,
    texture,
    azimuths,
    PREVIEW_DIR,
    width=PREVIEW_RENDER_SIZE,
    height=PREVIEW_RENDER_SIZE,
    elevation_deg=obliquity_deg,
    distance_scale=2.8,
    wall_mounted=wall_mounted,
    orbit_mode=ORBIT_MODE,
)
print("Manifest:", PREVIEW_DIR / "view_manifest.json")

strip = show_view_strip_colab(
    PREVIEW_DIR,
    thumb_height=PREVIEW_THUMB_HEIGHT,
    max_cols=PREVIEW_GRID_COLS,
    save=True,
    strip_name="view_strip_full360.png",
)
print("Kaydedildi:", PREVIEW_DIR / "view_strip_full360.png")
